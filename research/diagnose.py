#!/usr/bin/env python3
"""
Stage 1 of the repair: find every place that may be wrong, cut it out, index it.

Nothing is corrected here. Four detectors, all tuned for recall (a false alarm
costs a 20 s clip and one click; a miss costs a hole in the test set):

  spike        one- or two-frame jump of the telemetry increment on any axis
               (distance from the 5-frame median); tagged with whether the
               image confirms it and whether the image is trustworthy there
  tel_jitter   high-frequency (>4 Hz) disagreement between telemetry and image
               where the image is self-consistent
  render_jerk  burst of residual motion (>4 Hz) in a Gyroflow render made with
               the original telemetry -- what the viewer sees, whatever the cause
  image_bad    stretches where the image cannot be trusted: its two roll
               estimators (pure-rotation fit and affine curl) disagree, or too
               few points are tracked (darkness, foliage in wind, sky)

Each event gets a +-10 s window; overlapping windows merge into one segment
(at most 30 s; a longer chain starts a new, overlapping segment). The first and
last EDGE_S = 5 s of every file (take-off, landing) are never looked at, and
segments are kept inside that window on keyframes, since a cut is whole GOPs.
Every segment is cut from the source without re-encoding and with its
telemetry (src/trim_video.py), so it loads in Gyroflow as is; next to it a
1080p H.264 preview of the original stabilized by Gyroflow (defaults), a plot,
and a few event-free control segments per clip.

The clips come from a registry (research/clips.json: path, camera, lens
profile, what the eye says). Expensive steps are cached per clip in --cache:
the image pass (src/measure_rotation.py, with the registry's lens profile),
a Gyroflow render of the original and its residual motion. Event ids are
clip + source time + type, so verdicts survive a rebuild. The file header is
checked against the registry. meta.json records parameters and code version.

    python research/diagnose.py --out artifacts/segments --cache artifacts/cache --gyroflow PATH\Gyroflow.exe
    python research/diagnose.py --out artifacts/segments --reindex      # HTML only, from index.csv
    python research/check_segments.py artifacts/segments                # verify the base

The index (index.html in time order, index_by_severity.html worst first; one
row per event) has an empty verdict: "видно" / "зря поймали" / "не уверен" --
the eye is the ground truth that the metrics are not.
"""
import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from denoise import frame_increments      # noqa: E402
from fix_pipeline import clusters, image_rotations   # noqa: E402
from timing import load_sorted             # noqa: E402

TYPES = ('spike', 'tel_jitter', 'render_jerk', 'image_bad')
COLORS = {'spike': '#8e44ad', 'tel_jitter': '#c0392b', 'render_jerk': '#e67e22', 'image_bad': '#7f8c8d', 'control': '#27ae60'}


EDGE_S = 5.0     # take-off and landing: the first and last seconds of every file are not looked at

SKIP_NAME = ('_stabilized', '_stab', '_diag_orig', '_telemetry_fixed', '_00_control', '_preview', 'orig_render')


def usable(video):
    """(ok, reason): Gyroflow outputs, our own sidecars and clips without DJI motion data are skipped."""
    stem = os.path.splitext(os.path.basename(video))[0].lower()
    if any(stem.endswith(s) or (s + '_') in stem for s in SKIP_NAME):
        return False, 'derived file (%s)' % stem
    try:
        _, samples, frames, _, _ = load_sorted(video)
    except SystemExit as exc:
        return False, str(exc)
    except Exception as exc:                          # no djmd track, not an MP4, ...
        return False, '%s: %s' % (type(exc).__name__, exc)
    if not frames or len(samples) < 100:
        return False, 'no usable telemetry'
    return True, ''


# ----------------------------------------------------------------- measurement
def running_rms(x, win):
    k = np.ones(win) / win
    if x.ndim == 1:
        return np.sqrt(np.convolve(x ** 2, k, mode='same'))
    return np.sqrt(np.column_stack([np.convolve(x[:, i] ** 2, k, mode='same') for i in range(x.shape[1])]).sum(1))


def band_gain(tel, img, ok, fps, band=(2.0, 4.0), min_run=64, trim=15):
    runs = [r for r in clusters(ok, gap=0) if len(r) >= min_run]
    if sum(len(r) - 2 * trim for r in runs) < 300:
        return 1.0
    sos = butter(2, list(band), btype='band', fs=fps, output='sos')
    X = np.concatenate([sosfiltfilt(sos, tel[r[0]:r[-1] + 1])[trim:-trim] for r in runs])
    Y = np.concatenate([sosfiltfilt(sos, img[r[0]:r[-1] + 1])[trim:-trim] for r in runs])
    g = float(np.dot(X, Y) / np.dot(X, X))
    return g if 0.5 < g < 2.0 else 1.0


def render_measure(path, scale=0.5):
    """Per-pair residual motion of a render: shift x/y (px of this render) and roll (deg)."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, f = cap.read()
    H, W = f.shape[:2]
    g0 = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g1 = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(g0, 1500, 0.01, 8)
        row = (np.nan,) * 3
        if p0 is not None and len(p0) >= 30:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(31, 31), maxLevel=4)
            pb, sb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(31, 31), maxLevel=4)
            good = (st.ravel() == 1) & (sb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.5)
            a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
            if len(a) >= 30:
                M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.0)
                if M is not None:
                    c = np.array([W * scale / 2, H * scale / 2])
                    cc = M[:, :2] @ c + M[:, 2] - c
                    row = (cc[0] / scale, cc[1] / scale, np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        out.append(row)
        g0 = g1
    return fps, W, H, np.array(out)


def render_original(video, gyroflow, workdir):
    """Render the original at 1080p with Gyroflow's defaults and return the path."""
    base = os.path.splitext(video)[0]
    produced = base + '_diag_orig.mp4'
    params = "{ 'codec': 'H.265/HEVC', 'bitrate': 60, 'use_gpu': true, 'audio': false, 'output_width': 1920, 'output_height': 1080 }"
    for _ in range(3):
        with open(os.path.join(workdir, 'gyroflow_render.log'), 'w') as fh:
            subprocess.run([gyroflow, video, '-f', '-t', '_diag_orig', '-p', params], stdout=fh, stderr=subprocess.STDOUT)
        if os.path.isfile(produced) and os.path.getsize(produced) > 20e6:
            dst = os.path.join(workdir, os.path.basename(produced))
            shutil.move(produced, dst)
            return dst
    raise SystemExit('Gyroflow render failed for %s' % video)


def load_render(resid_npz, n):
    z = np.load(resid_npz)
    r = np.full((n, 3), np.nan)
    k = min(n, len(z['r']))
    r[:k] = z['r'][:k]
    s = 3840.0 / float(z['W'])                        # px of a 4K frame
    r[:, :2] *= s
    r[:, 2] = np.radians(r[:, 2]) * 3840.0 / 2        # roll as px at the frame edge
    return r


def analyse(video, image_npz, resid_npz=None, spike_deg=0.35, jitter_deg=0.05, kc_deg=0.03, min_pts=300,
            render_px=4.0, render_rel=3.0):
    clip, samples, frames, ts, qs = load_sorted(video)
    fps = float(clip.fps)
    n = len(frames)
    ft = np.arange(n + 1) * (1e6 / fps)
    inc = frame_increments(ts, qs, ft)
    rate = np.linalg.norm(inc, axis=1) * fps
    valid = (ft[:-1] >= ts[0]) & (ft[1:] <= ts[-1]) & (rate < 1000.0)
    valid[-10:] = False                                # the last DJI blocks carry junk
    edge = int(round(EDGE_S * fps))
    valid[:edge] = False                               # take-off
    valid[-edge:] = False                              # landing
    d = np.load(image_npz)
    inc_img, curl, _ = image_rotations(d, n)
    npts = np.full(n, np.nan)
    npts[:min(n, len(d['npts']))] = d['npts'][:n]
    has = np.isfinite(inc_img).all(1) & valid
    hp = butter(2, 4.0, btype='high', fs=fps, output='sos')
    H = lambda x: sosfiltfilt(hp, np.where(np.isfinite(x), x, 0.0), axis=0)   # noqa: E731
    win = max(5, int(round(0.5 * fps)))
    # image quality: the two independent roll estimators must agree above 4 Hz
    kc = running_rms(np.where(has, H(inc_img[:, 2]) - H(curl), 0.0), win)
    img_ok = has & (kc < kc_deg) & (np.nan_to_num(npts) >= min_pts)
    ok_fit = img_ok & (rate < 150)
    gains = np.array([band_gain(inc[:, k], np.nan_to_num(inc_img[:, k]), ok_fit, fps) for k in range(3)])
    img = np.where(has[:, None], inc_img / gains, np.nan)
    # telemetry-vs-image disagreement above 4 Hz
    e = np.where(has[:, None], H(img) - H(inc), 0.0)
    jit = running_rms(e, win)
    # spikes on all axes
    med = np.column_stack([median_filter(np.where(valid, inc[:, k], 0.0), size=5, mode='nearest') for k in range(3)])
    r = inc - med
    rn = np.linalg.norm(r, axis=1)
    witness = np.nan_to_num(img - inc)
    confirmed = np.sum(witness * r, axis=1) < -0.5 * rn ** 2
    spike = (rn >= spike_deg) & valid & (rate < 200)
    series = dict(fps=fps, n=n, t=np.arange(n) / fps, rate=rate, valid=valid, inc=inc, img=img, jit=jit, kc=kc,
                  npts=npts, img_ok=img_ok, rn=rn, gains=gains)
    flags = {'spike': spike, 'tel_jitter': (jit >= jitter_deg) & img_ok & (rate < 150) & valid,
             'image_bad': ~img_ok & valid & (rate < 100)}
    info = {'spike': dict(value=rn, confirmed=confirmed), 'tel_jitter': dict(value=jit), 'image_bad': dict(value=kc)}
    if resid_npz:
        rr = load_render(resid_npz, n)
        okr = np.isfinite(rr).all(1)
        bp = butter(2, [4.0, min(24.5, fps / 2 * 0.98)], btype='band', fs=fps, output='sos')
        pos = sosfiltfilt(bp, np.cumsum(np.where(okr[:, None], rr, 0.0), 0), axis=0)
        rj = running_rms(pos, win)
        calm = okr & (rate < 45) & valid
        base = float(np.median(rj[calm])) if calm.sum() > 100 else float(np.median(rj[okr]))
        series.update(render_jit=rj, render_base=base)
        # above ~60 deg/s the frames are blurred and the tracker itself disagrees with itself by 2x
        flags['render_jerk'] = (rj >= max(render_px, render_rel * base)) & okr & (rate < 60) & valid
        info['render_jerk'] = dict(value=rj)
    return clip, series, flags, info


# ----------------------------------------------------------------- events and segments
def find_events(series, flags, info):
    fps = series['fps']
    ev = []
    for kind, mask in flags.items():
        min_len = 1 if kind == 'spike' else int(0.2 * fps)
        if kind == 'image_bad':
            min_len = int(0.5 * fps)
        for run in clusters(mask, gap=int(0.25 * fps)):
            if len(run) < min_len:
                continue
            v = info[kind]['value'][run]
            k = run[int(np.argmax(v))]
            e = dict(type=kind, t=float(k / fps), t0=float(run[0] / fps), t1=float((run[-1] + 1) / fps),
                     severity=float(v.max()), rate=float(series['rate'][k]),
                     image_ok=float(series['img_ok'][run].mean()))
            if kind == 'spike':
                e['confirmed_by_image'] = bool(info['spike']['confirmed'][k])
                e['axis'] = 'xyz'[int(np.argmax(np.abs(series['inc'][k] - np.nan_to_num(np.median(series['inc'][max(0, k - 2):k + 3], axis=0)))))]
            ev.append(e)
    return sorted(ev, key=lambda e: e['t'])


def usable_window(video, duration):
    """(lo, hi): the stretch a segment may cover -- EDGE_S away from both ends of the file, on keyframes,
    because the cut is made of whole GOPs (the start moves back to a keyframe, the end forward to one)."""
    from mp4parse import open_mp4
    from trim_video import inspect as mp4_inspect
    f, boxes, moov, tracks = open_mp4(video)
    try:
        _, step, _, keys = mp4_inspect(f, boxes, moov, tracks)
    finally:
        f.close()
    times = [float(k * step) for k in keys]
    lo = min((t for t in times if t >= EDGE_S), default=EDGE_S)
    hi = max((t for t in times if t <= duration - EDGE_S), default=duration - EDGE_S)
    return lo, hi


def make_segments(events, duration, pad=10.0, max_len=30.0, lo=0.0, hi=None):
    hi = duration if hi is None else hi
    segs = []
    for e in events:
        a, b = max(lo, e['t0'] - pad), min(hi, e['t1'] + pad)
        if segs and a <= segs[-1]['end'] and max(b, segs[-1]['end']) - segs[-1]['start'] <= max_len:
            segs[-1]['end'] = max(segs[-1]['end'], b)
            segs[-1]['events'].append(e)
        else:
            segs.append(dict(start=a, end=b, events=[e]))
    return segs


def control_segments(events, duration, count=2, length=20.0, pad=2.0, lo=5.0, hi=None):
    """Event-free stretches: nothing flagged inside the segment or within `pad` s of it."""
    hi = duration - 5.0 if hi is None else hi
    busy = [(e['t0'] - pad, e['t1'] + pad) for e in events]
    out = []
    t = lo
    free = []
    while t + length <= hi:
        if not any(a < t + length and t < b for a, b in busy):
            free.append(t)
        t += 1.0
    if not free:
        return out
    picks = np.linspace(0, len(free) - 1, min(count, len(free))).round().astype(int)
    last = -1e9
    for p in picks:
        s = free[p]
        if s - last < length:
            continue
        out.append(dict(start=s, end=s + length, events=[dict(type='control', t=s + length / 2, t0=s, t1=s + length,
                                                              severity=0.0, rate=0.0, image_ok=1.0)]))
        last = s
    return out


# ----------------------------------------------------------------- outputs
def plot_segment(series, seg, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fps = series['fps']
    a, b = int(seg['start'] * fps), min(series['n'], int(seg['end'] * fps))
    t = series['t'][a:b]
    rows = 4 if 'render_jit' in series else 3
    fig, ax = plt.subplots(rows, 1, figsize=(12, 2.1 * rows), sharex=True)
    for k, name in enumerate(('pitch x', 'yaw y', 'roll z')):
        ax[0].plot(t, series['inc'][a:b, k] + 0.6 * k, lw=0.8, color='C%d' % k, label='telemetry ' + name)
        ax[0].plot(t, series['img'][a:b, k] + 0.6 * k, lw=0.8, color='k', alpha=0.45)
    ax[0].set_ylabel('deg/frame\n(offset per axis)')
    ax[0].legend(loc='upper right', fontsize=7, ncol=3)
    ax[0].set_title(title + '   (black: image)', fontsize=9)
    ax[1].plot(t, series['jit'][a:b], lw=0.9, color=COLORS['tel_jitter'], label='telemetry vs image >4 Hz')
    ax[1].plot(t, series['rn'][a:b], lw=0.6, color=COLORS['spike'], label='jump from 5-frame median')
    ax[1].set_ylabel('deg')
    ax[1].legend(loc='upper right', fontsize=7)
    ax[2].plot(t, series['kc'][a:b], lw=0.9, color=COLORS['image_bad'], label='image self-disagreement (kabsch vs curl)')
    ax2 = ax[2].twinx()
    ax2.plot(t, series['npts'][a:b], lw=0.6, color='C0', alpha=0.6)
    ax2.set_ylabel('points', fontsize=7)
    ax[2].set_ylabel('deg')
    ax[2].legend(loc='upper left', fontsize=7)
    if rows == 4:
        ax[3].plot(t, series['render_jit'][a:b], lw=0.9, color=COLORS['render_jerk'], label='Gyroflow render jitter >4 Hz (px, 4K)')
        ax[3].axhline(series['render_base'], color='k', lw=0.5, ls=':')
        ax[3].set_ylabel('px')
        ax[3].legend(loc='upper right', fontsize=7)
    for e in seg['events']:
        for x in ax:
            x.axvspan(e['t0'], max(e['t1'], e['t0'] + 1 / fps), color=COLORS[e['type']], alpha=0.18, lw=0)
    ax[-1].set_xlabel('time in the source clip, s')
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


class H264Writer:
    """Browser-playable H.264 MP4 (yuv420p, faststart) through PyAV; OpenCV's mp4v does not play in browsers."""

    def __init__(self, path, fps, width, height, crf=21):
        import av
        from fractions import Fraction
        self.av = av
        self.c = av.open(path, 'w', options={'movflags': 'faststart'})
        rate = Fraction(fps).limit_denominator(1001)
        for name, opts in (('h264_nvenc', {'preset': 'p5', 'cq': str(crf + 2)}), ('libx264', {'preset': 'veryfast', 'crf': str(crf)})):
            try:
                self.s = self.c.add_stream(name, rate=rate, options=opts)
                self.s.width, self.s.height, self.s.pix_fmt = width, height, 'yuv420p'
                self.s.codec_context.open()
                break
            except Exception:
                self.c.close()
                self.c = av.open(path, 'w', options={'movflags': 'faststart'})
        else:
            raise RuntimeError('no H.264 encoder available in PyAV')

    def write(self, bgr):
        for pkt in self.s.encode(self.av.VideoFrame.from_ndarray(bgr, format='bgr24')):
            self.c.mux(pkt)

    def close(self):
        for pkt in self.s.encode():
            self.c.mux(pkt)
        self.c.close()


def cut_preview(render_path, fps, begin_s, end_s, out):
    import cv2
    cap = cv2.VideoCapture(render_path)
    a, b = int(round(begin_s * fps)), int(round(end_s * fps))
    for _ in range(a):
        cap.grab()
    ok, f = cap.read()
    if not ok:
        return False
    h, w = f.shape[:2]
    vw = H264Writer(out, fps, w, h)
    i = a
    while ok and i < b:
        cv2.putText(f, '%.2f s' % (i / fps - begin_s), (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        vw.write(f)
        ok, f = cap.read()
        i += 1
    vw.close()
    return True


HTML_HEAD = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Problem segments</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--row:#f7f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#e8e8ea;--mut:#9a9aa0;--line:#333;--row:#1f1f22}}
body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:16px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:6px;vertical-align:top;text-align:left}
tr.seg td{background:var(--row);font-weight:600}.mut{color:var(--mut)}img{max-width:560px;width:100%}
.tag{display:inline-block;padding:1px 6px;border-radius:4px;color:#fff;font-size:12px}
select,input{font:inherit}button{font:inherit;padding:6px 12px;margin:8px 0}
</style></head><body>
<h1>Problem segments</h1>
<p class="mut">One row per detected event. Open the source segment in Gyroflow (it carries its telemetry) or watch the
stabilized preview, find the event at the given time, and set the verdict. Verdicts are kept in this browser; use
&laquo;Export CSV&raquo; to save them.</p>
<button onclick="exportCsv()">Export CSV</button>
<table><thead><tr><th>segment</th><th>time in segment</th><th>type</th><th>details</th><th>verdict</th><th>comment</th></tr></thead><tbody>
"""

HTML_TAIL = """</tbody></table><button onclick="exportCsv()">Export CSV</button>
<script>
function key(el){return 'verdict:'+el.dataset.id}
document.querySelectorAll('[data-id]').forEach(el=>{
  try{const v=localStorage.getItem(key(el)+':'+el.tagName);if(v!==null)el.value=v}catch(e){}
  el.addEventListener('change',()=>{try{localStorage.setItem(key(el)+':'+el.tagName,el.value)}catch(e){}});
});
function exportCsv(){
  const rows=[['event_id','verdict','comment']];
  document.querySelectorAll('select[data-id]').forEach(s=>{
    const c=document.querySelector('input[data-id="'+s.dataset.id+'"]');
    rows.push([s.dataset.id,s.value,(c?c.value:'').replace(/"/g,'""')]);});
  const csv=rows.map(r=>r.map(x=>'"'+x+'"').join(',')).join('\\n');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download='verdicts.csv';a.click();}
</script></body></html>"""


# how much one event counts towards a segment's badness: severity over the detector threshold;
# image_bad is not a defect, only a warning that the image cannot be trusted there
SCORE_UNIT = {'spike': 0.35, 'tel_jitter': 0.05, 'render_jerk': 4.0}


def event_score(r):
    unit = SCORE_UNIT.get(r['type'])
    return float(r['severity']) / unit if unit else 0.0


def write_html(rows, path, scores=None):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(HTML_HEAD)
        last = None
        for r in rows:
            if r['segment'] != last:
                last = r['segment']
                score = ('score %.1f (%d defects) &nbsp; ' % scores[last]) if scores else ''
                fh.write('<tr class="seg"><td colspan="6">%s%s &mdash; %s, %.1f&ndash;%.1f s &nbsp; '
                         '%s &middot; <a href="%s">stabilized preview</a><br>'
                         '<a href="%s"><img src="%s" loading="lazy"></a></td></tr>\n' % (
                             score, html.escape(r['segment']), html.escape(os.path.basename(r['source'])),
                             float(r['seg_start_s']), float(r['seg_end_s']),
                             ('<a href="%s">source segment</a>' % html.escape(r['segment_file'].replace('\\', '/'))) if r['segment_file']
                             else ('source: <a href="file:///%s">full file</a> from %.1f s (this layout cannot be cut losslessly)' % (
                                 html.escape(r['source']), float(r['seg_start_s']))),
                             html.escape(r['preview'].replace('\\', '/')),
                             html.escape(r['plot'].replace('\\', '/')), html.escape(r['plot'].replace('\\', '/'))))
            det = 'severity %s, %s deg/s, image ok %s' % (r['severity'], r['rate_deg_s'], r['image_ok_frac'])
            if r['type'] == 'spike':
                det += ', axis %s, image %s' % (r['axis'], 'confirms' if str(r['confirmed_by_image']) == 'True' else 'does not confirm')
            eid = html.escape(r['event_id'])
            fh.write('<tr><td class="mut">%s</td><td>%.2f s</td><td><span class="tag" style="background:%s">%s</span></td>'
                     '<td class="mut">%s</td><td><select data-id="%s"><option></option><option>видно</option>'
                     '<option>зря поймали</option><option>не уверен</option></select></td>'
                     '<td><input data-id="%s" size="30"></td></tr>\n' % (
                         eid, float(r['t_in_segment_s']), COLORS[r['type']], r['type'], det, eid, eid))
        fh.write(HTML_TAIL)


def write_indexes(rows, out):
    """index.html in time order; index_by_severity.html with the worst segments first."""
    write_html(rows, os.path.join(out, 'index.html'))
    scores = {}
    for r in rows:
        sc, nd = scores.get(r['segment'], (0.0, 0))
        scores[r['segment']] = (sc + event_score(r), nd + (r['type'] in SCORE_UNIT))
    order = sorted(scores, key=lambda s: -scores[s][0])
    rank = {s: i for i, s in enumerate(order)}
    ranked = sorted(rows, key=lambda r: (rank[r['segment']], -event_score(r), float(r['t_in_segment_s'])))
    write_html(ranked, os.path.join(out, 'index_by_severity.html'), scores)


CAMERA = {'DJI O4P': 'O4 Pro', 'DJI O4': 'O4 Lite', 'DJI FC8383': 'O3 (FC8383)'}
PARAMS = dict(edge_s=EDGE_S, pad_s=10.0, max_segment_s=30.0, spike_deg=0.35, jitter_deg=0.05, image_kc_deg=0.03,
              image_min_points=300, render_px=4.0, render_rel=3.0, render_rate_max=60.0, controls_per_clip=2, control_margin_s=2.0)


def image_complete(path, entry=None):
    """measure_rotation saves its cache as it goes; a finished file has values up to its end.
    With a registry entry, it must also have been measured on that video with that lens."""
    try:
        d = np.load(path)
        r = d['roll_deg']
        if not (len(r) > 100 and bool(np.isfinite(r[-max(10, len(r) // 20):]).any())):
            return False
        if entry is not None:
            m = json.loads(str(d['meta_json']))
            if os.path.basename(m['video']).lower() != os.path.basename(entry['path']).lower():
                return False
            lens = m.get('lens_source')
            custom = bool(lens) and lens != 'clip metadata'          # older files wrote None for the stock model
            if custom != bool(entry.get('lens_profile')):
                return False
        return True
    except Exception:
        return False


def prepare(entry, cache, gyroflow, root):
    """Fill the cache of one registry entry: image pass, render of the original, its residual motion."""
    d = os.path.join(cache, entry['name'])
    os.makedirs(d, exist_ok=True)
    image, render, resid = (os.path.join(d, f) for f in ('image.npz', 'orig_render.mp4', 'orig_resid.npz'))
    if not image_complete(image, entry):
        print('[%s] image pass (one decode of the whole clip)...' % entry['name'], flush=True)
        cmd = [sys.executable, os.path.join(root, 'src', 'measure_rotation.py'), entry['path'], '-o', image]
        if entry.get('lens_profile'):
            cmd += ['--lens', os.path.join(root, entry['lens_profile'])]
        subprocess.run(cmd, check=True)
    if not os.path.isfile(render):
        if not gyroflow:
            print('[%s] no render of the original and no --gyroflow: render_jerk and previews skipped' % entry['name'])
            return image, None, None
        print('[%s] rendering the original in Gyroflow...' % entry['name'], flush=True)
        os.replace(render_original(entry['path'], gyroflow, d), render)
    if not os.path.isfile(resid):
        print('[%s] measuring the render...' % entry['name'], flush=True)
        fps_r, W, H, r = render_measure(render)
        np.savez(resid, fps=fps_r, W=W, H=H, r=r)
    return image, resid, render


def header_check(entry, clip):
    h = clip.header or {}
    found = dict(camera=CAMERA.get(h.get('product_name'), h.get('product_name')), firmware=h.get('product_firmware_version'),
                 serial=h.get('product_sn'), readout_ms=clip.frame_readout_time_ms)
    problems = [k for k in ('camera', 'firmware') if entry.get(k) and found[k] and entry[k] != found[k]]
    if entry.get('serial') and found['serial'] and not str(found['serial']).startswith(entry['serial']):
        problems.append('serial')
    return found, problems


def git_rev(root):
    try:
        return subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--registry', default=os.path.join(HERE, 'clips.json'))
    ap.add_argument('--only', nargs='*', help='names from the registry (default: all)')
    ap.add_argument('--out', required=True, help='the segment base')
    ap.add_argument('--cache', required=True, help='per-clip cache: image pass, render of the original, its residual')
    ap.add_argument('--gyroflow', default=os.environ.get('GYROFLOW'), help='Gyroflow.exe, to render originals not yet in the cache')
    ap.add_argument('--no-cut', action='store_true', help='only detect and index, do not cut video')
    ap.add_argument('--reindex', action='store_true', help='only rebuild the HTML indexes from OUT/index.csv')
    a = ap.parse_args()
    if a.reindex:
        with open(os.path.join(a.out, 'index.csv'), encoding='utf-8-sig') as fh:
            write_indexes(list(csv.DictReader(fh)), a.out)
        print('indexes rebuilt in %s' % a.out)
        return
    from trim_video import trim
    root = os.path.abspath(os.path.join(HERE, '..'))
    with open(a.registry, encoding='utf-8') as fh:
        registry = json.load(fh)
    entries = [c for c in registry['clips'] if not a.only or c['name'] in a.only]
    os.makedirs(a.out, exist_ok=True)
    all_rows, meta_clips = [], {}
    for entry in entries:
        name, video = entry['name'], entry['path']
        ok, why = usable(video)
        if not ok:
            print('[%s] skipped: %s' % (name, why), flush=True)
            meta_clips[name] = dict(skipped=why)
            continue
        image_npz, resid, render_path = prepare(entry, a.cache, a.gyroflow, root)
        print('[%s] analysing...' % name, flush=True)
        clip, series, flags, info = analyse(video, image_npz, resid, spike_deg=PARAMS['spike_deg'],
                                            jitter_deg=PARAMS['jitter_deg'], kc_deg=PARAMS['image_kc_deg'],
                                            min_pts=PARAMS['image_min_points'], render_px=PARAMS['render_px'],
                                            render_rel=PARAMS['render_rel'])
        found, problems = header_check(entry, clip)
        if problems:
            print('[%s] WARNING: registry disagrees with the file header on %s: %s' % (name, problems, found), flush=True)
        duration = series['n'] / series['fps']
        try:
            lo, hi = usable_window(video, duration)
            cut_ok, cut_why = True, ''
        except ValueError as exc:
            # a layout src/trim_video.py refuses to cut losslessly (e.g. O3: variable-step djmd track);
            # such clips get preview and plot only, and the window is not keyframe-aligned
            lo, hi = EDGE_S, duration - EDGE_S
            cut_ok, cut_why = False, str(exc)
            print('[%s] no lossless cut for this layout: preview and plot only' % name, flush=True)
        # the usable window is keyframe-aligned (lo >= EDGE_S); an event outside it could only be
        # shown in a segment that reaches into the take-off or landing
        events = [e for e in find_events(series, flags, info) if lo <= e['t'] <= hi]
        segs = make_segments(events, duration, pad=PARAMS['pad_s'], max_len=PARAMS['max_segment_s'], lo=lo, hi=hi)
        segs += control_segments(events, duration, count=PARAMS['controls_per_clip'], pad=PARAMS['control_margin_s'], lo=lo, hi=hi)
        segs.sort(key=lambda s: s['start'])
        counts = {k: sum(e['type'] == k for e in events) for k in TYPES}
        print('[%s] %d events %s -> %d segments' % (name, len(events), counts, len(segs)), flush=True)
        cdir = os.path.join(a.out, name)
        os.makedirs(cdir, exist_ok=True)
        for seg in segs:
            begin, end = seg['start'], seg['end']
            src_cut = os.path.join(cdir, '_cut.MP4')
            if not a.no_cut and cut_ok:
                if os.path.exists(src_cut):
                    os.remove(src_cut)
                rep = trim(video, '%.3f' % seg['start'], '%.3f' % seg['end'], src_cut)
                begin, end = rep['start_s'], rep['end_s']
            sid = '%s_%06.1f' % (name, begin)                      # stable: clip + cut start in the source
            base = os.path.join(cdir, '%s-%06.1f' % (sid, end))
            if not a.no_cut:
                if cut_ok:
                    os.replace(src_cut, base + '.MP4')
                if render_path:
                    cut_preview(render_path, series['fps'], begin, end, base + '_preview.mp4')
            seg.update(id=sid, begin=begin, finish=end)
            plot_segment(series, seg, base + '.png', '%s  %.1f-%.1f s of %s' % (sid, begin, end, os.path.basename(video)))
            for e in seg['events']:
                all_rows.append(dict(
                    event_id='%s_t%07.2f_%s' % (name, e['t'], e['type']),   # stable across rebuilds
                    segment=sid, clip=name, eye=entry.get('eye', ''), camera=found['camera'], source=video,
                    segment_file=os.path.relpath(base + '.MP4', a.out) if cut_ok else '', preview=os.path.relpath(base + '_preview.mp4', a.out),
                    plot=os.path.relpath(base + '.png', a.out), seg_start_s=round(begin, 3), seg_end_s=round(end, 3),
                    t_source_s=round(e['t'], 2), t_in_segment_s=round(e['t'] - begin, 2), type=e['type'],
                    severity=round(e['severity'], 4), rate_deg_s=round(e['rate'], 1), image_ok_frac=round(e['image_ok'], 2),
                    confirmed_by_image=e.get('confirmed_by_image', ''), axis=e.get('axis', ''), verdict='', comment=''))
        meta_clips[name] = dict(entry=entry, header=found, header_problems=problems, duration_s=duration,
                                cut=cut_ok, cut_problem=cut_why,
                                window_s=[lo, hi], gains=series['gains'].tolist(), render_baseline_px=series.get('render_base'),
                                counts=counts, segments=len(segs), cache=os.path.join(a.cache, name))
        with open(os.path.join(cdir, 'events.json'), 'w', encoding='utf-8') as fh:
            json.dump(dict(meta_clips[name], events=events), fh, indent=1, ensure_ascii=False)
    import datetime
    meta_path, csv_path = os.path.join(a.out, 'meta.json'), os.path.join(a.out, 'index.csv')
    meta = dict(built=datetime.datetime.now().isoformat(timespec='seconds'), code=git_rev(root), params=PARAMS,
                registry=os.path.abspath(a.registry), clips={})
    if a.only and os.path.isfile(meta_path) and os.path.isfile(csv_path):
        # a partial build adds to the base: the other clips' rows and metadata are kept
        old = json.load(open(meta_path, encoding='utf-8'))
        if old.get('params') != PARAMS:
            print('WARNING: the base was built with other parameters: %s' % old.get('params'))
        meta['clips'] = {k: v for k, v in old['clips'].items() if k not in meta_clips}
        with open(csv_path, encoding='utf-8-sig') as fh:
            all_rows = [r for r in csv.DictReader(fh) if r['clip'] not in meta_clips] + all_rows
    meta['clips'].update(meta_clips)
    write_csv(all_rows, csv_path)
    write_indexes(all_rows, a.out)
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    print('index: %s (%d events)' % (os.path.join(a.out, 'index.html'), len(all_rows)))


if __name__ == '__main__':
    main()

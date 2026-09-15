#!/usr/bin/env python3
"""
Judge a telemetry file against the picture, without rendering anything.

    python src/verify_fix.py <video> --image <clip_image.npz> <sidecar.mp4> [<sidecar2.mp4> ...]

For every file (the source video's own telemetry is always the first column):

  timing   the content shift, in ms, that best matches the image roll and pitch,
           read at the frame instants Gyroflow uses (pts). A fixed file should sit
           at 0 +- 2 ms.
  roll     frame-to-frame jitter of the roll rate on calm / medium / fast frames
           against the image floor, plus the rms discrepancy against the image roll
           (the numbers HANDOFF.md section 9 uses).
  pitch/yaw  per 5-frame window: rms and p99 of |image - telemetry| where both image
           estimates agree, and the list of windows above the event threshold.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import slerp_series                    # noqa: E402
from denoise import frame_increments              # noqa: E402
from fix_pipeline import image_rotations, scalar_gain, window_disagreement   # noqa: E402
from timing import load_sorted                    # noqa: E402

CORNER_PX_PER_DEG = 2400 * np.pi / 180


def series_at_pts(path, fps, n_frames, shift_ms=0.0):
    clip, samples, frames, ts, qs = load_sorted(path)
    ft = np.arange(n_frames + 1) * (1e6 / fps) + shift_ms * 1000.0
    return slerp_series(ts, qs, ft), frame_increments(ts, qs, ft)


def timing_scan(path, fps, n_frames, img_roll, uh, ugain, rate_mask, shifts=np.arange(-12, 12.01, 1.0)):
    best = None
    res = []
    for sh in shifts:
        _, inc = series_at_pts(path, fps, n_frames, sh)
        tel_roll = inc @ uh
        m = rate_mask & np.isfinite(img_roll)
        r = np.sqrt(np.mean((img_roll[m] / ugain - tel_roll[m]) ** 2))
        res.append(r)
    res = np.array(res)
    k = int(np.argmin(res))
    if 0 < k < len(res) - 1:
        y0, y1, y2 = res[k - 1], res[k], res[k + 1]
        frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    else:
        frac = 0.0
    return float(shifts[k] + frac * (shifts[1] - shifts[0])), float(res[k]), float(res[len(res) // 2])


def jitter_table(inc, img_roll, uh, ugain, rate):
    tel_roll = inc @ uh
    im = img_roll / ugain
    out = {}
    for name, m in (('calm<20', rate < 20), ('20-60', (rate >= 20) & (rate < 60)), ('fast>60', rate >= 60)):
        mm = m[:-1] & m[1:] & np.isfinite(im[:-1]) & np.isfinite(im[1:])
        out[name] = dict(frames=int(mm.sum()),
                         tel_jitter=float(np.diff(tel_roll)[mm].std()) if mm.sum() > 5 else float('nan'),
                         img_jitter=float(np.diff(im)[mm].std()) if mm.sum() > 5 else float('nan'),
                         disc_rms=float((im - tel_roll)[m & np.isfinite(im)].std()) if (m & np.isfinite(im)).sum() > 5 else float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('--image', required=True)
    ap.add_argument('files', nargs='*')
    ap.add_argument('--event-thr', type=float, default=0.12)
    ap.add_argument('--json')
    a = ap.parse_intermixed_args()

    clip, samples, frames, ts, qs = load_sorted(a.video)
    fps, n = float(clip.fps), len(frames)
    d = np.load(a.image)
    gap = int(json.loads(str(d['meta_json']))['gap'])
    inc_img, roll_curl, src_name = image_rotations(d, n)
    img_roll = inc_img[:, 2].copy() if inc_img is not None else roll_curl

    # roll = z of the telemetry (camera) frame; gain from the source telemetry read the
    # native way (pts + readout/2) on the fast frames
    ft = np.arange(n + 1) * (1e6 / fps) + clip.frame_readout_time_ms * 500.0
    inc0 = frame_increments(ts, qs, ft)
    rate0 = np.linalg.norm(inc0, axis=1) * fps
    uh = np.array([0.0, 0.0, 1.0])
    ugain, ucorr, un = scalar_gain(inc0[:, 2], img_roll, rate0)
    if not np.isfinite(ucorr) or ucorr < 0.8:
        ugain = 1.0
    print('image roll from %s: gain %.3f corr %.3f (%d fast frames); image on %d/%d frames; gap %d' % (
        src_name, ugain, ucorr, un, np.isfinite(img_roll).sum(), n, gap))

    results = {}
    for path in [a.video] + list(a.files):
        label = os.path.basename(path)
        qf, inc = series_at_pts(path, fps, n)
        rate = np.linalg.norm(inc, axis=1) * fps
        rate[-8:] = 1e6                                  # junk end blocks, see fix_pipeline
        rate[rate >= 1000] = 1e6
        fast = (rate > 30) & (rate < 1000)
        sh, rbest, r0 = timing_scan(path, fps, n, img_roll, uh, ugain, fast)
        jit = jitter_table(inc, img_roll, uh, ugain, rate)
        w = window_disagreement(qf, d, fps, gap, rate)
        rel = w['reliable']
        ax_pt = w['D'] - np.outer(np.nan_to_num(w['D']) @ uh, uh)
        Dpt = np.linalg.norm(np.nan_to_num(ax_pt), axis=1) / gap
        ev = np.flatnonzero(rel & (Dpt >= a.event_thr))
        res = dict(timing_best_shift_ms=sh, timing_rms_at_best=rbest, timing_rms_at_zero=r0, roll=jit,
                   pitch_yaw=dict(windows=int(rel.sum()), rms_deg_per_frame=float(np.sqrt(np.mean(Dpt[rel] ** 2))) if rel.any() else None,
                                  p99_deg_per_frame=float(np.percentile(Dpt[rel], 99)) if rel.any() else None,
                                  windows_over_threshold=int(len(ev)),
                                  over_threshold_times_s=[round(float(i / fps), 2) for i in ev[::max(1, len(ev) // 40)]]))
        results[label] = res
        print('\n== %s' % label)
        print('  timing     : best content shift %+.2f ms at pts (rms %.3f vs %.3f at 0)' % (sh, rbest, r0))
        for band, v in jit.items():
            print('  roll %-8s: jitter tel %.4f  image %.4f  deg/frame (%.1f px corner)  disc rms %.4f  [%d frames]' % (
                band, v['tel_jitter'], v['img_jitter'], v['tel_jitter'] * CORNER_PX_PER_DEG, v['disc_rms'], v['frames']))
        print('  pitch/yaw  : %d vetted windows, |image-tel| rms %.4f p99 %.4f deg/frame; %d windows over %.2f deg/frame' % (
            rel.sum(), res['pitch_yaw']['rms_deg_per_frame'] or 0, res['pitch_yaw']['p99_deg_per_frame'] or 0, len(ev), a.event_thr))
        if len(ev):
            print('      at (s): %s' % ' '.join('%.2f' % (i / fps) for i in ev[::max(1, len(ev) // 25)]))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2)


if __name__ == '__main__':
    main()

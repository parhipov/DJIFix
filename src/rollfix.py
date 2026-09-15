#!/usr/bin/env python3
"""
Correct the telemetry's roll with the roll actually measured from the image.

DJI's fused attitude carries a roll error of about 0.117 deg per frame that is
uncorrelated with what the camera really did (see HANDOFF.md section 5.4). It sits
at 0.5-3 Hz, so no filter reaches it, and no model predicts it -- the information
simply is not in the file. The only source of truth is the picture, which
`measure_clip.py` extracts as the curl of the affine flow field, a roll
measurement that parallax cannot fake.

This applies the difference as an extra rotation about the optical axis, on top of
the frame-alignment fix from `align.py`:

  d[f]     = image_roll[f] - telemetry_roll[f]          per-frame discrepancy
  w[f]     = 1 / (1 + (rate[f]/R0)^2)                   trust the image where it is reliable
  theta[f] = bandpass(cumsum(w*d), lo, hi)              the attitude correction
  q_new(t) = q(t) (x) exp(n * theta(t) / 2)             n = optical axis, body frame

The artefact is **per-frame**, right up at Nyquist -- at 14.4-15.2 s the image roll
runs smoothly (-0.025, -0.026, -0.029, -0.035, ...) while the telemetry alternates
(-0.002, -0.083, +0.041, +0.013, -0.072, -0.374, ...) with 1200 tracked points and
the drone turning at 2-12 deg/s. So the low pass must stay near Nyquist: an earlier
version cut at 8 Hz and threw away exactly the component that matters, closing only
4% of the gap instead of 101%.

The rate weighting is the other half. At speed, motion blur (up to 409 px of
rotational smear at 19.6 ms exposure) and rolling shutter (13.58 ms of a 20 ms
frame) make the image the noisier source -- measured: at >60 deg/s the image jitters
by 1.449 deg/frame against the telemetry's 0.811 -- and the artefact is invisible
there anyway because real motion swamps it. R0 above ~80 starts leaking that noise
into the fast sections.

Defaults chosen on an independent criterion, not on the fit residual (which would be
circular): the frame-to-frame jitter of the corrected telemetry against the physical
floor set by the image. lo 4 Hz / hi 24.8 Hz / R0 40 brings the calm sections to
0.0460 deg against a floor of 0.0464, with a correction of only 0.031 deg rms.
"""
import argparse

import numpy as np
from scipy.signal import butter, sosfiltfilt

import quat as Q
import sidecar
from align import corrections, slerp_series
from rotmath import logv, qmul, qconj  # noqa: F401  (re-exported for older callers)

# Optical axis in the telemetry body frame, fitted on the two fast windows
# (correlation 0.990, gain 1.0000). See HANDOFF.md section 6.
U = np.array([0.0573, -0.2774, 1.0202])


def build(video, roll_npz, out_path, r0=40.0, lo=4.0, hi=None, gain=1.0,
          align_first=True, report=True, timing='exposure', bias_ms=0.0):
    from timing import shifts
    clip, samples, frames, delta_us = shifts(video, timing, bias_ms)
    n_frames = len(frames)
    fps = float(clip.fps)
    if hi is None:
        hi = fps / 2 * 0.992

    t = np.array([s.t_us for s in samples])
    q = np.array([s.q_cam for s in samples])
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, qs = ts[keep], qs[keep]

    # --- the telemetry's own per-frame roll, on Gyroflow's frame instants -----
    # It has to be the roll of what will actually end up in the file, i.e. AFTER
    # the alignment shift. Measuring the discrepancy against the unaligned
    # telemetry instead makes the correction fight a 0.242 deg/frame baseline
    # difference that is seven times its own size, and it does nothing.
    ft = np.arange(n_frames + 1) * (1e6 / fps)
    if align_first:
        dd = np.concatenate([delta_us, delta_us[-1:]])
        qf = slerp_series(ts, qs, ft + dd)
    else:
        qf = slerp_series(ts, qs, ft)
    q1 = qf[1:].copy()
    s = np.sign(np.sum(qf[:-1] * q1, axis=1))
    s[s == 0] = 1
    q1 = q1 * s[:, None]
    inc = logv(qmul(qconj(qf[:-1]), q1))
    tel_roll = inc @ U
    rate = np.linalg.norm(inc, axis=1) * fps             # deg/s

    # --- the discrepancy against the image ------------------------------------
    d = np.load(roll_npz)
    img_roll = d['roll'][:n_frames]
    disc = img_roll - tel_roll
    w = 1.0 / (1.0 + (rate / r0) ** 2)
    w[~np.isfinite(disc)] = 0.0
    disc = np.nan_to_num(disc)

    # theta[f] must be the sum of the corrections BEFORE frame f: the attitude at
    # f and f+1 both carry theta, so the increment between them picks up the
    # forward difference theta[f+1] - theta[f], and that has to equal the
    # correction for frame f. Getting this off by one inverts the phase of a
    # near-Nyquist correction, which makes the jitter worse instead of better.
    c = w * disc * gain
    theta = np.concatenate([[0.0], np.cumsum(c)[:-1]])
    sos = butter(2, [lo, min(hi, fps / 2 * 0.999)], btype='band', fs=fps, output='sos')
    theta = sosfiltfilt(sos, theta)

    # --- apply: alignment first, then the roll correction ---------------------
    frame_of = np.array([s.frame for s in samples])
    target = t + (delta_us[frame_of] if align_first else 0.0)
    newq = slerp_series(ts, qs, target)

    axis = U / np.linalg.norm(U)
    th = np.interp(t, ft[:-1], theta)                     # degrees, per sample
    half = np.radians(th) / 2.0
    dq = np.column_stack([np.cos(half),
                          np.sin(half)[:, None] * axis[None, :]])
    newq = qmul(newq, dq)
    newq /= np.linalg.norm(newq, axis=1, keepdims=True)

    inverted = np.array([s.inverted for s in samples])
    newq = np.where(inverted[:, None], -newq, newq)
    edits = {(s.frame, s.index_in_frame):
             Q.norm(Q.gyroflow_to_dji(tuple(map(float, qq))))
             for s, qq in zip(samples, newq)}

    info = sidecar.build(video, out_path, edits)
    if report:
        m = np.isfinite(img_roll)
        print('frames with an image measurement : %d / %d' % (m.sum(), n_frames))
        print('roll discrepancy                 : rms %.4f deg/frame (weighted %.4f)'
              % (disc[m].std(), (w * disc)[m].std()))
        print('correction angle theta           : rms %.4f deg, peak %.4f deg '
              '(%.1f / %.1f px at the frame corner)'
              % (theta.std(), np.abs(theta).max(),
                 theta.std() * 2400 * np.pi / 180,
                 np.abs(theta).max() * 2400 * np.pi / 180))
        print('band %.1f-%.1f Hz, R0 %.0f deg/s, gain %.2f, alignment %s'
              % (lo, hi, r0, gain, 'on' if align_first else 'off'))
        print('%s: %d samples, %.1f MB, %d records rewritten'
              % (info['path'], info['samples'], info['bytes'] / 1e6,
                 info['edited_records']))
    return info, theta


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('-r', '--roll', default='artifacts/main/roll_clip.npz')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--r0', type=float, default=40.0,
                    help='rate (deg/s) at which the image is trusted half as much')
    ap.add_argument('--lo', type=float, default=4.0, help='high-pass corner, Hz')
    ap.add_argument('--hi', type=float, default=24.8, help='low-pass corner, Hz')
    ap.add_argument('--gain', type=float, default=1.0)
    ap.add_argument('--no-align', action='store_true')
    a = ap.parse_args()
    build(a.video, a.roll, a.out, a.r0, a.lo, a.hi, a.gain, not a.no_align)


CORNER_PX_PER_DEG = 2400 * np.pi / 180


def clusters(sel, gap=5):
    idx = np.where(sel)[0]
    if not len(idx):
        return []
    out, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] > gap:
            out.append(cur)
            cur = [i]
        else:
            cur.append(i)
    out.append(cur)
    return out


def event_theta(disc, rate, thr_px=8.0, r0=40.0, rate_max=60.0, margin=8,
                gain=1.0, gap=5):
    """Correction angle that only touches the bursts, and nets to zero.

    The user's symptom is occasional, not continuous: the median roll discrepancy
    is 2.1 px at the frame corner, invisible, but there are ~73 bursts reaching
    8-41 px. A global correction would smear the image measurement's own noise
    across the whole clip to fix 11% of it, so instead each burst is corrected on
    its own: the discrepancy is tapered to zero at the window edges, integrated,
    and then de-ramped so the correction starts and ends at zero. Nothing outside
    a burst is touched and the camera is never left permanently rotated.
    """
    ok = np.isfinite(disc)
    w = 1.0 / (1.0 + (rate / r0) ** 2)
    d = np.nan_to_num(disc) * w
    sel = ok & (np.abs(np.nan_to_num(disc)) > thr_px / CORNER_PX_PER_DEG) & (rate < rate_max)
    theta = np.zeros(len(disc))
    events = clusters(sel, gap)
    for ev in events:
        a = max(0, ev[0] - margin)
        b = min(len(disc), ev[-1] + margin + 1)
        seg = d[a:b] * np.hanning(b - a)
        th = np.cumsum(seg)
        th = th - np.linspace(0.0, th[-1], len(th))     # start and end at zero
        theta[a:b] += th * gain
    return theta, events


def build_events(video, roll_npz, out_path, thr_px=8.0, r0=40.0, gain=1.0,
                 margin=8, align_first=True, report=True):
    clip, samples, frames, iref, i_used, shift_samples, delta_us = corrections(video)
    n_frames = len(frames)

    t = np.array([s.t_us for s in samples])
    q = np.array([s.q_cam for s in samples])
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, qs = ts[keep], qs[keep]

    ft = np.arange(n_frames + 1) * 20000.0
    qf = slerp_series(ts, qs, ft)
    q1 = qf[1:].copy()
    s = np.sign(np.sum(qf[:-1] * q1, axis=1))
    s[s == 0] = 1
    inc = logv(qmul(qconj(qf[:-1]), q1 * s[:, None]))
    tel_roll = inc @ U
    rate = np.linalg.norm(inc, axis=1) * 50.0

    img_roll = np.load(roll_npz)['roll'][:n_frames]
    disc = img_roll - tel_roll
    theta, events = event_theta(disc, rate, thr_px, r0, gain=gain, margin=margin)

    frame_of = np.array([s.frame for s in samples])
    target = t + (delta_us[frame_of] if align_first else 0.0)
    newq = slerp_series(ts, qs, target)

    axis = U / np.linalg.norm(U)
    half = np.radians(np.interp(t, ft[:-1], theta)) / 2.0
    newq = qmul(newq, np.column_stack([np.cos(half), np.sin(half)[:, None] * axis[None, :]]))
    newq /= np.linalg.norm(newq, axis=1, keepdims=True)

    inverted = np.array([s.inverted for s in samples])
    newq = np.where(inverted[:, None], -newq, newq)
    edits = {(s.frame, s.index_in_frame):
             Q.norm(Q.gyroflow_to_dji(tuple(map(float, qq))))
             for s, qq in zip(samples, newq)}

    info = sidecar.build(video, out_path, edits)
    if report:
        touched = np.sum(theta != 0)
        print('events over %.0f px : %d, covering %d frames (%.1f%% of the clip)'
              % (thr_px, len(events), touched, 100.0 * touched / n_frames))
        print('correction angle   : rms %.4f deg over the touched frames (%.2f px), '
              'peak %.4f deg (%.1f px)'
              % (theta[theta != 0].std(), theta[theta != 0].std() * CORNER_PX_PER_DEG,
                 np.abs(theta).max(), np.abs(theta).max() * CORNER_PX_PER_DEG))
        print('%s: %.1f MB, %d records rewritten'
              % (info['path'], info['bytes'] / 1e6, info['edited_records']))
    return info, theta, events

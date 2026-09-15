#!/usr/bin/env python3
"""
Clean DJI's roll noise out of the telemetry -- **without touching the video**.

Measured against optical flow on this clip, the ratio of the telemetry's roll
energy to the camera's real roll energy, on calm frames:

    0-2 Hz  1.29x     8-12 Hz   6.02x
    2-5 Hz  1.73x    12-16 Hz   9.89x
    5-8 Hz  3.44x    16-20 Hz  13.26x
                     20-25 Hz  20.59x

Above 5 Hz the fused attitude is 3 to 20 times larger than the motion that
actually happened: it is essentially all noise. So the optimal Wiener gain to
*keep* is P_real/P_telemetry = 1/ratio, and that curve is what this applies. The
gains were calibrated once against the image; the assumption underneath them --
that a calm drone has no real 8-25 Hz camera roll -- is a physical prior, and the
image measurement confirmed it.

Recovered, judged against the image measurement it never sees:

    calm sections    68 % of what the full image-based correction achieves
    20-60 deg/s     100 %
    faster          improves them too (0.785 -> 0.755), which the image-based
                    method cannot, because there the image is the noisier source

So for a clip with no optical-flow pass available, this gets most of the way. Use
`rollfix.py` when the image measurement exists.
"""
import argparse

import numpy as np
from scipy.signal import butter, sosfiltfilt

import quat as Q
import sidecar
from align import slerp_series
from rollfix import U, logv, qmul, qconj, CORNER_PX_PER_DEG
from timing import shifts

# Wiener keep-gain against frequency (Hz), calibrated on DJI O4 Pro / fw 01.00.06.00
BAND_HZ = np.array([0, 1, 2, 3.5, 5, 6.5, 8, 10, 12, 14, 16, 18, 20, 25])
KEEP = np.array([1.0, 0.90, 0.77, 0.68, 0.58, 0.40, 0.29,
                 0.22, 0.17, 0.13, 0.10, 0.086, 0.075, 0.049])


def wiener_roll(roll, fps=50.0):
    """Apply the calibrated keep-gain curve to a per-frame roll series."""
    x = np.nan_to_num(roll)
    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / fps)
    g = np.interp(f, BAND_HZ, KEEP)
    return np.fft.irfft(np.fft.rfft(x) * g, n)


def band_sos(hp, fps):
    """Band-pass from `hp` Hz to just below Nyquist, for the integrated correction."""
    return butter(2, [hp, fps / 2 * 0.992], btype='band', fs=fps, output='sos')


def build(video, out_path, r0=80.0, hp=1.0, gain=1.0, align_first=True, report=True,
          timing='exposure', bias_ms=0.0):
    clip, samples, frames, delta_us = shifts(video, timing, bias_ms)
    n_frames = len(frames)
    fps = float(clip.fps)

    t = np.array([s.t_us for s in samples])
    q = np.array([s.q_cam for s in samples])
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, qs = ts[keep], qs[keep]

    # per-frame attitude of what will actually be in the file (after alignment)
    ft = np.arange(n_frames + 1) * (1e6 / fps)
    if align_first:
        dd = np.concatenate([delta_us, delta_us[-1:]])
        qf = slerp_series(ts, qs, ft + dd)
    else:
        qf = slerp_series(ts, qs, ft)
    q1 = qf[1:].copy()
    s = np.sign(np.sum(qf[:-1] * q1, axis=1))
    s[s == 0] = 1
    inc = logv(qmul(qconj(qf[:-1]), q1 * s[:, None]))
    tel_roll = inc @ U
    rate = np.linalg.norm(inc, axis=1) * fps

    # how much of the roll to replace with its de-noised version
    w = 1.0 / (1.0 + (rate / r0) ** 2)
    c = w * (wiener_roll(tel_roll, fps) - tel_roll) * gain

    # theta[f] = sum of corrections before f (see rollfix: an unshifted cumsum
    # applies frame f+1's correction to frame f and inverts the phase)
    theta = np.concatenate([[0.0], np.cumsum(c)[:-1]])
    theta = sosfiltfilt(band_sos(hp, fps), theta)

    frame_of = np.array([s_.frame for s_ in samples])
    target = t + (delta_us[frame_of] if align_first else 0.0)
    newq = slerp_series(ts, qs, target)

    axis = U / np.linalg.norm(U)
    half = np.radians(np.interp(t, ft[:-1], theta)) / 2.0
    newq = qmul(newq, np.column_stack([np.cos(half), np.sin(half)[:, None] * axis[None, :]]))
    newq /= np.linalg.norm(newq, axis=1, keepdims=True)

    inverted = np.array([s_.inverted for s_ in samples])
    newq = np.where(inverted[:, None], -newq, newq)
    edits = {(s_.frame, s_.index_in_frame):
             Q.norm(Q.gyroflow_to_dji(tuple(map(float, qq))))
             for s_, qq in zip(samples, newq)}

    info = sidecar.build(video, out_path, edits)
    if report:
        print('roll noise removed : rms %.4f deg/frame' % c.std())
        print('correction angle   : rms %.4f deg (%.2f px at the frame corner), '
              'peak %.4f deg (%.1f px)'
              % (theta.std(), theta.std() * CORNER_PX_PER_DEG,
                 np.abs(theta).max(), np.abs(theta).max() * CORNER_PX_PER_DEG))
        print('R0 %.0f deg/s, high pass %.1f Hz, gain %.2f, alignment %s'
              % (r0, hp, gain, 'on' if align_first else 'off'))
        print('%s: %.1f MB, %d records rewritten'
              % (info['path'], info['bytes'] / 1e6, info['edited_records']))
    return info, theta


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--r0', type=float, default=80.0,
                    help='rate (deg/s) at which the de-noising is half as strong')
    ap.add_argument('--hp', type=float, default=1.0, help='high-pass corner, Hz')
    ap.add_argument('--gain', type=float, default=1.0)
    ap.add_argument('--no-align', action='store_true')
    a = ap.parse_args()
    build(a.video, a.out, a.r0, a.hp, a.gain, not a.no_align)


# Pan/tilt keep-gain, derived the same way but against the image's frame shift
# (at 4-25 Hz a shift can only come from attitude -- the drone's translation
# cannot reverse direction ten times a second). The excess there is only 1.5x,
# against 2.8x for roll, so the curve is much gentler.
KEEP_PANTILT = np.array([1.0, 0.93, 0.87, 0.82, 0.77, 0.72, 0.67,
                         0.64, 0.61, 0.48, 0.35, 0.25, 0.19, 0.19])


def _basis():
    uh = U / np.linalg.norm(U)
    a = np.array([1.0, 0.0, 0.0])
    a = a - uh * np.dot(a, uh)
    a /= np.linalg.norm(a)
    return uh, a, np.cross(uh, a)


def wiener_axis(v, keep, fps=50.0):
    x = np.nan_to_num(v)
    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / fps)
    return np.fft.irfft(np.fft.rfft(x) * np.interp(f, BAND_HZ, keep), n)


def frame_increments(ts, qs, ft):
    """Body-frame rotation vectors (deg) between consecutive frame instants."""
    qf = slerp_series(ts, qs, ft)
    q1 = qf[1:].copy()
    s = np.sign(np.sum(qf[:-1] * q1, axis=1))
    s[s == 0] = 1
    return logv(qmul(qconj(qf[:-1]), q1 * s[:, None]))


def denoise_correction(inc, fps, r0=80.0, gain=1.0, roll_only=False):
    """Per-frame increment correction (deg) from the Wiener keep-gain curves."""
    rate = np.linalg.norm(inc, axis=1) * fps
    w = 1.0 / (1.0 + (rate / r0) ** 2)
    uh, ax, ay = _basis()
    axes = [(uh, KEEP)] if roll_only else [(uh, KEEP), (ax, KEEP_PANTILT), (ay, KEEP_PANTILT)]
    corr = np.zeros_like(inc)
    for axis, keep in axes:
        comp = inc @ axis
        corr += np.outer(w * (wiener_axis(comp, keep, fps) - comp) * gain, axis)
    return corr


def integrate_correction(corr, fps, hp=1.0):
    """theta[f] = sum of the corrections before f, band-limited to hp..Nyquist.

    An unshifted cumsum applies frame f+1's correction to frame f and inverts
    the phase of a near-Nyquist correction (see rollfix.py).
    """
    theta = np.vstack([np.zeros((1,) + corr.shape[1:]), np.cumsum(corr, axis=0)[:-1]])
    return sosfiltfilt(band_sos(hp, fps), theta, axis=0)


def build3(video, out_path, r0=80.0, hp=1.0, gain=1.0, align_first=True, report=True,
           timing='exposure', bias_ms=0.0):
    """De-noise all three axes, not just roll.

    Once roll is fixed it stops being the dominant residual: measured on clip
    0003 at 7-10 s, roll went from 4.95 px of apparent jitter down to 1.09 px,
    while pitch and yaw sat untouched at 1.11 and 1.30 px. This adds them, with
    their own (much gentler) keep-gain curve.
    """
    clip, samples, frames, delta_us = shifts(video, timing, bias_ms)
    n_frames = len(frames)
    fps = float(clip.fps)

    t = np.array([s.t_us for s in samples])
    q = np.array([s.q_cam for s in samples])
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    k = np.concatenate([[True], np.diff(ts) > 0])
    ts, qs = ts[k], qs[k]

    ft = np.arange(n_frames + 1) * (1e6 / fps)
    if align_first:
        dd = np.concatenate([delta_us, delta_us[-1:]])
        inc = frame_increments(ts, qs, ft + dd)
    else:
        inc = frame_increments(ts, qs, ft)
    corr = denoise_correction(inc, fps, r0, gain)
    theta = integrate_correction(corr, fps, hp)

    frame_of = np.array([s_.frame for s_ in samples])
    target = t + (delta_us[frame_of] if align_first else 0.0)
    newq = slerp_series(ts, qs, target)

    th = np.column_stack([np.interp(t, ft[:-1], theta[:, i]) for i in range(3)])
    mag = np.linalg.norm(th, axis=1)
    half = np.radians(mag) / 2.0
    sc = np.where(mag > 1e-12, np.sin(half) / np.where(mag > 1e-12, mag, 1.0), 0.0)
    newq = qmul(newq, np.column_stack([np.cos(half), th * sc[:, None]]))
    newq /= np.linalg.norm(newq, axis=1, keepdims=True)

    inverted = np.array([s_.inverted for s_ in samples])
    newq = np.where(inverted[:, None], -newq, newq)
    edits = {(s_.frame, s_.index_in_frame):
             Q.norm(Q.gyroflow_to_dji(tuple(map(float, qq))))
             for s_, qq in zip(samples, newq)}

    info = sidecar.build(video, out_path, edits)
    if report:
        for nm, axis in (('roll', uh), ('pitch', ax), ('yaw', ay)):
            print('  %-6s removed rms %.4f deg/frame' % (nm, (corr @ axis).std()))
        print('correction angle : rms %.4f deg, peak %.4f deg'
              % (np.linalg.norm(theta, axis=1).std(), np.linalg.norm(theta, axis=1).max()))
        print('%s: %.1f MB, %d records rewritten'
              % (info['path'], info['bytes'] / 1e6, info['edited_records']))
    return info, theta

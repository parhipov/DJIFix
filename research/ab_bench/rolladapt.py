#!/usr/bin/env python3
"""
Research prototype, not project code: roll from the image with a rate-dependent crossover.

Above 4 Hz the image roll replaces the telemetry's (translation cannot oscillate
that fast, so parallax does not reach it), weight 1 / (1 + (rate / R0_HF)^2) with
R0_HF = 120 deg/s: full in ordinary flight, falling off in blurred fast passes. Between 1 and 4 Hz
it does so only on calm frames: weight 1 / (1 + (rate / R0_LF)^2), R0_LF = 15
deg/s -- when the camera barely moves there is no parallax to fake a roll, and
that band is where the residual wobble of the current fix lives. Below 1 Hz
the telemetry is kept. The 1-4 Hz part also needs a calm neighbourhood: no
frame faster than 30 deg/s within 1 s (the band-pass spreads a correction about
0.5 s both ways, and the image lies as blur builds up before a manoeuvre); frames
faster than 60 deg/s (+-0.1 s) are not used at all. Both weights are zero where the image is not
self-consistent (research/diagnose.py: its two roll estimators disagree or too
few points) and in the take-off/landing margins. The image gain is measured at
2-4 Hz. Pitch and yaw are not touched.

Defect gate: roll is only corrected where DJI's roll defect is measured. Its
signature is the telemetry-vs-image roll error above 4 Hz (where the image is
trustworthy): per calm second 0.0315 deg median on the one clip that has the
defect, 0.0065-0.0123 on the 16 that do not, whose isolated seconds above 0.02
are 0-6 %. Per calm second a robust value (MAD, so a one-frame spike does not
count); for each second the median of the nearest 7 calm seconds within 20 s
(fewer than 5: no decision, no correction) switches the correction on smoothly
between 0.018 and 0.025 deg; elsewhere the telemetry is left exactly as it is.

    python research/ab_bench/rolladapt.py VIDEO IMAGE.npz OUT.mp4 [EDGE_S]    # EDGE_S 0.02 for short cut examples
"""
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'research'))
import diagnose                    # noqa: E402
import fix_pipeline as fp          # noqa: E402
import quat as Q                   # noqa: E402
import sidecar                     # noqa: E402
from align import slerp_series     # noqa: E402
from denoise import frame_increments   # noqa: E402
from rollfix import qmul           # noqa: E402
from rotmath import rotation_vector_quat   # noqa: E402
from timing import load_sorted     # noqa: E402


def build(video, image_npz, out, r0_lf=15.0, r0_hf=120.0, lf_band=(1.0, 4.0), edge_s=None,
          defect_lo=0.018, defect_hi=0.025, defect_k=7, defect_min=5, defect_reach_s=20, fast_deg_s=60.0, fast_pad_s=0.1, calm_ctx_deg_s=30.0, calm_ctx_s=1.0):
    if edge_s is not None:
        diagnose.EDGE_S = edge_s
    clip, ser, _, _ = diagnose.analyse(video, image_npz, None)
    _, samples, frames, ts, qs = load_sorted(video)
    fps, n = ser['fps'], ser['n']
    ft = np.arange(n + 1) * (1e6 / fps)
    inc, rate, ok = ser['inc'], ser['rate'], ser['img_ok'] & ser['valid']
    d = np.load(image_npz)
    inc_img, _, _ = fp.image_rotations(d, n)
    g = diagnose.band_gain(inc[:, 2], np.nan_to_num(inc_img[:, 2]), ok & (rate < 150), fps)
    disc = np.where(ok, np.nan_to_num(inc_img[:, 2]) / g - inc[:, 2], 0.0)          # deg about z, per frame
    if fast_deg_s:
        # blurred fast passes: the image is not used at all there, nor just around them
        from scipy.ndimage import binary_dilation
        fast = binary_dilation(rate > fast_deg_s, iterations=max(1, int(round(fast_pad_s * fps))))
        ok = ok & ~fast
        disc = np.where(ok, disc, 0.0)
    w_hf = np.where(ok, 1.0 / (1.0 + (rate / r0_hf) ** 2), 0.0)
    w_lf = np.where(ok, 1.0 / (1.0 + (rate / r0_lf) ** 2), 0.0)
    if calm_ctx_s:
        # the 1-4 Hz part is taken only in a calm neighbourhood: the band-pass spreads a correction
        # about half a second both ways, and the image just before a manoeuvre (blur building up) lies
        from scipy.ndimage import binary_dilation
        busy = binary_dilation(rate > calm_ctx_deg_s, iterations=max(1, int(round(calm_ctx_s * fps))))
        w_lf = np.where(busy, 0.0, w_lf)
    hf = butter(2, [lf_band[1], fps / 2 * 0.992], btype='band', fs=fps, output='sos')
    lf = butter(2, list(lf_band), btype='band', fs=fps, output='sos')
    # --- defect gate: HF roll error per second, rolling median over 10 s ---
    hp = butter(2, 4.0, btype='high', fs=fps, output='sos')
    herr = np.where(ok, sosfiltfilt(hp, np.nan_to_num(inc_img[:, 2]) / g) - sosfiltfilt(hp, inc[:, 2]), 0.0)
    calm = ok & (rate < 45)
    sec = int(round(fps))
    per_s = np.full(n // sec + 1, np.nan)
    for i in range(len(per_s)):
        sl = slice(i * sec, min(n, (i + 1) * sec))
        m = calm[sl]
        if m.sum() > 0.5 * sec:
            per_s[i] = 1.4826 * np.median(np.abs(herr[sl][m]))       # robust: a one-frame spike does not move it
    # the decision for each second: median of the nearest `defect_k` calm seconds within `defect_reach_s`;
    # with fewer than `defect_min` of them there is no decision and nothing is corrected
    valid_s = np.flatnonzero(np.isfinite(per_s))
    med = np.zeros(len(per_s))
    for i in range(len(per_s)):
        if not len(valid_s):
            break
        dist = np.abs(valid_s - i)
        near = valid_s[np.argsort(dist)[:defect_k]]
        near = near[np.abs(near - i) <= defect_reach_s]
        med[i] = np.median(per_s[near]) if len(near) >= defect_min else 0.0
    gate_s = np.clip((med - defect_lo) / (defect_hi - defect_lo), 0.0, 1.0)
    gate = np.interp(np.arange(n) / fps, (np.arange(len(per_s)) + 0.5), gate_s)
    theta = sosfiltfilt(hf, fp.shifted_cumsum(gate * w_hf * disc)) + sosfiltfilt(lf, fp.shifted_cumsum(gate * w_lf * disc))
    u = np.array([0.0, 0.0, g])
    t = np.array([s.t_us for s in samples], dtype=float)
    base = slerp_series(ts, qs, t)
    th = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.interp(t, ft[:-1], theta)])
    newq = qmul(base, rotation_vector_quat(th))
    newq /= np.linalg.norm(newq, axis=1, keepdims=True)
    inverted = np.array([s.inverted for s in samples])
    qq = np.where(inverted[:, None], -newq, newq)
    edits = {(s.frame, s.index_in_frame): Q.norm(Q.gyroflow_to_dji(tuple(map(float, v)))) for s, v in zip(samples, qq)}
    sidecar.build(video, out, edits)
    return dict(out=out, roll_gain=float(u[2]), r0_lf=r0_lf, corr_rms_deg=round(float(theta.std()), 4),
                corr_peak_deg=round(float(np.abs(theta).max()), 3), defect_active_frac=round(float((gate > 0.5).mean()), 3),
                defect_median_deg=round(float(np.nanmedian(per_s)), 4))


if __name__ == '__main__':
    a = sys.argv[1:]
    print(json.dumps(build(a[0], a[1], a[2], edge_s=float(a[3]) if len(a) > 3 else None)))

"""Research prototype, not project code: two-sensor fusion on all three axes.

Above the crossover `fc` the telemetry's increments are replaced by the image's
(kabsch1, per pair), each axis scaled by the image/telemetry gain measured at
2-4 Hz, rate-weighted like the roll stage (R0). Below `fc` the telemetry is kept.
Telemetry spikes are removed first, exactly as fix_pipeline does. No timing
shift, no Wiener, no events.
"""
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, 'K:/Work/Python/DJI fix/src')
import fix_pipeline as fp          # noqa: E402
import quat as Q                   # noqa: E402
import sidecar                     # noqa: E402
from align import slerp_series     # noqa: E402
from denoise import frame_increments   # noqa: E402
from rollfix import qmul           # noqa: E402
from rotmath import rotation_vector_quat   # noqa: E402
from timing import load_sorted     # noqa: E402


def band_gain(tel, img, ok, fps, band=(2.0, 4.0), min_run=64, trim=15):
    runs = [r for r in fp.clusters(ok, gap=0) if len(r) >= min_run]
    if sum(len(r) - 2 * trim for r in runs) < 300:
        return 1.0, float('nan')
    sos = butter(2, list(band), btype='band', fs=fps, output='sos')
    X = np.concatenate([sosfiltfilt(sos, tel[r[0]:r[-1] + 1])[trim:-trim] for r in runs])
    Y = np.concatenate([sosfiltfilt(sos, img[r[0]:r[-1] + 1])[trim:-trim] for r in runs])
    return float(np.dot(X, Y) / np.dot(X, X)), float(np.corrcoef(X, Y)[0, 1])


def build(video, image_npz, out_path, fc=4.0, r0=40.0, axes=(0, 1, 2), spikes=True):
    clip, samples, frames, ts, qs = load_sorted(video)
    fps = float(clip.fps)
    n = len(frames)
    d = np.load(image_npz)
    inc_img, _, _ = fp.image_rotations(d, n)
    ft = np.arange(n + 1) * (1e6 / fps)
    inc = frame_increments(ts, qs, ft)
    rate = np.linalg.norm(inc, axis=1) * fps
    valid = (ft[:-1] >= ts[0]) & (ft[1:] <= ts[-1]) & (rate < 1000.0)
    rate = np.where(valid, rate, 1e6)
    has = np.isfinite(inc_img).all(1) & valid
    info = {}
    # spikes, as in fix_pipeline.build
    c_spike = np.zeros_like(inc)
    spike = np.zeros(n, bool)
    if spikes:
        fine = np.full((n, 3), np.nan)
        for k in range(3):
            g, corr, _ = fp.scalar_gain(inc[:, k], inc_img[:, k], rate)
            g = g if (np.isfinite(corr) and corr >= 0.6 and 0.3 < g < 3.0) else 1.0
            fine[:, k] = inc_img[:, k] / g - inc[:, k]
        fine[~valid] = np.nan
        fine_raw = inc_img - inc
        fine_raw[~valid] = np.nan
        spike, c_spike, _ = fp.detect_spikes(inc, fine, fine_raw, rate, np.array([0.0, 0.0, 1.0]), valid)
    inc_ds = inc + c_spike
    theta = np.zeros_like(inc)
    w = 1.0 / (1.0 + (rate / r0) ** 2)
    w[~has] = 0.0
    ok = has & (rate < 150.0)
    hp = butter(2, [fc, fps / 2 * 0.992], btype='band', fs=fps, output='sos')
    for k in axes:
        g, corr = band_gain(inc_ds[:, k], np.nan_to_num(inc_img[:, k]), ok, fps)
        if not (np.isfinite(corr) and corr >= 0.6 and 0.5 < g < 2.0):
            g = 1.0
        info['axis%d' % k] = dict(gain=g, corr=corr)
        disc = np.where(has, np.nan_to_num(inc_img[:, k]) / g - inc_ds[:, k], 0.0)
        theta[:, k] = sosfiltfilt(hp, fp.shifted_cumsum(w * disc))
    th_sp = fp.shifted_cumsum(c_spike)
    if spike.any():
        decay = np.exp(-1.0 / (1.5 * fps))
        y = np.zeros_like(th_sp)
        for f in range(1, n):
            y[f] = (y[f - 1] if spike[f] else y[f - 1] * decay) + (th_sp[f] - th_sp[f - 1])
        th_sp = y
    theta = theta + th_sp
    t = np.array([s.t_us for s in samples], dtype=float)
    base = slerp_series(ts, qs, t)
    th = np.column_stack([np.interp(t, ft[:-1], theta[:, k]) for k in range(3)])
    newq = qmul(base, rotation_vector_quat(th))
    newq /= np.linalg.norm(newq, axis=1, keepdims=True)
    inverted = np.array([s.inverted for s in samples])
    qq = np.where(inverted[:, None], -newq, newq)
    edits = {(s.frame, s.index_in_frame): Q.norm(Q.gyroflow_to_dji(tuple(map(float, v)))) for s, v in zip(samples, qq)}
    sidecar.build(video, out_path, edits)
    info.update(fc=fc, r0=r0, spikes=int(spike.sum()), corr_rms_deg=np.round(theta.std(0), 4).tolist(),
                corr_peak_deg=np.round(np.abs(theta).max(0), 3).tolist())
    return info


if __name__ == '__main__':
    print(json.dumps(build(sys.argv[1], sys.argv[2], sys.argv[3], fc=float(sys.argv[4]))))

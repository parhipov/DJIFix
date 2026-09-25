"""Telemetry-only fingerprint: per-frame jitter of the attitude increments on calm frames."""
import sys, numpy as np
sys.path.insert(0, 'K:/Work/Python/DJI fix/src')
from timing import load_sorted, exposure_ms
from denoise import frame_increments
from scipy.signal import welch
for v in sys.argv[1:]:
    c, s, f, ts, qs = load_sorted(v); fps = float(c.fps)
    ft = np.arange(len(f) + 1) * (1e6 / fps)
    inc = frame_increments(ts, qs, ft); rate = np.linalg.norm(inc, axis=1) * fps
    calm = (rate < 30) & (np.arange(len(inc)) < len(inc) - 10)
    d2 = np.diff(inc, axis=0); m = calm[1:] & calm[:-1]
    # spectrum of increments on calm runs: ratio of power 15-25 Hz to 2-6 Hz
    idx = np.flatnonzero(calm); runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1); runs = [r for r in runs if len(r) >= 128]
    P = None
    for r in runs:
        fr, p = welch(inc[r], fs=fps, nperseg=128, axis=0, detrend='linear'); P = p * len(r) if P is None else P + p * len(r)
    hi = P[(fr >= 15)].mean(0); lo = P[(fr >= 2) & (fr < 6)].mean(0)
    ex = exposure_ms(f)
    print('%-40s %s fw %s gyro? | calm frames %5d | per-frame jitter (2nd diff/sqrt2) deg x/y/z %s | HF/LF power %s | exposure %.1f-%.1f ms' % (
        v.split('/')[-1][-40:], c.header.get('product_name'), c.header.get('product_firmware_version'), calm.sum(),
        np.round(d2[m].std(0) / np.sqrt(2), 4), np.round(hi / lo, 2), ex.min(), ex.max()))

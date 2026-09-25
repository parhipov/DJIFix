"""Compare Gyroflow renders of the variants of one long clip.

Residual motion of each render (residual.py: shift x/y and roll, per frame pair)
is integrated to a position, band-passed, and compared with the render of the
original telemetry frame by frame. Above 4 Hz Gyroflow's smoothed path carries
nothing, so what is measured there is error. Confidence intervals: paired block
bootstrap over 2 s blocks. Units: px of a 4K frame (x, y, roll at the frame edge).
"""
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, 'K:/Work/Python/DJI fix/src')
HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, '..', 'full')
VIDEOS = {'pro': 'F:/36/DJI_20260905181949_0005_D.MP4', 'lite': 'F:/36/54-56-DJI_20260517170526_0020_D.MP4', 'good': 'F:/Май дом с красной крышей 2/DJI_20260521111216_0036_D.MP4', 'winter': 'F:/15 тест/DJI_20251207151759_0069_D.MP4'}
BANDS = ((4, 8), (8, 24.5))
AX = ('x', 'y', 'roll')


def tel_rate(clip):
    cache = os.path.join(FULL, '%s_rate.npy' % clip)
    if os.path.isfile(cache):
        return np.load(cache)
    from timing import load_sorted
    from denoise import frame_increments
    c, samples, frames, ts, qs = load_sorted(VIDEOS[clip])
    ft = np.arange(len(frames) + 1) * (1e6 / float(c.fps))
    r = np.linalg.norm(frame_increments(ts, qs, ft), axis=1) * float(c.fps)
    np.save(cache, r)
    return r


def load(clip, name):
    z = np.load(os.path.join(FULL, '%s_%s_resid.npz' % (clip, name)))
    r = z['r'].copy()
    k = 3840.0 / float(z['W'])
    r[:, :2] *= k
    r[:, 2] = np.radians(r[:, 2]) * 3840.0 / 2
    return float(z['fps']), r


def band_series(r, ok, fps, lo, hi):
    x = np.where(ok[:, None], r, 0.0)
    return sosfiltfilt(butter(2, [lo, hi], btype='band', fs=fps, output='sos'), np.cumsum(x, 0), axis=0)


def main(clip, names):
    fps, r0 = load(clip, 'orig')
    R = {n: load(clip, n)[1] for n in names}
    R['orig'] = r0
    n = min(len(v) for v in R.values())
    R = {k: v[:n] for k, v in R.items()}
    ok = np.all([np.isfinite(v).all(1) for v in R.values()], axis=0)
    rate = tel_rate(clip)[:n]
    edge = np.zeros(n, bool)
    edge[:25] = edge[-25:] = True
    blk = int(2 * fps)
    nb = n // blk
    rng = np.random.default_rng(0)
    boot = rng.integers(0, nb, size=(2000, nb))
    print('== %s: %d frame pairs, %d usable' % (clip, n, ok.sum()))
    for lo, hi in BANDS:
        S = {k: band_series(v, ok, fps, lo, hi) for k, v in R.items()}
        for sel_name, sel in (('<20/s', ok & ~edge & (rate < 20)), ('20-45/s', ok & ~edge & (rate >= 20) & (rate < 45)),
                              ('<45/s', ok & ~edge & (rate < 45)), ('45-100*', ok & ~edge & (rate >= 45) & (rate < 100))):
            if sel.sum() < 100:
                continue
            base = S['orig'] ** 2
            line = '%5g-%-4g Hz %-9s n=%5d  orig %s' % (lo, hi, sel_name, sel.sum(), ' '.join('%5.2f' % np.sqrt(base[sel, a].mean()) for a in range(3)))
            for k in names:
                cur = S[k] ** 2
                cells = []
                for a in range(3):
                    b = np.array([base[i * blk:(i + 1) * blk, a][sel[i * blk:(i + 1) * blk]].sum() for i in range(nb)])
                    c = np.array([cur[i * blk:(i + 1) * blk, a][sel[i * blk:(i + 1) * blk]].sum() for i in range(nb)])
                    ratio = np.sqrt(c.sum() / max(b.sum(), 1e-12))
                    br = np.sqrt(c[boot].sum(1) / np.maximum(b[boot].sum(1), 1e-12))
                    lo_ci, hi_ci = np.percentile(br, [2.5, 97.5])
                    mark = '+' if hi_ci < 0.97 else ('-' if lo_ci > 1.03 else ' ')
                    cells.append('%4.2f%s' % (ratio, mark))
                line += ' | %s %s' % (k, ' '.join(cells))
            print(line)
    print('* = render metric noisy there (tracker-dependent). ratios are variant/orig rms per axis x y roll; + = better with 95%% CI below 0.97, - = worse with CI above 1.03')


if __name__ == '__main__':
    clip = sys.argv[1]
    names = [n for n in sys.argv[2:] if os.path.isfile(os.path.join(FULL, '%s_%s_resid.npz' % (clip, n)))]
    main(clip, names)

import sys, numpy as np
sys.path.insert(0, 'K:/Work/Python/DJI fix/research/ab_bench')
from residual import measure
from scipy.signal import butter, sosfiltfilt
res = {}
for n in ('orig', 'fixed', 'Rinf', 'R1', 'Rfull'):
    for sc in (0.25, 0.5):
        fps, W, H, r = measure('v_%s_stab.mp4' % n, scale=sc)
        res[(n, sc)] = np.radians(np.nan_to_num(r[:, 2])) * W / 2       # roll, px at the frame edge (4K)
np.savez('roll_resid.npz', **{'%s_%g' % k: v for k, v in res.items()})
fps = 50.0
def bands(x):
    pos = np.cumsum(x)
    out = []
    for lo, hi in ((1, 4), (4, 8), (8, 24.5)):
        out.append(sosfiltfilt(butter(2, [lo, hi], btype='band', fs=fps, output='sos'), pos)[10:-10].std())
    out.append(np.diff(x)[5:-5].std() / np.sqrt(2))
    return out
print('roll residual in the Gyroflow render, px at the 4K frame edge (tracker scale 0.25 / 0.5)')
print('%-7s %-15s %-15s %-15s %-15s' % ('', '1-4 Hz', '4-8 Hz', '8-25 Hz', 'per-frame'))
for n in ('orig', 'fixed', 'Rinf', 'R1', 'Rfull'):
    a, b = bands(res[(n, 0.25)]), bands(res[(n, 0.5)])
    print('%-7s ' % n + ' '.join('%5.2f / %5.2f  ' % (x, y) for x, y in zip(a, b)))
d = bands(res[('orig', 0.25)] - res[('orig', 0.5)])
print('tracker noise (same render, two scales, difference/sqrt2):', ' '.join('%.2f' % (x / np.sqrt(2)) for x in d))

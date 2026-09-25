"""Which image-side indicators flag the seconds where the image is wrong? (good clip: telemetry is the reference)"""
import sys, numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr
S = sys.argv[1]
g = np.load(S + '_fix_diagnostics.npz'); d = np.load(S + '_image.npz')
fps = 50.0; inc = g['inc_deg']; rate = g['rate_deg_s']; n = len(inc)
k1 = np.full((n, 3), np.nan); k1[:len(d['kabsch1_deg'])] = d['kabsch1_deg'][:n]; k1 *= np.array([-1, 1, 1])
cu = np.full(n, np.nan); cu[:len(d['roll_n_deg'])] = d['roll_n_deg'][:n]
npts = np.full(n, np.nan); npts[:len(d['npts'])] = d['npts'][:n]
sx = np.full(n, np.nan); sx[:len(d['shift_xn_deg'])] = d['shift_xn_deg'][:n]
sy = np.full(n, np.nan); sy[:len(d['shift_yn_deg'])] = d['shift_yn_deg'][:n]
rr = np.full(n, np.nan); rr[:len(d['resid_r_px'])] = d['resid_r_px'][:n]
hp = butter(2, 4.0, btype='high', fs=fps, output='sos')
def H(x):
    x = np.nan_to_num(x); return sosfiltfilt(hp, x, axis=0)
err = np.linalg.norm(H(k1) - H(inc), axis=1)                       # image error proxy (telemetry fine here)
f_kc = np.abs(H(k1[:, 2]) - H(cu))                                  # two image roll estimators disagree
# shift channels vs kabsch pitch/yaw (another estimator from the same flow, different model)
f_ks = np.abs(H(k1[:, 0]) + H(sy) * 0) * 0
from numpy.linalg import lstsq
A = np.column_stack([H(sy), H(sx)]); ok = np.isfinite(A).all(1)
for k in (0, 1):
    pass
f_npts = -npts
f_rr = rr
sec = int(fps); ns = n // sec
def per_sec(x, fn=np.nanmean): return np.array([fn(x[i*sec:(i+1)*sec]) for i in range(ns)])
E = per_sec(err, lambda v: np.sqrt(np.nanmean(v**2)))
calm = per_sec(rate) < 45
feats = {'kabsch-vs-curl roll (HF)': per_sec(f_kc, lambda v: np.sqrt(np.nanmean(v**2))),
         'fewer tracked points': per_sec(f_npts), 'window fit residual px': per_sec(f_rr, np.nanmedian)}
print('%s: %d calm seconds; image-error proxy median %.4f, p90 %.4f deg' % (S.split('/')[-1], calm.sum(), np.median(E[calm]), np.percentile(E[calm], 90)))
worst = E[calm] >= np.percentile(E[calm], 90)
for name, f in feats.items():
    m = calm & np.isfinite(f)
    rho = spearmanr(f[m], E[m]).correlation
    # how many of the worst-10% image-error seconds are in the top 20% of the flag
    top = f[m] >= np.nanpercentile(f[m], 80)
    hit = (top & (E[m] >= np.percentile(E[m], 90))).sum() / max((E[m] >= np.percentile(E[m], 90)).sum(), 1)
    print('   %-28s spearman %.2f   catches %.0f%% of worst-10%% seconds in its top 20%%' % (name, rho, 100 * hit))
idx = np.argsort(E * calm)[::-1][:6]
print('   worst image-error calm seconds:', [(int(i), round(float(E[i]), 3), round(float(feats['kabsch-vs-curl roll (HF)'][i]), 3), int(-feats['fewer tracked points'][i])) for i in idx])

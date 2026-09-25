import numpy as np, sys
from scipy.signal import butter, sosfiltfilt
z = np.load(sys.argv[1]); fps = 50.0
GAPS = (1, 2, 3, 4, 6, 8); R1 = z['R1']
print('closure error |direct f->f+k  -  sum of k pairs|, rms per axis (deg), and implied noise')
rows = []
for k in GAPS[1:]:
    n = len(R1) - k
    Sk = np.array([R1[i:i + k].sum(0) for i in range(n)]); Dk = z['R%d' % k][:n]
    E = Dk - Sk
    m = np.isfinite(E).all(1)
    # remove anything slower than 1 Hz (drift, shared parallax differences)
    Ef = sosfiltfilt(butter(2, 1.0, btype='high', fs=fps, output='sos'), E[m], axis=0)[10:-10]
    rows.append((k, Ef.std(0)))
    print('  k=%d  rms %s  signal rms of Dk %s' % (k, np.round(Ef.std(0), 4), np.round(Dk[m].std(0), 3)))
# model var(E_k) = k*s1^2 + sk^2 with sk = s1*(a + b*k); simplest: assume sk = c*s1 const -> fit s1, c per axis
ks = np.array([r[0] for r in rows]); V = np.array([r[1] ** 2 for r in rows])
for ax in range(3):
    A_ = np.column_stack([ks, np.ones_like(ks)]); (s1sq, sksq), *_ = np.linalg.lstsq(A_, V[:, ax], rcond=None)
    print('  axis %d: pair noise s1 = %.4f deg, direct-gap noise (assumed k-independent) = %.4f deg' % (ax, np.sqrt(max(s1sq, 0)), np.sqrt(max(sksq, 0))))

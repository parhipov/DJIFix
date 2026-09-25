"""Per-second HF jitter (4-24.5 Hz, px of a 4K frame, |x,y,roll|) on calm seconds: distribution per variant."""
import sys, numpy as np
sys.path.insert(0, 'scripts')
from analyze_full import load, band_series, tel_rate
names = ['orig', 'control', 'fixed', 'R', 'RT', 'RW', 'RE', 'F4', 'F2']
for clip in sys.argv[1:]:
    D = {}
    for nm in names:
        try:
            fps, r = load(clip, nm); D[nm] = r
        except FileNotFoundError:
            pass
    n = min(len(v) for v in D.values()); D = {k: v[:n] for k, v in D.items()}
    ok = np.all([np.isfinite(v).all(1) for v in D.values()], axis=0)
    rate = tel_rate(clip)[:n]
    sec = int(fps); ns = n // sec
    calm_s = np.array([(rate[i*sec:(i+1)*sec] < 45).mean() > 0.9 and ok[i*sec:(i+1)*sec].mean() > 0.95 for i in range(ns)])
    print('== %s: %d calm seconds of %d' % (clip, calm_s.sum(), ns))
    base = None
    for nm, v in D.items():
        S = band_series(v, ok, fps, 4, 24.5)
        per = np.array([np.sqrt((S[i*sec:(i+1)*sec] ** 2).sum(1).mean()) for i in range(ns)])[calm_s]
        if nm == 'orig':
            base = per; worst = np.argsort(base)[-10:]
        print('  %-8s median %5.2f  p90 %5.2f  p99 %5.2f  max %5.2f | on orig\'s 10 worst seconds: %5.2f (orig %5.2f)' % (
            nm, np.median(per), np.percentile(per, 90), np.percentile(per, 99), per.max(), per[worst].mean(), base[worst].mean()))

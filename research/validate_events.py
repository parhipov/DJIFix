#!/usr/bin/env python3
"""Re-measure every detected event at full resolution, as an independent check.

The correction is only as trustworthy as the image measurement behind it, so each
event window is measured again at scale 1.0 with a denser feature set. If the
discrepancy reproduces, it is the telemetry; if it does not, it was my noise.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import numpy as np, time
from rollfix import clusters, CORNER_PX_PER_DEG as C
from imagerot import affine_flow

V = r'F:\36\DJI_20260905181949_0005_D.MP4'
d = np.load('artifacts/main/roll_clip.npz')
img = d['roll']
telroll = np.load('artifacts/main/telroll.npy')
rate = np.load('artifacts/main/rate_frame.npy')
n = min(len(img), len(telroll))
img, telroll, rate = img[:n], telroll[:n], rate[:n]
disc = img - telroll
sel = np.isfinite(disc) & (np.abs(np.nan_to_num(disc)) > 8 / C) & (rate < 60)
events = clusters(sel, 5)

MARGIN = 10
roll_hi = np.full(n, np.nan)
t0 = time.time()
todo = sum(min(n, e[-1] + MARGIN + 1) - max(0, e[0] - MARGIN) for e in events)
done = 0
for k, e in enumerate(events):
    a = max(0, e[0] - MARGIN)
    b = min(n, e[-1] + MARGIN + 1)
    r = affine_flow(V, a, b - a, scale=1.0, max_pts=3000, radius_frac=0.6)
    roll_hi[a:b] = r['roll_deg']
    done += b - a
    np.savez_compressed('artifacts/main/roll_events_fullres.npz', roll_hi=roll_hi)
    el = time.time() - t0
    print('event %2d/%d  frames %d-%d  %.0fs elapsed, ~%.0fs left'
          % (k + 1, len(events), a, b, el, el / done * (todo - done)), flush=True)
print('done in %.0fs' % (time.time() - t0))

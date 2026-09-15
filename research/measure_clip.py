#!/usr/bin/env python3
"""Measure the image roll (affine curl) for the whole clip, saving as it goes."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import sys, time
import numpy as np
from imagerot import affine_flow

V = sys.argv[1] if len(sys.argv) > 1 else r'F:\36\DJI_20260905181949_0005_D.MP4'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'artifacts/main/roll_clip.npz'
TOTAL = 4612
CHUNK = 400

roll = np.full(TOTAL, np.nan)
shx = np.full(TOTAL, np.nan)
shy = np.full(TOTAL, np.nan)
div = np.full(TOTAL, np.nan)
npts = np.zeros(TOTAL, dtype=int)

t0 = time.time()
for start in range(0, TOTAL, CHUNK):
    n = min(CHUNK, TOTAL - start)
    d = affine_flow(V, start, n, scale=0.5, max_pts=1200)
    roll[start:start + n] = d['roll_deg']
    shx[start:start + n] = d['shift_x_deg']
    shy[start:start + n] = d['shift_y_deg']
    div[start:start + n] = d['div']
    npts[start:start + n] = d['npts']
    np.savez_compressed(OUT, roll=roll, shift_x=shx, shift_y=shy, div=div, npts=npts)
    el = time.time() - t0
    print('%d/%d  %.0fs elapsed, ~%.0fs left' % (start + n, TOTAL, el,
          el / (start + n) * (TOTAL - start - n)), flush=True)
print('done in %.0fs -> %s' % (time.time() - t0, OUT))

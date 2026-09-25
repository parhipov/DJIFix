#!/usr/bin/env python3
"""
Research prototype, not project code: remove telemetry spikes, nothing else.

Rule: a spike is a jump of the telemetry increment from its 5-frame median of at
least 0.35 deg/frame on any axis (rate < 200 deg/s), at a frame where the image
is trustworthy (its two roll estimators agree above 4 Hz, >= 300 tracked points,
on the spike frame and its neighbours) and the image confirms the jump did not
happen (the image-minus-telemetry difference points against the jump). The
spike frame is put back on its median; the attitude offset this leaves decays
over 1.5 s, forwards only, as in fix_pipeline.

    python research/ab_bench/spikefix.py TELEMETRY.mp4 IMAGE.npz OUT.mp4

TELEMETRY is the source clip or any motion-data sidecar built from it (e.g. the
current fix), so the rule can be applied on top of another correction; the
image measurement is always the source clip's.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'research'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import diagnose                    # noqa: E402
import quat as Q                   # noqa: E402
import sidecar                     # noqa: E402
from align import slerp_series     # noqa: E402
from fix_pipeline import shifted_cumsum   # noqa: E402
from rollfix import qmul           # noqa: E402
from rotmath import rotation_vector_quat   # noqa: E402
from timing import load_sorted     # noqa: E402

SPIKE_DEG = 0.35
RATE_MAX = 200.0
RELEASE_S = 1.5


def build(telemetry, image_npz, out):
    diagnose.EDGE_S = 1.0 / 50.0         # short example clips: no take-off/landing margin, one frame at each end
    clip, s, flags, info = diagnose.analyse(telemetry, image_npz, None, spike_deg=SPIKE_DEG)
    fps, n = s['fps'], s['n']
    inc, rn = s['inc'], s['rn']
    ok_img = s['img_ok'].copy()
    ok_near = ok_img.copy()
    for k in (1, 2):                      # the neighbours carry the median the spike is judged against
        ok_near[k:] &= ok_img[:-k]
        ok_near[:-k] &= ok_img[k:]
    cand = (rn >= SPIKE_DEG) & s['valid'] & (s['rate'] < RATE_MAX)
    confirmed = info['spike']['confirmed']
    spike = cand & ok_near & confirmed
    from scipy.ndimage import median_filter
    med = np.column_stack([median_filter(np.where(s['valid'], inc[:, k], 0.0), size=5, mode='nearest') for k in range(3)])
    c = np.where(spike[:, None], med - inc, 0.0)
    th = shifted_cumsum(c)
    decay = np.exp(-1.0 / (RELEASE_S * fps))
    y = np.zeros_like(th)
    for f in range(1, n):
        y[f] = (y[f - 1] if spike[f] else y[f - 1] * decay) + (th[f] - th[f - 1])
    _, samples, frames, ts, qs = load_sorted(telemetry)
    t = np.array([x.t_us for x in samples], dtype=float)
    ft = np.arange(n) * (1e6 / fps)
    base = slerp_series(ts, qs, t)
    thq = np.column_stack([np.interp(t, ft, y[:, k]) for k in range(3)])
    newq = qmul(base, rotation_vector_quat(thq))
    newq /= np.linalg.norm(newq, axis=1, keepdims=True)
    inverted = np.array([x.inverted for x in samples])
    qq = np.where(inverted[:, None], -newq, newq)
    edits = {(x.frame, x.index_in_frame): Q.norm(Q.gyroflow_to_dji(tuple(map(float, v)))) for x, v in zip(samples, qq)}
    sidecar.build(telemetry, out, edits)
    report = []
    for f in np.flatnonzero(cand):
        report.append(dict(t_s=round(f / fps, 2), jump_deg=round(float(rn[f]), 3),
                           axis='xyz'[int(np.argmax(np.abs(inc[f] - med[f])))], rate_deg_s=round(float(s['rate'][f]), 1),
                           image_ok=bool(ok_near[f]), image_confirms=bool(confirmed[f]), removed=bool(spike[f])))
    return dict(telemetry=telemetry, out=out, removed=int(spike.sum()), candidates=report,
                peak_offset_deg=round(float(np.abs(y).max()), 3))


if __name__ == '__main__':
    print(json.dumps(build(*sys.argv[1:4]), indent=1, ensure_ascii=False))

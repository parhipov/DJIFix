"""
Lens model for the image measurement: from a Gyroflow lens profile, or fitted.

The rotation of the tracked rays depends on the lens model the pixels are
lifted through. The stock model comes from the clip's own metadata; when the
camera carries another lens (the test Lite had a Flywoo O4 Wide) the image's
pitch and yaw come out scaled by roughly f_true / f_assumed. Roll does not care
(rotation about the optical axis is invariant to radial distortion).

Two ways to get the right model:

  load_gyroflow_lens(path, w, h)   the same profile you pick in Gyroflow, rescaled
                                   from its calibration size to the video size
  fit_focal_scale(...)             one number: the focal scale at which the image
                                   pitch/yaw match the telemetry on fast frames
                                   (the telemetry's *scale* is right; its errors are
                                   spikes and jitter). Karpenko et al. 2011 did the
                                   same with a gyro to calibrate focal length.
"""
import json

import cv2
import numpy as np

from imagerot import Lens, _track, rotation_kabsch
from rotmath import logv, qconj, qmul, slerp_series

AXIS_MAP = ((0, 1, 2), (-1, 1, 1))     # OpenCV camera axes -> telemetry frame


def load_gyroflow_lens(path, width, height):
    """(f_px, cx, cy, D[4]) for a video of width x height from a Gyroflow profile JSON."""
    with open(path, encoding='utf-8') as fh:
        prof = json.load(fh)
    fp = prof['fisheye_params']
    K = np.array(fp['camera_matrix'], float)
    D = list(map(float, fp['distortion_coeffs']))[:4]
    cal = prof.get('calib_dimension') or {}
    cw, ch = float(cal.get('w') or width), float(cal.get('h') or height)
    sx, sy = width / cw, height / ch
    f = 0.5 * (K[0, 0] * sx + K[1, 1] * sy)
    return dict(f=float(f), cx=float(K[0, 2] * sx), cy=float(K[1, 2] * sy), D=D,
                name=prof.get('name') or prof.get('identifier') or path, source=path)


def _telemetry_pair_rotations(clip, samples, frames, pairs, fps):
    """Telemetry rotation vectors (deg, telemetry frame) for frame pairs (f -> f+1) at pts."""
    t = np.array([s.t_us for s in samples], float)
    q = np.array([s.q_cam for s in samples], float)
    o = np.argsort(t, kind='stable')
    ts, qs = t[o], q[o]
    k = np.concatenate([[True], np.diff(ts) > 0])
    ts, qs = ts[k], qs[k]
    ft = np.array([[f, f + 1] for f in pairs], float) * (1e6 / fps)
    qq = slerp_series(ts, qs, ft.ravel()).reshape(-1, 2, 4)
    s = np.sign(np.sum(qq[:, 0] * qq[:, 1], axis=1))
    s[s == 0] = 1
    return logv(qmul(qconj(qq[:, 0]), qq[:, 1] * s[:, None]))


def select_pairs(clip, samples, frames, fps, n_pairs=160, rate_lo=40.0, rate_hi=160.0):
    """Frame indices spread over the clip whose telemetry rate is in [rate_lo, rate_hi] deg/s."""
    n = len(frames)
    T = _telemetry_pair_rotations(clip, samples, frames, np.arange(n - 1), fps)
    rate = np.linalg.norm(T, axis=1) * fps
    ok = np.flatnonzero((rate > rate_lo) & (rate < rate_hi) & (np.arange(n - 1) < n - 10))
    if len(ok) == 0:
        return np.array([], int)
    if len(ok) <= n_pairs:
        return ok
    pick = np.linspace(0, len(ok) - 1, n_pairs).round().astype(int)
    return ok[np.unique(pick)]


def track_pairs(video, pairs, scale=0.25, max_pts=1500, radius_frac=0.6, progress=None):
    """Tracked point pairs (full-resolution pixels) for the given frame indices, by seeking."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit('cannot open %s' % video)
    out = {}
    mask = None
    for i, f in enumerate(pairs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok1, im1 = cap.read()
        ok2, im2 = cap.read()
        if not (ok1 and ok2):
            continue
        g1 = cv2.cvtColor(cv2.resize(im1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(im2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if mask is None:
            h, w = g1.shape
            yy, xx = np.mgrid[0:h, 0:w]
            mask = (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (radius_frac * min(w, h)) ** 2).astype(np.uint8) * 255
        a, b = _track(g1, g2, mask, max_pts)
        if a is not None and len(a) >= 150:
            out[int(f)] = (a / scale, b / scale)
        if progress and i % 20 == 0:
            progress(i, len(pairs))
    cap.release()
    return out


def rotations_with_lens(tracks, f, cx, cy, D):
    """Kabsch rotation (deg, telemetry frame) per tracked pair for one lens model."""
    lens = Lens(f=f, cx=cx, cy=cy, D=D, scale=1.0)
    keys = sorted(tracks)
    R = np.full((len(keys), 3), np.nan)
    for i, k in enumerate(keys):
        a, b = tracks[k]
        Rm, res, w = rotation_kabsch(lens.to_rays(a), lens.to_rays(b))
        rv = np.degrees(cv2.Rodrigues(Rm)[0].ravel())
        R[i] = rv[list(AXIS_MAP[0])] * np.array(AXIS_MAP[1], float)
    return np.array(keys), R


def fit_focal_scale(video, clip, samples, frames, fps, base, scales=np.linspace(0.6, 1.5, 19),
                    n_pairs=160, progress=None):
    """Focal scale that brings the image pitch/yaw gains against the telemetry to 1.

    base = dict(f, cx, cy, D) of the starting lens model. Returns a report dict with
    the chosen scale, the gains before and after, and whether the fit is trusted.
    """
    pairs = select_pairs(clip, samples, frames, fps, n_pairs)
    rep = dict(pairs_requested=int(len(pairs)), scale=1.0, trusted=False)
    if len(pairs) < 40:
        rep['reason'] = 'fewer than 40 fast frame pairs'
        return rep
    tracks = track_pairs(video, pairs, progress=progress)
    if len(tracks) < 40:
        rep.update(pairs_tracked=len(tracks), reason='tracking failed on most pairs')
        return rep
    keys = np.array(sorted(tracks))
    T = _telemetry_pair_rotations(clip, samples, frames, keys, fps)

    def gains(R):
        g = []
        for ax in (0, 1):
            m = np.isfinite(R[:, ax])
            g.append(float(np.polyfit(T[m, ax], R[m, ax], 1)[0]) if m.sum() >= 20 and np.std(T[m, ax]) > 1e-6 else np.nan)
        return g

    results = []
    for s in scales:
        _, R = rotations_with_lens(tracks, base['f'] * s, base['cx'], base['cy'], base['D'])
        gx, gy = gains(R)
        resid = float(np.nanmedian(np.linalg.norm(R[:, :2] - T[:, :2], axis=1)))
        results.append((s, gx, gy, resid))
    results = np.array(results, float)
    # cost: both gains at 1; residual as the tie-breaker
    cost = (results[:, 1] - 1) ** 2 + (results[:, 2] - 1) ** 2 + 0.1 * (results[:, 3] / np.nanmin(results[:, 3])) ** 2
    k = int(np.nanargmin(cost))
    s_best = float(results[k, 0])
    # parabolic refinement on the cost
    if 0 < k < len(results) - 1 and np.isfinite(cost[k - 1:k + 2]).all():
        y0, y1, y2 = cost[k - 1], cost[k], cost[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            s_best = float(results[k, 0] + 0.5 * (y0 - y2) / den * (results[1, 0] - results[0, 0]))
    base_row = results[np.argmin(np.abs(results[:, 0] - 1.0))]
    rep.update(pairs_tracked=len(tracks), scale=s_best,
               gains_at_1=[float(base_row[1]), float(base_row[2])], resid_at_1=float(base_row[3]),
               gains_at_best=[float(results[k, 1]), float(results[k, 2])], resid_at_best=float(results[k, 3]),
               table=[[round(float(v), 3) for v in row] for row in results])
    # trust: both gains move towards 1 and end within 15 %, and the optimum is inside the range
    gb = rep['gains_at_best']
    rep['trusted'] = bool(np.isfinite(gb).all() and max(abs(gb[0] - 1), abs(gb[1] - 1)) < 0.15
                          and scales[0] < s_best < scales[-1])
    if not rep['trusted']:
        rep['reason'] = 'gains do not converge to 1 within the scan'
        rep['scale'] = 1.0
    return rep

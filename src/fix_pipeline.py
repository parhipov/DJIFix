#!/usr/bin/env python3
"""
The one telemetry fix: DJI video + image measurement in, Gyroflow motion-data file out.

    python src/fix_pipeline.py "F:\\36\\clip.MP4" --image artifacts/main/clip/clip_image.npz \
                               -o artifacts/main/clip/clip_telemetry_fixed.mp4

Four things are done to the telemetry, in this order, and everything is measured
against the picture rather than assumed:

  1. Timing.  The constant is measured on the clip: the content shift at which the
     telemetry roll read at pts matches the image roll (+0.2 ms on the Pro,
     -1.2 ms on the Lite, i.e. Gyroflow's pts is right and no shift is needed);
     on top, the per-frame exposure variation (timing.py).
  2. Roll.    DJI's fused roll jitters by ~0.12 deg per frame regardless of what the
     camera did. The roll measured from the image (curl of the affine flow,
     parallax-immune) replaces it above 4 Hz where the image is reliable
     (rate-weighted, R0 = 40 deg/s), exactly as rollfix.py did.
  3. Pitch/yaw noise. The calibrated Wiener keep-gain curve (denoise.py), gentle.
  4. Pitch/yaw events. Where the 5-frame image rotation (homography and pure
     rotation fit, both from measure_rotation.py) disagrees with the telemetry
     for several consecutive windows, and the two image estimates agree with
     each other, the telemetry's pitch/yaw increments are replaced by the image's
     inside that window (tapered, zero-net so the attitude returns to the
     telemetry afterwards). Between the two image estimates the one closer to the
     telemetry is used: the repair is never larger than what both agree on.

The control file (timing only, no content change) is written next to the result
so an A/B in Gyroflow isolates the content repair from the timing.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quat as Q                                    # noqa: E402
import sidecar                                      # noqa: E402
from align import slerp_series                      # noqa: E402
from denoise import (BAND_HZ, KEEP, KEEP_PANTILT, band_sos, frame_increments,   # noqa: E402
                     wiener_axis)
from imagerot import best_axis_map                  # noqa: E402
from rollfix import U as U_DEFAULT, logv, qmul, qconj   # noqa: E402
from rotmath import limit_rotation, rotation_vector_quat    # noqa: E402
from timing import frame_shift_us, load_sorted     # noqa: E402

EXPECTED_AXIS_MAP = ((0, 1, 2), (-1, 1, 1))   # OpenCV camera axes -> Gyroflow camera frame


def image_rotations(d, n_frames, axis_map=EXPECTED_AXIS_MAP):
    """Per-frame image rotation vectors in the telemetry frame, plus the curl roll.

    Prefers the per-pair pure-rotation fit (`kabsch1_deg`, ray space, so a pan
    does not leak into the roll the way it does in the affine curl when the
    tracked points sit mostly in one half of the frame); falls back to the curl
    in undistorted coordinates, then to the raw curl. Returns (inc_img (N,3) or
    None, roll_curl (N,), source name).
    """
    n = n_frames
    inc_img = None
    if 'kabsch1_deg' in d:
        k = min(len(d['kabsch1_deg']), n)
        inc_img = np.full((n, 3), np.nan)
        inc_img[:k] = d['kabsch1_deg'][:k][:, list(axis_map[0])] * np.array(axis_map[1], float)
    key = 'roll_n_deg' if 'roll_n_deg' in d else 'roll_deg'
    k = min(len(d[key]), n)
    roll = np.full(n, np.nan)
    roll[:k] = d[key][:k]
    return inc_img, roll, ('kabsch1' if inc_img is not None else key)


def scalar_gain(inc_comp, img_comp, rate, lo=20.0, hi=150.0, min_frames=200):
    """Image/telemetry gain and correlation of one axis on the fast frames."""
    m = np.isfinite(img_comp) & np.isfinite(inc_comp) & (rate > lo) & (rate < hi)
    if m.sum() < min_frames:
        return 1.0, float('nan'), int(m.sum())
    g = float(np.polyfit(inc_comp[m], img_comp[m], 1)[0])
    c = float(np.corrcoef(inc_comp[m], img_comp[m])[0, 1])
    return g, c, int(m.sum())


def measure_timing_shift(ts, qs, fps, n_frames, img_roll, shifts=np.arange(-12.0, 12.01, 1.0),
                         rate_lo=30.0, rate_hi=400.0):
    """Content shift (ms) at which the telemetry roll, read at pts, best matches the image.

    Uses the roll (z) only: it is the one axis the image measures with unit gain
    on both cameras. Frames faster than `rate_lo` carry the timing information.
    Returns (shift_ms, rms_at_best, rms_at_zero, n_frames_used).
    """
    ft = np.arange(n_frames + 1) * (1e6 / fps)
    inc0 = frame_increments(ts, qs, ft)
    rate0 = np.linalg.norm(inc0, axis=1) * fps
    m = np.isfinite(img_roll) & (rate0 > rate_lo) & (rate0 < rate_hi)
    m[-8:] = False
    if m.sum() < 200:
        return 0.0, float('nan'), float('nan'), int(m.sum())
    g = float(np.polyfit(inc0[m, 2], img_roll[m], 1)[0])
    res = []
    for sh in shifts:
        inc = frame_increments(ts, qs, ft + sh * 1000.0)
        res.append(np.sqrt(np.mean((img_roll[m] - g * inc[m, 2]) ** 2)))
    res = np.array(res)
    k = int(np.argmin(res))
    frac = 0.0
    if 0 < k < len(res) - 1:
        y0, y1, y2 = res[k - 1], res[k], res[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den > 0 else 0.0
    step = shifts[1] - shifts[0]
    return float(shifts[k] + frac * step), float(res[k]), float(res[len(res) // 2]), int(m.sum())


def fit_roll_axis(inc, img_roll, rate, lo=20.0, hi=150.0):
    """Optical axis in the telemetry body frame, from the fast frames.

    Returns (unit axis, gain, correlation, n). Falls back to rollfix.U when the
    fit is not trustworthy (correlation below 0.9 or too few frames).
    """
    m = np.isfinite(img_roll) & (rate > lo) & (rate < hi)
    if m.sum() < 200:
        u = U_DEFAULT
        return u / np.linalg.norm(u), float(np.linalg.norm(u)), float('nan'), int(m.sum())
    u, *_ = np.linalg.lstsq(inc[m], img_roll[m], rcond=None)
    pred = inc[m] @ u
    corr = float(np.corrcoef(pred, img_roll[m])[0, 1])
    if not np.isfinite(corr) or corr < 0.9:
        u = U_DEFAULT
        return u / np.linalg.norm(u), float(np.linalg.norm(u)), corr, int(m.sum())
    return u / np.linalg.norm(u), float(np.linalg.norm(u)), corr, int(m.sum())


def perp_basis(uh):
    a = np.array([1.0, 0.0, 0.0])
    a = a - uh * np.dot(a, uh)
    a /= np.linalg.norm(a)
    return a, np.cross(uh, a)


def fit_axis(inc, img, rate, lo=20.0, hi=150.0, min_corr=0.8, min_frames=200):
    """Body-frame axis whose increment best predicts an image channel, or None."""
    m = np.isfinite(img) & (rate > lo) & (rate < hi)
    if m.sum() < min_frames:
        return None
    v, *_ = np.linalg.lstsq(inc[m], img[m], rcond=None)
    g = float(np.linalg.norm(v))
    if g < 1e-9:
        return None
    corr = float(np.corrcoef(inc[m] @ v, img[m])[0, 1])
    if not np.isfinite(corr) or corr < min_corr:
        return None
    return v / g, g, corr, int(m.sum())


def hf_only(keep, floor_hz=4.0):
    """A keep-gain curve that leaves everything below `floor_hz` alone."""
    k = np.array(keep, float).copy()
    k[BAND_HZ < floor_hz] = 1.0
    return k


def spectral_keep(tel, img, fps, calm, floor_hz=4.0, min_frames=400, nperseg=128):
    """Wiener keep-gain on BAND_HZ from the image/telemetry power ratio on calm runs.

    This is how denoise.KEEP was derived once for the Pro; here it is redone per
    clip, from contiguous calm stretches, so a different unit or firmware gets its
    own curve. Below `floor_hz` nothing is touched. Returns (keep, info) or None.
    """
    from scipy.signal import welch
    ok = calm & np.isfinite(img) & np.isfinite(tel)
    runs = [r for r in clusters(ok, gap=0) if len(r) >= nperseg]
    total = sum(len(r) for r in runs)
    if total < min_frames:
        return None
    pt = pi = None
    for r in runs:
        f, a = welch(tel[r], fs=fps, nperseg=nperseg, detrend='linear')
        _, b = welch(img[r], fs=fps, nperseg=nperseg, detrend='linear')
        pt = a * len(r) if pt is None else pt + a * len(r)
        pi = b * len(r) if pi is None else pi + b * len(r)
    ratio = pi / np.maximum(pt, 1e-18)
    keep_f = np.clip(ratio, 0.03, 1.0)
    keep = np.interp(BAND_HZ, f, keep_f)
    keep[BAND_HZ < floor_hz] = 1.0
    return keep, dict(frames=int(total), runs=len(runs),
                      excess_by_band={('%g-%gHz' % (lo, hi)): float(np.mean(1 / np.maximum(ratio[(f >= lo) & (f < hi)], 1e-9)))
                                      for lo, hi in ((4, 8), (8, 12), (12, 16), (16, 25)) if ((f >= lo) & (f < hi)).any()})


def shifted_cumsum(c):
    """theta[f] = sum of corrections before f (see rollfix.py on the off-by-one)."""
    return np.concatenate([np.zeros((1,) + c.shape[1:]), np.cumsum(c, axis=0)[:-1]])


def clusters(mask, gap=2):
    """Runs of True indices; holes of up to `gap` frames do not break a run."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] > gap + 1:
            out.append(cur)
            cur = [i]
        else:
            cur.append(i)
    out.append(cur)
    return out


def window_disagreement(qf, d, fps, gap, rate, fast=40.0, baseline_s=2.0, estimator='mean'):
    """Per-window image-vs-telemetry rotation difference, mapped and vetted.

    Returns dict with D (N,3) chosen disagreement in deg per window (NaN where
    unusable), reliable mask, dH, dK, axis map info.
    """
    n = len(qf) - 1
    nw = min(len(d['homog_deg']), n - gap)
    q0, q1 = qf[:nw], qf[gap:gap + nw]
    s = np.sign(np.sum(q0 * q1, axis=1))
    s[s == 0] = 1
    T = logv(qmul(qconj(q0), q1 * s[:, None]))              # telemetry, deg per window
    H = d['homog_deg'][:nw]
    K = d['kabsch_deg'][:nw]
    npts = d['npts_win'][:nw]
    ok = np.isfinite(H).all(1) & np.isfinite(K).all(1) & (npts >= 150)
    wrate = np.linalg.norm(T, axis=1) / gap * fps
    fit = ok & (wrate > fast) & (wrate < 300)
    if fit.sum() >= 100:
        err, perm, signs = best_axis_map(H[fit], T[fit])
        # the map must be a clear winner; otherwise use the known one
        alt = np.mean(np.linalg.norm(H[fit][:, EXPECTED_AXIS_MAP[0]] * np.array(EXPECTED_AXIS_MAP[1]) - T[fit], axis=1))
        if err > 0.9 * alt:
            perm, signs = EXPECTED_AXIS_MAP
            err = alt
    else:
        perm, signs = EXPECTED_AXIS_MAP
        err = float('nan')
    S = np.array(signs, float)
    Hm, Km = H[:, list(perm)] * S, K[:, list(perm)] * S
    dH, dK = Hm - T, Km - T
    # Both image estimates under-read pan and tilt by a slowly varying 25-50 %
    # (translation over the ground is confounded with rotation) and each does so
    # differently, so the raw differences disagree with each other during every
    # sustained pan. What an event looks like is a *change* in the disagreement
    # over a fraction of a second; so each estimator's own slow baseline (running
    # median over `baseline_s`) is removed first and everything below is judged
    # on the remaining detail.
    from scipy.ndimage import median_filter
    size = max(3, int(round(baseline_s * fps)) | 1)

    def detail(dx):
        out = np.full_like(dx, np.nan)
        idx = np.arange(len(dx))
        for k in range(3):
            col = dx[:, k].copy()
            good = ok & np.isfinite(col)
            if good.sum() < 10:
                continue
            col[~good] = np.interp(idx[~good], idx[good], col[good])
            out[:, k] = col - median_filter(col, size=size, mode='nearest')
        out[~ok] = np.nan
        return out
    dHd, dKd = detail(dH), detail(dK)
    nH, nK = np.linalg.norm(np.nan_to_num(dHd), axis=1), np.linalg.norm(np.nan_to_num(dKd), axis=1)
    # the two image estimates must agree with each other about the telemetry error
    # (same direction, magnitudes within a factor ~2)
    agree = ok & np.isfinite(dHd).all(1) & np.isfinite(dKd).all(1) & (np.sum(np.nan_to_num(dHd) * np.nan_to_num(dKd), axis=1) > 0) & \
        (np.linalg.norm(np.nan_to_num(dHd - dKd), axis=1) <= 0.6 * np.maximum(nH, nK) + 0.3)
    if estimator == 'max':
        # the image under-reads pan/tilt, so the larger of the two is the less wrong
        D = np.where((nH >= nK)[:, None], dHd, dKd)
    elif estimator == 'min':
        D = np.where((nH <= nK)[:, None], dHd, dKd)
    else:
        D = 0.5 * (dHd + dKd)
    D[~ok] = np.nan
    return dict(D=D, reliable=agree, dH=dH, dK=dK, dHd=dHd, dKd=dKd, T=T, ok=ok,
                axis_perm=list(perm), axis_signs=list(map(int, signs)), axis_fit_err=float(err),
                axis_fit_n=int(fit.sum()), wrate=wrate)


def detect_spikes(inc, fine, fine_raw, rate, uh, valid, spike_deg=0.35, rate_max=150.0, gap=5):
    """One- or two-frame jumps of the telemetry's pitch/yaw increment that the image does not show.

    The Lite at 55.72 s: +1.77 deg in one frame against 0.3 around it, i.e. 88 deg/s
    that never happened; Gyroflow then yanks the frame. Found on the telemetry
    itself (distance from the 5-frame median) and confirmed by the image (gain-
    normalised and raw) disagreeing with the jump. Returns (spike mask, c_spike
    (N,3) = the correction that puts the frame back on its median, contaminated
    mask = frames whose 5-frame window averages contain a spike).
    """
    n = len(inc)
    inc_pt = inc - np.outer(inc @ uh, uh)
    from scipy.ndimage import median_filter
    med5 = np.column_stack([median_filter(np.where(valid, inc_pt[:, k], 0.0), size=5, mode='nearest') for k in range(3)])
    r = inc_pt - med5
    rn = np.linalg.norm(r, axis=1)
    confirm = np.zeros(n, bool)
    for witness in (fine, fine_raw):
        if witness is None:
            continue
        w = np.nan_to_num(np.asarray(witness, float))
        w = w - np.outer(w @ uh, uh)
        confirm |= np.sum(w * r, axis=1) < -0.5 * rn ** 2      # the image says the jump did not happen
    spike = (rn >= spike_deg) & confirm & (rate < rate_max) & valid
    c_spike = np.where(spike[:, None], -r, 0.0)
    near = np.convolve(spike.astype(int), np.ones(2 * gap + 1, int), mode='same') > 0
    return spike, c_spike, near & ~spike


def event_correction(delta_pf, reliable_pf, rate, fps, uh, thr_deg=0.10, min_len_s=0.10,
                     margin=3, rate_max=150.0, gain=1.0, inc=None, ratio=0.5, baseline_s=2.0,
                     fine=None, merge_s=0.12, model='image', zero_net=True, fine_alt=None,
                     spike_deg=0.35, free_net_windows=None, fine_raw=None, spike=None, contaminated=None):
    """Per-frame increment correction (deg) that only touches vetted pitch/yaw events.

    delta_pf     per-frame disagreement image - telemetry (deg/frame), NaN = unknown
    reliable_pf  per-frame flag: both image estimates agree
    inc          telemetry increments; an event must disagree by at least `ratio`
                 of the telemetry's own pitch/yaw motion, because the image
                 under-reads pan and tilt by 25-50 % (translation over the ground
                 is confounded with rotation), so a plain difference would fire on
                 every pan. A telemetry swing the image does not show, or an image
                 motion the telemetry lacks, both pass; a scaled pan does not.
    baseline_s   a running median this long is removed from the disagreement
                 first: the parallax bias of forward flight is slow, the events
                 are not.
    """
    from scipy.ndimage import median_filter
    n = len(delta_pf)
    dpt = delta_pf - np.outer(np.nan_to_num(delta_pf) @ uh, uh)   # drop the roll component
    finite = np.isfinite(dpt).all(1)
    filled = dpt.copy()
    idx = np.arange(n)
    for k in range(3):
        col = filled[:, k]
        good = np.isfinite(col)
        if good.sum() >= 2:
            col[~good] = np.interp(idx[~good], idx[good], col[good])
        else:
            col[:] = 0.0
    size = max(3, int(round(baseline_s * fps)) | 1)
    base = np.column_stack([median_filter(filled[:, k], size=size, mode='nearest') for k in range(3)])
    dpt = np.where(finite[:, None], filled - base, np.nan)
    mag = np.linalg.norm(np.nan_to_num(dpt), axis=1)
    sel = reliable_pf & finite & (mag >= thr_deg) & (rate < rate_max)
    if inc is not None:
        inc_pt = inc - np.outer(inc @ uh, uh)
        sel &= mag >= ratio * np.linalg.norm(inc_pt, axis=1)
    # Telemetry spikes are detected in build() (detect_spikes) before the Wiener
    # stage and repaired there; here they only extend the event windows, and the
    # frames whose 5-frame window averages contain a spike are not allowed to use
    # those averages (the spike would be subtracted a second time).
    spike = np.zeros(n, bool) if spike is None else np.asarray(spike, bool)
    contaminated = np.zeros(n, bool) if contaminated is None else np.asarray(contaminated, bool)
    sel |= spike
    reliable_pf = reliable_pf | spike
    min_len = max(2, int(round(min_len_s * fps)))
    c = np.zeros((n, 3))
    events = []
    # Neighbouring detections are one event: separate windows with soft edges
    # would put dips in the correction exactly in the middle of the excursion.
    merge_gap = max(2, int(round(merge_s * fps)))
    runs = [r for r in clusters(sel, gap=2) if len(r) >= min_len or spike[r].any()]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][-1] <= merge_gap:
            merged[-1] = list(range(merged[-1][0], r[-1] + 1))
        else:
            merged.append(list(r))
    # Inside an event, per-frame detail if a per-pair image rotation is available
    # (`fine`): the 5-frame windows are a 100 ms moving average and blur a 250 ms
    # excursion into the wrong shape -- more gain then does not help.
    src = dpt
    ok_f = np.zeros(n, bool)
    fine = None if fine is None else np.asarray(fine, float)
    if fine is not None:
        f = fine - np.outer(np.nan_to_num(fine) @ uh, uh)
        ff = f.copy()
        for k in range(3):
            col = ff[:, k]
            good = np.isfinite(col)
            if good.sum() >= 2:
                col[~good] = np.interp(idx[~good], idx[good], col[good])
            else:
                col[:] = 0.0
        fbase = np.column_stack([median_filter(ff[:, k], size=size, mode='nearest') for k in range(3)])
        fdet = ff - fbase
        # A single-frame value is only trusted when the independent shift-channel
        # estimate (`fine_alt`) agrees with it; otherwise the 5-frame window value
        # is used for that frame. (A blanket 3-frame median here smeared away the
        # very thing that matters: the telemetry's own one-frame spikes.)
        ok_f = np.isfinite(f).all(1)
        if fine_alt is not None:
            alt = fine_alt - np.outer(np.nan_to_num(fine_alt) @ uh, uh)
            aa = np.nan_to_num(alt)
            abase = np.column_stack([median_filter(aa[:, k], size=size, mode='nearest') for k in range(3)])
            adet = aa - abase
            ok_f &= np.isfinite(alt).all(1) & (np.linalg.norm(fdet - adet, axis=1) <= 0.5 + 0.5 * np.linalg.norm(fdet, axis=1))
        src = np.where(ok_f[:, None], fdet, np.nan_to_num(dpt))
    src = np.where((contaminated & ~ok_f)[:, None], 0.0, src)
    src = np.where(spike[:, None], 0.0, src)          # fixed separately, see detect_spikes
    for ev in merged:
        a, b = max(0, ev[0] - margin), min(n, ev[-1] + margin + 1)
        ramp = min(3, (b - a) // 2)
        win = np.ones(b - a)
        if ramp > 0:
            edge = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ramp) + 1) / (ramp + 1))
            win[:ramp] = edge
            win[-ramp:] = edge[::-1]
        free_here = any(lo <= a / fps <= hi or lo <= b / fps <= hi for lo, hi in (free_net_windows or []))
        has_spike = bool(spike[a:b].any())
        free_here = free_here or has_spike
        if model == 'lowpass':
            # Local smoothing of the telemetry's own pitch/yaw inside the window
            # (the user's 8 Hz full-rate trial, but only where the image says the
            # telemetry is wrong, and lower): a spike is removed, a real slow
            # manoeuvre is kept. Zero-phase, computed on a padded stretch so the
            # window edges do not ring.
            from scipy.signal import butter as _butter, sosfiltfilt as _sosfiltfilt
            pad = int(round(2.0 * fps))
            lo_i, hi_i = max(0, a - pad), min(n, b + pad)
            inc_pt = inc - np.outer(inc @ uh, uh)
            sm = _sosfiltfilt(_butter(2, lowpass_hz, fs=fps, output='sos'), np.nan_to_num(inc_pt[lo_i:hi_i]), axis=0)
            seg = (sm[a - lo_i:b - lo_i] - inc_pt[a:b]) * win[:, None] * gain
        elif model == 'bridge':
            # The user's "straight line": inside the window the telemetry's
            # pitch/yaw rate is replaced by a straight line between the mean
            # rates just before and just after the window (the bug happens in
            # level flight). The image only says *where*; it does not shape it.
            ctx = max(3, int(round(0.2 * fps)))
            inc_pt = inc - np.outer(inc @ uh, uh)
            before = inc_pt[max(0, a - ctx):a]
            after = inc_pt[b:min(n, b + ctx)]
            v0 = before.mean(axis=0) if len(before) else inc_pt[a]
            v1 = after.mean(axis=0) if len(after) else inc_pt[b - 1]
            u = np.linspace(0.0, 1.0, b - a)[:, None]
            target = v0 + (v1 - v0) * u
            seg = (target - inc_pt[a:b]) * win[:, None] * gain
        else:
            seg = np.nan_to_num(src[a:b]) * win[:, None] * gain
            if has_spike and zero_net and not any(lo <= a / fps <= hi or lo <= b / fps <= hi for lo, hi in (free_net_windows or [])):
                # the spike frames keep their full correction (the spike is the
                # error); the rest of the window is shape-only, as usual
                sp = spike[a:b]
                if (~sp).sum() > 0:
                    seg[~sp] -= seg[~sp].mean(axis=0, keepdims=True)
            elif zero_net and not free_here:
                # The image's *shape* is trustworthy (it agrees with the shift
                # channel at 0.96), its net rotation over the window is not (it
                # under-reads pan/tilt by a rate-dependent 25-50 %); left free,
                # the offsets piled up to 20 deg on the Lite. So the shape is
                # kept and the net is removed: the correction is the deviation
                # from a straight line through the window.
                seg -= seg.mean(axis=0, keepdims=True)
        c[a:b] += seg
        events.append(dict(start_s=a / fps, end_s=b / fps, frames=[int(a), int(b)],
                           peak_deg_per_frame=float(mag[ev].max()),
                           mean_rate_deg_s=float(rate[ev].mean()), weight=1.0, _seg=seg,
                           telemetry_spike=has_spike, spike_frames_s=[round(float(i / fps), 2) for i in np.flatnonzero(spike[a:b]) + a],
                           free_net=bool(free_here)))
    return c, events


def weight_events(ev_list, n, confirmed=1.0, unclear=0.5, opposed=0.0):
    """Rebuild the event correction with per-event weights from the shift check."""
    c = np.zeros((n, 3))
    for e in ev_list:
        sc = e.get('shift_check')
        w = unclear if sc is None else {'confirmed': confirmed, 'unclear': unclear, 'opposed': opposed}[sc['verdict']]
        if e.get('telemetry_spike'):
            w = 1.0      # found on the telemetry itself and confirmed by the image frame by frame
        e['weight'] = float(w)
        a, b = e['frames']
        c[a:b] += e.pop('_seg') * w
    return c


def shift_check(ev_list, c_ev, inc, d, n_frames, nimg, rate, fps, uh, baseline_s=2.0):
    """Annotate events with their agreement to the affine-shift pitch/yaw estimate."""
    from scipy.ndimage import median_filter
    keys = ('shift_yn_deg', 'shift_xn_deg') if 'shift_yn_deg' in d else ('shift_y_deg', 'shift_x_deg')
    img = np.full((n_frames, 2), np.nan)
    for k, key in enumerate(keys):
        if key in d:
            img[:nimg, k] = d[key][:nimg]
    dis = np.full((n_frames, 2), np.nan)
    size = max(3, int(round(baseline_s * fps)) | 1)
    idx = np.arange(n_frames)
    for k in range(2):
        g, corr, nfit = scalar_gain(inc[:, k], img[:, k], rate)
        if not np.isfinite(corr) or corr < 0.5:
            continue
        col = img[:, k] / g - inc[:, k]
        good = np.isfinite(col)
        if good.sum() < 10:
            continue
        col[~good] = np.interp(idx[~good], idx[good], col[good])
        dis[:, k] = col - median_filter(col, size=size, mode='nearest')
    for e in ev_list:
        a, b = e['frames']
        x = c_ev[a:b, :2].ravel()
        y = dis[a:b, :].ravel()
        m = np.isfinite(y) & (x != 0)
        if m.sum() < 6 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
            e['shift_check'] = None
            continue
        corr = float(np.corrcoef(x[m], y[m])[0, 1])
        gain = float(np.dot(x[m], y[m]) / np.dot(x[m], x[m]))
        e['shift_check'] = dict(corr=corr, gain=gain, verdict='confirmed' if corr > 0.5 else ('opposed' if corr < -0.3 else 'unclear'))
    _ = uh


def build(video, image_npz, out_path, timing='image', bias_ms=0.0, r0_roll=40.0,
          r0_pt=80.0, gain=1.0, roll_lo_hz=4.0, hp_hz=1.0, events=True, event_thr=0.10,
          control_path=None, report=True, diagnostics_path=None, roll_axis='z', event_max_deg=0.0,
          event_gain=1.0, unclear_weight=0.0, event_estimator='max', event_fine=True, release_s=1.5,
          event_model='image', event_zero_net=True, spike_deg=0.35):
    clip, samples, frames, ts, qs = load_sorted(video)
    fps = float(clip.fps)
    n_frames = len(frames)
    d = np.load(image_npz)
    meta = json.loads(str(d['meta_json']))
    gap = int(meta['gap'])
    nimg = min(len(d['roll_deg']), n_frames)
    inc_img, roll_curl, img_source = image_rotations(d, n_frames)
    img_roll = inc_img[:, 2].copy() if inc_img is not None else roll_curl
    has_img = np.isfinite(img_roll)

    # ---- 1. timing ------------------------------------------------------------
    timing_info = dict(mode=timing, bias_ms=bias_ms)
    if timing == 'image':
        # Measured on this clip: the constant is the shift at which the telemetry
        # roll read at pts best matches the image roll (~0 on both cameras); the
        # per-frame part is the exposure variation around the clip's median (see
        # timing.py). The exposure-centre model is reported for comparison only.
        from timing import exposure_ms
        shift0, r_best, r_zero, n_used = measure_timing_shift(ts, qs, fps, n_frames, img_roll)
        # A minimum shallower than 0.3 % of the residual is noise (on the Lite three
        # scan variants spread over 4 ms with such a curve): then the constant is 0,
        # which is where both cameras measured within ~1 ms anyway.
        determined = np.isfinite(r_zero) and r_zero > 0 and (r_zero - r_best) / r_zero >= 0.003
        used = shift0 if determined else 0.0
        ex = exposure_ms(frames)
        e_med = float(np.median(ex))
        delta_us = (used + bias_ms + (e_med - ex) / 2.0) * 1000.0
        model = float(clip.frame_readout_time_ms or 0.0) / 2.0 - e_med / 2.0
        timing_info.update(measured_shift_ms=shift0, constant_used_ms=used, minimum_determined=bool(determined),
                           rms_at_best=r_best, rms_at_zero=r_zero, frames_used=n_used,
                           median_exposure_ms=e_med, exposure_model_shift_ms=model)
    else:
        delta_us = frame_shift_us(clip, frames, timing, bias_ms)
    dd = np.concatenate([delta_us, delta_us[-1:]])
    ft = np.arange(n_frames + 1) * (1e6 / fps)
    qf = slerp_series(ts, qs, ft + dd)                     # what the render will read, per frame
    inc = frame_increments(ts, qs, ft + dd)                # deg/frame, body frame
    rate = np.linalg.norm(inc, axis=1) * fps
    # The last blocks of a DJI file are short (21 or 37 samples) and carry junk
    # increments of thousands of deg/s, and the shifted instants can fall past the
    # last sample. Nothing is corrected or calibrated on those frames.
    valid = ((ft[:-1] + dd[:-1]) >= ts[0]) & ((ft[1:] + dd[1:]) <= ts[-1]) & (rate < 1000.0)
    rate = np.where(valid, rate, 1e6)                      # weights -> 0, fits exclude

    # ---- 2. roll from the image --------------------------------------------
    # The telemetry frame is the camera frame: the 5-frame homography rotations
    # map onto it by a signed permutation with 0.06-0.1 deg/frame residual, so
    # the roll axis is z. (A free fit of the axis on the affine curl drifts
    # towards the pan axis: the curl of a pan field is not zero when the tracked
    # points are not centred, and pan and roll are correlated in a banked turn.)
    free_uh, free_gain, free_corr, free_n = fit_roll_axis(inc, img_roll, rate)
    uh = np.array([0.0, 0.0, 1.0])
    if roll_axis == 'fit':
        uh, ugain, ucorr, un = free_uh, free_gain, free_corr, free_n
        tel_roll = inc @ uh
        disc = (np.nan_to_num(img_roll) / max(ugain, 1e-6)) - tel_roll   # deg about uh
    else:
        # Measurement model: image_roll = u . inc, with u fitted on the fast frames.
        # u_z is the roll gain (~1.0 on both cameras); the small u_y (0.2-0.3) is
        # yaw leaking into the measured roll -- a property of the measurement, so
        # it is removed from the image before the comparison, and the discrepancy
        # is attributed to the z axis only, which is where the correction goes.
        ugain, ucorr, un = scalar_gain(inc[:, 2], img_roll, rate)
        u_model = free_uh * free_gain if (np.isfinite(free_corr) and free_corr >= 0.85 and free_gain * free_uh[2] > 0.5) \
            else np.array([0.0, 0.0, ugain if (np.isfinite(ucorr) and ucorr >= 0.8) else 1.0])
        tel_roll = inc @ uh
        disc = (np.nan_to_num(img_roll) - inc @ u_model) / u_model[2]      # deg about z
        ugain = float(u_model[2])
    w_roll = 1.0 / (1.0 + (rate / r0_roll) ** 2)
    w_roll[~has_img | ~valid] = 0.0
    c_img = w_roll * disc * gain
    # Wiener fallback where the image measurement is missing (above 4 Hz only)
    w_pt = 1.0 / (1.0 + (rate / r0_pt) ** 2)
    w_pt[~valid] = 0.0
    # ---- telemetry spikes, before anything else looks at pitch/yaw ------------
    # per-pair image rotation minus telemetry (gain-normalised and raw)
    fine = fine_raw = None
    if inc_img is not None:
        fine = np.full((n_frames, 3), np.nan)
        for k in range(3):
            g, corr, nfit = scalar_gain(inc[:, k], inc_img[:, k], rate)
            g = g if (np.isfinite(corr) and corr >= 0.6 and 0.3 < g < 3.0) else 1.0
            fine[:, k] = inc_img[:, k] / g - inc[:, k]
        fine[~valid] = np.nan
        fine_raw = inc_img - inc
        fine_raw[~valid] = np.nan
    spike, c_spike, contaminated = detect_spikes(inc, fine, fine_raw, rate, uh, valid, spike_deg=spike_deg) \
        if (events and fine is not None) else (np.zeros(n_frames, bool), np.zeros_like(inc), np.zeros(n_frames, bool))
    inc_ds = inc + c_spike                                 # de-spiked telemetry increments
    inc_w = np.where(valid[:, None], inc_ds, 0.0)         # junk frames must not enter the filters
    c_wroll = w_pt * (wiener_axis(np.where(valid, tel_roll, 0.0), hf_only(KEEP, roll_lo_hz), fps) - tel_roll) * gain
    c_roll = np.where(has_img, c_img, c_wroll)
    theta_roll = shifted_cumsum(c_roll)
    theta_roll = sosfiltfilt(band_sos(roll_lo_hz, fps), theta_roll)

    # ---- 3. pitch / yaw Wiener, calibrated on this clip --------------------
    # The image shift channels are angle proxies contaminated by translation, but
    # above 4 Hz a drone's translation cannot reverse ten times a second, so their
    # spectrum there is the camera's real pitch/yaw. Each axis is fitted from the
    # fast frames (where rotation dominates the shift), then the keep-gain curve
    # is the image/telemetry power ratio on calm runs. Nothing below 4 Hz is
    # touched: the old curve's 0.93 at 1 Hz removed 7 % of a real pan's
    # acceleration, up to 1.8 deg on this clip.
    calm = rate < 20.0
    pt_axes, pt_info = [], []
    ax0, ay0 = perp_basis(uh)
    for axis_idx, axis_vec in ((0, ax0), (1, ay0)):
        if inc_img is not None:
            img = inc_img[:, axis_idx]
            chan = 'kabsch1[%d]' % axis_idx
        else:
            key = ('shift_yn_deg', 'shift_xn_deg')[axis_idx] if 'shift_yn_deg' in d else ('shift_y_deg', 'shift_x_deg')[axis_idx]
            img = np.full(n_frames, np.nan)
            img[:nimg] = d[key][:nimg]
            chan = key
        comp = inc_w @ axis_vec
        g, corr, nfit = scalar_gain(comp, img, rate)
        sk = None
        if np.isfinite(corr) and corr >= 0.8:
            sk = spectral_keep(comp, img / g, fps, calm, floor_hz=roll_lo_hz)
        keep = sk[0] if sk else hf_only(KEEP_PANTILT, roll_lo_hz)
        pt_axes.append((axis_vec, keep))
        pt_info.append(dict(channel=chan, used=True, axis=axis_vec.tolist(), image_gain=g, fit_corr=corr,
                            fit_frames=nfit, keep=np.round(keep, 3).tolist(),
                            calibrated=bool(sk), **(sk[1] if sk else {})))
    c_pt = np.zeros_like(inc)
    for axis, keep in pt_axes:
        comp = inc_w @ axis
        c_pt += np.outer(w_pt * (wiener_axis(comp, keep, fps) - comp) * gain, axis)
    theta_pt = sosfiltfilt(band_sos(hp_hz, fps), shifted_cumsum(c_pt), axis=0)

    # ---- 4. pitch / yaw events from the 5-frame windows --------------------
    ev_list, c_ev, winfo = [], np.zeros_like(inc), None
    if events and 'homog_deg' in d:
        winfo = window_disagreement(qf, d, fps, gap, rate, estimator=event_estimator)
        D = winfo['D']
        nw = len(D)
        # per-frame disagreement: mean of the windows covering pair f, per frame
        delta_pf = np.full((n_frames, 3), np.nan)
        rel_pf = np.zeros(n_frames, bool)
        acc = np.zeros((n_frames, 3))
        cnt = np.zeros(n_frames)
        relc = np.zeros(n_frames)
        for wdx in range(nw):
            if not np.isfinite(D[wdx]).all():
                continue
            acc[wdx:wdx + gap] += D[wdx] / gap
            cnt[wdx:wdx + gap] += 1
            relc[wdx:wdx + gap] += winfo['reliable'][wdx]
        m = cnt > 0
        delta_pf[m] = acc[m] / cnt[m, None]
        rel_pf = m & (relc >= np.maximum(1, 0.6 * cnt)) & valid
        # independent per-frame pitch/yaw from the affine shift channels, image gain taken out
        fine_alt = None
        keys = ('shift_yn_deg', 'shift_xn_deg') if 'shift_yn_deg' in d else ('shift_y_deg', 'shift_x_deg')
        if all(k in d for k in keys):
            fine_alt = np.full((n_frames, 3), np.nan)
            fitted = 0
            for k, key in enumerate(keys):
                img = np.full(n_frames, np.nan)
                img[:nimg] = d[key][:nimg]
                g, corr, nfit = scalar_gain(inc[:, k], img, rate)
                if np.isfinite(corr) and corr >= 0.5 and 0.3 < abs(g) < 3.0:
                    fitted += 1
                else:
                    g = 1.0
                fine_alt[:, k] = img / g - inc[:, k]
            fine_alt[:, 2] = 0.0
            fine_alt[~valid] = np.nan
            if fitted < 2:
                fine_alt = None      # an uncalibrated witness would veto every per-frame value (short clips)
        fine_ds = None if fine is None else fine - c_spike   # image minus the de-spiked telemetry
        c_ev, ev_list = event_correction(delta_pf, rel_pf, rate, fps, uh, thr_deg=event_thr,
                                         gain=gain * event_gain, inc=inc_w,
                                         fine=fine_ds if event_fine else None, fine_alt=fine_alt,
                                         model=event_model, zero_net=event_zero_net,
                                         spike=spike, contaminated=contaminated)
    if np.any(c_ev):
        # Independent check of every event against the affine shift channels
        # (another estimator of pitch/yaw from the same frames): sign and gain of
        # the proposed correction against the shift-based disagreement detail.
        # Confirmed events are applied in full, unclear ones at half strength,
        # opposed ones not at all.
        shift_check(ev_list, c_ev, inc_w, d, n_frames, nimg, rate, fps, uh)
        c_ev = weight_events(ev_list, n_frames, unclear=unclear_weight)
    theta_ev = shifted_cumsum(c_ev)
    if np.any(c_ev):
        # Any net offset an event leaves behind fades afterwards with a ~1.5 s time
        # constant, forwards in time only: a zero-phase high-pass would start
        # rotating the picture half a second *before* the event.
        in_event = np.zeros(n_frames, bool)
        for e in ev_list:
            a, b = e['frames']
            in_event[a:b] = True
        decay = np.exp(-1.0 / (release_s * fps))
        y = np.zeros_like(theta_ev)
        for f in range(1, n_frames):
            step = theta_ev[f] - theta_ev[f - 1]
            y[f] = (y[f - 1] if in_event[f] else y[f - 1] * decay) + step
        theta_ev = y
        # a wrong event must stay cheap: smooth bound on the added rotation (0 = no bound)
        if event_max_deg and event_max_deg > 0:
            theta_ev = limit_rotation(theta_ev, event_max_deg)
        for e in ev_list:
            a, b = e['frames']
            e['applied_peak_deg'] = float(np.linalg.norm(theta_ev[a:b], axis=1).max()) if b > a else 0.0

    theta_spike = shifted_cumsum(c_spike * gain)
    if np.any(c_spike):
        decay = np.exp(-1.0 / (release_s * fps))
        y = np.zeros_like(theta_spike)
        for f in range(1, n_frames):
            y[f] = (y[f - 1] if spike[f] else y[f - 1] * decay) + (theta_spike[f] - theta_spike[f - 1])
        theta_spike = y
    spike_times = [round(float(i / fps), 2) for i in np.flatnonzero(spike)]

    theta = np.outer(theta_roll, uh) + theta_pt + theta_ev + theta_spike   # (n_frames, 3) deg at frame instants

    # ---- apply -------------------------------------------------------------
    t = np.array([s.t_us for s in samples], dtype=float)
    frame_of = np.array([s.frame for s in samples])
    base = slerp_series(ts, qs, t + delta_us[frame_of])
    th = np.column_stack([np.interp(t, ft[:-1], theta[:, k]) for k in range(3)])
    newq = qmul(base, rotation_vector_quat(th))
    newq /= np.linalg.norm(newq, axis=1, keepdims=True)
    inverted = np.array([s.inverted for s in samples])

    def to_edits(qarr):
        qq = np.where(inverted[:, None], -qarr, qarr)
        return {(s.frame, s.index_in_frame): Q.norm(Q.gyroflow_to_dji(tuple(map(float, v))))
                for s, v in zip(samples, qq)}

    info = sidecar.build(video, out_path, to_edits(newq))
    info_ctrl = None
    if control_path:
        info_ctrl = sidecar.build(video, control_path, to_edits(base))

    mag = np.linalg.norm(theta, axis=1)
    rep = dict(
        video=os.path.abspath(video), image_npz=os.path.abspath(image_npz), output=os.path.abspath(out_path),
        control=os.path.abspath(control_path) if control_path else None,
        product=clip.header, fps=fps, frames=n_frames, samples=len(samples),
        readout_ms=clip.frame_readout_time_ms,
        timing=dict(timing_info, shift_ms_mean=float(delta_us.mean() / 1000),
                    shift_ms_min=float(delta_us.min() / 1000), shift_ms_max=float(delta_us.max() / 1000)),
        roll=dict(axis=uh.tolist(), axis_mode=roll_axis, image_source=img_source, image_gain=ugain,
                  fit_corr=ucorr, fit_frames=un,
                  free_fit=dict(axis=free_uh.tolist(), gain=free_gain, corr=free_corr, frames=free_n,
                                angle_to_z_deg=float(np.degrees(np.arccos(abs(free_uh[2]))))),
                  frames_with_image=int(has_img.sum()), r0_deg_s=r0_roll, band_hz=[roll_lo_hz, fps / 2 * 0.992],
                  discrepancy_rms_deg_per_frame=float(np.nanstd(disc[has_img])),
                  correction_rms_deg=float(theta_roll.std()), correction_peak_deg=float(np.abs(theta_roll).max())),
        pitch_yaw=dict(r0_deg_s=r0_pt, hp_hz=hp_hz, floor_hz=roll_lo_hz, axes=pt_info,
                       correction_rms_deg=float(np.linalg.norm(theta_pt, axis=1).std()),
                       correction_peak_deg=float(np.linalg.norm(theta_pt, axis=1).max())),
        spikes=dict(count=int(spike.sum()), times_s=spike_times, threshold_deg_per_frame=spike_deg,
                    peak_deg=float(np.linalg.norm(c_spike, axis=1).max()) if np.any(c_spike) else 0.0),
        events=dict(count=len(ev_list), list=ev_list, threshold_deg_per_frame=event_thr, max_deg=event_max_deg,
                    event_gain=event_gain, unclear_weight=unclear_weight, estimator=event_estimator,
                    model=event_model, zero_net=event_zero_net,
                    correction_peak_deg=float(np.linalg.norm(theta_ev, axis=1).max()),
                    windows_total=int(winfo['ok'].sum()) if winfo else 0,
                    windows_reliable=int(winfo['reliable'].sum()) if winfo else 0,
                    axis_map=dict(perm=winfo['axis_perm'], signs=winfo['axis_signs'],
                                  fit_err_deg=winfo['axis_fit_err'], fit_windows=winfo['axis_fit_n']) if winfo else None),
        total=dict(correction_rms_deg=float(mag.std()), correction_peak_deg=float(mag.max()),
                   edited_records=int(info['edited_records']), bytes=int(info['bytes'])),
    )
    if diagnostics_path:
        np.savez_compressed(diagnostics_path, frame_time_s=ft[:-1] / 1e6, inc_deg=inc, rate_deg_s=rate, valid=valid,
                            img_roll_deg=img_roll, disc_deg=disc, theta_roll_deg=theta_roll,
                            theta_pt_deg=theta_pt, theta_ev_deg=theta_ev, theta_spike_deg=theta_spike, theta_deg=theta,
                            delta_us=delta_us, uh=uh,
                            **({'win_D_deg': winfo['D'], 'win_reliable': winfo['reliable'],
                                'win_T_deg': winfo['T'], 'win_dH': winfo['dH'], 'win_dK': winfo['dK']} if winfo else {}))
    if report:
        r = rep
        print('  timing     : %s, shift %+.2f ms mean (%+.2f..%+.2f)' % (
            timing, r['timing']['shift_ms_mean'], r['timing']['shift_ms_min'], r['timing']['shift_ms_max']))
        if timing == 'image':
            ti = r['timing']
            print('               measured %+.2f ms at pts on %d fast frames (rms %.4f vs %.4f at 0, %s), used %+.2f; '
                  'exposure model would say %+.2f ms (median exposure %.1f ms)' % (
                      ti['measured_shift_ms'], ti['frames_used'], ti['rms_at_best'], ti['rms_at_zero'],
                      'clear minimum' if ti['minimum_determined'] else 'too shallow to trust',
                      ti['constant_used_ms'], ti['exposure_model_shift_ms'], ti['median_exposure_ms']))
        print('  roll axis  : %s (%s)  image %s gain %.3f  corr %.3f on %d fast frames; image on %d/%d frames' % (
            np.round(uh, 4), roll_axis, img_source, ugain, ucorr, un, has_img.sum(), n_frames))
        print('               free fit would be %s (gain %.3f, corr %.3f, %.1f deg from z)' % (
            np.round(free_uh, 3), free_gain, free_corr, np.degrees(np.arccos(abs(free_uh[2])))))
        print('  roll       : discrepancy %.4f deg/frame rms -> correction rms %.4f deg, peak %.3f' % (
            r['roll']['discrepancy_rms_deg_per_frame'], r['roll']['correction_rms_deg'], r['roll']['correction_peak_deg']))
        for pi_ in pt_info:
            if pi_.get('used') and 'fit_corr' in pi_:
                print('  pitch/yaw  : %s axis %s corr %.3f; keep above %g Hz %s%s' % (
                    pi_['channel'], np.round(pi_['axis'], 3), pi_['fit_corr'], roll_lo_hz,
                    'calibrated on %d calm frames' % pi_['frames'] if pi_.get('calibrated') else 'default curve',
                    ('; telemetry excess ' + ', '.join('%s x%.1f' % kv for kv in pi_['excess_by_band'].items()))
                    if pi_.get('excess_by_band') else ''))
            elif pi_.get('used'):
                print('  pitch/yaw  : %s' % pi_.get('note', pi_['channel']))
        print('  pitch/yaw  : Wiener correction rms %.4f deg, peak %.3f' % (
            r['pitch_yaw']['correction_rms_deg'], r['pitch_yaw']['correction_peak_deg']))
        if winfo:
            print('  windows    : %d usable, %d with both image estimates agreeing; axis map %s %s (fit err %.3f deg on %d)' % (
                winfo['ok'].sum(), winfo['reliable'].sum(), winfo['axis_perm'], winfo['axis_signs'],
                winfo['axis_fit_err'], winfo['axis_fit_n']))
        print('  spikes     : %d telemetry spikes removed (>= %.2f deg/frame, image-confirmed)%s' % (
            spike.sum(), spike_deg, (' at ' + ', '.join('%.2f' % x for x in spike_times[:20]) + ' s') if spike_times else ''))
        print('  events     : %d repaired, peak correction %.3f deg' % (len(ev_list), r['events']['correction_peak_deg']))
        for e in ev_list:
            sc = e.get('shift_check')
            print('      %7.2f-%7.2f s  peak %.3f deg/frame  at %3.0f deg/s   shift check: %-28s weight %.1f  applied %.2f deg%s' % (
                e['start_s'], e['end_s'], e['peak_deg_per_frame'], e['mean_rate_deg_s'],
                ('%s (corr %+.2f, gain %.2f)' % (sc['verdict'], sc['corr'], sc['gain'])) if sc else 'n/a', e['weight'],
                e.get('applied_peak_deg', 0.0),
                ('  TELEMETRY SPIKE at %s' % e['spike_frames_s']) if e.get('telemetry_spike') else ''))
        print('  total      : correction rms %.4f deg, peak %.3f deg; %d records -> %s (%.1f MB)' % (
            r['total']['correction_rms_deg'], r['total']['correction_peak_deg'], info['edited_records'],
            out_path, info['bytes'] / 1e6))
        if info_ctrl:
            print('  control    : %s' % control_path)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('--image', required=True, help='npz from measure_rotation.py')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--control', help='also write a timing-only control sidecar here')
    ap.add_argument('--timing', default='image', choices=['image', 'exposure', 'constant', 'none', 'dbgi'],
                    help='image: constant measured against the image roll + per-frame exposure (default); '
                         'exposure: readout/2 - exposure/2; dbgi: the old alignment')
    ap.add_argument('--bias-ms', type=float, default=0.0)
    ap.add_argument('--gain', type=float, default=1.0)
    ap.add_argument('--r0-roll', type=float, default=40.0)
    ap.add_argument('--r0-pt', type=float, default=80.0)
    ap.add_argument('--event-thr', type=float, default=0.10, help='deg/frame of pitch/yaw disagreement')
    ap.add_argument('--no-events', action='store_true')
    ap.add_argument('--event-gain', type=float, default=1.0, help='multiply the event correction (image under-reads pan/tilt)')
    ap.add_argument('--event-max-deg', type=float, default=0.0, help='smooth bound on the event rotation; 0 = none (default)')
    ap.add_argument('--unclear-weight', type=float, default=0.0, help='weight of events the shift check calls unclear (default 0: only confirmed events and telemetry spikes are applied)')
    ap.add_argument('--event-estimator', default='max', choices=['mean', 'max', 'min'],
                    help='which of the two image estimates drives an event')
    ap.add_argument('--event-coarse', action='store_true',
                    help='shape the event from the 5-frame windows instead of the per-pair rotation')
    ap.add_argument('--spike-deg', type=float, default=0.35, help='one-frame telemetry jump (deg/frame) treated as a spike when the image confirms')
    ap.add_argument('--event-model', default='image', choices=['image', 'bridge'],
                    help='image: shape from the image; bridge: straight-line pitch/yaw rate through the window')
    ap.add_argument('--event-free-net', action='store_true',
                    help='do not remove the net rotation of an image-shaped event (unsafe: offsets accumulate)')
    ap.add_argument('--roll-axis', default='z', choices=['z', 'fit'],
                    help='roll axis in the telemetry frame: z (camera frame, default) or a free fit on the image roll')
    ap.add_argument('--diagnostics', help='npz with all intermediate series')
    ap.add_argument('--report-json')
    a = ap.parse_args()
    rep = build(a.video, a.image, a.out, timing=a.timing, bias_ms=a.bias_ms, r0_roll=a.r0_roll, r0_pt=a.r0_pt,
                gain=a.gain, events=not a.no_events, event_thr=a.event_thr, control_path=a.control,
                diagnostics_path=a.diagnostics, roll_axis=a.roll_axis, event_max_deg=a.event_max_deg,
                event_gain=a.event_gain, unclear_weight=a.unclear_weight, event_estimator=a.event_estimator,
                event_fine=not a.event_coarse, event_model=a.event_model, event_zero_net=not a.event_free_net, spike_deg=a.spike_deg)
    if a.report_json:
        with open(a.report_json, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)

#!/usr/bin/env python3
"""
Research prototype, not project code: a second, image-free witness for the roll defect gate.

roll_adaptive (src/fix_pipeline.py) switches the roll correction on where the
telemetry-vs-image roll error above 4 Hz is high. That decision uses the same
image the correction is taken from, so where the image is wrong (near branches
passing through the frame, lite39_0005 at 21-23 s) it both "finds" the defect
and "fixes" it. DJI's defect is also visible in the telemetry alone: its roll
carries 15-25 Hz power that other units do not have (share 15-25 Hz / 2-6 Hz on
calm frames: pro 0.99, the others 0.04-0.59). Parallax cannot touch that number.

This script:
  1. captures the inputs of roll_adaptive from fix_pipeline.build() for every clip
     in artifacts/clips.json that has an image measurement in the base cache
     (build is stopped right after, nothing is written next to the videos);
  2. runs the current roll_adaptive (reference) and the same with its gate
     multiplied by a telemetry-only gate: per calm second the HF share of the
     telemetry roll, median of the same neighbourhood the image gate uses;
  3. prints per clip: where each gate is on and how big the correction is.

    python research/ab_bench/roll_gate_tel.py [--cache artifacts/cache] [--work DIR] [--lo 0.6 --hi 0.8] [--only NAMES]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
import fix_pipeline as fp   # noqa: E402


class _Captured(Exception):
    pass


def capture(video, image_npz, out_npz):
    """Inputs of roll_adaptive as build() computes them."""
    real = fp.roll_adaptive

    def spy(tel_roll, img_roll, curl_roll, npts, rate, valid, fps, **kw):
        np.savez_compressed(out_npz, tel_roll=tel_roll, img_roll=img_roll,
                            curl_roll=np.full_like(tel_roll, np.nan) if curl_roll is None else curl_roll,
                            npts=npts, rate=rate, valid=valid, fps=fps)
        raise _Captured()
    fp.roll_adaptive = spy
    try:
        fp.build(video, image_npz, os.path.join(os.path.dirname(out_npz), '_unused.mp4'), report=False)
    except _Captured:
        pass
    finally:
        fp.roll_adaptive = real


def per_second(values_ok, x, fps):
    sec = max(1, int(round(fps)))
    return sec, np.array([np.nan if values_ok[i * sec:(i + 1) * sec].sum() <= 0.5 * sec else
                          x(slice(i * sec, (i + 1) * sec), values_ok[i * sec:(i + 1) * sec])
                          for i in range(len(values_ok) // sec + 1)])


def neighbourhood(per_s, k=7, reach=20, min_n=5):
    have = np.flatnonzero(np.isfinite(per_s))
    out = np.full(len(per_s), np.nan)
    for i in range(len(per_s)):
        if not len(have):
            break
        near = have[np.argsort(np.abs(have - i))[:k]]
        near = near[np.abs(near - i) <= reach]
        if len(near) >= min_n:
            out[i] = np.median(per_s[near])
    return out


def tel_hf_share(tel_roll, rate, valid, fps, calm_deg_s=30.0):
    """Per calm second: power of the telemetry roll at 15-25 Hz over 2-6 Hz (no image involved)."""
    x = np.where(valid, tel_roll, 0.0)
    hi = sosfiltfilt(butter(2, [15.0, min(25.0, fps / 2 * 0.95)], btype='band', fs=fps, output='sos'), x)
    lo = sosfiltfilt(butter(2, [2.0, 6.0], btype='band', fs=fps, output='sos'), x)
    calm = valid & (rate < calm_deg_s)
    # mean power per Hz in each band, as a Welch ratio would give
    bw_hi = min(25.0, fps / 2 * 0.95) - 15.0
    sec, s = per_second(calm, lambda sl, m: (np.mean(hi[sl][m] ** 2) / bw_hi) / max(np.mean(lo[sl][m] ** 2) / 4.0, 1e-12), fps)
    return sec, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--registry', default=os.path.join(ROOT, 'artifacts', 'clips.json'))
    ap.add_argument('--cache', default=os.path.join(ROOT, 'artifacts', 'cache'))
    ap.add_argument('--work', default=os.path.join(ROOT, 'artifacts', 'roll_gate'))
    ap.add_argument('--lo', type=float, default=0.6, help='telemetry HF share where the new gate starts to open')
    ap.add_argument('--hi', type=float, default=0.8, help='... and where it is fully open')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    reg = json.load(open(a.registry, encoding='utf-8'))
    ramp = lambda v: np.clip((np.nan_to_num(v) - a.lo) / (a.hi - a.lo), 0.0, 1.0)   # noqa: E731
    print('%-15s %6s %6s | %-24s | %-24s | %s' % ('clip', 'HFclip', 'HFmax', 'current: on%  rms  peak', 'with tel gate: on% rms peak', 'HF share where current gate is on'))
    for c in reg['clips']:
        if a.only and c['name'] not in a.only:
            continue
        img = os.path.join(a.cache, c['name'], 'image.npz')
        if not os.path.isfile(img):
            continue
        cap = os.path.join(a.work, c['name'] + '_roll_inputs.npz')
        if not os.path.isfile(cap):
            capture(c['path'], img, cap)
        d = np.load(cap)
        fps = float(d['fps'])
        args = [d['tel_roll'], d['img_roll'], None if np.isnan(d['curl_roll']).all() else d['curl_roll'],
                d['npts'], d['rate'], d['valid'].astype(bool), fps]
        theta0, gate0 = fp.roll_adaptive(*args)
        sec, hs = tel_hf_share(d['tel_roll'], d['rate'], d['valid'].astype(bool), fps)
        hs_n = neighbourhood(hs)
        gate_t = np.interp(np.arange(len(gate0)) / fps, np.arange(len(hs_n)) + 0.5, ramp(hs_n))
        # the same correction with the gate multiplied: rerun with the image gate scaled
        theta1 = gated(args, gate_t)
        on0, on1 = gate0 > 0.01, (gate0 * gate_t) > 0.01
        where = hs_n[np.unique(np.clip((np.flatnonzero(on0) / fps).astype(int), 0, len(hs_n) - 1))] if on0.any() else []
        print('%-15s %6.2f %6.2f | %5.1f%% %6.3f %6.3f     | %5.1f%% %6.3f %6.3f     | %s' % (
            c['name'], np.nanmedian(hs), np.nanmax(hs_n) if np.isfinite(hs_n).any() else np.nan,
            100 * on0.mean(), np.sqrt(np.mean(theta0 ** 2)), np.abs(theta0).max(),
            100 * on1.mean(), np.sqrt(np.mean(theta1 ** 2)), np.abs(theta1).max(),
            ('%.2f-%.2f' % (np.nanmin(where), np.nanmax(where))) if len(where) else '-'))


def gated(args, extra_gate):
    """roll_adaptive with its per-frame gate multiplied by `extra_gate` (monkeypatched np.interp result)."""
    real_interp = fp.np.interp
    calls = {'n': 0}

    def interp(x, xp, fp_, *k, **kw):
        out = real_interp(x, xp, fp_, *k, **kw)
        calls['n'] += 1
        if calls['n'] == 1 and len(out) == len(extra_gate):   # the gate is roll_adaptive's only np.interp call
            out = out * extra_gate
        return out
    fp.np.interp = interp
    try:
        theta, _ = fp.roll_adaptive(*args)
    finally:
        fp.np.interp = real_interp
    return theta


if __name__ == '__main__':
    main()

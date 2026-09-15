"""Compare original DJI attitudes with independently tracked image points.

python research/track_telemetry.py "source.MP4" --start 52 --end 58
No video rendering or telemetry editing. Results: tracks.npz, comparison.csv/png,
report.json. Camera axes below are rotation-vector axes, not Euler columns.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import argparse
import csv
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from align import slerp_series
from dji_o4 import read_telemetry
from imagerot import Lens, rotation_kabsch, best_axis_map

FILE = r'F:\36\54-56-DJI_20260517170526_0020_D.MP4'


def estimate(a, b, lens):
    """Image-only rotation fits; essential model explicitly includes translation."""
    ra, rb = lens.to_rays(a), lens.to_rays(b)
    rot, residual, weights = rotation_kabsch(ra, rb)
    pure = Rotation.from_matrix(rot).as_rotvec()
    n1, n2 = lens.to_normalised(a), lens.to_normalised(b)
    essential = np.full(3, np.nan)
    count = 0
    if np.median(residual) < .15 / lens.K[0, 0]:
        return pure, essential, count, np.median(residual), ra, rb
    E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC,
                                  prob=.999, threshold=.7 / lens.K[0, 0])
    if E is not None:
        for candidate in np.asarray(E).reshape(-1, 3, 3):
            num, R, _, pose_mask, _ = cv2.recoverPose(
                candidate, n1, n2, np.eye(3), distanceThresh=1000., mask=mask.copy())
            kept = pose_mask.ravel() != 0
            parallax = np.degrees(np.arccos(np.clip(np.sum((ra @ R.T)*rb, axis=1), -1, 1)))
            if not kept.any() or np.median(parallax[kept]) < .1:
                continue
            if num > count:
                count = num
                essential = Rotation.from_matrix(R).as_rotvec()
    # Low cheirality support means the rotation/translation separation is unreliable.
    if count < max(40, .4 * len(a)):
        essential[:] = np.nan
    return pure, essential, count, np.median(residual), ra, rb


def track(video, first, last, gap, scale, lens):
    cap = cv2.VideoCapture(str(video))
    history = deque(maxlen=gap + 1)
    results, points_a, points_b, offsets = [], [], [], [0]
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError('Cannot open/seek input video')
        for frame in range(first, last + 1):
            ok, img = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode frame {frame}')
            if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (frame + 1)) > .1:
                raise ValueError('Decoder frame position mismatch')
            gray = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            history.append(gray)
            if len(history) < gap + 1:
                continue
            prev = history[0]
            h, w = gray.shape
            mask = np.zeros_like(gray)
            # Avoid fisheye edges and distribute features across a broad central field.
            mask[int(h*.15):int(h*.85), int(w*.15):int(w*.85)] = 255
            p = cv2.goodFeaturesToTrack(prev, 1200, .01, 8, mask=mask, blockSize=7)
            a, b = np.empty((0, 2)), np.empty((0, 2))
            pure, ess, support, res = np.full(3, np.nan), np.full(3, np.nan), 0, np.nan
            if p is not None and len(p) >= 40:
                lk = dict(winSize=(31, 31), maxLevel=4,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p, None, **lk)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None, **lk)
                a, b = p.reshape(-1, 2), nxt.reshape(-1, 2)
                valid = (st.ravel() != 0) & (st2.ravel() != 0)
                valid &= np.linalg.norm(a-back.reshape(-1, 2), axis=1) < .5
                valid &= np.isfinite(b).all(axis=1)
                valid &= (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h)
                a, b = a[valid], b[valid]
                if len(a) >= 40:
                    pure, ess, support, res, _, _ = estimate(a, b, lens)
            results.append([frame-gap, frame, len(a), support, res, *pure, *ess])
            points_a.extend(a); points_b.extend(b); offsets.append(len(points_a))
            if (frame-first) % 50 == 0:
                print(f'Tracked {frame-first}/{last-first} frames', flush=True)
    finally:
        cap.release()
    return dict(rows=np.array(results), points_a=np.array(points_a),
                points_b=np.array(points_b), offsets=np.array(offsets))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('video', nargs='?', default=FILE)
    ap.add_argument('--start', type=float, default=52.)
    ap.add_argument('--end', type=float, default=58.)
    ap.add_argument('--exclude', type=float, nargs=2, default=[54., 56.],
                    help='Suspect interval excluded from axis-convention fitting')
    ap.add_argument('--gap', type=int, default=5, help='Frame separation, default 5 (100 ms at 50 fps)')
    ap.add_argument('--scale', type=float, default=.25)
    ap.add_argument('--offset-ms', type=float, default=0., help='Extra telemetry offset; positive samples later')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--reuse-tracks', action='store_true')
    args = ap.parse_args()
    if not (0 <= args.start < args.end and 0 < args.scale <= 1 and args.gap >= 1
            and np.isfinite(args.offset_ms) and args.exclude[0] < args.exclude[1]):
        ap.error('Invalid time interval, scale, gap or offset')
    video = Path(args.video).resolve()
    out = (args.out or Path(__file__).resolve().parent.parent / 'artifacts' /
           'analysis' / (video.stem + '_tracking'))
    out.mkdir(parents=True, exist_ok=True)
    clip, samples, _ = read_telemetry(str(video))
    cap = cv2.VideoCapture(str(video))
    width, height, fps, count = [cap.get(p) for p in (cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT)]
    cap.release()
    if fps <= 0 or abs(fps-clip.fps) > .001 or args.end*fps >= count:
        raise ValueError('Invalid video properties or interval beyond last frame')
    if not clip.focal_length or len(clip.distortion_coeffs or []) != 4:
        raise ValueError('Missing DJI fisheye lens calibration')
    lens = Lens(f=clip.focal_length, cx=width/2, cy=height/2,
                D=clip.distortion_coeffs, scale=args.scale)
    first, last = round(args.start*fps), round(args.end*fps)
    if last-first <= args.gap:
        raise ValueError('Interval must exceed gap')
    config = dict(video=str(video), size=video.stat().st_size,
                  mtime_ns=video.stat().st_mtime_ns, first=first, last=last,
                  gap=args.gap, scale=args.scale)
    cache = out / 'tracks.npz'
    if args.reuse_tracks:
        with np.load(cache) as saved:
            if json.loads(str(saved['config'])) != config:
                raise ValueError('Cached tracks do not match these inputs/settings')
            data = {k: saved[k] for k in ('rows', 'points_a', 'points_b', 'offsets')}
        # Refit cached correspondences so solver changes do not require video decoding.
        cv2.setRNGSeed(0)
        for i, row in enumerate(data['rows']):
            lo, hi = data['offsets'][i:i+2]
            if hi-lo >= 40:
                pure, ess, support, res, _, _ = estimate(
                    data['points_a'][lo:hi], data['points_b'][lo:hi], lens)
                row[3:] = [support, res, *pure, *ess]
    else:
        cv2.setRNGSeed(0)
        data = track(video, first, last, args.gap, args.scale, lens)
        np.savez_compressed(cache, config=json.dumps(config), **data)
    rows = data['rows']
    times = rows[:, :2] / fps
    mid = times.mean(axis=1)
    ts = np.array([s.t_us/1e6 for s in samples])
    qs = np.array([s.q_cam for s in samples])
    order = np.argsort(ts, kind='stable'); ts, qs = ts[order], qs[order]
    unique = np.r_[True, np.diff(ts) > 0]; ts, qs = ts[unique], qs[unique]
    # Match Gyroflow's native center-of-readout sampling; offset is explicit, not fitted.
    offset = (clip.frame_readout_time_ms or 0)/2000 + args.offset_ms/1000
    query = times + offset
    if query.min() < ts[0] or query.max() > ts[-1]:
        raise ValueError('Telemetry does not cover requested timestamps')
    q = slerp_series(ts, qs, query.ravel()).reshape(-1, 2, 4)
    r0 = Rotation.from_quat(q[:, 0][:, [1, 2, 3, 0]])
    r1 = Rotation.from_quat(q[:, 1][:, [1, 2, 3, 0]])
    raw = (r1.inv()*r0).as_rotvec()
    control = (times[:, 1] < args.exclude[0]) | (times[:, 0] > args.exclude[1])
    good = np.isfinite(rows[:, 8:11]).all(axis=1) & (rows[:, 2] >= 80)
    if np.sum(control & good) < 20:
        raise ValueError('Need at least 20 good frame pairs outside --exclude to fit camera axes')
    fit_error, perm, signs = best_axis_map(raw[control & good], rows[control & good, 8:11])
    predicted = raw[:, perm]*np.array(signs)
    dt = args.gap/fps
    telemetry, visual, essential = [np.degrees(x)/dt for x in
                                   (predicted, rows[:, 5:8], rows[:, 8:11])]
    difference = essential-telemetry
    score = np.linalg.norm(difference, axis=1)
    fields = ['time_s', 'frame_a', 'frame_b', 'tracks', 'essential_support', 'ray_fit_residual']
    fields += [f'{kind}_{axis}_deg_s' for kind in ('telemetry', 'image_rotation_only', 'image_with_translation', 'difference')
               for axis in ('x', 'y', 'z')]
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh); writer.writerow(fields)
        writer.writerows(np.column_stack([mid, rows[:, :5], telemetry, visual, essential, difference]))
    valid = np.isfinite(score)
    peaks = np.flatnonzero(valid)[np.argsort(score[valid])[::-1]]
    chosen = []
    for i in peaks:
        if all(abs(mid[i]-mid[j]) >= .2 for j in chosen):
            chosen.append(i)
        if len(chosen) == 8:
            break
    report = dict(source=str(video), interval_s=[args.start, args.end], gap_frames=args.gap,
        scale=args.scale, focal_length_px=clip.focal_length, distortion=clip.distortion_coeffs,
        telemetry_sample_offset_ms=offset*1000, axis_permutation=list(perm), axis_signs=list(signs),
        axis_fit_excluded_interval_s=args.exclude, axis_fit_mean_error_deg=float(np.degrees(fit_error)),
        axis_fit_reference='image rotation with translation, context only',
        pairs=len(rows), valid_rotation_translation_pairs=int(valid.sum()),
        candidate_peaks=[dict(time_s=float(mid[i]), mismatch_deg_s=float(score[i]),
            difference_xyz_deg_s=difference[i].tolist(), tracks=int(rows[i, 2]),
            essential_support=int(rows[i, 3])) for i in chosen],
        limitations=['Differences are candidates, not proven telemetry errors.',
            'Essential-matrix rotation can be ambiguous at low parallax or planar scenes.',
            'Rotation-only fit includes translation bias; it is a diagnostic reference.',
            'Rolling shutter is not corrected per point; center readout timing is used.',
            'Axis convention is fitted on context, no time-offset optimization.',
            'DJI fused attitudes are original telemetry, not raw gyro measurements.'])
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for k, ax in enumerate(axes[:3]):
        ax.plot(mid, telemetry[:, k], label='Original telemetry', lw=1.2)
        ax.plot(mid, essential[:, k], label='Points: rotation + translation', lw=1.)
        ax.plot(mid, visual[:, k], label='Points: rotation-only (biased by translation)', alpha=.4, lw=.7)
        ax.set_ylabel(f'{"XYZ"[k]} deg/s'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[3].plot(mid, score, label='Rotation + translation mismatch', color='crimson')
    axes[3].set_ylabel('Mismatch deg/s'); axes[3].set_xlabel('Original video time, seconds')
    for ax in axes:
        ax.axvspan(*args.exclude, color='orange', alpha=.1)
    fig.suptitle('Original image tracking vs original DJI attitudes — candidate discrepancies')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()

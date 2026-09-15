"""Build a conservative local quaternion correction from cached image tracks.

Uses repeated image-only fits to estimate sensitivity to point selection.
Writes a curve, diagnostic plot, and experimental telemetry; no video rendering.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from align import slerp_series
from dji_o4 import read_telemetry
from imagerot import Lens
from track_telemetry import estimate
import quat
import sidecar

ROOT = Path(__file__).resolve().parent.parent
TRACK = ROOT / 'artifacts/analysis/54-56-DJI_20260517170526_0020_D_tracking'
OUT = ROOT / 'artifacts/experiments/lite_tracking_curve'
START, END = 54.65, 56.15


def as_rotation(q):
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]])


def main(reuse_fits=False):
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads((TRACK/'report.json').read_text(encoding='utf-8'))
    baseline = ROOT/'artifacts/experiments/lite_v2/fullrate_8hz.mp4'
    _, samples, _ = read_telemetry(str(baseline))
    clip, _, _ = read_telemetry(report['source'])
    ts = np.array([s.t_us/1e6 for s in samples])
    qs = np.array([s.q_cam for s in samples])
    order = np.argsort(ts, kind='stable'); st, sq = ts[order], qs[order]
    unique = np.r_[True, np.diff(st)>0]; st, sq = st[unique], sq[unique]
    cache = np.load(TRACK/'tracks.npz')
    config = json.loads(str(cache['config']))
    cap = cv2.VideoCapture(report['source'])
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    lens = Lens(f=clip.focal_length, cx=width/2, cy=height/2,
                D=clip.distortion_coeffs, scale=config['scale'])
    rows = cache['rows']; pair_time = rows[:, :2]/clip.fps
    selected = np.flatnonzero((pair_time[:, 0]>=START) & (pair_time[:, 1]<=END))
    rng = np.random.default_rng(42)
    all_fits = np.full((len(selected), 7, 3), np.nan)
    if reuse_fits:
        saved = np.load(OUT/'point_subset_fits.npz')
        np.testing.assert_array_equal(saved['times'], pair_time[selected])
        all_fits = saved['fits_deg']
    for j, i in ([] if reuse_fits else enumerate(selected)):
        lo, hi = cache['offsets'][i:i+2]
        a, b = cache['points_a'][lo:hi], cache['points_b'][lo:hi]
        if len(a)<100:
            continue
        for k in range(7):
            subset = np.arange(len(a)) if k==0 else rng.choice(len(a), int(.7*len(a)), replace=False)
            cv2.setRNGSeed(100+j*7+k)
            _, r, _, _, _, _ = estimate(a[subset], b[subset], lens)
            all_fits[j, k] = np.degrees(r)
        if j%15==0:
            print(f'Checking point subsets: {j}/{len(selected)}', flush=True)
    tpair = pair_time[selected]; mid = tpair.mean(axis=1)
    np.savez_compressed(OUT/'point_subset_fits.npz', times=tpair, fits_deg=all_fits)
    support = np.isfinite(all_fits).all(axis=2).sum(axis=1)
    median = np.full((len(selected), 3), np.nan)
    spread = np.full_like(median, np.nan)
    for i in range(len(selected)):
        good = np.isfinite(all_fits[i]).all(axis=1)
        if good.any():
            median[i] = np.median(all_fits[i, good], axis=0)
            spread[i] = 1.4826*np.median(abs(all_fits[i, good]-median[i]), axis=0)
    reliable = (support>=5) & (np.max(spread, axis=1)<1.)
    if reliable.sum()<15:
        raise ValueError('Too few consistent pairs; subset diagnostics saved, no correction produced')
    # Invert the empirically calibrated camera-axis map, not Euler column labels.
    target_vec = np.empty_like(median)
    perm, signs = report['axis_permutation'], np.array(report['axis_signs'])
    target_vec[:, perm] = median*signs
    target = Rotation.from_rotvec(np.radians(target_vec[reliable]))
    base = as_rotation(slerp_series(st, sq, tpair.ravel()))
    base_a, base_b = base[::2], base[1::2]
    knots = np.linspace(START, END, 9)

    def curve(parameters):
        return CubicSpline(knots, np.vstack([np.zeros(3), parameters.reshape(-1, 3), np.zeros(3)]),
                           bc_type=((1, np.zeros(3)), (1, np.zeros(3))))

    def predicted(parameters):
        c = curve(parameters)
        a = base_a * Rotation.from_rotvec(np.radians(c(tpair[:, 0])))
        b = base_b * Rotation.from_rotvec(np.radians(c(tpair[:, 1])))
        return b.inv()*a

    sigma = np.maximum(np.linalg.norm(spread[reliable], axis=1), .3)

    def residual(parameters):
        rotation_error = np.degrees((target.inv()*predicted(parameters)[reliable]).as_rotvec())
        nodes = parameters.reshape(-1, 3)
        # Explicit conservative priors: small corrections and low curvature.
        return np.r_[(rotation_error/sigma[:, None]).ravel(),
                     parameters/.6, (np.diff(np.vstack([np.zeros(3), nodes, np.zeros(3)]), n=2, axis=0)/.4).ravel()]

    initial = np.zeros(21)
    fit = least_squares(residual, initial, bounds=(-1., 1.), loss='soft_l1', f_scale=1., max_nfev=150)
    if not fit.success:
        raise ValueError(f'Curve fit failed: {fit.message}')
    spline = curve(fit.x)
    active = np.flatnonzero((ts>START)&(ts<END))
    delta = spline(ts[active])
    edited = (as_rotation(qs[active])*Rotation.from_rotvec(np.radians(delta))).as_quat()[:, [3, 0, 1, 2]]
    # Preserve the source's quaternion sign branch.
    edited *= np.where(np.sum(edited*qs[active], axis=1)<0, -1., 1.)[:, None]
    edits = {}
    for idx, q in zip(active, edited):
        s = samples[idx]
        if s.inverted:
            q = -q
        edits[s.frame, s.index_in_frame] = quat.norm(quat.gyroflow_to_dji(q))
    shutil.copyfile(baseline, OUT/'00_control.mp4')
    dest = OUT/'tracking_curve.mp4'
    sidecar.build(str(baseline), str(dest), edits)
    zero = predicted(initial)
    after = predicted(fit.x)
    before_error = np.degrees((target.inv()*zero[reliable]).magnitude())
    after_error = np.degrees((target.inv()*after[reliable]).magnitude())
    grid = np.linspace(START, END, 301)
    values = spline(grid)
    np.testing.assert_allclose(spline([START, END]), 0, atol=1e-10)
    np.testing.assert_allclose(spline([START, END], 1), 0, atol=1e-10)
    np.savetxt(OUT/'correction.csv', np.column_stack([grid, values]), delimiter=',',
               header='video_time_s,local_quat_x_deg,local_quat_y_deg,local_quat_z_deg', comments='')
    peak = int(np.argmax(np.linalg.norm(values, axis=1)))
    summary = dict(source=report['source'], baseline=str(baseline), windows=[[START, END]], fade_s=1e-9,
        method='bounded local quaternion spline fit to point-subset consensus; robust loss and small/smooth priors',
        reliable_pairs=int(reliable.sum()), total_pairs=len(selected), subset_fits=7,
        median_image_mismatch_before_deg=float(np.median(before_error)),
        median_image_mismatch_after_deg=float(np.median(after_error)),
        peak_correction_deg=float(np.linalg.norm(values[peak])), peak_time_s=float(grid[peak]),
        peak_components_deg=np.max(abs(values), axis=0).tolist(),
        timing_changed=False, note='Experimental conservative curve, not ground truth. '
        'Subset agreement does not remove shared lens/rolling-shutter/tracking bias. '
        'Fit improvement is on fitting data, not independent validation. '
        'Outside the window equals 8 Hz baseline; previous Euler bridges are not used.',
        variants={'00_control':dict(path=str(OUT/'00_control.mp4')),
                  'tracking_curve':dict(path=str(dest))})
    (OUT/'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    baseline_vec = np.degrees(zero.as_rotvec())[:, perm]*signs
    for k, name in enumerate('XYZ'):
        axes[0].plot(mid, median[:, k]-baseline_vec[:, k], label=name, alpha=.7)
        axes[1].plot(grid, values[:, k], label=f'local quaternion {name}')
    axes[0].scatter(mid[~reliable], np.zeros((~reliable).sum()), marker='x', c='black', label='rejected pair')
    axes[0].set_ylabel('Image minus baseline, deg / frame pair')
    axes[1].set_ylabel('Applied orientation correction, deg')
    for ax in axes:
        ax.axvline(55.09, c='gray', ls=':'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.set_xlabel('Original video time, seconds')
    fig.tight_layout(); fig.savefig(OUT/'correction.png', dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-fits', action='store_true')
    main(reuse_fits=parser.parse_args().reuse_fits)

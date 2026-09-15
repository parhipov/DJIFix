"""Compare DJI telemetry with image motion before and after Gyroflow stabilization."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Edit these parameters and run: python compare_stabilization_telemetry.py
# ---------------------------------------------------------------------------
SOURCE_VIDEO = r'F:\36\54-56-DJI_20260517170526_0020_D.MP4'
STABILIZED_VIDEO = r'F:\36\54-56-DJI_20260517170526_0020_D_stabilized.mp4'
START_S = 0.0
END_S = None                 # None = common end of both videos
#TRACKING_SCALE = 0.25        # recommended; 0.125 is a faster preview
TRACKING_SCALE = 0.125        # recommended; 0.125 is a faster preview
SESSION_ROOT = 'artifacts/sessions'

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'research'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import slerp_series
from dji_o4 import read_telemetry
from imagerot import affine_flow, best_axis_map
from analyze_telemetry_outliers import analyze_session


FIELDS = ('time_s', 'frame_a', 'frame_b', 'x_deg_s', 'y_deg_s', 'z_deg_s',
          'confidence', 'support')
AXES = ('X', 'Y', 'Z/roll')


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frames < 2:
        raise ValueError(f'Invalid video properties: {path}')
    return {'path': str(path), 'width': width, 'height': height,
            'fps': fps, 'frames': frames, 'duration_s': frames / fps}


def new_session(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f'{stamp}_{suffix:02d}'
        suffix += 1
    candidate.mkdir()
    return candidate


def write_series(path: Path, time_s: np.ndarray, frames: np.ndarray,
                 values: np.ndarray, confidence: np.ndarray,
                 support: np.ndarray) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for i in range(len(time_s)):
            writer.writerow((f'{time_s[i]:.9f}', int(frames[i]), int(frames[i] + 1),
                             *(f'{value:.9g}' for value in values[i]),
                             f'{confidence[i]:.6f}', int(support[i])))


def image_series(video: Path, first: int, count: int, fps: float,
                 scale: float, focal_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = affine_flow(str(video), first, count, scale=scale, progress=True,
                         focal_px=focal_px, max_pts=1000)
    # These are image-space angular-motion channels. Z/roll is least affected by
    # translation; X/Y can include parallax and Gyroflow's crop movement.
    values = np.column_stack([
        result['shift_x_deg'], result['shift_y_deg'], result['roll_deg']
    ]) * fps
    support = result['npts'].astype(int)
    confidence = np.clip(support / 300.0, 0, 1)
    confidence[~np.isfinite(values).all(1)] = 0
    return values, confidence, support


def gyro_series(source: Path, frame_a: np.ndarray, fps: float,
                image_source: np.ndarray, image_support: np.ndarray):
    clip, samples, _ = read_telemetry(str(source))
    raw_time = np.array([sample.t_us / 1e6 for sample in samples])
    quaternions = np.array([sample.q_cam for sample in samples])
    order = np.argsort(raw_time, kind='stable')
    raw_time, quaternions = raw_time[order], quaternions[order]
    unique = np.r_[True, np.diff(raw_time) > 0]
    raw_time, quaternions = raw_time[unique], quaternions[unique]
    offset_s = (clip.frame_readout_time_ms or 0) / 2000
    pair_times = np.column_stack([frame_a / fps, (frame_a + 1) / fps]) + offset_s
    if pair_times.min() < raw_time[0] or pair_times.max() > raw_time[-1]:
        raise ValueError('DJI telemetry does not cover the selected video frame range')
    q = slerp_series(raw_time, quaternions, pair_times.ravel()).reshape(-1, 2, 4)
    first_q = Rotation.from_quat(q[:, 0][:, [1, 2, 3, 0]])
    second_q = Rotation.from_quat(q[:, 1][:, [1, 2, 3, 0]])
    raw_velocity = np.degrees((second_q.inv() * first_q).as_rotvec()) * fps
    reliable = np.isfinite(image_source).all(1) & (image_support >= 80)
    activity = np.linalg.norm(raw_velocity, axis=1)
    if reliable.sum() > 200:
        cutoff = np.nanpercentile(activity[reliable], 85)
        reliable &= activity <= cutoff
    if reliable.sum() < 30:
        raise ValueError('Too few reliable source-image tracks to align telemetry axes')
    fit_error, permutation, signs = best_axis_map(
        raw_velocity[reliable], image_source[reliable])
    values = raw_velocity[:, permutation] * np.asarray(signs)
    metadata = {
        'source_type': 'DJI fused attitude quaternions (not raw gyro samples)',
        'frame_readout_center_offset_ms': offset_s * 1000,
        'axis_permutation': list(permutation),
        'axis_signs': list(signs),
        'axis_fit_mean_error_deg_s': float(fit_error),
        'axis_fit_pairs': int(reliable.sum()),
        'focal_length_px': float(clip.focal_length),
        'distortion_coefficients': list(clip.distortion_coeffs or []),
    }
    return values, metadata


def plot_overview(time_s: np.ndarray, data: dict[str, np.ndarray], output: Path) -> None:
    fig, panels = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    colors = ('#2878b5', '#e07a1f', '#3a923a')
    titles = ('DJI gyro telemetry', 'Image: source', 'Image: stabilized')
    for panel, key, title in zip(panels, ('gyro', 'source', 'stabilized'), titles):
        for axis, color, label in zip(range(3), colors, AXES):
            panel.plot(time_s, data[key][:, axis], color=color, linewidth=.75, label=label)
        panel.set_title(title)
        panel.set_xlabel('time, s')
        panel.grid(alpha=.22)
    panels[0].set_ylabel('angular velocity, deg/s')
    panels[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('Gyroflow stabilization telemetry comparison')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(source: Path, stabilized: Path, start_s: float, end_s: float | None,
        scale: float, output_root: Path) -> Path:
    source, stabilized = source.resolve(), stabilized.resolve()
    if not source.is_file() or not stabilized.is_file():
        missing = [str(path) for path in (source, stabilized) if not path.is_file()]
        raise FileNotFoundError('Missing input: ' + ', '.join(missing))
    source_info, stabilized_info = video_info(source), video_info(stabilized)
    if not np.isclose(source_info['fps'], stabilized_info['fps'], rtol=0, atol=1e-4):
        raise ValueError('Source and stabilized videos have different frame rates')
    if (source_info['width'], source_info['height']) != (
            stabilized_info['width'], stabilized_info['height']):
        raise ValueError('Source and stabilized videos have different dimensions')
    fps = source_info['fps']
    common_frames = min(source_info['frames'], stabilized_info['frames'])
    end_s = min(end_s if end_s is not None else common_frames / fps,
                (common_frames - 1) / fps)
    if not (0 <= start_s < end_s) or not (0 < scale <= 1):
        raise ValueError('Invalid START_S, END_S or TRACKING_SCALE')
    first = int(round(start_s * fps))
    stop = min(int(round(end_s * fps)), common_frames - 1)
    frame_a = np.arange(first, stop, dtype=int)
    if len(frame_a) < 5:
        raise ValueError('Selected interval is too short')
    time_s = (frame_a + .5) / fps

    session = new_session(output_root)
    print(f'Session: {session}', flush=True)
    clip, samples, _ = read_telemetry(str(source))
    telemetry_end = max(sample.t_us for sample in samples) / 1e6
    readout_offset = (clip.frame_readout_time_ms or 0) / 2000
    covered = (frame_a + 1) / fps + readout_offset <= telemetry_end
    frame_a, time_s = frame_a[covered], time_s[covered]
    if len(frame_a) < 5:
        raise ValueError('DJI telemetry does not cover the selected interval')

    focal = float(clip.focal_length or max(source_info['width'], source_info['height']) / 2)
    print('Tracking source image ...', flush=True)
    image_source, source_confidence, source_support = image_series(
        source, int(frame_a[0]), len(frame_a), fps, scale, focal)
    print('Tracking stabilized image ...', flush=True)
    image_stabilized, stabilized_confidence, stabilized_support = image_series(
        stabilized, int(frame_a[0]), len(frame_a), fps, scale, focal)
    print('Sampling DJI telemetry ...', flush=True)
    gyro, gyro_metadata = gyro_series(source, frame_a, fps, image_source, source_support)

    ones = np.ones(len(frame_a))
    write_series(session / 'telemetry_gyro.csv', time_s, frame_a, gyro, ones, ones)
    write_series(session / 'telemetry_image_source.csv', time_s, frame_a, image_source,
                 source_confidence, source_support)
    write_series(session / 'telemetry_image_stabilized.csv', time_s, frame_a,
                 image_stabilized, stabilized_confidence, stabilized_support)
    plot_overview(time_s, {'gyro': gyro, 'source': image_source,
                           'stabilized': image_stabilized},
                  session / 'telemetry_overview.png')
    metadata = {
        'format_version': 1,
        'created_at': datetime.now().astimezone().isoformat(),
        'source_video': source_info,
        'stabilized_video': stabilized_info,
        'analyzed_interval_s': [float(time_s[0] - .5 / fps),
                                float(time_s[-1] + .5 / fps)],
        'samples': len(time_s),
        'tracking_scale': scale,
        'telemetry_columns': list(FIELDS),
        'axes': {'x': 'horizontal image-motion angular proxy',
                 'y': 'vertical image-motion angular proxy',
                 'z': 'image-plane roll angular velocity'},
        'gyro_mapping': gyro_metadata,
        'notes': [
            'All three CSV files use the same frame-pair time grid and schema.',
            'Image X/Y can contain translation, parallax and stabilization crop motion.',
            'Stabilized video uses the source focal length as a common angular scale.',
        ],
    }
    (session / 'session.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    report = analyze_session(session)
    print(f'Detected bug ranges: {report["event_count"]}', flush=True)
    print(f'Results: {session}', flush=True)
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(SOURCE_VIDEO))
    parser.add_argument('--stabilized', type=Path, default=Path(STABILIZED_VIDEO))
    parser.add_argument('--start', type=float, default=START_S)
    parser.add_argument('--end', type=float, default=END_S)
    parser.add_argument('--scale', type=float, default=TRACKING_SCALE)
    parser.add_argument('--output-root', type=Path, default=ROOT / SESSION_ROOT)
    args = parser.parse_args()
    run(args.source, args.stabilized, args.start, args.end, args.scale, args.output_root)


if __name__ == '__main__':
    main()

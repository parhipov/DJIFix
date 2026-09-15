"""Find stabilization-only motion spikes in a telemetry comparison session.

Edit SESSION_DIR below and run:
    python analyze_telemetry_outliers.py

Or override the folder from the command line:
    python analyze_telemetry_outliers.py artifacts/sessions/YYYYMMDD_HHMMSS
"""
from __future__ import annotations

# Folder containing the three telemetry_*.csv files. Reports and pictures are
# written back into this same folder. Relative paths start at the project root.
SESSION_DIR = r'k:\Work\Python\DJI fix\artifacts\sessions\o4pro'

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter


ROOT = Path(__file__).resolve().parents[2]

FILES = {
    'gyro': 'telemetry_gyro.csv',
    'source': 'telemetry_image_source.csv',
    'stabilized': 'telemetry_image_stabilized.csv',
}
FIELDS = ('time_s', 'frame_a', 'frame_b', 'x_deg_s', 'y_deg_s', 'z_deg_s',
          'confidence', 'support')
AXES = ('X', 'Y', 'Z/roll')

# Detector parameters. They can be edited here without rerunning video tracking.
LOCAL_WINDOW_S = 0.12
AGREEMENT_Z_MAX = 4.0
AGREEMENT_ABS_MAX_DEG_S = 15.0
OUTLIER_Z_MIN = 8.0
MIN_IMPULSE_DEG_S = 5.0
# A smaller oscillation is visible as jitter when it survives for several
# neighbouring frames, even though no individual sample reaches the main limit.
MICRO_OUTLIER_Z_MIN = 4.0
MICRO_MIN_IMPULSE_DEG_S = 2.0
MICRO_NEIGHBORHOOD_S = 0.10
MICRO_MIN_HITS = 2
MICRO_EXCLUSION_FROM_LARGE_S = 0.25
# A faulty gyro impulse can make gyro/source disagree, so it must not be
# rejected by the agreement gate. In that case Gyroflow's output normally
# jumps opposite to the gyro error while the source image remains calm.
GYRO_FAULT_MIN_DEG_S = 8.0
GYRO_FAULT_OUTPUT_MIN_DEG_S = 5.0
GYRO_FAULT_ANTI_ALIGNMENT_MIN = 0.70
GYRO_FAULT_SOURCE_DETAIL_MAX_DEG_S = 8.0
GYRO_FAULT_MIN_HITS = 2
GYRO_FAULT_SUPPORT_Z_MIN = 20.0
GYRO_FAULT_SUPPORT_IMPULSE_DEG_S = 5.0
GYRO_FAULT_EXPAND_S = 0.25
MIN_CONFIDENCE = 0.20
PAD_S = 0.06
MERGE_GAP_S = 0.12


def bug_strength(peak_score: float, peak_impulse: float) -> tuple[float, str]:
    """Map threshold exceedance and physical impulse to a readable 0..100 scale."""
    score_ratio = max(peak_score / OUTLIER_Z_MIN, 1.0)
    impulse_ratio = max(peak_impulse / MIN_IMPULSE_DEG_S, 1.0)
    combined = float(np.sqrt(score_ratio * impulse_ratio))
    strength = float(np.clip(10 + 90 * (1 - np.exp(-.65 * (combined - 1))), 10, 100))
    if strength < 35:
        level = 'weak'
    elif strength < 65:
        level = 'medium'
    elif strength < 85:
        level = 'strong'
    else:
        level = 'extreme'
    return round(strength, 1), level


def load_series(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8')
    data = np.atleast_1d(data)
    if data.dtype.names != FIELDS:
        raise ValueError(f'{path}: expected columns {FIELDS}, got {data.dtype.names}')
    return {name: np.asarray(data[name], dtype=float) for name in FIELDS}


def _mad_scale(values: np.ndarray, floor: float) -> np.ndarray:
    center = np.nanmedian(values, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(values - center), axis=0)
    return np.maximum(scale, floor)


def _filled(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    x = np.arange(len(out))
    for axis in range(out.shape[1]):
        good = np.isfinite(out[:, axis])
        if good.sum() < 2:
            out[:, axis] = 0
        else:
            out[:, axis] = np.interp(x, x[good], out[good, axis])
    return out


def _local_detail(values: np.ndarray, samples: int) -> np.ndarray:
    size = samples if samples % 2 else samples + 1
    filled = _filled(values)
    baseline = np.column_stack([
        median_filter(filled[:, axis], size=size, mode='nearest') for axis in range(3)
    ])
    detail = filled - baseline
    detail[~np.isfinite(values)] = np.nan
    return detail


def _ranges(mask: np.ndarray, time_s: np.ndarray, merge_samples: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > merge_samples + 1:
            groups.append((start, previous))
            start = index
        previous = index
    groups.append((start, previous))
    return groups


def plot_three(series: dict[str, dict[str, np.ndarray]], events: list[dict],
               output: Path) -> None:
    fig, panels = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    colors = ('#2878b5', '#e07a1f', '#3a923a')
    titles = ('DJI gyro telemetry', 'Image: source', 'Image: stabilized')
    for panel, key, title in zip(panels, FILES, titles):
        values = np.column_stack([series[key][f'{axis}_deg_s'] for axis in 'xyz'])
        for axis, color, label in zip(range(3), colors, AXES):
            panel.plot(series[key]['time_s'], values[:, axis], color=color,
                       linewidth=.75, label=label)
        for event in events:
            strength = event['strength_0_100'] / 100
            color = plt.cm.Reds(.22 + .73 * strength)
            panel.axvspan(event['start_s'], event['end_s'], color=color,
                         alpha=.22 + .48 * strength)
        panel.set_title(title)
        panel.set_xlabel('time, s')
        panel.grid(alpha=.22)
    panels[0].set_ylabel('angular velocity, deg/s')
    panels[0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'Telemetry comparison; detected bug ranges: {len(events)}')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def analyze_session(session: Path) -> dict:
    session = Path(session).resolve()
    series = {key: load_series(session / filename) for key, filename in FILES.items()}
    time_s = series['gyro']['time_s']
    for key in ('source', 'stabilized'):
        if len(series[key]['time_s']) != len(time_s) or not np.allclose(
                series[key]['time_s'], time_s, rtol=0, atol=1e-9):
            raise ValueError(f'{key} telemetry is not on the gyro time grid')
    if len(time_s) < 5:
        raise ValueError('Need at least five synchronized frame pairs')
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid telemetry time grid')

    gyro = np.column_stack([series['gyro'][f'{axis}_deg_s'] for axis in 'xyz'])
    source = np.column_stack([series['source'][f'{axis}_deg_s'] for axis in 'xyz'])
    stabilized = np.column_stack([series['stabilized'][f'{axis}_deg_s'] for axis in 'xyz'])
    valid = np.isfinite(gyro).all(1) & np.isfinite(source).all(1) & np.isfinite(stabilized).all(1)
    valid &= series['source']['confidence'] >= MIN_CONFIDENCE
    valid &= series['stabilized']['confidence'] >= MIN_CONFIDENCE

    window = max(3, round(LOCAL_WINDOW_S / dt))
    gyro_detail = _local_detail(gyro, window)
    source_detail = _local_detail(source, window)
    stabilized_detail = _local_detail(stabilized, window)
    agreement_delta = gyro_detail - source_detail
    agreement_scale = _mad_scale(agreement_delta[valid], floor=.35)
    agreement_z = np.sqrt(np.nanmean((agreement_delta / agreement_scale) ** 2, axis=1))
    agreement_abs = np.linalg.norm(gyro - source, axis=1)

    trusted = valid & (agreement_z <= AGREEMENT_Z_MAX)
    trusted &= agreement_abs <= AGREEMENT_ABS_MAX_DEG_S
    if trusted.sum() < 10:
        raise ValueError('Too few trustworthy samples where gyro and source image agree')
    # A stabilization bug must be a local impulse in the stabilized image itself.
    # Comparing it directly with the noisier reference detail marks ordinary smooth
    # stabilization as an error, so the references are used only as an agreement gate.
    score_scale = _mad_scale(stabilized_detail[trusted], floor=.20)
    axis_score = np.abs(stabilized_detail) / score_scale
    outlier_score = np.nanmax(axis_score, axis=1)
    impulse = np.linalg.norm(stabilized_detail, axis=1)
    large_trigger = (trusted & (outlier_score >= OUTLIER_Z_MIN) &
                     (impulse >= MIN_IMPULSE_DEG_S))
    micro_candidate = (trusted & (outlier_score >= MICRO_OUTLIER_Z_MIN) &
                       (impulse >= MICRO_MIN_IMPULSE_DEG_S))
    micro_window = max(3, round(MICRO_NEIGHBORHOOD_S / dt))
    micro_hits = np.convolve(micro_candidate.astype(int),
                             np.ones(micro_window, int), mode='same')
    micro_trigger = micro_candidate & (micro_hits >= MICRO_MIN_HITS)
    exclusion = max(1, round(MICRO_EXCLUSION_FROM_LARGE_S / dt))
    near_large = np.convolve(large_trigger.astype(int),
                             np.ones(2 * exclusion + 1, int), mode='same') > 0
    micro_trigger &= ~near_large

    gyro_fault = agreement_delta
    gyro_fault_magnitude = np.linalg.norm(gyro_fault, axis=1)
    source_detail_magnitude = np.linalg.norm(source_detail, axis=1)
    anti_denominator = gyro_fault_magnitude * impulse
    gyro_fault_anti_alignment = np.divide(
        -np.sum(gyro_fault * stabilized_detail, axis=1), anti_denominator,
        out=np.zeros_like(anti_denominator), where=anti_denominator > 1e-9)
    gyro_fault_candidate = (
        valid &
        (gyro_fault_magnitude >= GYRO_FAULT_MIN_DEG_S) &
        (impulse >= GYRO_FAULT_OUTPUT_MIN_DEG_S) &
        (gyro_fault_anti_alignment >= GYRO_FAULT_ANTI_ALIGNMENT_MIN) &
        (source_detail_magnitude <= GYRO_FAULT_SOURCE_DETAIL_MAX_DEG_S)
    )
    gyro_fault_window = max(3, round(MICRO_NEIGHBORHOOD_S / dt))
    gyro_fault_hits = np.convolve(
        gyro_fault_candidate.astype(int), np.ones(gyro_fault_window, int),
        mode='same')
    gyro_fault_seed = gyro_fault_candidate & (
        gyro_fault_hits >= GYRO_FAULT_MIN_HITS)
    fault_expand = max(1, round(GYRO_FAULT_EXPAND_S / dt))
    near_gyro_fault = np.convolve(
        gyro_fault_seed.astype(int), np.ones(2 * fault_expand + 1, int),
        mode='same') > 0
    gyro_fault_support = (
        valid & (outlier_score >= GYRO_FAULT_SUPPORT_Z_MIN) &
        (impulse >= GYRO_FAULT_SUPPORT_IMPULSE_DEG_S)
    )
    gyro_fault_trigger = gyro_fault_support & near_gyro_fault
    core_trigger = large_trigger | micro_trigger | gyro_fault_trigger

    pad = max(0, round(PAD_S / dt))
    trigger = core_trigger.copy()
    if pad and core_trigger.any():
        trigger = np.convolve(core_trigger.astype(int), np.ones(2 * pad + 1, int), mode='same') > 0
        trigger &= valid
    merged = _ranges(trigger, time_s, max(0, round(MERGE_GAP_S / dt)))
    events = []
    for number, (lo, hi) in enumerate(merged, 1):
        core = np.flatnonzero(core_trigger & (np.arange(len(time_s)) >= lo)
                              & (np.arange(len(time_s)) <= hi))
        if not len(core):
            continue
        peak = int(core[np.nanargmax(outlier_score[core])])
        axis = int(np.nanargmax(axis_score[peak]))
        strength, strength_level = bug_strength(outlier_score[peak], impulse[peak])
        if np.any(gyro_fault_trigger[lo:hi + 1]):
            trigger_kind = 'gyro_fault'
        elif np.any(large_trigger[lo:hi + 1]):
            trigger_kind = 'large_impulse'
        else:
            trigger_kind = 'micro_jitter'
        events.append({
            'id': number,
            'start_s': float(max(0, time_s[lo] - dt / 2)),
            'end_s': float(time_s[hi] + dt / 2),
            'duration_s': float((hi - lo + 1) * dt),
            'peak_time_s': float(time_s[peak]),
            'peak_axis': AXES[axis],
            'peak_score': float(outlier_score[peak]),
            'peak_impulse_deg_s': float(impulse[peak]),
            'strength_0_100': strength,
            'strength_level': strength_level,
            'trigger_kind': trigger_kind,
            'agreement_score_at_peak': float(agreement_z[peak]),
            'gyro_source_difference_deg_s': float(agreement_abs[peak]),
            'gyro_fault_detail_deg_s': float(gyro_fault_magnitude[peak]),
            'gyro_fault_anti_alignment': float(gyro_fault_anti_alignment[peak]),
            'source_detail_deg_s': float(source_detail_magnitude[peak]),
            'gyro_xyz_deg_s': gyro[peak].tolist(),
            'source_xyz_deg_s': source[peak].tolist(),
            'stabilized_xyz_deg_s': stabilized[peak].tolist(),
            'source_support': int(series['source']['support'][peak]),
            'stabilized_support': int(series['stabilized']['support'][peak]),
        })

    report = {
        'format_version': 1,
        'session': str(session),
        'criterion': ('stabilized-image outlier with agreeing gyro/source references, '
                      'or a high-confidence gyro/source fault whose stabilization '
                      'response points in the opposite direction'),
        'event_count': len(events),
        'events': events,
        'detector': {
            'local_window_s': LOCAL_WINDOW_S,
            'agreement_z_max': AGREEMENT_Z_MAX,
            'agreement_absolute_max_deg_s': AGREEMENT_ABS_MAX_DEG_S,
            'outlier_z_min': OUTLIER_Z_MIN,
            'minimum_impulse_deg_s': MIN_IMPULSE_DEG_S,
            'micro_outlier_z_min': MICRO_OUTLIER_Z_MIN,
            'micro_minimum_impulse_deg_s': MICRO_MIN_IMPULSE_DEG_S,
            'micro_neighborhood_s': MICRO_NEIGHBORHOOD_S,
            'micro_minimum_hits': MICRO_MIN_HITS,
            'micro_exclusion_from_large_s': MICRO_EXCLUSION_FROM_LARGE_S,
            'gyro_fault_minimum_deg_s': GYRO_FAULT_MIN_DEG_S,
            'gyro_fault_output_minimum_deg_s': GYRO_FAULT_OUTPUT_MIN_DEG_S,
            'gyro_fault_anti_alignment_minimum': GYRO_FAULT_ANTI_ALIGNMENT_MIN,
            'gyro_fault_source_detail_maximum_deg_s': GYRO_FAULT_SOURCE_DETAIL_MAX_DEG_S,
            'gyro_fault_minimum_hits': GYRO_FAULT_MIN_HITS,
            'gyro_fault_support_z_minimum': GYRO_FAULT_SUPPORT_Z_MIN,
            'gyro_fault_support_impulse_deg_s': GYRO_FAULT_SUPPORT_IMPULSE_DEG_S,
            'gyro_fault_expand_s': GYRO_FAULT_EXPAND_S,
            'minimum_confidence': MIN_CONFIDENCE,
            'padding_s': PAD_S,
            'merge_gap_s': MERGE_GAP_S,
            'robust_axis_scale_deg_s': score_scale.tolist(),
            'agreement_axis_scale_deg_s': agreement_scale.tolist(),
            'strength_scale': ('0..100 from geometric mean of robust-score and '
                               'impulse-amplitude threshold exceedance'),
        },
        'limitations': [
            'Image X/Y motion is an angular proxy and may include parallax or crop motion.',
            'Detected ranges are candidates for inspection, not proof of a Gyroflow defect.',
            'Gyro-fault events bypass the normal reference-agreement gate.',
            'Slow stabilization drift is intentionally ignored by the impulse detector.',
        ],
    }
    (session / 'telemetry_outliers.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    plot_three(series, events, session / 'telemetry_outliers.png')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path, nargs='?', default=Path(SESSION_DIR),
                        help='session folder (default: SESSION_DIR at top of script)')
    args = parser.parse_args()
    session = args.session if args.session.is_absolute() else ROOT / args.session
    report = analyze_session(session)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

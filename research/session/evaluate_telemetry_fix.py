"""Evaluate stabilization before/after a telemetry bug fix, event by event."""
from __future__ import annotations

# Edit these two folders and run: python evaluate_telemetry_fix.py
BEFORE_SESSION = r'K:\Work\Python\DJI fix\artifacts\sessions\o4pro'
AFTER_SESSION = r'K:\Work\Python\DJI fix\artifacts\sessions\o4pro_after_fix_2'

# Analysis settings. Original cases always come from BEFORE_SESSION.
CONTEXT_S = 0.50
LOCAL_WINDOW_S = 0.12
HIGH_PASS_HZ = 5.0

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('time_s', 'frame_a', 'frame_b', 'x_deg_s', 'y_deg_s', 'z_deg_s',
          'confidence', 'support')
AXES = ('X', 'Y', 'Z/roll')


def load_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8'))
    if data.dtype.names != FIELDS:
        raise ValueError(f'{path}: unexpected telemetry columns {data.dtype.names}')
    return {name: np.asarray(data[name], dtype=float) for name in FIELDS}


def vectors(series: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([series[f'{axis}_deg_s'] for axis in 'xyz'])


def fill(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    index = np.arange(len(result))
    for axis in range(3):
        valid = np.isfinite(result[:, axis])
        if valid.sum() < 2:
            raise ValueError(f'Not enough valid values on axis {AXES[axis]}')
        result[:, axis] = np.interp(index, index[valid], result[valid, axis])
    return result


def local_detail(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(3, samples if samples % 2 else samples + 1)
    clean = fill(values)
    baseline = np.column_stack([
        median_filter(clean[:, axis], size=samples, mode='nearest') for axis in range(3)
    ])
    return clean - baseline


def high_pass(values: np.ndarray, fps: float) -> np.ndarray:
    clean = fill(values)
    sos = butter(2, HIGH_PASS_HZ, btype='highpass', fs=fps, output='sos')
    return sosfiltfilt(sos, clean, axis=0)


def metric_set(values: np.ndarray, detail: np.ndarray, high: np.ndarray,
               mask: np.ndarray, dt: float) -> dict[str, float]:
    if mask.sum() < 2:
        raise ValueError('A bug interval contains fewer than two samples')
    value = values[mask]
    impulse = detail[mask]
    hf = high[mask]
    jerk = np.diff(value, axis=0) / dt
    norm = lambda data: np.linalg.norm(data, axis=1)
    # Absolute excursion of the stabilized image inside the window: the integral
    # of its angular velocity minus the linear trend. A smooth multi-degree
    # swing over 0.3 s has almost no impulse or jerk but is a visible lurch.
    excursion = np.cumsum(value - value.mean(axis=0), axis=0) * dt
    return {
        'excursion_peak_deg': float(np.max(norm(excursion))),
        'speed_peak_deg_s': float(np.max(norm(value))),
        'speed_rms_deg_s': float(np.sqrt(np.mean(norm(value) ** 2))),
        'impulse_peak_deg_s': float(np.max(norm(impulse))),
        'impulse_rms_deg_s': float(np.sqrt(np.mean(norm(impulse) ** 2))),
        'jerk_peak_deg_s2': float(np.max(norm(jerk))),
        'jerk_rms_deg_s2': float(np.sqrt(np.mean(norm(jerk) ** 2))),
        'high_frequency_rms_deg_s': float(np.sqrt(np.mean(norm(hf) ** 2))),
    }


def improvement(before: float, after: float) -> float | None:
    if not np.isfinite(before) or abs(before) < 1e-12:
        return None
    return float((before - after) / before * 100)


def rounded(value: float | None, digits: int = 3):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def align_to(reference_t: np.ndarray, series: dict[str, np.ndarray], name: str):
    source_t = series['time_s']
    if len(source_t) == len(reference_t) and np.allclose(source_t, reference_t, atol=1e-9, rtol=0):
        return series
    if reference_t.min() < source_t.min() or reference_t.max() > source_t.max():
        raise ValueError(f'{name} does not cover the before-session timeline')
    aligned = {'time_s': reference_t.copy()}
    for field in FIELDS[1:]:
        aligned[field] = np.interp(reference_t, source_t, series[field])
    return aligned


def plot_case(path: Path, case: dict, time_s: np.ndarray, before: np.ndarray,
              after: np.ndarray, gyro: np.ndarray, source: np.ndarray) -> None:
    context = (time_s >= case['context_start_s']) & (time_s <= case['context_end_s'])
    fig, panels = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for axis, panel in enumerate(panels[:3]):
        panel.plot(time_s[context], before[context, axis], label='stabilized before',
                   color='#c9483b', linewidth=1.1)
        panel.plot(time_s[context], after[context, axis], label='stabilized after fix',
                   color='#2878b5', linewidth=1.1)
        panel.plot(time_s[context], gyro[context, axis], label='gyro reference',
                   color='.45', linewidth=.65, alpha=.65)
        panel.plot(time_s[context], source[context, axis], label='source image',
                   color='#3a923a', linewidth=.65, alpha=.65)
        panel.axvspan(case['start_s'], case['end_s'], color='orange', alpha=.13)
        panel.set_ylabel(f'{AXES[axis]} deg/s')
        panel.grid(alpha=.2)
    before_detail = np.linalg.norm(case['_before_detail'][context], axis=1)
    after_detail = np.linalg.norm(case['_after_detail'][context], axis=1)
    panels[3].plot(time_s[context], before_detail, color='#c9483b', label='impulse before')
    panels[3].plot(time_s[context], after_detail, color='#2878b5', label='impulse after')
    panels[3].axvspan(case['start_s'], case['end_s'], color='orange', alpha=.13)
    panels[3].set_ylabel('local impulse, deg/s')
    panels[3].set_xlabel('time, s')
    panels[3].grid(alpha=.2)
    panels[0].legend(ncol=4, fontsize=8)
    panels[3].legend(fontsize=8)
    fig.suptitle(
        f'Case {case["id"]}: {case["start_s"]:.3f}–{case["end_s"]:.3f} s | '
        f'{case["verdict"]} | score {case["improvement_score_percent"]:+.1f}%')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary(path: Path, cases: list[dict]) -> None:
    if not cases:
        fig, panel = plt.subplots(figsize=(10, 4))
        panel.text(.5, .5, 'No original bug cases in BEFORE_SESSION', ha='center', va='center')
        panel.axis('off')
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        return
    ids = [str(case['id']) for case in cases]
    keys = ('impulse_peak_deg_s', 'impulse_rms_deg_s', 'jerk_rms_deg_s2',
            'high_frequency_rms_deg_s')
    labels = ('impulse peak', 'impulse RMS', 'jerk RMS', 'HF RMS')
    values = np.array([[
        case['improvement_percent'][key]
        if case['improvement_percent'][key] is not None else np.nan
        for key in keys] for case in cases], float)
    scores = np.array([case['improvement_score_percent'] for case in cases])
    fig, (bars, heat) = plt.subplots(1, 2, figsize=(16, max(5, .35 * len(cases) + 2)))
    positions = np.arange(len(cases))
    colors = np.where(scores >= 15, '#3a923a', np.where(scores < -10, '#c9483b', '#d39b2a'))
    bars.barh(positions, scores, color=colors)
    bars.axvline(0, color='.25', linewidth=.8)
    bars.set_yticks(positions, ids)
    bars.invert_yaxis()
    bars.set_xlabel('combined improvement, % (higher is better)')
    bars.set_ylabel('original bug case')
    bars.grid(axis='x', alpha=.2)
    image = heat.imshow(values, aspect='auto', cmap='RdYlGn', vmin=-100, vmax=100)
    heat.set_yticks(positions, ids)
    heat.set_xticks(np.arange(len(labels)), labels, rotation=30, ha='right')
    heat.set_xlabel('metric improvement, %')
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            heat.text(column, row, f'{values[row, column]:+.0f}', ha='center', va='center', fontsize=8)
    fig.colorbar(image, ax=heat, label='improvement, %')
    fig.suptitle('Telemetry bug-fix evaluation by original detected case')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(before_folder: Path, after_folder: Path) -> dict:
    before_folder, after_folder = Path(before_folder).resolve(), Path(after_folder).resolve()
    original_report = json.loads(
        (before_folder / 'telemetry_outliers.json').read_text(encoding='utf-8'))
    after_report_path = after_folder / 'telemetry_outliers.json'
    after_report = json.loads(after_report_path.read_text(encoding='utf-8')) if after_report_path.exists() else None

    gyro_series = load_csv(before_folder / 'telemetry_gyro.csv')
    source_series = load_csv(before_folder / 'telemetry_image_source.csv')
    before_series = load_csv(before_folder / 'telemetry_image_stabilized.csv')
    after_series = align_to(before_series['time_s'],
                            load_csv(after_folder / 'telemetry_image_stabilized.csv'), 'after telemetry')
    after_gyro_series = align_to(before_series['time_s'],
                                 load_csv(after_folder / 'telemetry_gyro.csv'), 'after gyro')
    after_source_series = align_to(before_series['time_s'],
                                   load_csv(after_folder / 'telemetry_image_source.csv'), 'after source')
    fixed_path = before_folder / 'telemetry_gyro_fixed.csv'
    fixed_series = align_to(before_series['time_s'], load_csv(fixed_path), 'fixed gyro') \
        if fixed_path.exists() else None

    time_s = before_series['time_s']
    dt = float(np.median(np.diff(time_s)))
    fps = 1 / dt
    gyro, source = fill(vectors(gyro_series)), fill(vectors(source_series))
    before, after = fill(vectors(before_series)), fill(vectors(after_series))
    after_gyro, after_source = fill(vectors(after_gyro_series)), fill(vectors(after_source_series))
    fixed = fill(vectors(fixed_series)) if fixed_series else None
    samples = max(3, round(LOCAL_WINDOW_S / dt))
    detail_before, detail_after = local_detail(before, samples), local_detail(after, samples)
    hf_before, hf_after = high_pass(before, fps), high_pass(after, fps)
    bug_union = np.zeros(len(time_s), dtype=bool)
    cases = []
    case_dir = after_folder / 'fix_evaluation_cases'
    case_dir.mkdir(exist_ok=True)
    sample_rows = []

    for fallback_id, event in enumerate(original_report.get('events', []), 1):
        case_id = int(event.get('id', fallback_id))
        start, end = float(event['start_s']), float(event['end_s'])
        inside = (time_s >= start) & (time_s <= end)
        if inside.sum() < 2:
            raise ValueError(f'Original case {case_id} is outside the common timeline')
        bug_union |= inside
        before_metrics = metric_set(before, detail_before, hf_before, inside, dt)
        after_metrics = metric_set(after, detail_after, hf_after, inside, dt)
        percentages = {key: improvement(before_metrics[key], after_metrics[key])
                       for key in before_metrics}
        score_keys = ('impulse_peak_deg_s', 'impulse_rms_deg_s',
                      'jerk_rms_deg_s2', 'high_frequency_rms_deg_s',
                      'excursion_peak_deg')
        score = float(np.mean([percentages[key] for key in score_keys
                              if percentages[key] is not None]))
        verdict = 'improved' if score >= 15 else ('worsened' if score < -10 else 'mixed_or_unchanged')
        context_start, context_end = max(time_s[0], start - CONTEXT_S), min(time_s[-1], end + CONTEXT_S)
        context = (time_s >= context_start) & (time_s <= context_end)
        shoulders = context & ~inside
        collateral = np.linalg.norm(after[shoulders] - before[shoulders], axis=1)
        intended = (np.linalg.norm(fixed[inside] - gyro[inside], axis=1)
                    if fixed is not None else np.array([]))
        case = {
            'id': case_id,
            'start_s': start,
            'end_s': end,
            'duration_s': end - start,
            'context_start_s': float(context_start),
            'context_end_s': float(context_end),
            'original_strength_0_100': event.get('strength_0_100'),
            'original_strength_level': event.get('strength_level'),
            'original_trigger_kind': event.get('trigger_kind', 'large_impulse'),
            'verdict': verdict,
            'improvement_score_percent': score,
            'before': before_metrics,
            'after': after_metrics,
            'improvement_percent': percentages,
            'remaining_impulse_percent': rounded(
                100 - percentages['impulse_peak_deg_s'] if percentages['impulse_peak_deg_s'] is not None else None),
            'intended_telemetry_change_rms_deg_s': rounded(
                np.sqrt(np.mean(intended ** 2)) if len(intended) else None),
            'outside_event_change_rms_deg_s': rounded(
                np.sqrt(np.mean(collateral ** 2)) if len(collateral) else None),
            'mean_tracking_confidence': rounded(float(np.mean(np.minimum(
                before_series['confidence'][inside], after_series['confidence'][inside])))),
        }
        case['_before_detail'], case['_after_detail'] = detail_before, detail_after
        plot_case(case_dir / f'case_{case_id:03d}.png', case, time_s,
                  before, after, gyro, source)
        for index in np.flatnonzero(context):
            sample_rows.append([
                case_id, time_s[index], int(inside[index]),
                *gyro[index], *source[index], *before[index], *after[index],
                *(fixed[index] if fixed is not None else [np.nan] * 3),
                np.linalg.norm(detail_before[index]), np.linalg.norm(detail_after[index]),
                before_series['confidence'][index], after_series['confidence'][index],
            ])
        del case['_before_detail'], case['_after_detail']
        cases.append(case)

    unaffected = ~bug_union
    outside_change = np.linalg.norm(after[unaffected] - before[unaffected], axis=1)
    valid_scores = [case['improvement_score_percent'] for case in cases]
    trigger_kinds = sorted({case['original_trigger_kind'] for case in cases})
    report = {
        'format_version': 1,
        'before_session': str(before_folder),
        'after_session': str(after_folder),
        'source_case_count': len(cases),
        'after_detector_event_count': after_report.get('event_count') if after_report else None,
        'verdict_counts': {name: sum(case['verdict'] == name for case in cases)
                           for name in ('improved', 'mixed_or_unchanged', 'worsened')},
        'results_by_trigger_kind': {
            kind: {
                'case_count': sum(case['original_trigger_kind'] == kind for case in cases),
                'mean_improvement_score_percent': rounded(np.mean([
                    case['improvement_score_percent'] for case in cases
                    if case['original_trigger_kind'] == kind
                ])),
            } for kind in trigger_kinds
        },
        'aggregate': {
            'mean_improvement_score_percent': rounded(np.mean(valid_scores) if valid_scores else None),
            'median_improvement_score_percent': rounded(np.median(valid_scores) if valid_scores else None),
            'unaffected_timeline_change_rms_deg_s': rounded(
                np.sqrt(np.mean(outside_change ** 2)) if len(outside_change) else None),
            'unaffected_timeline_change_peak_deg_s': rounded(
                np.max(outside_change) if len(outside_change) else None),
        },
        'comparability_checks': {
            'gyro_before_after_rms_difference_deg_s': rounded(
                np.sqrt(np.mean(np.linalg.norm(after_gyro - gyro, axis=1) ** 2))),
            'source_image_before_after_rms_difference_deg_s': rounded(
                np.sqrt(np.mean(np.linalg.norm(after_source - source, axis=1) ** 2))),
            'note': ('Values near zero confirm that the two sessions differ mainly in '
                     'the stabilized output being evaluated.'),
        },
        'settings': {'context_s': CONTEXT_S, 'local_window_s': LOCAL_WINDOW_S,
                     'high_pass_hz': HIGH_PASS_HZ},
        'cases': cases,
        'interpretation': ('Positive improvement means less stabilized-image impulse, jerk, '
                           'or high-frequency energy after the telemetry fix.'),
    }
    (after_folder / 'fix_evaluation.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    summary_fields = ['id', 'start_s', 'end_s', 'original_trigger_kind',
                      'original_strength_0_100', 'verdict',
                      'improvement_score_percent', 'remaining_impulse_percent',
                      'intended_telemetry_change_rms_deg_s', 'outside_event_change_rms_deg_s',
                      'mean_tracking_confidence']
    with (after_folder / 'fix_evaluation.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({key: case.get(key) for key in summary_fields} for case in cases)
    sample_fields = ['case_id', 'time_s', 'inside_event',
                     *[f'gyro_{axis}' for axis in 'xyz'], *[f'source_{axis}' for axis in 'xyz'],
                     *[f'before_{axis}' for axis in 'xyz'], *[f'after_{axis}' for axis in 'xyz'],
                     *[f'fixed_gyro_{axis}' for axis in 'xyz'],
                     'before_impulse', 'after_impulse', 'before_confidence', 'after_confidence']
    with (after_folder / 'fix_evaluation_samples.csv').open(
            'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(sample_fields); writer.writerows(sample_rows)
    plot_summary(after_folder / 'fix_evaluation.png', cases)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path, nargs='?', default=Path(BEFORE_SESSION))
    parser.add_argument('after', type=Path, nargs='?', default=Path(AFTER_SESSION))
    args = parser.parse_args()
    before = args.before if args.before.is_absolute() else ROOT / args.before
    after = args.after if args.after.is_absolute() else ROOT / args.after
    report = evaluate(before, after)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

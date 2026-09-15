"""Trace the user-identified output reversal back to original DJI attitudes.

No image analysis, no new correction, no inferred camera/IMU synchronization.
Euler columns use Gyroflow camera-export labels, not claimed physical axes.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align import slerp_series
from dji_o4 import read_telemetry
from telemetry import qeuler

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / 'artifacts/analysis/gyroflow_output'
OUT = ROOT / 'artifacts/analysis/wobble_trace'


def euler(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    # qeuler: mathematical ZYX yaw,pitch,roll. Gyroflow export: X,Y,Z angles
    # labelled pitch,yaw,roll, respectively. Validate against exported org_euler.
    e = qeuler(q)[:, ::-1]
    return np.degrees(np.unwrap(np.radians(e), axis=0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trial = json.loads((ROOT / 'artifacts/experiments/lite/report.json').read_text(encoding='utf-8'))
    clip, samples, _ = read_telemetry(trial['source'])
    raw_time = np.array([s.t_us / 1e6 for s in samples])
    q = np.array([s.q_cam for s in samples])
    order = np.argsort(raw_time, kind='stable')
    ts, qs = raw_time[order], q[order]
    unique = np.r_[True, np.diff(ts) > 0]
    ts, qs = ts[unique], qs[unique]
    # Map the existing baseline construction, not a newly estimated latency.
    shift_s = trial['shift_ms'] / 1000
    raw_video_time = raw_time - shift_s
    raw_e = euler(q)
    rows = json.loads((EXPORT / 'camera_render.json').read_text(encoding='utf-8'))
    native = json.loads((EXPORT / 'camera_native.json').read_text(encoding='utf-8'))
    t = np.array([r['frame'] / clip.fps for r in rows])
    raw_at_frames = slerp_series(ts, qs, t + shift_s)
    raw_frame_e = euler(raw_at_frames)
    input_e = euler(np.array([r['org_quat'] for r in rows]))
    output_e = euler(np.array([r['stab_quat'] for r in rows]))
    # Remove only 360-degree branch choices when comparing conventions.
    exported_e = np.array([r['org_euler'] for r in rows])
    angle_diff = (input_e - exported_e + 180) % 360 - 180
    assert np.max(abs(angle_diff)) < 1e-5, 'Euler convention mismatch'
    native_q = np.array([r['org_quat'] for r in native])
    native_q /= np.linalg.norm(native_q, axis=1, keepdims=True)
    sign = np.where(np.sum(raw_at_frames * native_q, axis=1) < 0, -1, 1)
    core = (t >= 54) & (t <= 56.5)
    max_error = float(np.max(abs(raw_at_frames[core] * sign[core, None] - native_q[core])))
    assert max_error < 1e-5, 'raw timeline does not reproduce native export'

    stages = [('raw_frame', t, raw_frame_e), ('input_8hz', t, input_e),
              ('output_stabilized', t, output_e)]
    events = {}
    for label, tt, ee in stages:
        events[label] = {}
        for k, name, direction in [(1, 'yaw', 1), (2, 'roll', -1)]:
            search = np.flatnonzero((tt >= 55) & (tt <= 56))
            i = search[np.argmax(direction * ee[search, k])]
            value_at_56 = float(np.interp(56, tt, ee[:, k]))
            events[label][name] = dict(time_s=float(tt[i]), angle_deg=float(ee[i, k]),
                return_to_56_deg=value_at_56 - float(ee[i, k]))
    events['raw_records'] = {}
    for k, name, direction in [(1, 'yaw', 1), (2, 'roll', -1)]:
        search = np.flatnonzero((raw_video_time >= 55) & (raw_video_time <= 56))
        i = search[np.argmax(direction * raw_e[search, k])]
        s = samples[i]
        events['raw_records'][name] = dict(
            mapped_video_time_s=float(raw_video_time[i]), original_telemetry_time_s=float(raw_time[i]),
            metadata_frame_zero_based=s.frame, slot_zero_based=s.index_in_frame,
            angle_deg=float(raw_e[i, k]))
    report = dict(source=trial['source'], source_type='original fused attitudes, not raw gyro',
        mapped_time_formula='original telemetry time minus existing control shift',
        existing_control_shift_ms=trial['shift_ms'],
        raw_to_native_max_quaternion_component_error=max_error,
        event=events, candidate_inspection_window_s=[55.4, 56.0],
        context_window_s=[55., 56.3],
        limitation='Output reversal traced to an input reversal, not proof that all motion in the interval is erroneous. Extrema displacement is not an estimated sync offset.')
    (OUT / 'trace.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    with (OUT / 'raw_records_55-56_3.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['mapped_video_time_s', 'original_telemetry_time_s', 'metadata_frame_zero_based',
                    'slot_zero_based', 'gf_pitch_deg', 'gf_yaw_deg', 'gf_roll_deg', 'qw', 'qx', 'qy', 'qz'])
        for i in np.flatnonzero((raw_video_time >= 55) & (raw_video_time <= 56.3)):
            s = samples[i]
            w.writerow([raw_video_time[i], raw_time[i], s.frame, s.index_in_frame, *raw_e[i], *q[i]])
    with (OUT / 'stages_54-57.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['video_time_s'] + [f'{label}_{a}_deg' for label, _, _ in stages for a in ('pitch', 'yaw', 'roll')])
        for i in np.flatnonzero((t >= 54) & (t <= 57)):
            w.writerow([t[i], *raw_frame_e[i], *input_e[i], *output_e[i]])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, k, name in zip(axes, [1, 2], ['Gyroflow yaw column', 'Gyroflow roll column']):
        m = (raw_video_time >= 55) & (raw_video_time <= 56.3)
        ax.plot(raw_video_time[m], raw_e[m, k], color='.7', linewidth=.7, label='Original DJI records')
        for label, tt, ee, color in [('8 Hz input', t, input_e, '#1266b0'),
                                    ('Calculated output', t, output_e, '#ce5020')]:
            m = (tt >= 55) & (tt <= 56.3)
            ax.plot(tt[m], ee[m, k], label=label, color=color)
        ax.axvspan(55.4, 56.0, color='orange', alpha=.12)
        ax.set_ylabel('degrees'); ax.set_title(name); ax.grid(alpha=.25)
    axes[0].legend()
    axes[-1].set_xlabel('Video timeline under the existing baseline mapping, seconds')
    fig.suptitle('Tracing the calculated output reversal back to original telemetry')
    fig.tight_layout(); fig.savefig(OUT / 'trace.png', dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

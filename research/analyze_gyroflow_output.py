"""Flatten calculated camera exports and compare their temporal variation.

The >4 Hz statistic describes the exported path, not measured image jitter.
Quaternion order is w,x,y,z in Gyroflow camera exports (different from type 2).
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
from scipy.signal import butter, sosfiltfilt

from rollfix import logv, qconj, qmul

OUT = Path(__file__).resolve().parent.parent / 'artifacts/analysis/gyroflow_output'


def analyze(name):
    rows = json.loads((OUT / (name + '.json')).read_text(encoding='utf-8'))
    frame = np.array([r['frame'] for r in rows])
    assert np.array_equal(frame, np.arange(4644)), 'unexpected frame coverage'
    time = frame / 50.
    signals, summary = {}, {}
    for key in ('org', 'stab'):
        q = np.array([r[key + '_quat'] for r in rows])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        rate = logv(qmul(qconj(q[:-1]), q[1:])) * 50
        hf = sosfiltfilt(butter(2, 4, fs=50, btype='highpass', output='sos'), rate, axis=0)
        signals[key] = dict(rate=rate, hf=hf)
        sel = (time[:-1] >= 54) & (time[:-1] < 56)
        peak_idx = np.flatnonzero(sel)[np.argmax(np.linalg.norm(hf[sel], axis=1))]
        summary[key] = dict(rate_rms_dps=float(np.sqrt(np.mean(np.sum(rate[sel] ** 2, axis=1)))),
            highpass_4hz_rate_rms_dps=float(np.sqrt(np.mean(np.sum(hf[sel] ** 2, axis=1)))),
            highpass_peak_dps=float(np.linalg.norm(hf[peak_idx])),
            highpass_peak_video_time_s=float(time[peak_idx]))
    fields = ['frame', 'video_time_s', 'timestamp_ms']
    for key in ('org', 'stab'):
        fields += [key + '_quat_' + a for a in ('w', 'x', 'y', 'z')]
        fields += [key + '_' + a + '_deg' for a in ('pitch', 'yaw', 'roll')]
        fields += [key + '_rate_' + a + '_dps' for a in ('x', 'y', 'z')]
    fields += ['fov_scale', 'minimal_fov_scale']
    flattened = []
    for i, r in enumerate(rows):
        row = dict(frame=r['frame'], video_time_s=time[i], timestamp_ms=r['timestamp_ms'])
        for key in ('org', 'stab'):
            row.update(zip((key + '_quat_' + a for a in ('w', 'x', 'y', 'z')), r[key + '_quat']))
            row.update(zip((key + '_' + a + '_deg' for a in ('pitch', 'yaw', 'roll')), r[key + '_euler']))
            rate = signals[key]['rate'][i] if i < len(rows) - 1 else [float('nan')] * 3
            row.update(zip((key + '_rate_' + a + '_dps' for a in ('x', 'y', 'z')), rate))
        row.update(fov_scale=r['fov_scale'], minimal_fov_scale=r['minimal_fov_scale'])
        flattened.append(row)
    for suffix, selected in [('', flattened), ('_51-60', [r for r in flattened if 51 <= r['video_time_s'] < 60])]:
        with (OUT / (name + suffix + '.csv')).open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    return time, rows, signals, summary


def main():
    results = {n: analyze(n) for n in ('camera_native', 'camera_control', 'camera_8hz', 'camera_render')}
    report = dict(window_s=[54, 56], fps=50, frames=4644,
        metric='RMS of the 4 Hz high-pass filtered body angular rate, calculated from frame quaternions',
        caveat='Not a measurement of rendered image motion; source and target share the same potentially faulty telemetry.',
        results={k: v[3] for k, v in results.items()})
    (OUT / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for name, label, color in [('camera_control', 'External control telemetry', '#777777'),
                               ('camera_8hz', 'Our 8 Hz telemetry', '#1266b0')]:
        t, rows, signals, _ = results[name]
        sel = (t[:-1] >= 51) & (t[:-1] <= 60)
        for i, key in enumerate(('org', 'stab')):
            magnitude = np.linalg.norm(signals[key]['hf'], axis=1)
            ax[i].plot(t[:-1][sel], magnitude[sel], label=label, color=color, linewidth=1)
        m = (t >= 51) & (t <= 60)
        ax[2].plot(t[m], np.array([r['fov_scale'] for r in rows])[m], label=label, color=color)
    for a in ax:
        a.axvspan(54, 56, alpha=.12, color='orange')
        a.grid(alpha=.25)
    ax[0].set_title('Exported input orientation: fast angular-rate component (>4 Hz)')
    ax[1].set_title('Calculated stabilized path: fast angular-rate component (>4 Hz)')
    ax[0].set_ylabel('deg/s'); ax[1].set_ylabel('deg/s')
    ax[2].set_title('Calculated crop scale'); ax[2].set_ylabel('FOV scale')
    ax[2].set_xlabel('Time in original video (seconds)')
    ax[0].legend()
    fig.suptitle('Gyroflow calculation exports — NOT motion measured from the output image')
    fig.tight_layout()
    fig.savefig(OUT / 'camera_comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report['results'], indent=2))


if __name__ == '__main__':
    main()

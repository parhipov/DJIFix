"""Replace the identified local attitude excursion with a smooth endpoint bridge.

An experimental shape edit on top of the user-tested 8 Hz telemetry, not another
frequency or synchronization sweep. The bridge is a motion hypothesis to test.
Gyroflow export Euler labels are X/Y/Z angles named pitch/yaw/roll respectively.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from align import slerp_series
from dji_o4 import read_telemetry
from telemetry import qeuler
import quat
import sidecar

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'artifacts/experiments/lite_bridge'


def smootherstep(x):
    x = np.clip(x, 0., 1.)
    return x ** 3 * (10. - 15. * x + 6. * x ** 2)


def build(include_pitch=False, export_camera=True):
    OUT = ROOT / ('artifacts/experiments/lite_bridge_all_axes' if include_pitch else 'artifacts/experiments/lite_bridge')
    baseline = ROOT / 'artifacts/experiments/lite_v2/fullrate_8hz.mp4'
    prior = json.loads((baseline.parent / 'report.json').read_text(encoding='utf-8'))
    _, samples, _ = read_telemetry(str(baseline))
    t = np.array([s.t_us / 1e6 for s in samples])
    q = np.array([s.q_cam for s in samples])
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    unique = np.r_[True, np.diff(ts) > 0]
    ts, qs = ts[unique], qs[unique]
    start, end, fade = 55.3, 56.0, 0.1
    active = np.flatnonzero((t > start) & (t < end))
    # Work in one continuous Euler branch around the short affected interval.
    ep = qeuler(slerp_series(ts, qs, np.array([start, end])))[:, ::-1]
    ep = np.degrees(np.unwrap(np.radians(ep), axis=0))
    original = qeuler(q[active])[:, ::-1]
    original = ep[0] + (original - ep[0] + 180) % 360 - 180
    assert np.max(abs(original[:, 1])) < 70, 'bridge too close to Euler singularity'
    blend = smootherstep((t[active] - start) / (end - start))
    target = ep[0] + blend[:, None] * (ep[1] - ep[0])
    weight = smootherstep(np.minimum(t[active] - start, end - t[active]) / fade)
    edited = original.copy()
    first_axis = 0 if include_pitch else 1
    edited[:, first_axis:] += weight[:, None] * (target[:, first_axis:] - original[:, first_axis:])
    if include_pitch:
        previous = np.load(ROOT / 'artifacts/experiments/lite_bridge/bridge_diagnostics.npz')
        np.testing.assert_array_equal(t[active], previous['time_s'])
        np.testing.assert_allclose(edited[:, 1:], previous['edited_euler_deg'][:, 1:],
                                   rtol=0, atol=1e-10)
    edits = {}
    for idx, angles in zip(active, edited):
        s = samples[idx]
        cam = quat.from_euler_deg(*map(float, angles[::-1]))
        if s.inverted:
            cam = quat.neg(cam)
        edits[s.frame, s.index_in_frame] = quat.norm(quat.gyroflow_to_dji(cam))
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(baseline, OUT / '00_control.mp4')
    dest = OUT / 'local_wobble_bridge.mp4'
    info = sidecar.build(str(baseline), str(dest), edits)
    np.savez_compressed(OUT / 'bridge_diagnostics.npz', time_s=t[active],
                        original_euler_deg=original, edited_euler_deg=edited, weight=weight)
    report = dict(source=prior['source'], baseline=str(baseline), windows=[[start, end]],
        fade_s=fade, method='quintic endpoint bridge of Gyroflow ' +
            ('pitch/yaw/roll columns' if include_pitch else 'yaw/roll columns'),
        unchanged_euler_column=None if include_pitch else 'Gyroflow pitch', timing_changed=False,
        note='Experimental replacement of an excursion, not measured ground truth. '
             'Outside this interval records equal the existing 8 Hz baseline.',
        peak_euler_change_deg=np.max(abs(edited - original), axis=0).tolist(),
        variants={'00_control': {'path': str(OUT / '00_control.mp4')},
                  'local_wobble_bridge': dict(path=str(dest), bytes=info['bytes'])})
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    # Reuse the EXACT project of the viewed excerpt; change only motion source
    # and diagnostic output destination. No rendering or new synchronization.
    project = json.loads((ROOT / 'artifacts/analysis/gyroflow_output/lite_excerpt.gyroflow').read_text(encoding='utf-8'))
    project['gyro_source']['filepath'] = dest.as_uri()
    project['output']['output_folder'] = OUT.as_uri() + '/'
    project['output']['output_filename'] = 'diagnostic_only.mp4'
    project_path = OUT / 'bridge.gyroflow'
    project_path.write_text(json.dumps(project, indent=2), encoding='utf-8')
    if not export_camera:
        return dest
    camera = OUT / 'camera_bridge.json'
    command = [r'F:\Gyroflow\Gyroflow.exe', str(project_path),
               '--export-metadata', '3:' + str(camera), '-f']
    with (OUT / 'export.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=log, cwd=OUT, timeout=90,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or not camera.exists() or camera.stat().st_size < 100000:
        raise RuntimeError('Gyroflow camera export failed; inspect export.log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-pitch', action='store_true', help='Also bridge Pitch; save as a separate variant')
    parser.add_argument('--skip-export', action='store_true', help='Build telemetry and project only')
    args = parser.parse_args()
    build(include_pitch=args.include_pitch, export_camera=not args.skip_export)

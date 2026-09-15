"""Export Gyroflow camera trajectories and render one small diagnostic video.

These trajectories are calculated from motion data, not measured from the
rendered image. Rendering is explicitly requested for this experiment.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'artifacts' / 'analysis' / 'gyroflow_output'
EXE = Path(r'F:\Gyroflow\Gyroflow.exe')
VIDEO = Path(r'F:\36\54-56-DJI_20260517170526_0020_D.MP4')
MOTION = ROOT / 'artifacts/experiments/lite_v2/fullrate_8hz.mp4'


def run(label, cmd, expected, timeout=120):
    (OUT / (label + '_command.json')).write_text(json.dumps(cmd, indent=2), encoding='utf-8')
    started = time.time()
    with (OUT / (label + '.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(cmd, stdout=log, stderr=log, cwd=OUT,
                                timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
    size = expected.stat().st_size if expected.exists() else 0
    print(label, 'exit', result.returncode, 'bytes', size,
          'seconds', round(time.time() - started, 1), flush=True)
    if result.returncode or size < 1000:
        raise RuntimeError(f'{label} failed; inspect {label}.log')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    options = dict(codec='H.264/AVC', bitrate=0.35, use_gpu=False, audio=False,
                   output_width=320, output_height=240, pixel_format='yuv420p',
                   encoder_options='-preset ultrafast', interpolation='Bilinear',
                   preserve_other_tracks=False,
                   output_path=str(OUT / 'lite_low.mp4'))
    (OUT / 'output_options.json').write_text(json.dumps(options, indent=2), encoding='utf-8')
    base = [str(EXE), str(VIDEO), '-p', json.dumps(options), '-f']
    project = OUT / 'lite_low.gyroflow'
    run('project', base + ['-g', str(MOTION), '--export-project', '1'], project)
    # Native loading may overwrite -p bitrate with source bitrate. Persist the
    # actual small output settings and apply them as a preset after loading.
    data = json.loads(project.read_text(encoding='utf-8'))
    data['output']['bitrate'] = .35
    data['output']['encoder_options'] = '-preset ultrafast -crf 35'
    project.write_text(json.dumps(data, indent=2), encoding='utf-8')
    camera = OUT / 'camera_8hz.json'
    project_cmd = [str(EXE), str(project), '-p', json.dumps(options), '-f']
    run('camera_8hz', project_cmd + ['--export-metadata', '3:' + str(camera)], camera)
    control_data = json.loads(project.read_text(encoding='utf-8'))
    control_data['gyro_source']['filepath'] = (ROOT / 'artifacts/experiments/lite/00_control.mp4').as_uri()
    control_data['output']['output_filename'] = 'lite_control_low.mp4'
    control_project = OUT / 'lite_control.gyroflow'
    control_project.write_text(json.dumps(control_data, indent=2), encoding='utf-8')
    camera = OUT / 'camera_control.json'
    run('camera_control', [str(EXE), str(control_project), '--export-metadata',
                          '3:' + str(camera), '-f'], camera)
    native = OUT / 'camera_native.json'
    run('camera_native', base + ['--export-metadata', '3:' + str(native)], native)
    # The complete camera exports above retain all frames. Decode/render only
    # the diagnostic interval to avoid spending minutes on unrelated footage.
    data['output']['output_filename'] = 'lite_51-60_low.mp4'
    data['trim_ranges_ms'] = [[51000, 60000]]
    for param in data['stabilization']['smoothing_params']:
        if param['name'] == 'trim_range_only':
            param['value'] = 0.
    excerpt = OUT / 'lite_excerpt.gyroflow'
    excerpt.write_text(json.dumps(data, indent=2), encoding='utf-8')
    preset = OUT / 'render_preset.json'
    preset.write_text(json.dumps(dict(version=2, output=data['output']), indent=2), encoding='utf-8')
    cmd = [str(EXE), str(excerpt), '--preset', str(preset), '-f']
    camera = OUT / 'camera_render.json'
    run('camera_render', cmd + ['--export-metadata', '3:' + str(camera)], camera)
    run('render_excerpt', cmd + ['--stdout-progress'], OUT / 'lite_51-60_low.mp4', timeout=300)


if __name__ == '__main__':
    main()

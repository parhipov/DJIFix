import sys, os, subprocess, json, numpy as np
sys.path.insert(0, 'research/ab_bench'); sys.path.insert(0, 'research')
from rolladapt import build
from diagnose import render_measure
from scipy.signal import butter, sosfiltfilt
GF = 'F:/Gyroflow/Gyroflow.exe'
P = 'src/dji_o4.py'
clips = {'house36': 'F:/Май дом с красной крышей 2/DJI_20260521111216_0036_D.MP4', 'winter': 'F:/15 тест/DJI_20251207151759_0069_D.MP4'}
out = 'artifacts/roll_test/good'
os.makedirs(out, exist_ok=True)
for name, video in clips.items():
    side = '%s/%s_RA.mp4' % (out, name)
    print(name, build(video, 'artifacts/cache/%s/image.npz' % name, side), flush=True)
    subprocess.run([sys.executable, P, 'extract', side, '-o', out + '/csv'], capture_output=True)
    patched = '%s/%s_RA_patched.MP4' % (out, name)
    subprocess.run([sys.executable, P, 'patch', video, '%s/csv/%s_RA_quaternions.csv' % (out, name), patched], capture_output=True, check=True)
    params = "{ 'codec': 'H.265/HEVC', 'bitrate': 60, 'use_gpu': true, 'audio': false, 'output_width': 1920, 'output_height': 1080 }"
    with open(out + '/gf.log', 'w') as fh:
        subprocess.run([GF, patched, '-f', '-t', '_stab', '-p', params], stdout=fh, stderr=subprocess.STDOUT)
    os.remove(patched)
    fps, W, H, r = render_measure('%s/%s_RA_patched_stab.mp4' % (out, name))
    np.savez('%s/%s_RA_resid.npz' % (out, name), fps=fps, W=W, H=H, r=r)
    print(name, 'measured', flush=True)

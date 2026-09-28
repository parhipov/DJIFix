import sys, os, subprocess, numpy as np
sys.path.insert(0, 'research')
from diagnose import render_measure
def run(side, src, out_dir, name, size):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([sys.executable, 'src/dji_o4.py', 'extract', side, '-o', out_dir + '/csv'], capture_output=True)
    base = os.path.splitext(os.path.basename(side))[0]
    patched = '%s/%s_patched.MP4' % (out_dir, name)
    subprocess.run([sys.executable, 'src/dji_o4.py', 'patch', src, '%s/csv/%s_quaternions.csv' % (out_dir, base), patched], capture_output=True, check=True)
    params = "{ 'codec': 'H.265/HEVC', 'bitrate': %d, 'use_gpu': true, 'audio': false%s }" % ((60, ", 'output_width': 1920, 'output_height': 1080") if size == '1080' else (150, ''))
    with open(out_dir + '/gf.log', 'w') as fh:
        subprocess.run(['F:/Gyroflow/Gyroflow.exe', patched, '-f', '-t', '_stab', '-p', params], stdout=fh, stderr=subprocess.STDOUT)
    os.remove(patched)
    stab = '%s/%s_patched_stab.mp4' % (out_dir, name)
    fps, W, H, r = render_measure(stab, scale=0.5 if size == '1080' else 0.25)
    np.savez('%s/%s_resid.npz' % (out_dir, name), fps=fps, W=W, H=H, r=r)
    print(name, 'done', flush=True)
run('artifacts/roll_test/side_RA5.mp4', 'examples/o4pro_13-17s/o4pro_13-17s.MP4', 'artifacts/roll_test/ex_v5', 'RA5', '4k')
run('artifacts/roll_test/all/pro_RA_v5.mp4', 'F:/36/DJI_20260905181949_0005_D.MP4', 'artifacts/roll_test/pro_full', 'pro_RA_v5', '1080')

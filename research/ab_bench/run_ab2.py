"""Full-clip A/B: build variants, patch copies, render in Gyroflow at 1080p, measure residual motion."""
import os, sys, subprocess, shutil, numpy as np, time
sys.path.insert(0, 'K:/Work/Python/DJI fix/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_pipeline as fp
from residual import measure
S = os.path.dirname(os.path.abspath(__file__)) + '/../full'
GF = 'F:/Gyroflow/Gyroflow.exe'
P = 'K:/Work/Python/DJI fix/src/dji_o4.py'
CLIPS = {'pro': ('F:/36/DJI_20260905181949_0005_D.MP4', S + '/pro_image.npz'),
         'lite': ('F:/36/54-56-DJI_20260517170526_0020_D.MP4', S + '/lite_image.npz'),
         'good': ('F:/Май дом с красной крышей 2/DJI_20260521111216_0036_D.MP4', S + '/good_image.npz'),
         'winter': ('F:/15 тест/DJI_20251207151759_0069_D.MP4', S + '/winter_image.npz')}
orig_hf, orig_sk, orig_we = fp.hf_only, fp.spectral_keep, fp.weight_events
def no_wiener():
    fp.hf_only = lambda keep, floor_hz=4.0: np.ones(len(fp.BAND_HZ)); fp.spectral_keep = lambda *a, **k: None
def wiener():
    fp.hf_only, fp.spectral_keep = orig_hf, orig_sk
def spikes_only():
    fp.weight_events = lambda ev, n, confirmed=1.0, unclear=0.5, opposed=0.0: orig_we(ev, n, confirmed=0.0, unclear=0.0, opposed=0.0)
def all_events():
    fp.weight_events = orig_we
VARIANTS_OLD = [('fixed', {}, wiener, all_events),
            ('R', dict(timing='none'), no_wiener, spikes_only),
            ('RW', dict(timing='none'), wiener, spikes_only),
            ('RE', dict(timing='none'), no_wiener, all_events)]
def render(src, tag):
    out = os.path.splitext(src)[0] + '_' + tag + '.mp4'
    for _ in range(3):
        with open(S + '/gf.log', 'w') as fh:
            subprocess.run([GF, src, '-f', '-t', '_' + tag, '-p',
                            "{ 'codec': 'H.265/HEVC', 'bitrate': 60, 'use_gpu': true, 'audio': false, 'output_width': 1920, 'output_height': 1080 }"],
                           stdout=fh, stderr=subprocess.STDOUT)
        if os.path.isfile(out) and os.path.getsize(out) > 20e6:
            return out
    raise SystemExit('render failed ' + src)
import fusion
def b_control(video, npz, side):
    tmp = side.replace('.mp4', '_unused.mp4'); fp.build(video, npz, tmp, report=False, control_path=side, events=False); os.remove(tmp)
def b_RT(video, npz, side):
    no_wiener(); spikes_only(); fp.build(video, npz, side, report=False)
def b_F(fc):
    def f(video, npz, side):
        print(fusion.build(video, npz, side, fc=fc), flush=True)
    return f
VARIANTS = [('control', b_control), ('RT', b_RT), ('F4', b_F(4.0)), ('F2', b_F(2.0))]
only = sys.argv[1:] or list(CLIPS)
for clip in only:
    video, npz = CLIPS[clip]
    todo = [(v[0], v) for v in VARIANTS]
    for name, v in todo:
        res = '%s/%s_%s_resid.npz' % (S, clip, name)
        if os.path.isfile(res):
            continue
        t0 = time.time()
        if v is None:
            src = video
            dst = '%s/%s_orig_stab.mp4' % (S, clip)
            if not os.path.isfile(dst):
                shutil.move(render(src, 'ab_orig_%s' % clip), dst)
            stab = dst
        else:
            side = '%s/side_%s_%s.mp4' % (S, clip, name)
            v[1](video, npz, side)
            subprocess.run([sys.executable, P, 'extract', side, '-o', S + '/csv'], capture_output=True)
            csvp = '%s/csv/side_%s_%s_quaternions.csv' % (S, clip, name)
            src = '%s/%s_%s.MP4' % (S, clip, name)
            stab = os.path.splitext(src)[0] + '_stab.mp4'
            if not os.path.isfile(stab):
                subprocess.run([sys.executable, P, 'patch', video, csvp, src], capture_output=True, check=True)
                stab = render(src, 'stab')
                os.remove(src)
        fps, W, H, r = measure(stab, scale=0.5)
        np.savez(res, fps=fps, W=W, H=H, r=r)
        print('%s %s done in %.0f s' % (clip, name, time.time() - t0), flush=True)

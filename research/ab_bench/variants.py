import sys, os, numpy as np
sys.path.insert(0,'K:/Work/Python/DJI fix/src')
import fix_pipeline as fp
S=sys.argv[1]
clips={'o4pro_13-17s':'ex_pro','o4lite_53-58s':'ex_lite'}
orig_hf, orig_sk = fp.hf_only, fp.spectral_keep
for clip,d in clips.items():
    video='K:/Work/Python/DJI fix/examples/%s/%s.MP4'%(clip,clip)
    npz='%s/%s/%s_image.npz'%(S,d,clip)
    for name,kw,nowiener in (('B_notiming',dict(timing='none'),False),
                             ('C_notiming_noev',dict(timing='none',events=False),False),
                             ('D_roll_spikes',dict(timing='none',events=False),True)):
        fp.hf_only = (lambda keep, floor_hz=4.0: np.ones(len(fp.BAND_HZ))) if nowiener else orig_hf
        fp.spectral_keep = (lambda *a, **k: None) if nowiener else orig_sk
        out='%s/render/side_%s_%s.mp4'%(S,clip,name)
        fp.build(video, npz, out, report=False, **kw)
        print('built',out)

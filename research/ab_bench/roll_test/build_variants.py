import sys, os
sys.path.insert(0, 'src')
import fix_pipeline as fp
V = 'examples/o4pro_13-17s/o4pro_13-17s.MP4'
IMG = sys.argv[1]
OUT = 'artifacts/roll_test'
variants = {
    'Rinf': dict(r0_roll=1e9, roll_lo_hz=4.0),
    'R1':   dict(r0_roll=1e9, roll_lo_hz=1.0),
    'Rfull': dict(r0_roll=1e9, roll_lo_hz=0.3),
}
for name, kw in variants.items():
    rep = fp.build(V, IMG, '%s/side_%s.mp4' % (OUT, name), report=False, events=False, diagnostics_path='%s/diag_%s.npz' % (OUT, name), **kw)
    print(name, 'roll corr rms %.4f peak %.3f' % (rep['roll']['correction_rms_deg'], rep['roll']['correction_peak_deg']))
rep = fp.build(V, IMG, '%s/side_fixed_check.mp4' % OUT, report=False, diagnostics_path='%s/diag_fixed.npz' % OUT)
print('fixed(default) roll corr rms %.4f peak %.3f' % (rep['roll']['correction_rms_deg'], rep['roll']['correction_peak_deg']))

#!/usr/bin/env python3
"""
Generate a set of telemetry variants to load in Gyroflow under "Motion data".

Each variant is a ~5 MB MP4 holding only the DJI telemetry track, parsed by
Gyroflow through its native DJI path. Load one, look at the moment that shakes,
note it down, load the next -- the video and every stabilization setting stay
untouched, so the only thing that changes between runs is the telemetry.

    python research/variants.py "F:\\36\\DJI_20260905181949_0005_D.MP4" -o artifacts/main/variants

The sync sweep is the interesting part: a constant timing error between the
telemetry and the frames shows up as shake proportional to how fast the drone is
rotating. If one of the shifted files is visibly better than `00_baseline`, the
problem is timing and we then know the sign and roughly the size.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import argparse, os

import sidecar

VARIANTS = [
    # name,              time_shift_ms, offset_mode
    ('00_baseline',                0.0, 'keep'),
    ('01_shift_minus20',         -20.0, 'keep'),
    ('02_shift_minus15',         -15.0, 'keep'),
    ('03_shift_minus10',         -10.0, 'keep'),
    ('04_shift_minus5',           -5.0, 'keep'),
    ('05_shift_plus5',             5.0, 'keep'),
    ('06_shift_plus10',           10.0, 'keep'),
    ('07_offset_const',            0.0, 'const'),
    ('08_offset_zero',             0.0, 'zero'),
]

NOTES = {
    '00_baseline': 'unmodified — must look exactly like using no file at all',
    '01_shift_minus20': 'telemetry 20 ms earlier (one whole frame)',
    '02_shift_minus15': 'telemetry 15 ms earlier',
    '03_shift_minus10': 'telemetry 10 ms earlier (half a frame)',
    '04_shift_minus5': 'telemetry 5 ms earlier',
    '05_shift_plus5': 'telemetry 5 ms later',
    '06_shift_plus10': 'telemetry 10 ms later (half a frame)',
    '07_offset_const': "DJI's per-frame sub-frame offset replaced by its median",
    '08_offset_zero': "DJI's sub-frame offset ignored entirely",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('-o', '--outdir', default='artifacts/main/variants')
    ap.add_argument('--only', help='comma separated variant names to build')
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    wanted = set(a.only.split(',')) if a.only else None

    lines = []
    for name, shift, mode in VARIANTS:
        if wanted and name not in wanted:
            continue
        out = os.path.join(a.outdir, name + '.mp4')
        info = sidecar.build(a.video, out, time_shift_ms=shift, offset_mode=mode)
        lines.append('%-20s %+7.1f ms  offset=%-6s  %s' % (name, shift, mode, NOTES[name]))
        print('%s  (%.1f MB)' % (out, info['bytes'] / 1e6))

    manifest = os.path.join(a.outdir, 'README.txt')
    with open(manifest, 'w', encoding='utf-8') as fh:
        fh.write('Load each file in Gyroflow: Motion data -> open file.\n')
        fh.write('Keep every other setting as it is; only the telemetry changes.\n\n')
        fh.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print('\nmanifest: ' + manifest)


if __name__ == '__main__':
    main()

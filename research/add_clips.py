#!/usr/bin/env python3
"""
Add every usable DJI clip found in some folders to the study registry (artifacts/clips.json).

    python research/add_clips.py F:\\36 F:\\37 "F:\\some folder" [--dry-run]

Scans the folders recursively for .MP4 files, skips Gyroflow/derived outputs
(*_stabilized, *_stab, our sidecars) and files without DJI motion data, reads
camera, firmware and serial from the file header, and appends one entry per new
file (paths already in the registry are left alone). Names are
<folder>_<file number>, made unique. `eye` is "unknown"; `lens_profile` is null
(clip metadata) -- an O4 Lite wearing the Flywoo lens needs it set by hand, the
script lists the O4 Lite entries it added so they can be checked.

At the end it prints the command that builds the new clips into the base in the
background (research/diagnose.py --only ...); existing clips are kept.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from diagnose import CAMERA, SKIP_NAME   # noqa: E402


def slug(s):
    return re.sub(r'[^0-9a-z]+', '', s.lower()) or 'clip'


def header(path):
    """(ok, info or reason) from the telemetry header only."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if any(stem.endswith(s) or (s + '_') in stem for s in SKIP_NAME) or '_trim_' in stem:
        return False, 'derived file'
    try:
        from dji_o4 import read_telemetry
        clip, samples, frames = read_telemetry(path)
    except SystemExit as exc:
        return False, str(exc)
    except Exception as exc:                  # no djmd track, not a DJI file, ...
        return False, '%s: %s' % (type(exc).__name__, exc)
    if not frames or len(samples) < 100:
        return False, 'no usable telemetry'
    h = clip.header or {}
    return True, dict(camera=CAMERA.get(h.get('product_name'), h.get('product_name')),
                      firmware=h.get('product_firmware_version'), serial=h.get('product_sn'),
                      seconds=round(len(frames) / float(clip.fps), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+')
    ap.add_argument('--registry', default=os.path.join(HERE, '..', 'artifacts', 'clips.json'))
    ap.add_argument('--min-seconds', type=float, default=20.0, help='skip clips shorter than this (default 20 s)')
    ap.add_argument('--dry-run', action='store_true', help='list what would be added, write nothing')
    a = ap.parse_args()
    reg = json.load(open(a.registry, encoding='utf-8'))
    known = {os.path.normcase(os.path.abspath(c['path'])) for c in reg['clips']}
    names = {c['name'] for c in reg['clips']}
    added = []
    for folder in a.folders:
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if not f.lower().endswith('.mp4'):
                    continue
                path = os.path.join(root, f)
                if os.path.normcase(os.path.abspath(path)) in known:
                    continue
                ok, info = header(path)
                if not ok:
                    print('  skip %s: %s' % (path, info))
                    continue
                if info['seconds'] < a.min_seconds:
                    print('  skip %s: %.1f s, shorter than %g s' % (path, info['seconds'], a.min_seconds))
                    continue
                num = re.findall(r'(\d{3,4})(?:_D)?$', os.path.splitext(f)[0])
                base = '%s_%s' % (slug(os.path.basename(os.path.normpath(root)))[:12], num[-1] if num else slug(f)[:8])
                name, k = base, 2
                while name in names:
                    name, k = '%s_%d' % (base, k), k + 1
                names.add(name)
                entry = dict(name=name, path=path.replace('\\', '/'), camera=info['camera'], drone=None, lens='unknown',
                             lens_profile=None, firmware=info['firmware'], serial=info['serial'], eye='unknown',
                             note='added by add_clips.py from %s (%.0f s)' % (folder, info['seconds']))
                added.append(entry)
                print('  add  %-18s %-12s %s  %s' % (name, info['camera'], info['firmware'], path))
    if not added:
        print('nothing new')
        return
    lite = [e['name'] for e in added if e['camera'] == 'O4 Lite']
    if lite:
        print('\nO4 Lite clips (set "lens_profile" in the registry if they wear the Flywoo lens): %s' % ', '.join(lite))
    if a.dry_run:
        print('\n(dry run: registry not written)')
        return
    reg['clips'].extend(added)
    with open(a.registry, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(reg, ensure_ascii=False, indent=1))
    print('\n%d clips added to %s. Build them into the base (one image pass + one Gyroflow render per clip):' % (len(added), a.registry))
    print('  python research/diagnose.py --out artifacts/segments --cache artifacts/cache --gyroflow PATH\\Gyroflow.exe --only %s'
          % ' '.join(e['name'] for e in added))
    print('then: python research/check_segments.py artifacts/segments && python research/summarize_base.py artifacts/segments')


if __name__ == '__main__':
    main()

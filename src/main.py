#!/usr/bin/env python3
"""
One entry point: DJI O4 video in, fixed telemetry file out.

    python src/main.py "F:\\36\\DJI_20260905181503_0003_D.MP4"

Writes <name>_telemetry_fixed.mp4 next to the video -- a ~5 MB file holding only
the telemetry track, which you load in Gyroflow under "Motion data". The video
and every stabilization setting stay untouched. Intermediates (the image
measurement cache, a <name>_00_control.mp4 with the same timing but untouched
content, diagnostics) are deleted afterwards unless --keep (or --artifacts / -o);
the small <name>_fix_report.json (what was changed, every event with its verdict)
stays.

What is fixed (see fix_pipeline.py for the details and the numbers):

  1. Timing.  Measured against the image: on both cameras Gyroflow's own frame
     instant (pts) is right to ~1 ms, so the constant is ~0; only the per-frame
     exposure variation is applied (timing.py). The old dbgi alignment
     (+10.5 ms) is available as --timing dbgi for A/B.
  2. Roll.    DJI's fused roll jitters by ~0.12 deg per frame regardless of the
     real motion; the roll measured from the image replaces it above 4 Hz.
  3. Pitch/yaw noise. Wiener keep-gain curve, gentle (denoise.py).
  4. Pitch/yaw events. Where the 5-frame image rotation disagrees with the
     telemetry for a sustained stretch and both image estimates agree with each
     other, the telemetry is replaced by the image inside that window.

Steps 2 and 4 need one decoding pass over the video (measure_rotation.py,
~0.35 s per 4K frame, cached as <name>_image.npz next to the output and resumed
if interrupted). `--no-image` skips it and falls back to timing + Wiener only.

Options worth knowing:
    --plots         also write four PNG plots of telemetry vs image vs fixed
    --keep          keep the intermediates next to the video (control file for A/B, image cache)
    --artifacts     write into artifacts/main/<name>/ and keep everything (development)
    --lens flywoo   Gyroflow lens profile for the image pass when the camera has another lens:
                    a JSON path or part of a file name in lens_profiles/ (see lens_profiles/README.md)
    --fit-focal     diagnostic focal-scale fit against the telemetry (see lenscal.py)
    --no-image      no video decoding: timing + three-axis Wiener de-noise only
    --gain 0.7      gentler correction, if the result looks over-smoothed
    --no-events     skip the pitch/yaw event repair
    --timing dbgi   the old dbgi-based alignment, for A/B against the new timing
    --extract       also dump the telemetry as CSV + npz for analysis
    --no-verify     skip the image-based judgement and the Gyroflow cross-check
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Gyroflow is only used for the final cross-check (its CLI re-reads the sidecar).
# Pass --gyroflow PATH, or set the GYROFLOW environment variable; without either
# the default install locations are tried and the check is skipped if none exists.
GYROFLOW_CANDIDATES = [
    os.path.expandvars(r'%ProgramFiles%\Gyroflow\Gyroflow.exe'),
    os.path.expandvars(r'%LOCALAPPDATA%\Programs\Gyroflow\Gyroflow.exe'),
]


def find_gyroflow():
    env = os.environ.get('GYROFLOW')
    for p in ([env] if env else []) + GYROFLOW_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return None


def gyroflow_export(exe, video, sidecar, kind, out_json):
    """Run Gyroflow's CLI export. Returns True if it produced real content.

    Never run this in a tight loop: some runs exit after under a second having
    loaded nothing and leave a few-hundred-byte stub, which is why the size is
    checked here. Every path must be absolute (Gyroflow runs with its own cwd).
    With `-g` the CLI never parses the video's own telemetry, so the project has
    frame_readout_time = 0 and the type-3 export timestamps are exactly pts.
    """
    video = os.path.abspath(video)
    out_json = os.path.abspath(out_json)
    cmd = [exe, video]
    if sidecar:
        cmd += ['-g', os.path.abspath(sidecar)]
    cmd += ['--export-metadata', '%d:%s' % (kind, out_json), '-f']
    with tempfile.TemporaryFile() as log:
        subprocess.run(cmd, stdout=log, stderr=log, cwd=os.path.dirname(exe))
    stray = os.path.splitext(video)[0] + '.gyroflow'
    if os.path.isfile(stray):
        try:
            os.remove(stray)
        except OSError:
            pass
    return os.path.isfile(out_json) and os.path.getsize(out_json) > 10000


def gyroflow_check(video, sidecar_path, exe, outdir):
    """Does Gyroflow read from the sidecar exactly what we wrote, at pts?"""
    import numpy as np
    from align import slerp_series
    from timing import load_sorted

    j = os.path.join(outdir, '_verify_fixed.json')
    print('  Gyroflow cross-check ...')
    if not gyroflow_export(exe, video, sidecar_path, 3, j):
        print('  ! Gyroflow produced no export for the sidecar; check the file')
        return
    rows = json.load(open(j, encoding='utf-8'))
    oq = np.array([r['org_quat'] for r in rows])          # w, x, y, z in this export
    tms = np.array([r['timestamp_ms'] for r in rows])
    clip, samples, frames, ts, qs = load_sorted(sidecar_path)
    ours = slerp_series(ts, qs, tms * 1000.0)
    ang = 2 * np.degrees(np.arccos(np.abs(np.sum(oq * ours, axis=1)).clip(0, 1)))
    print('    Gyroflow reads %d frames at %.3f, %.3f, ... ms; attitude vs our file p50 %.2e deg, p99 %.2e deg'
          % (len(rows), tms[0], tms[1], np.percentile(ang, 50), np.percentile(ang, 99)))
    try:
        os.remove(j)
    except OSError:
        pass


def resolve_lens(arg):
    """A path, or any part of a profile file name under lens_profiles/ (and external/lens_profiles/)."""
    if os.path.isfile(arg):
        return arg
    hits = []
    # the project's own folder first; the full external database only if nothing matches there
    for folder in (os.path.join(PROJECT_ROOT, 'lens_profiles'), os.path.join(PROJECT_ROOT, 'external', 'lens_profiles')):
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith('.json') and arg.lower() in f.lower():
                    hits.append(os.path.join(root, f))
        if hits:
            break
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit('lens profile not found: %r (a file path, or part of a name in lens_profiles/)' % arg)
    raise SystemExit('lens profile %r is ambiguous: ' % arg + ', '.join(os.path.basename(h) for h in hits[:20]))


def write_control(video, control_path, timing_mode):
    """Sidecar with the chosen timing and no content change."""
    import numpy as np
    import quat as Q
    import sidecar as sc
    from timing import load_sorted, retime, shifts
    clip, samples, frames, delta_us = shifts(video, timing_mode)
    _, _, _, ts, qs = load_sorted((clip, samples, frames))
    base = retime(samples, ts, qs, delta_us)
    inv = np.array([s.inverted for s in samples])
    base = np.where(inv[:, None], -base, base)
    return sc.build(video, control_path, {(s.frame, s.index_in_frame): Q.norm(Q.gyroflow_to_dji(tuple(map(float, q))))
                                          for s, q in zip(samples, base)})


def process(video, args):
    import dji_o4
    import denoise
    import fix_pipeline
    import measure_rotation
    import sidecar as sc

    video = os.path.abspath(video)
    if not os.path.isfile(video):
        print('!! not found: %s' % video)
        return False
    base = os.path.splitext(os.path.basename(video))[0]
    if args.outdir:
        outdir = args.outdir
    elif args.artifacts:
        outdir = os.path.join('artifacts', 'main', base)
    else:
        outdir = os.path.dirname(video)
    keep = args.keep or args.artifacts or bool(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print('=' * 72)
    print(base)
    try:
        clip, samples, frames = dji_o4.read_telemetry(video)
    except SystemExit as e:
        print('!! %s' % e)
        print('   This file has no DJI telemetry track. A re-encoded or remuxed')
        print('   copy loses it -- use the original straight off the drone.')
        return False
    hdr = clip.header
    print('  %s  %s fw %s  sn %s' % (hdr.get('product_name', '?'), hdr.get('proto_file_name', '?'),
                                     hdr.get('product_firmware_version', '?'), hdr.get('product_sn', '?')))
    print('  %d frames, %.2f s, %dx%d @ %.4f fps, readout %.3f ms, %d quaternions'
          % (len(frames), samples[-1].t_us / 1e6, clip.video.get('resolution_width', 0),
             clip.video.get('resolution_height', 0), clip.sensor_fps, clip.frame_readout_time_ms, len(samples)))

    if args.extract:
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'src', 'dji_o4.py'), 'extract', video, '-o', outdir])

    out = os.path.join(outdir, base + '_telemetry_fixed.mp4')
    control = os.path.join(outdir, base + '_00_control.mp4')
    image_npz = os.path.join(outdir, base + '_image.npz')
    report_path = os.path.join(outdir, base + '_fix_report.json')

    if args.no_image:
        mode = 'exposure_var' if args.timing == 'image' else args.timing   # no image -> constant 0, per-frame exposure part only
        print('  building %s (timing + Wiener only, no image pass)' % out)
        denoise.build3(video, out, r0=args.r0, gain=args.gain, report=False, timing=mode)
        write_control(video, control, mode)
        rep = None
    else:
        lens = None
        if args.lens:
            from lenscal import load_gyroflow_lens
            lens_path = resolve_lens(args.lens)
            lens = load_gyroflow_lens(lens_path, int(clip.video.get('resolution_width') or 3840),
                                      int(clip.video.get('resolution_height') or 2880))
            print('  lens profile: %s  (f %.1f px, D %s)' % (lens['name'], lens['f'], [round(x, 4) for x in lens['D']]))
        if args.fit_focal:
            from lenscal import fit_focal_scale
            base = lens or dict(f=float(clip.focal_length), cx=(clip.video.get('resolution_width') or 3840) / 2.0,
                                cy=(clip.video.get('resolution_height') or 2880) / 2.0, D=list(clip.distortion_coeffs))
            print('  fitting the focal scale on fast frames (diagnostic; applied only if the gains reach 1) ...')
            rep = fit_focal_scale(video, clip, samples, frames, float(clip.fps), base)
            print('  focal fit: scale %.3f, trusted %s, gains x/y at 1: %s, at best: %s%s' % (
                rep['scale'], rep['trusted'], rep.get('gains_at_1'), rep.get('gains_at_best'),
                ('  (%s)' % rep['reason']) if rep.get('reason') else ''))
            if rep['trusted']:
                lens = dict(base, f=base['f'] * rep['scale'], name='%s x %.3f (fitted)' % (base.get('name', 'clip lens'), rep['scale']))
        # a cached measurement made through another lens model must not be reused
        if os.path.isfile(image_npz) and not args.remeasure:
            try:
                import numpy as np
                cached = json.loads(str(np.load(image_npz)['meta_json'])).get('lens_source', 'clip metadata')
                wanted = lens['name'] if lens else 'clip metadata'
                if cached != wanted:
                    print('  image cache was measured with lens %r, now %r -> measuring again' % (cached, wanted))
                    args.remeasure = True
            except Exception as e:  # noqa: BLE001
                print('  image cache unreadable (%s) -> measuring again' % e)
                args.remeasure = True
        if not os.path.isfile(image_npz) or args.remeasure:
            print('  measuring the image (one decoding pass, ~0.35 s per frame) -> %s' % image_npz)
            measure_rotation.measure(video, image_npz, scale=args.scale, lens=lens)
        else:
            print('  image measurement found: %s' % image_npz)
        print('  building %s' % out)
        rep = fix_pipeline.build(video, image_npz, out, timing=args.timing, gain=args.gain,
                                 r0_pt=args.r0, events=not args.no_events, control_path=control,
                                 diagnostics_path=os.path.join(outdir, base + '_fix_diagnostics.npz'),
                                 report=True)
        with open(report_path, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
    print('  -> %.1f MB' % (os.path.getsize(out) / 1e6))

    if args.verify:
        if not args.no_image:
            print('  judging against the image (verify_fix.py) ...')
            subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'src', 'verify_fix.py'), video,
                            '--image', image_npz, control, out,
                            '--json', os.path.join(outdir, base + '_verify.json')])
        exe = args.gyroflow or find_gyroflow()
        if exe:
            gyroflow_check(video, out, exe, outdir)
        else:
            print('  (Gyroflow.exe not found, skipping the cross-check; pass --gyroflow PATH '
                  'or set the GYROFLOW environment variable)')

    diag_path = os.path.join(outdir, base + '_fix_diagnostics.npz')
    if args.plots and not args.no_image and os.path.isfile(diag_path):
        import plot_report
        for p in plot_report.make_plots(diag_path, image_npz, report_path, outdir, base):
            print('  plot: %s' % p)
    if not keep:
        removed = []
        for path in (image_npz, control, diag_path,
                     os.path.join(outdir, base + '_verify.json'), os.path.splitext(video)[0] + '.gyroflow'):
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed.append(os.path.basename(path))
                except OSError:
                    pass
        if removed:
            print('  removed intermediates: %s  (--keep retains them; the image cache saves the ~25 min next time)'
                  % ', '.join(removed))
            print('  kept: %s (what was fixed, and every event with its verdict)' % os.path.basename(report_path))
    print('\n  Load in Gyroflow: Motion data -> open file ->')
    print('  %s' % out)
    if keep:
        print('  (control with the same timing but no content change: %s)' % control)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('videos', nargs='+', help='DJI .MP4 files straight off the drone')
    ap.add_argument('-o', '--outdir', help='output directory (default: next to the source video)')
    ap.add_argument('--artifacts', action='store_true', help='write into artifacts/main/<name>/ and keep everything')
    ap.add_argument('--keep', action='store_true', help='keep intermediate files (image cache, control, diagnostics)')
    ap.add_argument('--plots', action='store_true', help='write PNG plots (overview, roll, pitch/yaw, correction) next to the result')
    ap.add_argument('--no-image', action='store_true', help='skip the video pass: timing + Wiener only')
    ap.add_argument('--remeasure', action='store_true', help='redo the image pass even if cached')
    ap.add_argument('--lens', help='Gyroflow lens profile JSON (the profile you use in Gyroflow) for the image pass')
    ap.add_argument('--fit-focal', action='store_true',
                    help='diagnostic: fit a focal scale against the telemetry on fast frames; applied only if the '
                         'pitch/yaw gains reach 1 (they do not when parallax, not the lens, is the cause)')
    ap.add_argument('--scale', type=float, default=0.25, help='image pass working scale (default 0.25)')
    ap.add_argument('--no-events', action='store_true', help='skip the pitch/yaw event repair')
    ap.add_argument('--gain', type=float, default=1.0, help='correction strength (default 1.0)')
    ap.add_argument('--r0', type=float, default=80.0, help='rate in deg/s at which the Wiener part is half as strong')
    ap.add_argument('--timing', default='image', choices=['image', 'exposure', 'none', 'dbgi'],
                    help='frame timing: image = measured on this clip + per-frame exposure (default), '
                         'exposure = readout/2 - exposure/2 without the image, dbgi = the old alignment')
    ap.add_argument('--extract', action='store_true', help='also dump CSV + npz')
    ap.add_argument('--no-verify', dest='verify', action='store_false', help='skip the checks')
    ap.add_argument('--gyroflow', help='path to Gyroflow.exe for the final cross-check '
                '(default: GYROFLOW env var, then the standard install locations)')
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    files = []
    for v in args.videos:
        files.extend(glob.glob(v) or [v])
    ok = sum(bool(process(v, args)) for v in files)
    print('\n%d of %d file(s) done' % (ok, len(files)))
    return 0 if ok == len(files) else 1


if __name__ == '__main__':
    sys.exit(main())

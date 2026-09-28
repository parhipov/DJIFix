#!/usr/bin/env python3
"""
Run fix_pipeline.build() on every clip of the base and keep what it changed, for a before/after
comparison of a code change (research tool, writes nothing next to the videos).

    python research/ab_bench/bench_build.py OUT.json [--src DIR] [--cache artifacts/cache] [--work DIR] [--only NAMES]

--src points at a copy of src/ to benchmark an older version of the code. The result holds per
clip: roll gate and correction, removed spikes, events (window, weight, applied peak), totals.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--src', default=os.path.join(ROOT, 'src'))
    ap.add_argument('--registry', default=os.path.join(ROOT, 'artifacts', 'clips.json'))
    ap.add_argument('--cache', default=os.path.join(ROOT, 'artifacts', 'cache'))
    ap.add_argument('--work', default=os.path.join(ROOT, 'artifacts', 'roll_gate', 'bench'))
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.src))
    import fix_pipeline as fp
    os.makedirs(a.work, exist_ok=True)
    reg = json.load(open(a.registry, encoding='utf-8'))
    res = json.load(open(a.out, encoding='utf-8')) if os.path.isfile(a.out) else {}
    for c in reg['clips']:
        img = os.path.join(a.cache, c['name'], 'image.npz')
        if (a.only and c['name'] not in a.only) or not os.path.isfile(img) or c['name'] in res:
            continue
        print('[%s]' % c['name'], flush=True)
        try:
            rep = fp.build(c['path'], img, os.path.join(a.work, '_out.mp4'), report=False,
                           diagnostics_path=os.path.join(a.work, '%s_%s_diag.npz' % (c['name'], os.path.basename(a.out)[:-5])))
        except SystemExit as exc:            # e.g. a >4 GB joined file (co64) the sidecar writer refuses
            print('  skipped: %s' % exc, flush=True)
            continue
        ev = rep.get('events', {})
        res[c['name']] = dict(
            roll={k: rep['roll'].get(k) for k in ('defect_active_frac', 'defect_touched_frac', 'correction_rms_deg', 'correction_peak_deg')},
            spikes=rep.get('spikes', {}).get('times_s', []),
            events=[dict(t=[e['start_s'], e['end_s']], w=e.get('weight'), peak=e.get('applied_peak_deg'),
                         rate=e.get('mean_rate_deg_s'), spike=e.get('telemetry_spike')) for e in ev.get('list', [])],
            total={k: rep['total'].get(k) for k in ('correction_rms_deg', 'correction_peak_deg')})
        json.dump(res, open(a.out, 'w', encoding='utf-8'), indent=1)
    print('written %s (%d clips)' % (a.out, len(res)))


if __name__ == '__main__':
    main()

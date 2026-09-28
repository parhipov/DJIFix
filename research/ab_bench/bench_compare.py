#!/usr/bin/env python3
"""
Compare two bench_build.py results clip by clip: roll, spikes and events that appeared or vanished.

    python research/ab_bench/bench_compare.py BEFORE.json AFTER.json
"""
import json
import sys


def ev_key(e):
    return (round(e['t'][0], 2), round(e['t'][1], 2))


def main():
    a, b = (json.load(open(p, encoding='utf-8')) for p in sys.argv[1:3])
    same = []
    for name in a:
        if name not in b:
            continue
        x, y = a[name], b[name]
        out = []
        r0, r1 = x['roll'], y['roll']
        if abs((r0['correction_rms_deg'] or 0) - (r1['correction_rms_deg'] or 0)) > 1e-6 or abs((r0['correction_peak_deg'] or 0) - (r1['correction_peak_deg'] or 0)) > 1e-6:
            out.append('roll: on %.1f%% rms %.3f peak %.2f -> on %.1f%% rms %.3f peak %.2f' % (
                100 * (r0['defect_active_frac'] or 0), r0['correction_rms_deg'] or 0, r0['correction_peak_deg'] or 0,
                100 * (r1['defect_active_frac'] or 0), r1['correction_rms_deg'] or 0, r1['correction_peak_deg'] or 0))
        gone = sorted(set(x['spikes']) - set(y['spikes']))
        new = sorted(set(y['spikes']) - set(x['spikes']))
        if gone or new:
            out.append('spikes %d -> %d%s%s' % (len(x['spikes']), len(y['spikes']),
                                                ('; no longer removed: %s' % gone) if gone else '', ('; new: %s' % new) if new else ''))
        applied = lambda L: {ev_key(e): e for e in L if (e['w'] or 0) > 0 and (e['peak'] or 0) > 0.01}   # noqa: E731
        ea, eb = applied(x['events']), applied(y['events'])
        g = [k for k in ea if k not in eb]
        n = [k for k in eb if k not in ea]
        if g or n:
            out.append('applied events %d -> %d; gone: %s%s' % (
                len(ea), len(eb), ', '.join('%.2f-%.2f (%.0f deg/s, %.2f deg)' % (k[0], k[1], ea[k]['rate'], ea[k]['peak']) for k in g) or '-',
                ('; new: ' + ', '.join('%.2f-%.2f' % k for k in n)) if n else ''))
        out.append('total rms %.3f peak %.2f -> rms %.3f peak %.2f' % (x['total']['correction_rms_deg'], x['total']['correction_peak_deg'],
                                                                      y['total']['correction_rms_deg'], y['total']['correction_peak_deg']))
        if len(out) == 1 and abs(x['total']['correction_rms_deg'] - y['total']['correction_rms_deg']) < 1e-9:
            same.append(name)
            continue
        print('%s\n  %s' % (name, '\n  '.join(out)))
    print('\nunchanged: %s' % (', '.join(same) or '-'))


if __name__ == '__main__':
    main()

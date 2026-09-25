#!/usr/bin/env python3
"""
Verify a segment base built by research/diagnose.py; prints every violation and a summary.

    python research/check_segments.py artifacts/segments

Checked for every segment: the source cut exists, carries its telemetry and has
the expected number of frames; it lies inside the clip's usable window (first
and last EDGE_S seconds excluded, keyframe-aligned); the preview is H.264,
browser-playable, has the same number of frames; the plot exists. For every
event: its id is unique, its source time is inside the window, its time in the
segment is inside the segment. The registry-vs-header check from the build is
repeated in the summary.
"""
import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from diagnose import EDGE_S, usable   # noqa: E402
from timing import load_sorted       # noqa: E402


def preview_info(path):
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        n = s.frames or sum(1 for _ in c.decode(video=0))
        return s.codec_context.name, s.codec_context.pix_fmt, s.codec_context.width, n, float(s.average_rate or 50)


def main(base):
    meta = json.load(open(os.path.join(base, 'meta.json'), encoding='utf-8'))
    rows = list(csv.DictReader(open(os.path.join(base, 'index.csv'), encoding='utf-8-sig')))
    bad = []
    tol = 1e-3
    ids = collections.Counter(r['event_id'] for r in rows)
    bad += ['duplicate event id %s (x%d)' % (k, v) for k, v in ids.items() if v > 1]
    segs = {}
    for r in rows:
        segs.setdefault(r['segment'], r)
    print('checking %d segments, %d events...' % (len(segs), len(rows)), flush=True)
    for sid, r in segs.items():
        m = meta['clips'][r['clip']]
        lo, hi = m['window_s']
        dur = m['duration_s']
        a, b = float(r['seg_start_s']), float(r['seg_end_s'])
        if a < lo - tol or b > hi + tol or a < EDGE_S - tol or b > dur - EDGE_S + tol:
            bad.append('%s: %.2f-%.2f s outside the window %.2f-%.2f (file %.1f s)' % (sid, a, b, lo, hi, dur))
        cut = os.path.join(base, r['segment_file'])
        if m.get('cut') is False:
            if r['segment_file']:
                bad.append('%s: clip marked as not cuttable but has a segment file' % sid)
        elif not r['segment_file'] or not os.path.isfile(cut):
            bad.append('%s: source cut missing' % sid)
        else:
            ok, why = usable(cut)
            if not ok:
                bad.append('%s: source cut has no usable telemetry (%s)' % (sid, why))
            else:
                c, _, frames, _, _ = load_sorted(cut)
                want = round((b - a) * float(c.fps))
                if abs(len(frames) - want) > 1:
                    bad.append('%s: cut has %d frames, expected %d' % (sid, len(frames), want))
        pv = os.path.join(base, r['preview'])
        if not os.path.isfile(pv):
            bad.append('%s: preview missing' % sid)
        else:
            codec, pix, w, n, rate = preview_info(pv)
            want = round((b - a) * rate)
            if codec != 'h264' or pix != 'yuv420p':
                bad.append('%s: preview is %s/%s, not h264/yuv420p' % (sid, codec, pix))
            if abs(n - want) > 2:
                bad.append('%s: preview has %d frames, expected ~%d' % (sid, n, want))
        if not os.path.isfile(os.path.join(base, r['plot'])):
            bad.append('%s: plot missing' % sid)
    for r in rows:
        m = meta['clips'][r['clip']]
        t, ti = float(r['t_source_s']), float(r['t_in_segment_s'])
        a, b = float(r['seg_start_s']), float(r['seg_end_s'])
        if not (EDGE_S - tol <= t <= m['duration_s'] - EDGE_S + tol):
            bad.append('%s: event inside the take-off/landing margin (t=%.2f)' % (r['event_id'], t))
        if not (-tol <= ti <= b - a + tol):
            bad.append('%s: time in segment %.2f outside 0-%.2f' % (r['event_id'], ti, b - a))
    print()
    print('%-9s %-8s %-8s %5s %5s | %s' % ('clip', 'camera', 'eye', 'segs', 'events', 'spike tel_jitter render_jerk image_bad control'))
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r['clip']][r['type']] += 1
    for name, m in meta['clips'].items():
        if 'skipped' in m:
            print('%-9s skipped: %s' % (name, m['skipped']))
            continue
        cnt = by[name]
        print('%-9s %-8s %-8s %5d %5d | %5d %10d %11d %9d %7d%s%s' % (
            name, m['header']['camera'], m['entry'].get('eye', ''), m['segments'], sum(cnt.values()),
            cnt['spike'], cnt['tel_jitter'], cnt['render_jerk'], cnt['image_bad'], cnt['control'],
            ('   HEADER MISMATCH: %s' % m['header_problems']) if m['header_problems'] else '',
            '   (no lossless cut: preview and plot only)' if m.get('cut') is False else ''))
    print()
    if bad:
        print('%d PROBLEMS:' % len(bad))
        for x in bad:
            print('  ' + x)
        return 1
    print('OK: every segment and event passes')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))

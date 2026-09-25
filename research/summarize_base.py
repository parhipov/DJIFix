#!/usr/bin/env python3
"""
Per-clip and per-group numbers for a segment base built by research/diagnose.py.

    python research/summarize_base.py artifacts/segments        # writes artifacts/segments/SUMMARY.md

Per clip, on the usable window (take-off and landing excluded):
  events per minute of each detector type;
  telemetry error vs the image above 4 Hz on calm frames (< 30 deg/s) where the
  image is self-consistent: median of the 0.5 s rms, all axes and roll alone --
  the most direct measure of DJI's own error the base has;
  high-frequency share of the telemetry itself on calm stretches (15-25 Hz power
  over 2-6 Hz, per axis), which needs no image;
  the calm baseline of residual motion in the Gyroflow render of the original.
Groups: by air-unit serial, camera generation, and the user's eye verdict.
"""
import collections
import csv
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from diagnose import analyse, running_rms   # noqa: E402
from fix_pipeline import clusters            # noqa: E402

TYPES = ('spike', 'tel_jitter', 'render_jerk', 'image_bad')


def clip_numbers(m, rows):
    entry = m['entry']
    cache = m['cache']
    image, resid = os.path.join(cache, 'image.npz'), os.path.join(cache, 'orig_resid.npz')
    clip, s, flags, info = analyse(entry['path'], image, resid if os.path.isfile(resid) else None)
    fps = s['fps']
    lo, hi = m['window_s']
    t = s['t']
    win = (t >= lo) & (t <= hi) & s['valid']
    minutes = win.sum() / fps / 60.0
    calm = win & (s['rate'] < 30)
    good = calm & s['img_ok']
    hp = butter(2, 4.0, btype='high', fs=fps, output='sos')
    has = np.isfinite(s['img']).all(1)
    e = np.where(has[:, None], sosfiltfilt(hp, np.nan_to_num(s['img']), axis=0) - sosfiltfilt(hp, s['inc'], axis=0), 0.0)
    w = max(5, int(round(0.5 * fps)))
    roll = running_rms(e[:, 2], w)
    axes = [running_rms(e[:, k], w) for k in range(3)]
    # telemetry-only high-frequency share on calm runs
    runs = [r for r in clusters(calm, gap=0) if len(r) >= 128]
    P = None
    for r in runs:
        fr, p = welch(s['inc'][r], fs=fps, nperseg=128, axis=0, detrend='linear')
        P = p * len(r) if P is None else P + p * len(r)
    hf = (P[fr >= 15].mean(0) / P[(fr >= 2) & (fr < 6)].mean(0)) if P is not None else np.full(3, np.nan)
    counts = collections.Counter(r['type'] for r in rows if r['clip'] == entry['name'])
    return dict(minutes=minutes, calm_frac=calm.sum() / max(win.sum(), 1), img_ok_frac=good.sum() / max(calm.sum(), 1),
                err_all=float(np.median(s['jit'][good])) if good.sum() > 50 else np.nan,
                err_axes=[float(np.median(a[good])) if good.sum() > 50 else np.nan for a in axes],
                err_roll_p90=float(np.percentile(roll[good], 90)) if good.sum() > 50 else np.nan,
                hf=[float(x) for x in hf], render_base=m.get('render_baseline_px'),
                per_min={k: counts[k] / minutes if minutes > 0 else np.nan for k in TYPES}, counts=dict(counts))


def main(base):
    meta = json.load(open(os.path.join(base, 'meta.json'), encoding='utf-8'))
    rows = list(csv.DictReader(open(os.path.join(base, 'index.csv'), encoding='utf-8-sig')))
    res = {}
    for name, m in meta['clips'].items():
        if 'skipped' in m:
            continue
        print('  %s...' % name, file=sys.stderr, flush=True)
        res[name] = (m, clip_numbers(m, rows))
    out = []
    out.append('# Сводка по базе проблемных мест\n')
    out.append('Собрано %s, код %s. Клипов: %d, кусков: %d, событий: %d.\n' % (
        meta['built'], meta['code'], len(res), len({r['segment'] for r in rows}), len(rows)))
    out.append('Ошибка телеметрии — медиана 0.5‑секундного rms разницы «телеметрия − изображение» выше 4 Гц, '
               'на спокойных кадрах (< 30 °/с) с надёжным изображением, градусы. Доля ВЧ — мощность 15–25 Гц '
               'к 2–6 Гц в самой телеметрии на спокойных участках (изображение не нужно). События — в минуту рабочего '
               'окна (без первых и последних 5 с).\n')
    out.append('| клип | камера | блок | прошивка | глаз | мин | ошибка, ° (x y z) | крен p90 | доля ВЧ (x y z) | фон рендера, px | спайк/мин | дрожание/мин | рывок/мин | изобр. плохо/мин |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for name, (m, r) in res.items():
        e = m['entry']
        out.append('| %s | %s | %s | %s | %s | %.1f | %s | %.3f | %s | %s | %.2f | %.2f | %.2f | %.2f |' % (
            name, m['header']['camera'], (m['header']['serial'] or '')[:7], m['header']['firmware'], e.get('eye', ''),
            r['minutes'], ' '.join('%.3f' % x for x in r['err_axes']), r['err_roll_p90'],
            ' '.join('%.2f' % x for x in r['hf']), ('%.2f' % r['render_base']) if r['render_base'] else '—',
            r['per_min']['spike'], r['per_min']['tel_jitter'], r['per_min']['render_jerk'], r['per_min']['image_bad']))
    out.append('')

    def group(title, key):
        g = collections.defaultdict(list)
        for name, (m, r) in res.items():
            g[key(name, m)].append((name, r))
        out.append('## %s\n' % title)
        out.append('| группа | клипы | мин | ошибка крена, ° | ошибка x/y, ° | доля ВЧ крена | спайк/мин | дрожание/мин |')
        out.append('|---|---|---|---|---|---|---|---|')
        for k, items in sorted(g.items(), key=lambda kv: str(kv[0])):
            mins = np.array([r['minutes'] for _, r in items])
            wavg = lambda f: float(np.nansum([f(r) * r['minutes'] for _, r in items]) / mins.sum())   # noqa: E731
            out.append('| %s | %s | %.1f | %.3f | %.3f / %.3f | %.2f | %.2f | %.2f |' % (
                k, ', '.join(n for n, _ in items), mins.sum(), wavg(lambda r: r['err_axes'][2]),
                wavg(lambda r: r['err_axes'][0]), wavg(lambda r: r['err_axes'][1]), wavg(lambda r: r['hf'][2]),
                wavg(lambda r: r['per_min']['spike']), wavg(lambda r: r['per_min']['tel_jitter'])))
        out.append('')

    group('По блоку (серийник, первые 7 знаков)', lambda n, m: (m['header']['serial'] or '?')[:7])
    group('По поколению камеры', lambda n, m: m['header']['camera'])
    group('По оценке на глаз', lambda n, m: m['entry'].get('eye', 'unknown'))
    group('По прошивке', lambda n, m: m['header']['firmware'])
    out.append('Оговорки: пары 18.07 и 19.07 сняты на двух разных, хотя очень похожих дронах — блок и рама '
               'смешаны. Серийник относится к блоку передачи, не к камере. Метрики по изображению ненадёжны '
               'на быстрых кадрах, в темноте и у веток — поэтому считаются только на спокойных кадрах с '
               'самосогласованным изображением. Вердикты глазом — единственная правда; эти числа их не заменяют.\n')
    path = os.path.join(base, 'SUMMARY.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out) + '\n')
    print('written %s' % path, file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1])

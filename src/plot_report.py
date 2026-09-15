#!/usr/bin/env python3
"""
Pictures of what the fix did, from the diagnostics a run leaves behind.

    python src/plot_report.py <clip>_fix_diagnostics.npz <clip>_image.npz <clip>_fix_report.json -o <folder>

Writes three PNGs next to the clip (main.py --plots does this for you):

  <clip>_roll.png        per-frame roll: telemetry, image, corrected; whole clip and a calm 3 s zoom
  <clip>_pitch_yaw.png   per-frame pitch/yaw: telemetry vs image vs corrected, events and spikes marked
  <clip>_correction.png  the applied correction angle per axis over time
  <clip>_overview.png    three panels (telemetry / image / fixed) of angular velocity with the
                         repaired events shaded, in the style of the session pipeline's telemetry_outliers.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_pipeline import image_rotations   # noqa: E402

C_TEL, C_IMG, C_FIX, C_EV, C_SPK = '#c0392b', '#2c3e50', '#1e8449', '#f5b041', '#8e44ad'


def _load(diag_path, image_path, report_path):
    d = np.load(diag_path)
    im = np.load(image_path)
    rep = json.load(open(report_path, encoding='utf-8')) if report_path and os.path.isfile(report_path) else {}
    t = d['frame_time_s']
    n = len(t)
    inc = d['inc_deg']
    theta = d['theta_deg']
    inc_img, roll_curl, _ = image_rotations(im, n)
    img = inc_img if inc_img is not None else np.column_stack([np.full(n, np.nan), np.full(n, np.nan), roll_curl])
    # corrected increment: telemetry plus the forward difference of the correction
    dth = np.vstack([np.diff(theta, axis=0), np.zeros((1, 3))])
    fixed = inc + dth
    events = (rep.get('events') or {}).get('list', [])
    return t, inc, img, fixed, theta, d, events, rep


def _shade_events(ax, events):
    for e in events:
        if e.get('weight', 1.0) <= 0:
            continue
        ax.axvspan(e['start_s'], e['end_s'], color=C_EV, alpha=0.25, lw=0)
        for s in e.get('spike_frames_s', []):
            ax.axvline(s, color=C_SPK, alpha=0.6, lw=0.8)


def plot_roll(t, inc, img, fixed, d, events, out, title):
    rate = d['rate_deg_s']
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [1, 1]})
    a1.plot(t, inc[:, 2], color=C_TEL, lw=0.6, label='telemetry (DJI fused attitude)')
    a1.plot(t, img[:, 2], color=C_IMG, lw=0.6, alpha=0.8, label='image (measured from the video)')
    a1.plot(t, fixed[:, 2], color=C_FIX, lw=0.6, alpha=0.9, label='fixed telemetry')
    _shade_events(a1, events)
    a1.set_ylabel('roll, deg per frame')
    a1.set_title(title + ' — roll increment per frame')
    a1.legend(loc='upper right', fontsize=8, ncol=3)
    a1.grid(alpha=0.25)
    # calm zoom: the 3 s window with the lowest mean rate and image available
    ok = np.isfinite(img[:, 2]) & (rate < 1e5)
    w = max(3, int(round(3.0 / np.median(np.diff(t)))))
    best, best_v = 0, np.inf
    for s in range(0, len(t) - w, w // 6):
        seg = slice(s, s + w)
        if ok[seg].mean() < 0.9:
            continue
        v = np.mean(rate[seg])
        if v < best_v:
            best, best_v = s, v
    seg = slice(best, best + w)
    a2.plot(t[seg], inc[seg, 2], color=C_TEL, lw=1.0, marker='.', ms=3, label='telemetry')
    a2.plot(t[seg], img[seg, 2], color=C_IMG, lw=1.0, marker='.', ms=3, alpha=0.8, label='image')
    a2.plot(t[seg], fixed[seg, 2], color=C_FIX, lw=1.0, marker='.', ms=3, alpha=0.9, label='fixed')
    j = lambda x: np.nanstd(np.diff(x[seg]))
    a2.set_title('calm 3 s zoom (%.1f–%.1f s, mean rate %.0f deg/s): frame-to-frame jitter telemetry %.3f, image %.3f, fixed %.3f deg'
                 % (t[seg][0], t[seg][-1], best_v, j(inc[:, 2]), j(img[:, 2]), j(fixed[:, 2])), fontsize=9)
    a2.set_xlabel('time, s')
    a2.set_ylabel('roll, deg per frame')
    a2.grid(alpha=0.25)
    a2.legend(loc='upper right', fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_pitch_yaw(t, inc, img, fixed, events, out, title):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, k, name in ((axes[0], 0, 'pitch (x)'), (axes[1], 1, 'yaw (y)')):
        ax.plot(t, inc[:, k], color=C_TEL, lw=0.6, label='telemetry')
        ax.plot(t, img[:, k], color=C_IMG, lw=0.6, alpha=0.8, label='image')
        ax.plot(t, fixed[:, k], color=C_FIX, lw=0.6, alpha=0.9, label='fixed telemetry')
        _shade_events(ax, events)
        ax.set_ylabel('%s, deg per frame' % name)
        ax.grid(alpha=0.25)
    axes[0].set_title(title + ' — pitch/yaw increment per frame; shaded = repaired events, purple lines = telemetry spikes')
    axes[0].legend(loc='upper right', fontsize=8, ncol=3)
    axes[1].set_xlabel('time, s')
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_correction(t, theta, d, events, out, title, rep):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for k, name, c in ((0, 'pitch', '#2874a6'), (1, 'yaw', '#b9770e'), (2, 'roll', '#1e8449')):
        ax.plot(t, theta[:, k], color=c, lw=0.8, label=name)
    _shade_events(ax, events)
    ax.set_xlabel('time, s')
    ax.set_ylabel('added rotation, deg')
    tm = rep.get('timing', {})
    ax.set_title('%s — correction applied to the telemetry (timing shift %+.2f ms mean; roll rms %.3f deg; %d events)'
                 % (title, tm.get('shift_ms_mean', 0.0), rep.get('roll', {}).get('correction_rms_deg', float('nan')),
                    rep.get('events', {}).get('count', len(events))), fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_overview(t, inc, img, fixed, d, events, rep, out, title):
    """Three panels in the style of the session pipeline's telemetry_outliers.png:
    angular velocity per axis for the telemetry, the image and the fixed telemetry,
    repaired events shaded red (darker = larger correction), spikes as purple lines."""
    fps = 1.0 / float(np.median(np.diff(t)))
    fig, panels = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    colors = ('#2878b5', '#e07a1f', '#3a923a')
    names = ('X (pitch)', 'Y (yaw)', 'Z (roll)')
    data = (('DJI telemetry', inc), ('Image (measured from the source video)', img), ('Fixed telemetry', fixed))
    applied = [e.get('applied_peak_deg', 0.0) for e in events if e.get('weight', 1.0) > 0] or [1.0]
    top = max(max(applied), 1e-6)
    for panel, (name, series) in zip(panels, data):
        for k in range(3):
            panel.plot(t, series[:, k] * fps, color=colors[k], linewidth=0.75, label=names[k])
        for e in events:
            if e.get('weight', 1.0) <= 0:
                continue
            strength = min(1.0, e.get('applied_peak_deg', 0.0) / top)
            panel.axvspan(e['start_s'], e['end_s'], color=plt.cm.Reds(0.25 + 0.7 * strength), alpha=0.22 + 0.45 * strength, lw=0)
        for s_ in rep.get('spikes', {}).get('times_s', []):
            panel.axvline(s_, color=C_SPK, alpha=0.7, lw=0.9)
        panel.set_title(name)
        panel.set_xlabel('time, s')
        panel.grid(alpha=0.22)
    panels[0].set_ylabel('angular velocity, deg/s')
    panels[0].legend(loc='upper right', fontsize=8)
    n_ev = sum(1 for e in events if e.get('weight', 1.0) > 0)
    n_sp = rep.get('spikes', {}).get('count', 0)
    fig.suptitle('%s — repaired events: %d (red, darker = larger), telemetry spikes removed: %d (purple)' % (title, n_ev, n_sp))
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def make_plots(diag_path, image_path, report_path, outdir, base):
    t, inc, img, fixed, theta, d, events, rep = _load(diag_path, image_path, report_path)
    os.makedirs(outdir, exist_ok=True)
    title = base
    paths = []
    p = os.path.join(outdir, base + '_roll.png'); plot_roll(t, inc, img, fixed, d, events, p, title); paths.append(p)
    p = os.path.join(outdir, base + '_pitch_yaw.png'); plot_pitch_yaw(t, inc, img, fixed, events, p, title); paths.append(p)
    p = os.path.join(outdir, base + '_correction.png'); plot_correction(t, theta, d, events, p, title, rep); paths.append(p)
    p = os.path.join(outdir, base + '_overview.png'); plot_overview(t, inc, img, fixed, d, events, rep, p, title); paths.append(p)
    return paths


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('diagnostics')
    ap.add_argument('image')
    ap.add_argument('report', nargs='?')
    ap.add_argument('-o', '--outdir', default='.')
    ap.add_argument('--name', help='base name for the PNGs (default: from the diagnostics file)')
    a = ap.parse_args()
    base = a.name or os.path.basename(a.diagnostics).replace('_fix_diagnostics.npz', '')
    for p in make_plots(a.diagnostics, a.image, a.report, a.outdir, base):
        print(p)

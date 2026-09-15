"""Local all-axis quaternion smoothing trials on the full telemetry timeline.

Unlike selective.py, this does not use the Pro-fitted roll axis or spectral
curves, and does not reconstruct a correction from one attitude per video frame.
It is a visual diagnostic trial: smoothing can remove real camera motion too.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from align import slerp_series
from dji_o4 import read_telemetry
from rollfix import logv, qconj, qmul
from selective import rotation_vector_quat, window_gate
import quat
import sidecar


def smooth_quaternions(q, fs, cutoff):
    """Zero-phase component filtering of continuous unit quaternions, normalized.

    Not an intrinsic SO(3) mean; suitable for local modest rotations, with an
    explicit check against a degenerate component average.
    """
    q = np.asarray(q, dtype=float).copy()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    signs = np.where(np.sum(q[1:] * q[:-1], axis=1) < 0, -1., 1.)
    q[1:] *= np.cumprod(signs)[:, None]
    filtered = sosfiltfilt(butter(2, cutoff, fs=fs, output='sos'), q, axis=0)
    norms = np.linalg.norm(filtered, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or norms.min() < 0.5:
        raise ValueError('degenerate quaternion average; do not write this trial')
    return filtered / norms


def apply_correction(q, target, gate, gain=1.0):
    """Scale the relative rotation, independently of the filter's cutoff.

    gain=1 reaches the filtered target in the full-strength part of the window.
    gain>1 extrapolates past it; it does not mean stronger low-pass filtering.
    """
    if not np.isfinite(gain) or gain < 0:
        raise ValueError('gain must be finite and nonnegative')
    difference = logv(qmul(qconj(q), target))
    applied = difference * np.asarray(gate)[:, None] * gain
    return qmul(q, rotation_vector_quat(applied)), applied


def sample_shifted(grid, filtered, times, extra_ms):
    """Positive extra_ms uses a later attitude: output(t) = filtered(t + dt).

    This is a content shift, not a change to the stored sample timestamps.
    """
    if not np.isfinite(extra_ms):
        raise ValueError('extra_ms must be finite')
    target_times = times + extra_ms / 1000
    if np.min(target_times) < grid[0] or np.max(target_times) > grid[-1]:
        raise ValueError('insufficient telemetry context for the requested shift')
    return slerp_series(grid, filtered, target_times)


def build(folder, outdir, cutoffs=(8., 2.), gain=1.0, windows=None, extra_ms=0.0):
    if not np.isfinite(gain) or gain < 0:
        raise ValueError('gain must be finite and nonnegative')
    if not np.isfinite(extra_ms):
        raise ValueError('extra_ms must be finite')
    folder, outdir = Path(folder).resolve(), Path(outdir).resolve()
    if folder == outdir:
        raise ValueError('use a separate output folder to preserve the previous trials')
    prior = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    control = folder / '00_control.mp4'
    clip, samples, frames = read_telemetry(str(control))
    t = np.array([s.t_us / 1e6 for s in samples])
    q = np.array([s.q_cam for s in samples])
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    unique = np.r_[True, np.diff(ts) > 0]
    ts, qs = ts[unique], qs[unique]
    windows, fade = prior['windows'] if windows is None else windows, prior['fade_s']
    if not windows or any(len(w) != 2 for w in windows):
        raise ValueError('provide START:END windows')
    if any(end > len(frames) / clip.fps for _, end in windows):
        raise ValueError('window extends past the clip')
    fs = float(clip.imu_sampling_rate)
    if not cutoffs or not np.isfinite(fs) or fs <= 0:
        raise ValueError('invalid sampling rate or cutoffs')
    if any(not np.isfinite(c) or c <= 0 or c >= fs / 2 for c in cutoffs):
        raise ValueError('cutoffs must be positive and below Nyquist')
    gate = window_gate(t, windows, fade)
    active = np.flatnonzero(gate > 0)
    if not len(active):
        raise ValueError('no samples in the requested windows')
    # Include context on both sides to keep filter edge effects out of the trial.
    # Keep the same context as the original 8/2 Hz pair when testing 6 Hz alone.
    pad = max(2.5, 5. / min(cutoffs), abs(extra_ms) / 1000 + 1.)
    start = max(ts[0], min(w[0] for w in windows) - pad)
    end = min(ts[-1], max(w[1] for w in windows) + pad)
    grid = np.arange(start, end + 0.5 / fs, 1 / fs)
    dense = slerp_series(ts, qs, grid)
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(control, outdir / '00_control.mp4')
    report = dict(source=prior['source'], reference_control=str(control),
                  fps=clip.fps, resample_hz=fs, windows=windows, fade_s=fade,
                  shift_ms=prior['shift_ms'], extra_ms=extra_ms, gain=gain,
                  shift_convention='filtered(t + extra_ms/1000); local content shift, timestamps unchanged',
                  rate_gating=False, angle_limit=None,
                  method='full-rate normalized quaternion component low-pass',
                  note='Visual trial, not proof that removed motion is erroneous.',
                  variants={'00_control': {'path': str(outdir / '00_control.mp4')}})
    for cutoff in cutoffs:
        filtered = smooth_quaternions(dense, fs, cutoff)
        target = sample_shifted(grid, filtered, t[active], extra_ms)
        corrected, applied = apply_correction(q[active], target, gate[active], gain)
        edits = {}
        for idx, qq in zip(active, corrected):
            s = samples[idx]
            edits[s.frame, s.index_in_frame] = quat.norm(quat.gyroflow_to_dji(
                tuple(-qq if s.inverted else qq)))
        name = 'fullrate_%ghz' % cutoff
        if gain != 1.0:
            name += '_gain%g' % (100 * gain)
        if extra_ms:
            name += '_shift_%s%gms' % ('plus' if extra_ms > 0 else 'minus', abs(extra_ms))
        dest = outdir / (name + '.mp4')
        info = sidecar.build(str(control), str(dest), edits)
        magnitude = np.linalg.norm(applied, axis=1)
        report['variants'][name] = dict(path=str(dest), cutoff_hz=cutoff, gain=gain, extra_ms=extra_ms,
            bytes=info['bytes'], peak_correction_deg=float(magnitude.max()),
            changed_records=len(edits))
        np.savez_compressed(outdir / (name + '_diagnostics.npz'),
                            sample_time_s=t[active], applied_deg=applied)
        print(name, report['variants'][name], flush=True)
    (outdir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('previous_trial_folder')
    ap.add_argument('-o', '--outdir', required=True)
    ap.add_argument('--cutoff', type=float, nargs='+', default=[8., 2.])
    ap.add_argument('--gain', type=float, default=1.0,
                    help='relative rotation strength; 1 reaches filtered target, >1 extrapolates')
    ap.add_argument('--window', action='append', metavar='START:END',
                    help='override correction windows in seconds; repeat if needed')
    ap.add_argument('--extra-ms', type=float, default=0.,
                    help='local content shift: positive uses later filtered attitude; timestamps unchanged')
    a = ap.parse_args()
    windows = [tuple(map(float, w.split(':'))) for w in a.window] if a.window else None
    build(a.previous_trial_folder, a.outdir, a.cutoff, a.gain, windows, a.extra_ms)

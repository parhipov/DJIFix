"""Build local telemetry-only A/B trials; never decode or render video.

The spectral curves and roll axis are inherited from the earlier Pro experiment.
They are hypotheses to test visually, especially on Lite, not a noise detector.
The control compensates the previously observed external-sidecar readout/2
offset with one constant shift. No dbgi reference or blockwise time warp is used.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from align import slerp_series
from denoise import BAND_HZ, KEEP, KEEP_PANTILT, _basis
from dji_o4 import read_telemetry
from rotmath import logv, qconj, qmul, rotation_vector_quat, limit_rotation, window_gate  # noqa: F401
import quat
import sidecar


def spectral_filter(x, keep, fps):
    # Reflection avoids connecting the end of the flight to its beginning.
    pad = min(len(x) - 1, max(1, int(2 * fps)))
    xp = np.pad(x, (pad, pad), mode='reflect')
    f = np.fft.rfftfreq(len(xp), 1 / fps)
    y = np.fft.irfft(np.fft.rfft(xp) * np.interp(f, BAND_HZ, keep), len(xp))
    return y[pad:-pad]


def correction(qf, fps, axes, gain, hp=1.0):
    """Small body-frame correction; length equals the frame-attitude series."""
    inc = logv(qmul(qconj(qf[:-1]), qf[1:]))
    # Diagnostic only. The user observed no corresponding abrupt real motion.
    # Neither raw nor smoothed telemetry speed can veto a visually marked error.
    rate = np.linalg.norm(inc, axis=1) * fps
    speed = np.r_[rate, rate[-1]]
    delta = np.zeros_like(inc)
    for axis, keep in axes:
        v = inc @ axis
        delta += np.outer(spectral_filter(v, keep, fps) - v, axis)
    theta = np.vstack([np.zeros(3), np.cumsum(delta, axis=0)])
    # Apply gating AFTER integration and filtering: no filter tail outside it.
    sos = butter(2, hp, btype='highpass', fs=fps, output='sos')
    theta = sosfiltfilt(sos, theta, axis=0) * gain
    return theta, speed


def build_trials(video, outdir, windows, fade=0.5, gain=1.0, shift_ms=None,
                 max_angle=1.0):
    if not windows or not np.isfinite([fade, gain]).all() or fade <= 0 or gain < 0:
        raise ValueError('provide windows, positive fade and nonnegative gain')
    if not np.isfinite(max_angle) or max_angle <= 0:
        raise ValueError('max_angle must be positive and finite')
    video, outdir = Path(video).resolve(), Path(outdir).resolve()
    clip, samples, frames = read_telemetry(str(video))
    fps = float(clip.fps)
    if len(frames) < 32 or not np.isfinite(fps) or fps <= 2:
        raise ValueError('need at least 32 frames and fps > 2')
    t = np.array([s.t_us for s in samples])
    q = np.array([s.q_cam for s in samples])
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    keep = np.r_[True, np.diff(ts) > 0]
    ts, qs = ts[keep], qs[keep]
    if not np.isfinite(qs).all():
        raise ValueError('nonfinite input quaternions')
    shift_ms = clip.frame_readout_time_ms / 2 if shift_ms is None else shift_ms
    if not np.isfinite(shift_ms):
        raise ValueError('shift must be finite')
    # Constant shift is shared by all trials. Setting --shift-ms 0 retains raw
    # quaternion records in the control and outside the correction intervals.
    base = slerp_series(ts, qs, t + shift_ms * 1000) if shift_ms else q.copy()
    ft = np.arange(len(frames)) / fps
    if any(end > len(frames) / fps for _, end in windows):
        raise ValueError('window extends past the clip')
    window_gate(ft, windows, fade)  # validate before creating any output
    qf = slerp_series(ts, base[order][keep], ft * 1e6)
    outdir.mkdir(parents=True, exist_ok=True)
    uh, ax, ay = _basis()
    report = dict(source=str(video), fps=fps, product=clip.header,
                  windows=windows, fade_s=fade, gain=gain, shift_ms=shift_ms,
                  rate_gating=False, max_angle_deg=max_angle,
                  note='Visual trials, not a confirmed repair. Outside windows '
                       'quaternion records equal control, not necessarily source.',
                  variants={})
    base_edits = {}
    if shift_ms:
        for s, qq in zip(samples, base):
            base_edits[s.frame, s.index_in_frame] = quat.norm(quat.gyroflow_to_dji(
                tuple(-qq if s.inverted else qq)))
    sample_gate = window_gate(t / 1e6, windows, fade)
    for name, axes in [('00_control', []), ('01_local_roll', [(uh, KEEP)]),
                       ('02_local_3axis', [(uh, KEEP), (ax, KEEP_PANTILT), (ay, KEEP_PANTILT)])]:
        edits = base_edits.copy()
        applied = np.zeros((len(t), 3))
        if axes:
            theta, speed = correction(qf, fps, axes, gain)
            applied = np.column_stack([np.interp(t / 1e6, ft, theta[:, k]) for k in range(3)])
            # Bound the trial's intervention without labeling any motion real.
            applied = limit_rotation(applied, max_angle)
            applied *= sample_gate[:, None]
            changed = np.flatnonzero(np.any(applied != 0, axis=1))
            newq = qmul(base[changed], rotation_vector_quat(applied[changed]))
            for j, qq in zip(changed, newq):
                s = samples[j]
                edits[s.frame, s.index_in_frame] = quat.norm(quat.gyroflow_to_dji(
                    tuple(-qq if s.inverted else qq)))
        dest = outdir / (name + '.mp4')
        if dest == video:
            raise ValueError('output must differ from source')
        info = sidecar.build(str(video), str(dest), edits)
        magnitude = np.linalg.norm(applied, axis=1)
        report['variants'][name] = dict(path=str(dest), bytes=info['bytes'],
            changed_vs_control=int(np.count_nonzero(magnitude)),
            peak_correction_deg=float(magnitude.max()))
        print(name, report['variants'][name], flush=True)
        if axes:
            np.savez_compressed(outdir / (name + '_diagnostics.npz'),
                frame_time_s=ft, telemetry_rate_dps=speed, window_gate=sample_gate,
                theta_deg=theta, sample_time_s=t / 1e6, applied_deg=applied)
    (outdir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('video')
    ap.add_argument('-o', '--outdir', required=True)
    ap.add_argument('--window', action='append', required=True, metavar='START:END',
                    help='seconds; repeat for multiple windows, ramps are inside')
    ap.add_argument('--fade', type=float, default=0.5)
    ap.add_argument('--gain', type=float, default=1.0)
    ap.add_argument('--max-angle', type=float, default=1.0,
                    help='smooth bound on correction magnitude in degrees (default 1)')
    ap.add_argument('--shift-ms', type=float, default=None,
                    help='constant sidecar time compensation; default readout/2, 0 for raw')
    a = ap.parse_args()
    windows = [tuple(map(float, w.split(':'))) for w in a.window]
    if any(len(w) != 2 for w in windows):
        ap.error('each window must be START:END')
    build_trials(a.video, a.outdir, windows, a.fade, a.gain, a.shift_ms, a.max_angle)


if __name__ == '__main__':
    main()

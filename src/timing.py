"""
Per-frame telemetry timing for a sidecar that Gyroflow reads as external motion data.

What Gyroflow actually does (gyroflow 1.6.3, `stabilization/frame_transform.rs`):
the frame's reference attitude is sampled at the container timestamp `pts`, and with
rolling-shutter correction on, the rows are spread from `pts - readout/2` to
`pts + readout/2`. So the *middle row* of every frame is read at `pts`, for the
video's own telemetry and for an external file alike. (The type-3 metadata export
reports `pts + readout/2`; that is the export, not the render.)

Which instant is right was measured against the image (measure_rotation.py, the
per-pair rotation in ray space; fix_pipeline.measure_timing_shift), on frames
faster than 30 deg/s, with the telemetry read at pts:

    O4 Pro  (exposure 19.6 ms):  best shift +0.2 .. +0.35 ms   (roll, pitch/yaw, curl all agree)
    O4 Lite (exposure 5-10 ms):  best shift -1.3 .. +1.3 ms

So `pts` itself is the right instant on both cameras to about a millisecond.
That means the video's own telemetry is timed correctly by Gyroflow, an
unshifted sidecar is too, and the two earlier theories are dead: the DJI `dbgi`
reference index (`align.py`, +10.5 ms on the Pro) was ~10 ms late, and the
exposure-centre model `pts + readout/2 - exposure/2` (-3.0 ms on the Pro,
+5.5 ms on the Lite) is off by 3-6 ms -- DJI evidently accounts for the exposure
in its own frame timestamps. (An earlier scan on a coarser 0.125-scale roll
series had suggested the exposure model; the finer measurement does not.)

What does survive, weakly (163 short-exposure frames on the Pro, +3.9 ms
observed against +4.5 ms predicted), is the *per-frame* part: when the
auto-exposure shortens the exposure mid-clip the optimum moves later by half the
difference. The default therefore is: constant measured on the clip from the
image (falls back to 0), plus `(median_exposure - exposure_f)/2` per frame.

The sidecar sample stored at time t carries the attitude from `t + delta_f`.
"""
import numpy as np

from align import slerp_series
from dji_o4 import read_telemetry


def exposure_ms(frames, default=None):
    """Per-frame exposure in ms; gaps filled with the nearest known value."""
    ex = np.array([fr['camera'].get('exposure_time_ms', np.nan) for fr in frames], dtype=float)
    ok = np.isfinite(ex)
    if not ok.any():
        if default is None:
            raise SystemExit('no exposure_time in the telemetry; pass a default')
        return np.full(len(frames), float(default))
    idx = np.arange(len(ex))
    ex[~ok] = np.interp(idx[~ok], idx[ok], ex[ok])
    return ex


def frame_shift_us(clip, frames, mode='exposure', bias_ms=0.0):
    """Content shift per frame, microseconds, positive = take a later attitude.

    mode 'exposure'      readout/2 - exposure/2 + bias   (the physical model, see module doc)
         'exposure_var'  (median exposure - exposure)/2 + bias: only the per-frame part.
                         Measured on both cameras, the constant is ~0 at pts (see
                         fix_pipeline.measure_timing_shift), so this is the default
                         when the image is not available to measure it.
         'constant'      bias only
         'none'          0
    """
    n = len(frames)
    if mode == 'none':
        return np.zeros(n)
    if mode == 'constant':
        return np.full(n, bias_ms * 1000.0)
    if mode == 'exposure_var':
        ex = exposure_ms(frames)
        return ((np.median(ex) - ex) / 2.0 + bias_ms) * 1000.0
    if mode != 'exposure':
        raise ValueError('mode must be exposure, exposure_var, constant or none')
    readout = float(clip.frame_readout_time_ms or 0.0)
    ex = exposure_ms(frames)
    return (readout / 2.0 - ex / 2.0 + bias_ms) * 1000.0


def shifts(video, mode='exposure', bias_ms=0.0):
    """(clip, samples, frames, delta_us) for a video, in the chosen timing mode.

    'dbgi' keeps the legacy `align.corrections` behaviour for A/B comparisons.
    """
    if mode == 'dbgi':
        from align import corrections
        clip, samples, frames, _, _, _, delta_us = corrections(video, read_instant_ms=0.0)
        return clip, samples, frames, delta_us
    clip, samples, frames = read_telemetry(video)
    return clip, samples, frames, frame_shift_us(clip, frames, mode, bias_ms)


def load_sorted(video_or_samples):
    """(clip, samples, frames, ts_us, qs) with a strictly increasing timeline."""
    if isinstance(video_or_samples, str):
        clip, samples, frames = read_telemetry(video_or_samples)
    else:
        clip, samples, frames = video_or_samples
    t = np.array([s.t_us for s in samples], dtype=float)
    q = np.array([s.q_cam for s in samples], dtype=float)
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    return clip, samples, frames, ts[keep], qs[keep]


def retime(samples, ts, qs, delta_us):
    """Quaternion per sample after the per-frame content shift (Gyroflow frame)."""
    t = np.array([s.t_us for s in samples], dtype=float)
    frame_of = np.array([s.frame for s in samples])
    return slerp_series(ts, qs, t + delta_us[frame_of])


def frame_instants_us(n_frames, fps):
    """Container timestamps of frames 0..n_frames, i.e. where Gyroflow reads them."""
    return np.arange(n_frames + 1) * (1e6 / fps)

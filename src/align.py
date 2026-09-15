"""
Fix the frame <-> telemetry alignment using DJI's own statement of it.

The `dbgi` track names, per frame, the one `djmd` sample that DJI considers that
frame's attitude (a bit-exact copy, index 19-20 of the 40-sample block). Gyroflow
cannot see that; it reconstructs the timeline from the `offset` field as

    t(frame f, slot i) = frame_ts_f + ((i - offset_f) / n_f) * (1000 / sensor_fps)

and then reads each frame's attitude at `frame_ts_f + readout/2`, i.e. at slot

    i_used_f = offset_f + n_f * readout / (2 * vsync)

Measured on this clip over the 600 frames where the block spans more than 2 deg
(so the slot is unambiguous): i_used = 11.97 on average while DJI's reference is
19.48 -- Gyroflow reads the attitude **3.77 ms too early**, and the error wanders
with a standard deviation of 2.02 ms. A misalignment tau leaves a residual
rotation of tau * omega(t) in the stabilized image: a scaled copy of the rate
signal, which is exactly a shake that grows with how fast the drone is turning.

Rewriting `offset` would fix the arithmetic but Gyroflow re-bases the whole
timeline as soon as a timestamp goes negative, which moves the frame times too.
So instead this shifts the *contents* of each block: slot i is filled with the
attitude interpolated at its own nominal time plus the per-frame correction. The
timing fields are left exactly as DJI wrote them.
"""
import math

import numpy as np

import quat as Q
import sidecar
from dji_o4 import read_telemetry, reference_indices
from rotmath import slerp_series  # noqa: F401  (re-exported for older callers)


def corrections(video, read_instant_ms=0.0):
    """Per-frame time correction in microseconds, plus the numbers behind it.

    `read_instant_ms` is when, inside the frame, Gyroflow samples the attitude:
    readout/2 for a file's own built-in telemetry, but **0** for telemetry loaded
    as an external motion-data file -- loading one makes Gyroflow drop the
    per-frame time offsets ("Not a main video, clearing per-frame offsets"), so
    the frame timestamps become 0, 20, 40 ms instead of 6.79, 26.79, 46.79 ms.
    Measured: the same telemetry via -g lands 0.189 deg (4.8 px) off the built-in
    path in the median, 0.89 deg (22.7 px) at p90.
    """
    clip, samples, frames = read_telemetry(video)
    iref, _ = reference_indices(video)
    iref = np.array(iref, dtype=float)
    good = iref[~np.isnan(iref)]
    iref = np.where(np.isnan(iref), np.median(good), iref)

    n = np.array([fr['num_quaternions'] for fr in frames], dtype=float)
    off = np.array([fr['sample_offset'] for fr in frames], dtype=float)
    vsync_ms = 1000.0 / clip.sensor_fps
    readout_ms = clip.frame_readout_time_ms

    i_used = off + n * read_instant_ms / vsync_ms
    shift_samples = iref - i_used
    delta_us = shift_samples / n * vsync_ms * 1000.0
    return clip, samples, frames, iref, i_used, shift_samples, delta_us


def build(video, out_path, extra_ms=0.0, per_frame=True, report=True,
          read_instant_ms=0.0):
    """Write a sidecar whose blocks carry the attitude for the right instant.

    extra_ms   additional constant shift on top of the derived correction
    per_frame  True  = remove both the mean bias and the wander (uses DJI's
                       reference index for every frame)
               False = remove only the mean bias (one constant correction)
    """
    clip, samples, frames, iref, i_used, shift_samples, delta_us = corrections(
        video, read_instant_ms)

    t = np.array([s.t_us for s in samples])
    q = np.array([s.q_cam for s in samples])
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, qs = ts[keep], qs[keep]

    frame_of = np.array([s.frame for s in samples])
    d = delta_us if per_frame else np.full_like(delta_us, np.median(delta_us))
    target = t + d[frame_of] + extra_ms * 1000.0
    newq = slerp_series(ts, qs, target)

    inverted = np.array([s.inverted for s in samples])
    newq = np.where(inverted[:, None], -newq, newq)
    edits = {}
    for s, qq in zip(samples, newq):
        edits[(s.frame, s.index_in_frame)] = Q.norm(Q.gyroflow_to_dji(tuple(map(float, qq))))

    info = sidecar.build(video, out_path, edits)
    if report:
        print('DJI reference index : mean %.3f' % iref.mean())
        print('index Gyroflow uses : mean %.3f  (offset + n*readout/(2*vsync))' % i_used.mean())
        print('correction applied  : mean %+.3f ms  std %.3f ms  (per_frame=%s, extra %+.1f ms)'
              % (delta_us.mean() / 1000, delta_us.std() / 1000, per_frame, extra_ms))
        print('%s: %d samples, %.1f MB, %d records rewritten'
              % (info['path'], info['samples'], info['bytes'] / 1e6, info['edited_records']))
    return info


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--extra-ms', type=float, default=0.0)
    ap.add_argument('--mean-only', action='store_true',
                    help='apply one constant correction instead of a per-frame one')
    ap.add_argument('--read-instant-ms', type=float, default=0.0,
                    help='0 for an external motion-data file (default), readout/2 '
                         'for telemetry read from the video itself')
    a = ap.parse_args()
    build(a.video, a.out, a.extra_ms, not a.mean_only, read_instant_ms=a.read_instant_ms)

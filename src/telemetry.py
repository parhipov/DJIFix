"""
Load / edit / write back the DJI O4 gyro dump.

    from telemetry import load
    tel = load('artifacts/main/DJI_20260905181949_0005_D_telemetry.npz')

    tel.t_us          (N,)   Gyroflow timeline, microseconds from video start
    tel.q_cam         (N, 4) camera-frame quaternion (w, x, y, z) -- what Gyroflow uses
    tel.q_dji         (N, 4) quaternion exactly as stored in the file
    tel.frame, tel.idx (N,)  which video frame and which slot inside it
    tel.frames        dict of per-frame arrays (timestamps, iso, exposure, ...)
    tel.meta          dict of clip metadata

    tel.euler()              yaw / pitch / roll in degrees, (N, 3)
    tel.rates()              angular velocity in deg/s, (N, 3)
    tel.at(12.5)             index of the sample nearest to 12.5 s
    tel.window(10, 12)       slice covering 10..12 s

Edit `tel.q_cam` (or `tel.q_dji`) in place, then:

    tel.write_sidecar('F:/36/DJI_....MP4', 'artifacts/main/edited_telemetry.mp4')

and load that file in Gyroflow under "Motion data".
"""
import json

import numpy as np

import quat as _q


# ------------------------------------------------------------ vector quat math
def qmul(a, b):
    """Quaternion product of (..., 4) arrays, (w, x, y, z)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(a):
    a = np.asarray(a, dtype=np.float64)
    return np.stack([a[..., 0], -a[..., 1], -a[..., 2], -a[..., 3]], axis=-1)


def qnorm(a):
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n == 0, 1.0, n)


def qeuler(a):
    """ZYX intrinsic yaw/pitch/roll in degrees, (..., 3)."""
    a = np.asarray(a, dtype=np.float64)
    w, x, y, z = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees(np.stack([yaw, pitch, roll], axis=-1))


def from_axis_angle(axis, deg):
    """Quaternion for a rotation of `deg` degrees about `axis`."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.concatenate([[np.cos(a)], np.sin(a) * axis])


class Telemetry:
    def __init__(self, data):
        self._d = data
        self.t_us = data['t_us']
        self.frame = data['frame']
        self.idx = data['idx']
        self.q_dji = data['q_dji'].astype(np.float64)
        self.q_cam = data['q_cam']
        self.inverted = data['inverted']
        self.file_offset = data['file_offset']
        self.meta = json.loads(str(data['meta_json']))
        self.frames = {k[len('frame_'):]: data[k] for k in data.files
                       if k.startswith('frame_')}

    def __len__(self):
        return len(self.t_us)

    def __repr__(self):
        return ('<Telemetry %d samples, %.3f s, %d frames, %s>'
                % (len(self), self.t_us[-1] / 1e6, len(self.frames.get('frame', [])),
                   self.meta.get('clip_meta_header', {}).get('product_name', '?')))

    @property
    def t_s(self):
        return self.t_us / 1e6

    def euler(self, which='cam'):
        return qeuler(self.q_cam if which == 'cam' else self.q_dji)

    def rates(self, which='cam', span=2):
        """Angular velocity in the body frame, deg/s, (N, 3).

        Central difference over `span` samples. The default of 2 matters: the
        stream is 2000 Hz but the fusion only updates at 1000 Hz, so every value
        appears twice and a one-sample difference would be zero half the time.
        """
        q = self.q_cam if which == 'cam' else self.q_dji
        t = self.t_us / 1e6
        n = len(q)
        lo = np.clip(np.arange(n) - span // 2, 0, n - 1)
        hi = np.clip(lo + span, 0, n - 1)
        q0, q1 = q[lo], q[hi]
        dt = t[hi] - t[lo]
        flip = np.sum(q0 * q1, axis=-1) < 0
        q1 = np.where(flip[:, None], -q1, q1)
        dq = qmul(qconj(q0), q1)
        w = np.clip(dq[:, 0], -1.0, 1.0)
        sn = np.sqrt(np.maximum(0.0, 1.0 - w * w))
        ok = (sn > 1e-12) & (dt > 0)
        scale = np.zeros(n)
        scale[ok] = 2 * np.arctan2(sn[ok], w[ok]) / dt[ok] / sn[ok]
        return np.degrees(dq[:, 1:] * scale[:, None])

    def unique(self, which='cam'):
        """(t_us, q) with the 2x duplicates collapsed -- the real 1000 Hz series."""
        q = self.q_cam if which == 'cam' else self.q_dji
        keep = np.ones(len(q), dtype=bool)
        keep[1:] = np.any(q[1:] != q[:-1], axis=1)
        order = np.argsort(self.t_us[keep], kind='stable')
        return self.t_us[keep][order], q[keep][order]

    def at(self, seconds):
        return int(np.searchsorted(self.t_us, seconds * 1e6))

    def window(self, t0, t1):
        return slice(self.at(t0), self.at(t1))

    def frame_slice(self, frame):
        """Indices belonging to one video frame."""
        lo = int(np.searchsorted(self.frame, frame, 'left'))
        hi = int(np.searchsorted(self.frame, frame, 'right'))
        return slice(lo, hi)

    def rotate(self, q, which='cam'):
        """Left-multiply every sample by `q` (a fixed extra rotation)."""
        if which == 'cam':
            self.q_cam = qmul(q, self.q_cam)
        else:
            self.q_dji = qmul(q, self.q_dji)

    # ------------------------------------------------------------------ output
    def edits(self, source='cam'):
        """{(frame, idx): (w, x, y, z)} in DJI's frame, ready for sidecar.build."""
        if source == 'cam':
            q = qnorm(self.q_cam)
            q = np.where(self.inverted[:, None], -q, q)
            q = qmul(qmul(_q.conj(_q.CAM_FLIP_Y), q), _q.conj(_q.DJI_TO_CAM))
            q = qnorm(q)
        else:
            q = self.q_dji
        return {(int(fr), int(ix)): tuple(map(float, row))
                for fr, ix, row in zip(self.frame, self.idx, q)}

    def write_sidecar(self, video, out_path, source='cam'):
        import sidecar
        info = sidecar.build(video, out_path, self.edits(source))
        print('%s: %d samples, %.1f MB, %d records replaced'
              % (info['path'], info['samples'], info['bytes'] / 1e6, info['edited_records']))
        return info

    def write_csv(self, path, source='cam'):
        eul = self.euler(source)
        with open(path, 'w', encoding='utf-8', newline='') as fh:
            fh.write('t_us,frame,idx,qw,qx,qy,qz,cam_qw,cam_qx,cam_qy,cam_qz,'
                     'yaw_deg,pitch_deg,roll_deg\n')
            for i in range(len(self)):
                fh.write('%.3f,%d,%d,%s,%s,%.6f,%.6f,%.6f\n' % (
                    self.t_us[i], self.frame[i], self.idx[i],
                    ','.join('%.9g' % v for v in self.q_dji[i]),
                    ','.join('%.9g' % v for v in self.q_cam[i]),
                    eul[i, 0], eul[i, 1], eul[i, 2]))


def load(path):
    return Telemetry(np.load(path, allow_pickle=False))

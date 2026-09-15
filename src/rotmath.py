"""Vectorised rotation helpers shared by the pipeline. Quaternions are (w, x, y, z)."""
import numpy as np


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def qconj(a):
    return np.stack([a[..., 0], -a[..., 1], -a[..., 2], -a[..., 3]], axis=-1)


def logv(a):
    """Rotation vector of a quaternion array, degrees (shortest arc)."""
    w = np.clip(a[:, 0], -1, 1)
    sn = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    ang = 2 * np.arctan2(sn, w)
    ang = np.where(ang > np.pi, ang - 2 * np.pi, ang)
    k = np.where(sn > 1e-12, ang / np.where(sn > 1e-12, sn, 1.0), 2.0)
    return np.degrees(a[:, 1:] * k[:, None])


def rotation_vector_quat(v):
    """Quaternions from rotation vectors in degrees, (N,3) -> (N,4)."""
    angle = np.linalg.norm(v, axis=1)
    half = np.deg2rad(angle) / 2
    scale = np.divide(np.sin(half), angle, out=np.full_like(angle, np.pi / 360), where=angle > 1e-12)
    return np.column_stack([np.cos(half), v * scale[:, None]])


def limit_rotation(v, max_angle):
    """Smooth amplitude bound (tanh) on rotation vectors, direction preserved."""
    magnitude = np.linalg.norm(v, axis=1)
    ratio = magnitude / max_angle
    scale = np.divide(np.tanh(ratio), ratio, out=np.ones_like(ratio), where=ratio > 0)
    return v * scale[:, None]


def window_gate(t, windows, fade):
    """Cosine ramps INSIDE each interval, exactly zero outside their union."""
    gate = np.zeros_like(t, dtype=float)
    for start, end in windows:
        if not (np.isfinite([start, end]).all() and 0 <= start < end):
            raise ValueError('windows must satisfy 0 <= start < end')
        ramp = min(fade, (end - start) / 2)
        phase = np.clip(np.minimum(t - start, end - t) / ramp, 0, 1)
        gate = np.maximum(gate, 0.5 - 0.5 * np.cos(np.pi * phase))
    return gate


def slerp_series(t, q, tq):
    """Interpolate a quaternion series (t ascending, q (N,4)) at times tq."""
    idx = np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2)
    t0, t1 = t[idx], t[idx + 1]
    w = np.clip((tq - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0, 1.0)
    q0, q1 = q[idx], q[idx + 1]
    flip = np.sum(q0 * q1, axis=1) < 0
    q1 = np.where(flip[:, None], -q1, q1)
    dot = np.clip(np.sum(q0 * q1, axis=1), -1.0, 1.0)
    ang = np.arccos(dot)
    small = ang < 1e-7
    s = np.where(small, 1.0, np.sin(np.where(small, 1.0, ang)))
    a = np.where(small, 1.0 - w, np.sin((1.0 - w) * ang) / s)
    b = np.where(small, w, np.sin(w * ang) / s)
    out = a[:, None] * q0 + b[:, None] * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)

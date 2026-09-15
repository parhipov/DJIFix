"""Quaternion helpers. Quaternions are (w, x, y, z) tuples."""
import math

def mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw)

def conj(q):
    return (q[0], -q[1], -q[2], -q[3])

def norm(q):
    n = math.sqrt(sum(c*c for c in q))
    return tuple(c/n for c in q) if n else q

def neg(q):
    return tuple(-c for c in q)

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

# telemetry-parser applies these two fixed rotations to DJI's raw quaternion
DJI_TO_CAM   = (0.5, -0.5, -0.5, 0.5)   # right-multiplied
CAM_FLIP_Y   = (0.0, 0.0, 1.0, 0.0)     # left-multiplied (180 deg about Y)

def dji_to_gyroflow(q):
    return mul(CAM_FLIP_Y, mul(q, DJI_TO_CAM))

def gyroflow_to_dji(q):
    return mul(mul(conj(CAM_FLIP_Y), q), conj(DJI_TO_CAM))

def to_euler_deg(q):
    """ZYX intrinsic (yaw, pitch, roll) in degrees."""
    w, x, y, z = q
    sinr = 2*(w*x + y*z); cosr = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.asin(sinp)
    siny = 2*(w*z + x*y); cosy = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

def from_euler_deg(yaw, pitch, roll):
    """Inverse of to_euler_deg (ZYX intrinsic, degrees)."""
    cy, sy = math.cos(math.radians(yaw)/2),   math.sin(math.radians(yaw)/2)
    cp, sp = math.cos(math.radians(pitch)/2), math.sin(math.radians(pitch)/2)
    cr, sr = math.cos(math.radians(roll)/2),  math.sin(math.radians(roll)/2)
    return (cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy)

#!/usr/bin/env python3
"""
DJI O4 (Air Unit) telemetry tool.

DJI does not record raw gyro/accel. The video's `djmd` metadata track carries a
protobuf (dvtm_O4P.proto) whose IMU section holds *fused attitude quaternions*
at the IMU sampling rate (2000 Hz nominal on the O4 Pro), 40 per video frame.

Subcommands:
  extract  video.MP4                  -> readable CSV + metadata JSON + .npz
  dump     video.MP4                  -> .npz only (the working format, see telemetry.py)
  sidecar  video.MP4                  -> small MP4 with just the telemetry track,
                                         for Gyroflow's "Motion data"; --from-csv to edit
  gcsv     video.MP4                  -> Gyroflow IMU LOG (.gcsv), gyro derived from
                                         the quaternions (lossy, see README)
  patch    video.MP4 edited.csv out   -> copy of the whole video with quaternions replaced
  verify   video.MP4 native.json      -> compare our parse with Gyroflow's own export

The time base replicates telemetry-parser (the library Gyroflow uses) exactly, so
timestamps line up with Gyroflow's native parse of the same file.
"""
import argparse, csv, json, math, os, shutil, struct, sys

import pb, quat
from mp4parse import open_mp4

# ---------------------------------------------------------------- protobuf map
# Field paths in dvtm_O4P.proto (product proto 02.00.06).
F_CLIP, F_STREAM, F_FRAME = 1, 2, 3
CLIP_HEADER, CLIP_STREAMS, CLIP_DISTORTION = 1, 2, 3
CLIP_READOUT_TIME, CLIP_READ_DIRECTION = 4, 5
CLIP_FOCAL_LENGTH, CLIP_EIS, CLIP_IMU_RATE, CLIP_SENSOR_FPS = 8, 9, 10, 11
STREAM_HEADER, STREAM_VIDEO = 1, 3
FRAME_HEADER, FRAME_CAMERA, FRAME_IMU = 1, 2, 3
FRAME_TIMESTAMP = 2                 # in FrameMetaHeader
IMU_ATTITUDE_AFTER_FUSION = 2       # in FrameMetaOfIMU
ATT_TIMESTAMP, ATT_VSYNC, ATT_QUAT, ATT_OFFSET = 1, 2, 3, 4   # in DeviceAttitude

QUAT_RECORD_LEN = 20                # 4 x (1 byte tag + 4 byte float)

CLIP_HEADER_NAMES = {
    1: 'proto_file_name', 2: 'library_proto_version', 3: 'product_proto_version',
    5: 'product_sn', 6: 'product_firmware_version', 7: 'meta_encryption_type',
    8: 'meta_compression_type', 9: 'clip_timestamp', 10: 'product_name',
}
VIDEO_STREAM_NAMES = {
    1: 'resolution_width', 2: 'resolution_height', 3: 'framerate',
    4: 'is_bit_depth_valid', 5: 'bit_depth', 6: 'bit_format',
    7: 'video_stream_type', 8: 'video_codec_type',
}
CAMERA_FRAME_NAMES = {
    2: 'exposure_index', 3: 'iso', 4: 'exposure_time', 5: 'digital_zoom_ratio',
    6: 'white_balance_cct', 7: 'orientation', 12: 'temperature', 15: 'focal_length',
}


def field(data, num, idx=0):
    """First (or idx-th) occurrence of field `num`; None if absent."""
    k = 0
    for fn, wt, v, _ in pb.iter_fields(data):
        if fn == num:
            if k == idx:
                return v
            k += 1
    return None


def child(data, base, num):
    """Sub-message `num` of `data` plus its absolute file offset.

    `base` is the absolute offset of `data` itself, so the returned offset can be
    used to rewrite bytes directly in the file.
    """
    for fn, wt, v, span in pb.iter_fields(data):
        if fn == num and wt == 2:
            return v, base + span[1] - len(v)
    return None, None


def scalar(msg, num=1, kind='varint'):
    """DJI wraps every scalar in a single-field submessage."""
    if msg is None:
        return None
    v = field(msg, num)
    if v is None:
        return None
    if kind == 'f32':
        return struct.unpack('<f', v)[0]
    return v


def packed_f32(msg, num=1):
    if msg is None:
        return None
    v = field(msg, num)
    if v is None:
        return None
    return list(struct.unpack('<%df' % (len(v) // 4), v))


def packed_varints(msg, num=1):
    if msg is None:
        return None
    v = field(msg, num)
    if v is None:
        return None
    out, p = [], 0
    while p < len(v):
        x, p = pb.read_varint(v, p)
        out.append(x)
    return out


# ------------------------------------------------------------------- clip meta
class Clip:
    """Everything we need from the clip-level metadata."""

    def __init__(self):
        self.header = {}
        self.distortion_coeffs = None
        self.readout_time_ns = None
        self.read_direction = 0
        self.focal_length = None
        self.eis_enabled = None
        self.imu_sampling_rate = 2000.0
        self.sensor_fps = 59.969295501708984
        self.fps = 59.94
        self.video = {}

    @property
    def fps_ratio(self):
        return self.fps / self.sensor_fps

    @property
    def frame_readout_time_ms(self):
        """Same value Gyroflow reports (rolling-shutter readout, milliseconds)."""
        if self.readout_time_ns is None:
            return None
        return (self.readout_time_ns / 1e6) / self.fps_ratio

    def as_dict(self):
        return {
            'clip_meta_header': self.header,
            'distortion_coefficients': self.distortion_coeffs,
            'sensor_readout_time_ns': self.readout_time_ns,
            'frame_readout_time_ms': self.frame_readout_time_ms,
            'sensor_read_direction': 'TopToBottom' if self.read_direction == 0 else str(self.read_direction),
            'digital_focal_length_px': self.focal_length,
            'eis_enabled': self.eis_enabled,
            'imu_sampling_rate_hz': self.imu_sampling_rate,
            'sensor_fps': self.sensor_fps,
            'video_stream_meta': self.video,
        }


def parse_clip(data, clip):
    cm = field(data, F_CLIP)
    if cm is not None:
        hdr = field(cm, CLIP_HEADER)
        if hdr is not None:
            for fn, wt, v, _ in pb.iter_fields(hdr):
                name = CLIP_HEADER_NAMES.get(fn, 'field_%d' % fn)
                clip.header[name] = v.decode('utf-8', 'replace') if wt == 2 else v
        clip.distortion_coeffs = packed_f32(field(cm, CLIP_DISTORTION)) or clip.distortion_coeffs
        rt = scalar(field(cm, CLIP_READOUT_TIME))
        if rt is not None:
            clip.readout_time_ns = rt
        rd = scalar(field(cm, CLIP_READ_DIRECTION))
        if rd is not None:
            clip.read_direction = rd
        fl = scalar(field(cm, CLIP_FOCAL_LENGTH), kind='f32')
        if fl is not None:
            clip.focal_length = fl
        eis = scalar(field(cm, CLIP_EIS))
        if eis is not None:
            clip.eis_enabled = bool(eis)
        ir = scalar(field(cm, CLIP_IMU_RATE))
        if ir is not None:
            clip.imu_sampling_rate = float(ir)
        sf = scalar(field(cm, CLIP_SENSOR_FPS), kind='f32')
        if sf is not None:
            clip.sensor_fps = sf
    sm = field(data, F_STREAM)
    if sm is not None:
        vs = field(sm, STREAM_VIDEO)
        if vs is not None:
            for fn, wt, v, _ in pb.iter_fields(vs):
                name = VIDEO_STREAM_NAMES.get(fn, 'field_%d' % fn)
                clip.video[name] = struct.unpack('<f', v)[0] if wt == 5 else v
            if clip.video.get('framerate'):
                clip.fps = float(clip.video['framerate'])


def parse_camera_frame(fm):
    """Per-frame camera settings (exposure, ISO, white balance...)."""
    if fm is None:
        return {}
    cf = field(fm, FRAME_CAMERA)
    if cf is None:
        return {}
    out = {}
    for fn, wt, v, _ in pb.iter_fields(cf):
        name = CAMERA_FRAME_NAMES.get(fn)
        if name is None or wt != 2:
            continue
        if name == 'exposure_time':
            nums = packed_varints(v)
            if nums and len(nums) >= 2 and nums[1]:
                out['exposure_time_s'] = nums[0] / nums[1]
                out['exposure_time_ms'] = 1000.0 * nums[0] / nums[1]
            continue
        inner = field(v, 1)
        if inner is None:
            continue
        if isinstance(inner, bytes) and len(inner) == 4:
            out[name] = struct.unpack('<f', inner)[0]
        else:
            out[name] = inner
    return out


# ------------------------------------------------------------------ extraction
class Sample:
    """One quaternion sample placed on Gyroflow's timeline."""
    __slots__ = ('t_us', 'frame', 'index_in_frame', 'q_dji', 'q_cam', 'file_offset', 'inverted')

    def __init__(self, t_us, frame, index_in_frame, q_dji, q_cam, file_offset, inverted):
        self.t_us = t_us
        self.frame = frame
        self.index_in_frame = index_in_frame
        self.q_dji = q_dji
        self.q_cam = q_cam
        self.file_offset = file_offset
        self.inverted = inverted


def read_telemetry(path):
    """Returns (clip, samples, frames)."""
    f, boxes, moov, tracks = open_mp4(path)
    try:
        tr = next(t for t in tracks if t.formats and t.formats[0] == b'djmd')
    except StopIteration:
        raise SystemExit('no DJI `djmd` metadata track in this file')

    clip = Clip()
    samples, frames = [], []
    first_ts = None
    prev_q = None
    inv = False

    for si, (dts, off, data) in enumerate(tr.samples(f)):
        if si == 0:
            parse_clip(data, clip)

        fm, fm_off = child(data, off, F_FRAME)
        if fm is None:
            continue
        hdr = field(fm, FRAME_HEADER)
        frame_ts = field(hdr, FRAME_TIMESTAMP) if hdr is not None else None
        if frame_ts is None:
            continue
        if first_ts is None:
            first_ts = frame_ts
        frame_ts_ms = (frame_ts - first_ts) / 1000.0

        imu, imu_off = child(fm, fm_off, FRAME_IMU)
        att, att_off = child(imu, imu_off, IMU_ATTITUDE_AFTER_FUSION) if imu is not None else (None, None)
        if att is None:
            continue

        quats, offsets = [], []
        for fn, wt, v, span in pb.iter_fields(att):
            if fn == ATT_QUAT and wt == 2:
                quats.append(v)
                offsets.append(att_off + span[0])       # first byte of the record

        att_ts = field(att, ATT_TIMESTAMP)
        vsync = field(att, ATT_VSYNC)
        offset_raw = field(att, ATT_OFFSET)
        q_offset = struct.unpack('<f', offset_raw)[0] if offset_raw else 0.0

        n = len(quats)
        if not n:
            continue
        vsync_duration_ms = 1000.0 / max(clip.sensor_fps, 1.0)
        frames.append({
            'frame': si,
            'video_time_s': dts / tr.timescale if dts is not None else None,
            'frame_timestamp_us': frame_ts,
            'imu_timestamp_us': att_ts,
            'vsync': vsync,
            'sample_offset': q_offset,
            'num_quaternions': n,
            'camera': parse_camera_frame(fm),
        })

        for i, raw in enumerate(quats):
            vals = {}
            for qf, qwt, qv, _ in pb.iter_fields(raw):
                if qwt == 5:
                    vals[qf] = struct.unpack('<f', qv)[0]
            q = (vals.get(1, 0.0), vals.get(2, 0.0), vals.get(3, 0.0), vals.get(4, 0.0))
            if any(math.isnan(c) for c in q):
                continue
            q_cam = quat.dji_to_gyroflow(q)
            if q_cam == (0.0, 0.0, 0.0, 0.0):
                continue
            # Gyroflow flips the sign when the quaternion jumps to the other
            # hemisphere, so the series stays continuous; mirror that here.
            if prev_q is not None:
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(prev_q, q_cam)))
                if d > 1.5:
                    inv = not inv
            prev_q = q_cam

            idx = i - q_offset
            quat_ts_ms = frame_ts_ms + (idx / n) * vsync_duration_ms
            t_ms = quat_ts_ms / clip.fps_ratio
            samples.append(Sample(
                t_us=t_ms * 1000.0, frame=si, index_in_frame=i,
                q_dji=q, q_cam=quat.neg(q_cam) if inv else q_cam,
                file_offset=offsets[i], inverted=inv))

    f.close()
    if not samples:
        detail = 'DJI metadata track contains no fused-attitude quaternion samples'
        if clip.eis_enabled:
            detail += ' (the clip metadata reports camera stabilization/EIS enabled)'
        raise SystemExit(detail)
    return clip, samples, frames


# -------------------------------------------------------------- gyro from quat
def quaternions_to_gyro(samples, dedupe=True, stamp='start', min_dt_us=200.0):
    """Angular velocity (rad/s, camera body frame) from the quaternion series.

    The stream is 2000 Hz nominal but fusion updates at 1000 Hz, so every value
    appears twice; dedupe first, otherwise every other difference is zero.
    """
    # A handful of samples per clip land slightly out of order at frame
    # boundaries (the per-frame sub-sample offset shifts); sort before
    # differentiating so no segment is lost.
    ordered = sorted(((s.t_us, s.q_cam) for s in samples), key=lambda x: x[0])
    seq = []
    for t, q in ordered:
        # a few samples per clip sit almost on top of the previous one at frame
        # boundaries; keeping them would divide a real rotation by a near-zero dt
        # and spike the rate to thousands of deg/s
        if seq and t - seq[-1][0] < min_dt_us:
            continue
        if dedupe and seq and seq[-1][1] == q:
            continue
        seq.append((t, q))

    out = []
    for i in range(len(seq) - 1):
        (t0, q0), (t1, q1) = seq[i], seq[i + 1]
        dt = (t1 - t0) / 1e6
        if quat.dot(q0, q1) < 0:            # shortest arc
            q1 = quat.neg(q1)
        dq = quat.mul(quat.conj(q0), q1)    # relative rotation in q0's body frame
        w = max(-1.0, min(1.0, dq[0]))
        sn = math.sqrt(max(0.0, 1.0 - w * w))
        if sn < 1e-12:
            wx = wy = wz = 0.0
        else:
            k = 2.0 * math.atan2(sn, w) / dt / sn
            wx, wy, wz = dq[1] * k, dq[2] * k, dq[3] * k
        t = t0 if stamp == 'start' else (t0 + t1) / 2.0
        out.append((t, wx, wy, wz, dt))
    if out and out[0][0] > 0:
        # Gyroflow shifts the whole log so the first sample lands at t=0, so make
        # sure the first sample really is at 0 and the rest keep absolute timing
        first = out[0]
        out.insert(0, (0.0, first[1], first[2], first[3], first[0] / 1e6))
    return out


def integrate_gyro(gyro, q0=(1.0, 0.0, 0.0, 0.0)):
    """Reference re-integration, used by `verify` to prove the gyro is faithful."""
    q = q0
    out = [(gyro[0][0], q)] if gyro else []
    for i in range(len(gyro) - 1):
        t0, wx, wy, wz = gyro[i][:4]
        dt = (gyro[i + 1][0] - t0) / 1e6
        mag = math.sqrt(wx * wx + wy * wy + wz * wz)
        if mag * dt < 1e-12:
            dq = (1.0, 0.0, 0.0, 0.0)
        else:
            a = mag * dt / 2.0
            sn = math.sin(a) / mag
            dq = (math.cos(a), wx * sn, wy * sn, wz * sn)
        q = quat.norm(quat.mul(q, dq))
        out.append((gyro[i + 1][0], q))
    return out


# --------------------------------------------------------------------- outputs
def write_csv(path, samples):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['t_us', 'frame', 'idx', 'qw', 'qx', 'qy', 'qz',
                    'cam_qw', 'cam_qx', 'cam_qy', 'cam_qz',
                    'yaw_deg', 'pitch_deg', 'roll_deg'])
        for s in samples:
            yaw, pitch, roll = quat.to_euler_deg(s.q_cam)
            # %.9g keeps 9 significant digits, the minimum for an exact
            # float32 round-trip when the values are written back
            w.writerow(['%.3f' % s.t_us, s.frame, s.index_in_frame]
                       + ['%.9g' % c for c in s.q_dji]
                       + ['%.9g' % c for c in s.q_cam]
                       + ['%.6f' % yaw, '%.6f' % pitch, '%.6f' % roll])


def write_npz(path, clip, samples, frames):
    """numpy dump: the working format for analysis and for editing in Python."""
    import numpy as np

    cols = {
        't_us': np.array([s.t_us for s in samples], dtype=np.float64),
        'frame': np.array([s.frame for s in samples], dtype=np.int32),
        'idx': np.array([s.index_in_frame for s in samples], dtype=np.int16),
        'q_dji': np.array([s.q_dji for s in samples], dtype=np.float32),
        'q_cam': np.array([s.q_cam for s in samples], dtype=np.float64),
        'inverted': np.array([s.inverted for s in samples], dtype=bool),
        'file_offset': np.array([s.file_offset for s in samples], dtype=np.int64),
    }

    meta = clip.as_dict()
    meta['num_frames'] = len(frames)
    meta['num_quaternion_samples'] = len(samples)
    meta['duration_s'] = samples[-1].t_us / 1e6 if samples else 0
    meta['quaternion_columns'] = 'w, x, y, z'
    meta['q_cam_note'] = ("q_dji left/right multiplied the way telemetry-parser does "
                          "(x (0,0,1,0), x (0.5,-0.5,-0.5,0.5)), with Gyroflow's "
                          "sign-continuity flip applied; this is what Gyroflow stabilises with")
    cols['meta_json'] = np.array(json.dumps(meta, ensure_ascii=False))

    keys = ['frame', 'video_time_s', 'frame_timestamp_us', 'imu_timestamp_us',
            'vsync', 'sample_offset', 'num_quaternions']
    for k in keys:
        vals = [fr.get(k) for fr in frames]
        dt = np.float64 if k in ('video_time_s', 'sample_offset') else np.int64
        cols['frame_' + k] = np.array([v if v is not None else 0 for v in vals], dtype=dt)
    for k in sorted({k for fr in frames for k in fr['camera']}):
        vals = [fr['camera'].get(k) for fr in frames]
        cols['frame_' + k] = np.array([v if v is not None else np.nan for v in vals],
                                      dtype=np.float64)

    np.savez_compressed(path, **cols)
    return cols


def write_frames_csv(path, frames):
    keys = ['frame', 'video_time_s', 'frame_timestamp_us', 'imu_timestamp_us',
            'vsync', 'sample_offset', 'num_quaternions']
    cam_keys = sorted({k for fr in frames for k in fr['camera']})
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(keys + cam_keys)
        for fr in frames:
            w.writerow([fr.get(k) for k in keys] + [fr['camera'].get(k) for k in cam_keys])


def write_gcsv(path, gyro, clip, videoname, orientation='XYZ', gscale=1.0):
    rd = 'TtB' if clip.read_direction == 0 else 'BtT'
    with open(path, 'w', newline='\n', encoding='utf-8') as fh:
        fh.write('GYROFLOW IMU LOG\n')
        fh.write('version,1.3\n')
        fh.write('id,dji_o4p_quat_derived\n')
        fh.write('vendor,DJI\n')
        fh.write('note,derived from djmd fused quaternions (%.0f Hz nominal)\n' % clip.imu_sampling_rate)
        fh.write('fwversion,%s\n' % clip.header.get('product_firmware_version', ''))
        fh.write('videofilename,%s\n' % videoname)
        fh.write('orientation,%s\n' % orientation)
        if clip.frame_readout_time_ms:
            fh.write('frame_readout_time,%.6f\n' % clip.frame_readout_time_ms)
            fh.write('frame_readout_direction,%s\n' % rd)
        fh.write('tscale,0.000001\n')
        fh.write('gscale,%.10g\n' % gscale)
        fh.write('t,gx,gy,gz\n')
        k = 1.0 / gscale
        for row in gyro:
            t, gx, gy, gz = row[:4]
            fh.write('%.0f,%.6f,%.6f,%.6f\n' % (t, gx * k, gy * k, gz * k))


# -------------------------------------------------------------------- patching
def patch(video, edited_csv, out_path, source='quat', normalize=False):
    """Rewrite quaternion payloads in a copy of the video.

    Each record is 4 float32 behind fixed-size tags, so the edit is byte-for-byte
    in place: no box sizes, chunk offsets or sample tables change.
    """
    clip, samples, frames = read_telemetry(video)
    by_key = {(s.frame, s.index_in_frame): s for s in samples}

    with open(edited_csv, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))

    if os.path.abspath(video) == os.path.abspath(out_path):
        raise SystemExit('refusing to patch the source video in place; give a different output path')
    if not os.path.exists(out_path):
        print('copying %s -> %s ...' % (video, out_path))
        shutil.copyfile(video, out_path)
    else:
        print('output exists, patching it in place: %s' % out_path)

    written = skipped = 0
    expect = bytes([(ATT_QUAT << 3) | 2, QUAT_RECORD_LEN])
    with open(out_path, 'r+b') as fh:
        for row in rows:
            s = by_key.get((int(row['frame']), int(row['idx'])))
            if s is None:
                skipped += 1
                continue
            if source == 'euler':
                q_cam = quat.from_euler_deg(float(row['yaw_deg']), float(row['pitch_deg']),
                                            float(row['roll_deg']))
                q = quat.norm(quat.gyroflow_to_dji(quat.neg(q_cam) if s.inverted else q_cam))
            elif source == 'cam':
                q_cam = (float(row['cam_qw']), float(row['cam_qx']),
                         float(row['cam_qy']), float(row['cam_qz']))
                q = quat.norm(quat.gyroflow_to_dji(quat.neg(q_cam) if s.inverted else q_cam))
            else:
                # take the values as they stand: DJI's own quaternions are not
                # exactly unit either, and renormalising would flip low bits
                q = (float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz']))
                if normalize:
                    q = quat.norm(q)
            fh.seek(s.file_offset)
            hdr = fh.read(2)
            if hdr != expect:
                raise SystemExit('unexpected record header at %d: %s' % (s.file_offset, hdr.hex()))
            fh.write(b''.join(struct.pack('<Bf', (i << 3) | 5, q[i - 1]) for i in (1, 2, 3, 4)))
            written += 1
    print('patched %d quaternion records (%d csv rows had no match)' % (written, skipped))


# ---------------------------------------------------------------------- verify
def verify(video, native_json):
    clip, samples, frames = read_telemetry(video)
    ref = json.load(open(native_json, encoding='utf-8'))
    refq = {int(k): v for k, v in ref['quaternions'].items()}
    print('ours: %d samples   gyroflow: %d samples' % (len(samples), len(refq)))

    worst_q = 0.0
    matched = 0
    for s in samples:
        key = int(s.t_us)
        cand = refq.get(key) or refq.get(key + 1) or refq.get(key - 1)
        if cand is None:
            continue
        matched += 1
        rq = (cand[3], cand[0], cand[1], cand[2])   # gyroflow stores [x, y, z, w]
        worst_q = max(worst_q, max(abs(a - b) for a, b in zip(s.q_cam, rq)))
    print('matched %d/%d by timestamp; worst quaternion component error = %.3e'
          % (matched, len(samples), worst_q))

    print('frame_readout_time: ours %.9f ms   gyroflow %.9f ms'
          % (clip.frame_readout_time_ms, ref['frame_readout_time']))
    fp = (ref.get('lens_profile') or {}).get('fisheye_params') or {}
    print('focal length: ours %s   gyroflow %s' % (clip.focal_length, fp.get('camera_matrix', [[None]])[0][0]))
    print('distortion:   ours %s' % (clip.distortion_coeffs,))
    print('              gyroflow %s' % (fp.get('distortion_coeffs'),))

    gyro = quaternions_to_gyro(samples)
    span = (gyro[-1][0] - gyro[0][0]) / 1e6
    print('derived gyro: %d samples over %.3fs = %.1f Hz' % (len(gyro), span, len(gyro) / span))
    mx = max(max(abs(g[1]), abs(g[2]), abs(g[3])) for g in gyro)
    print('peak rate: %.1f deg/s' % math.degrees(mx))

    integ = integrate_gyro(gyro, q0=samples[0].q_cam)
    seq = [(s.t_us, s.q_cam) for s in samples]
    worst = 0.0
    j = 0
    for t, q in integ[::97]:
        while j + 1 < len(seq) and seq[j + 1][0] < t:
            j += 1
        qr = seq[j][1]
        d = min(1.0, abs(quat.dot(q, qr)))
        worst = max(worst, 2 * math.degrees(math.acos(d)))
    print('gyro round-trip: max attitude error over the whole clip %.4f deg' % worst)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)

    p = sp.add_parser('extract', help='dump telemetry to CSV + JSON')
    p.add_argument('video')
    p.add_argument('-o', '--outdir', default='.')

    p = sp.add_parser('gcsv', help='write a Gyroflow IMU LOG (.gcsv)')
    p.add_argument('video')
    p.add_argument('-o', '--out')
    p.add_argument('--orientation', default='XYZ')
    p.add_argument('--gscale', type=float, default=1.0)
    p.add_argument('--no-dedupe', action='store_true')
    p.add_argument('--stamp', choices=['start', 'mid'], default='start')
    p.add_argument('--from-csv', help='read (possibly edited) quaternions from this CSV instead')
    p.add_argument('--source', choices=['quat', 'cam', 'euler'], default='quat')

    p = sp.add_parser('patch', help='write a copy of the video with edited quaternions')
    p.add_argument('video')
    p.add_argument('csv')
    p.add_argument('out')
    p.add_argument('--source', choices=['quat', 'cam', 'euler'], default='quat')
    p.add_argument('--normalize', action='store_true',
                   help='renormalise quaternions read from the csv (quat source only)')

    p = sp.add_parser('dump', help='numpy dump (.npz) for analysis in Python')
    p.add_argument('video')
    p.add_argument('-o', '--out')

    p = sp.add_parser('sidecar', help='small MP4 with only the DJI telemetry track')
    p.add_argument('video')
    p.add_argument('-o', '--out')
    p.add_argument('--from-csv', help='replace the quaternions with those from this CSV')
    p.add_argument('--source', choices=['quat', 'cam', 'euler'], default='quat')

    p = sp.add_parser('verify', help="compare against Gyroflow's --export-metadata output")
    p.add_argument('video')
    p.add_argument('native_json')

    a = ap.parse_args()

    if a.cmd == 'extract':
        clip, samples, frames = read_telemetry(a.video)
        os.makedirs(a.outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(a.video))[0]
        qcsv = os.path.join(a.outdir, base + '_quaternions.csv')
        fcsv = os.path.join(a.outdir, base + '_frames.csv')
        mj = os.path.join(a.outdir, base + '_metadata.json')
        write_csv(qcsv, samples)
        write_frames_csv(fcsv, frames)
        write_npz(os.path.join(a.outdir, base + '_telemetry.npz'), clip, samples, frames)
        meta = clip.as_dict()
        meta['num_frames'] = len(frames)
        meta['num_quaternion_samples'] = len(samples)
        meta['duration_s'] = samples[-1].t_us / 1e6 if samples else 0
        meta['effective_quaternion_rate_hz'] = (
            len(samples) / (samples[-1].t_us / 1e6) if samples else 0)
        with open(mj, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        print('%s\n%s\n%s' % (qcsv, fcsv, mj))
        print('%d frames, %d quaternions, sensor fps %.4f, readout %.3f ms'
              % (len(frames), len(samples), clip.sensor_fps, clip.frame_readout_time_ms))

    elif a.cmd == 'gcsv':
        clip, samples, frames = read_telemetry(a.video)
        if a.from_csv:
            samples = apply_csv(samples, a.from_csv, a.source)
        gyro = quaternions_to_gyro(samples, dedupe=not a.no_dedupe, stamp=a.stamp)
        out = a.out or os.path.splitext(a.video)[0] + '.gcsv'
        write_gcsv(out, gyro, clip, os.path.basename(a.video), a.orientation, a.gscale)
        span = (gyro[-1][0] - gyro[0][0]) / 1e6
        print('%s: %d samples over %.3fs = %.1f Hz' % (out, len(gyro), span, len(gyro) / span))

    elif a.cmd == 'dump':
        clip, samples, frames = read_telemetry(a.video)
        out = a.out or os.path.splitext(os.path.basename(a.video))[0] + '_telemetry.npz'
        write_npz(out, clip, samples, frames)
        print('%s: %d samples, %d frames, %.1f MB'
              % (out, len(samples), len(frames), os.path.getsize(out) / 1e6))

    elif a.cmd == 'sidecar':
        import sidecar as sc
        out = a.out or os.path.splitext(a.video)[0] + '_telemetry.mp4'
        edits = None
        if a.from_csv:
            _, samples, _ = read_telemetry(a.video)
            edits = sc.load_edits(a.from_csv, a.source, samples)
            print('read %d edited quaternions from %s' % (len(edits), a.from_csv))
        info = sc.build(a.video, out, edits)
        print('%s: %d samples, %.1f MB, %d records replaced'
              % (info['path'], info['samples'], info['bytes'] / 1e6, info['edited_records']))

    elif a.cmd == 'patch':
        patch(a.video, a.csv, a.out, a.source, a.normalize)

    elif a.cmd == 'verify':
        verify(a.video, a.native_json)


def apply_csv(samples, path, source):
    """Overwrite the quaternions of `samples` from an edited CSV."""
    by_key = {(s.frame, s.index_in_frame): s for s in samples}
    n = 0
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            s = by_key.get((int(row['frame']), int(row['idx'])))
            if s is None:
                continue
            if source == 'euler':
                q_cam = quat.from_euler_deg(float(row['yaw_deg']), float(row['pitch_deg']),
                                            float(row['roll_deg']))
            elif source == 'cam':
                q_cam = (float(row['cam_qw']), float(row['cam_qx']),
                         float(row['cam_qy']), float(row['cam_qz']))
            else:
                q_cam = quat.dji_to_gyroflow((float(row['qw']), float(row['qx']),
                                              float(row['qy']), float(row['qz'])))
            s.q_cam = quat.norm(q_cam)
            n += 1
    print('applied %d edited rows from %s' % (n, path))
    return samples


if __name__ == '__main__':
    main()


# ------------------------------------------------- DJI's own frame reference
def reference_indices(path):
    """Which sample of each frame's block DJI itself calls that frame's attitude.

    The `dbgi` track stores, per frame, a quaternion (2.1.11.7.4) that is a
    bit-exact copy of one record in the *same* frame's `djmd` block -- normally
    index 19 or 20 of 40, i.e. the block centre. That is DJI stating the
    frame <-> sample alignment outright, where Gyroflow can only infer it from
    the `offset` field.

    Returns (indices, block_sizes) as float arrays; NaN where no match was found.
    """
    f, boxes, moov, tracks = open_mp4(path)
    try:
        djm = next(t for t in tracks if t.formats and t.formats[0] == b'djmd')
        dbg = next(t for t in tracks if t.formats and t.formats[0] == b'dbgi')
    except StopIteration:
        f.close()
        raise SystemExit('this file has no djmd + dbgi pair')

    blocks = []
    for _, off, data in djm.samples(f):
        fm, fm_off = child(data, off, F_FRAME)
        att = None
        if fm is not None:
            imu, _ = child(fm, 0, FRAME_IMU)
            if imu is not None:
                att, _ = child(imu, 0, IMU_ATTITUDE_AFTER_FUSION)
        blocks.append([bytes(v) for fn, wt, v, _ in pb.iter_fields(att)
                       if fn == ATT_QUAT and wt == 2] if att is not None else [])

    refs = []
    for _, off, data in dbg.samples(f):
        q = None
        m = data
        for path_ in ((2, 1, 11, 7),):
            cur = m
            for num in path_:
                cur, _ = child(cur, 0, num)
                if cur is None:
                    break
            if cur is not None:
                for fn, wt, v, _ in pb.iter_fields(cur):
                    if fn == 4 and wt == 2 and len(v) == 16:
                        q = bytes(v)
        refs.append(q)
    f.close()

    import math as _m
    idx = []
    for qs, r in zip(blocks, refs):
        found = _m.nan
        if r is not None:
            for k, rec in enumerate(qs):
                # a record is 4 x (1 byte tag + 4 byte float); drop the tags
                if rec[1:5] + rec[6:10] + rec[11:15] + rec[16:20] == r:
                    found = float(k)
                    break
        idx.append(found)
    return idx, [float(len(q)) for q in blocks]

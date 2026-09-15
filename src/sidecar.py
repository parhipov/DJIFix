"""
Build a small sidecar MP4 that contains nothing but the DJI `djmd` telemetry
track, so Gyroflow can load it under "Motion data" and parse it through its own
native DJI path -- the quaternions arrive bit-identical, with the original
timestamps and rolling-shutter readout time, and nothing is re-integrated.

The track box is copied verbatim (hdlr, stsd, stts, stsz all preserved); only
the chunk offsets in `stco` are rewritten to point into the new mdat.
"""
import csv, struct

import pb, quat
from mp4parse import open_mp4, raw, walk
from dji_o4 import (F_FRAME, FRAME_IMU, IMU_ATTITUDE_AFTER_FUSION, ATT_QUAT,
                    ATT_OFFSET, QUAT_RECORD_LEN, child)

FTYP = struct.pack('>I4s4sI4s', 20, b'ftyp', b'isom', 0x200, b'isom')


def _attitude(sample):
    fm, fm_off = child(sample, 0, F_FRAME)
    if fm is None:
        return None, None
    imu, imu_off = child(fm, fm_off, FRAME_IMU)
    if imu is None:
        return None, None
    return child(imu, imu_off, IMU_ATTITUDE_AFTER_FUSION)


def quat_record_offsets(sample):
    """Byte offsets of every quaternion record inside one djmd sample."""
    att, att_off = _attitude(sample)
    if att is None:
        return []
    return [att_off + span[0] for fn, wt, v, span in pb.iter_fields(att)
            if fn == ATT_QUAT and wt == 2]


def offset_field(sample):
    """(byte offset, value) of the sub-frame `offset` float, or (None, None).

    Gyroflow places sample i of a frame at
        frame_ts + ((i - offset) / n) * (1000 / sensor_fps)
    so this one float shifts the frame's whole quaternion block along the
    timeline. Rewriting it is how `time_shift_ms` is applied.
    """
    att, att_off = _attitude(sample)
    if att is None:
        return None, None
    for fn, wt, v, span in pb.iter_fields(att):
        if fn == ATT_OFFSET and wt == 5:
            return att_off + span[1] - 4, struct.unpack('<f', v)[0]
    return None, None


def load_edits(path, source='quat', samples=None):
    """{(frame, idx): (w, x, y, z)} in DJI's own frame, from an edited CSV.

    The cam_* / yaw-pitch-roll columns carry Gyroflow's sign-continuity flip, so
    `samples` is used to undo it and keep the stored quaternion sign as DJI had it.
    """
    flipped = {(s.frame, s.index_in_frame) for s in (samples or []) if s.inverted}
    edits = {}
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            key = (int(row['frame']), int(row['idx']))
            if source == 'euler':
                q_cam = quat.from_euler_deg(float(row['yaw_deg']), float(row['pitch_deg']),
                                            float(row['roll_deg']))
            elif source == 'cam':
                q_cam = (float(row['cam_qw']), float(row['cam_qx']),
                         float(row['cam_qy']), float(row['cam_qz']))
            else:
                edits[key] = (float(row['qw']), float(row['qx']),
                              float(row['qy']), float(row['qz']))
                continue
            if key in flipped:
                q_cam = quat.neg(q_cam)
            edits[key] = quat.norm(quat.gyroflow_to_dji(q_cam))
    return edits


def box(typ, payload):
    return struct.pack('>I4s', 8 + len(payload), typ) + payload


EMPTY_STTS = box(b'stts', struct.pack('>II', 0, 0))
EMPTY_STSC = box(b'stsc', struct.pack('>II', 0, 0))
EMPTY_STSZ = box(b'stsz', struct.pack('>III', 0, 0, 0))
EMPTY_STCO = box(b'stco', struct.pack('>II', 0, 0))


def empty_video_trak(f, tracks):
    """A video track with the real frame size but no samples.

    telemetry-parser reads width/height off the video track's tkhd to build the
    official DJI lens profile; without one the profile is simply not emitted.
    """
    vid = next((t for t in tracks if t.formats and t.formats[0] in (b'hvc1', b'avc1', b'hev1')), None)
    if vid is None:
        return b''
    trak = vid.trak
    mdia = trak.find(b'mdia')
    minf = mdia.find(b'minf')
    stbl = minf.find(b'stbl')

    def cp(b_):
        return raw(f, b_) if b_ is not None else b''

    stbl_new = box(b'stbl', cp(stbl.find(b'stsd')) + EMPTY_STTS + EMPTY_STSC
                   + EMPTY_STSZ + EMPTY_STCO)
    minf_new = box(b'minf', cp(minf.find(b'vmhd')) + cp(minf.find(b'dinf')) + stbl_new)
    mdia_new = box(b'mdia', cp(mdia.find(b'mdhd')) + cp(mdia.find(b'hdlr')) + minf_new)
    return box(b'trak', cp(trak.find(b'tkhd')) + mdia_new)


def build(video, out_path, edits=None, with_video_track=True,
          time_shift_ms=0.0, offset_mode='keep', sensor_fps=None,
          readout_ms=None):
    """Write the sidecar.

    edits         {(frame, idx): (w, x, y, z)} quaternions in DJI's own frame
    time_shift_ms move the whole telemetry timeline (negative = telemetry earlier)
    offset_mode   'keep'    leave DJI's per-frame sub-frame offset alone
                  'const'   replace it with its median (kills the dither)
                  'zero'    ignore it entirely
                  'dji_ref' derive it, per frame, from DJI's own reference sample
                            in the dbgi track, so that the sample DJI calls this
                            frame's attitude lands exactly on the instant Gyroflow
                            reads the frame at (frame_ts + readout/2)
    """
    f, boxes, moov, tracks = open_mp4(video)

    trak_box = None
    for t in tracks:
        if t.formats and t.formats[0] == b'djmd':
            trak_box = t.trak
            track = t
            break
    if trak_box is None:
        raise SystemExit('no DJI `djmd` metadata track in this file')

    mvhd = moov.find(b'mvhd')
    mvhd_bytes = raw(f, mvhd)
    trak_bytes = bytearray(raw(f, trak_box))
    video_trak = empty_video_trak(f, tracks) if with_video_track else b''

    stbl = trak_box.find(b'mdia', b'minf', b'stbl')
    stco = stbl.find(b'stco')
    if stco is None:
        raise SystemExit('metadata track uses co64; not handled')
    stco_rel = stco.offset - trak_box.offset

    samples = [(off, data) for _, off, data in track.samples(f)]
    f.close()

    if sensor_fps is None:
        import dji_o4
        clip = dji_o4.Clip()
        dji_o4.parse_clip(samples[0][1], clip)
        sensor_fps = clip.sensor_fps
    vsync_ms = 1000.0 / sensor_fps

    median_offset = None
    if offset_mode == 'const':
        vals = [v for v in (offset_field(d)[1] for _, d in samples) if v is not None]
        median_offset = sorted(vals)[len(vals) // 2] if vals else 0.0

    ref_idx = None
    if offset_mode == 'dji_ref':
        import math
        import dji_o4
        if readout_ms is None:
            clip2 = dji_o4.Clip()
            dji_o4.parse_clip(samples[0][1], clip2)
            readout_ms = clip2.frame_readout_time_ms
        ref_idx, _ = dji_o4.reference_indices(video)
        good = [v for v in ref_idx if not math.isnan(v)]
        fallback = sorted(good)[len(good) // 2] if good else 19.5
        ref_idx = [fallback if math.isnan(v) else v for v in ref_idx]

    # collect (possibly edited) sample payloads
    expect = bytes([(ATT_QUAT << 3) | 2, QUAT_RECORD_LEN])
    payloads = []
    edited = shifted = 0
    touch = bool(edits) or time_shift_ms or offset_mode != 'keep'
    for si, (off, data) in enumerate(samples):
        if not touch:
            payloads.append(data)
            continue
        buf = bytearray(data)
        recs = quat_record_offsets(data)
        if edits:
            for i, rel in enumerate(recs):
                q = edits.get((si, i))
                if q is None:
                    continue
                if bytes(buf[rel:rel + 2]) != expect:
                    raise SystemExit('unexpected record header in frame %d' % si)
                buf[rel + 2:rel + 2 + QUAT_RECORD_LEN] = b''.join(
                    struct.pack('<Bf', (n << 3) | 5, q[n - 1]) for n in (1, 2, 3, 4))
                edited += 1
        if time_shift_ms or offset_mode != 'keep':
            pos, cur = offset_field(data)
            if pos is not None and recs:
                if offset_mode == 'dji_ref':
                    # want ts(ref_idx) == frame_ts + readout/2, and
                    # ts(i) = frame_ts + ((i - offset)/n)*vsync
                    base = ref_idx[si] - len(recs) * readout_ms / (2.0 * vsync_ms)
                else:
                    base = {'keep': cur, 'const': median_offset, 'zero': 0.0}[offset_mode]
                # ts = frame_ts + ((i - offset)/n)*vsync  =>  +S ms means -S*n/vsync
                new = base - time_shift_ms * len(recs) / vsync_ms
                struct.pack_into('<f', buf, pos, new)
                shifted += 1
        payloads.append(bytes(buf))

    sizes = [len(p) for p in payloads]
    if sizes != track.sizes:
        raise SystemExit('sample sizes changed; refusing to write')

    moov_size = 8 + len(mvhd_bytes) + len(video_trak) + len(trak_bytes)
    mdat_data_start = len(FTYP) + moov_size + 8

    pos = mdat_data_start
    offsets = []
    for s in sizes:
        offsets.append(pos)
        pos += s

    count = struct.unpack('>I', bytes(trak_bytes[stco_rel + 12:stco_rel + 16]))[0]
    if count != len(offsets):
        raise SystemExit('stco entry count %d != %d samples' % (count, len(offsets)))
    struct.pack_into('>%dI' % count, trak_bytes, stco_rel + 16, *offsets)

    with open(out_path, 'wb') as o:
        o.write(FTYP)
        o.write(struct.pack('>I4s', moov_size, b'moov'))
        o.write(mvhd_bytes)
        o.write(video_trak)
        o.write(bytes(trak_bytes))
        o.write(struct.pack('>I4s', 8 + sum(sizes), b'mdat'))
        for p in payloads:
            o.write(p)

    return {'path': out_path, 'samples': len(payloads), 'bytes': pos,
            'edited_records': edited, 'shifted_frames': shifted}

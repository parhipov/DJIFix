"""Tiny protobuf wire-format reader/inspector (no .proto needed)."""
import struct

def read_varint(d, p):
    r = 0; s = 0
    while True:
        if p >= len(d): raise EOFError
        b = d[p]; p += 1
        r |= (b & 0x7f) << s
        if not (b & 0x80): break
        s += 7
        if s > 70: raise ValueError('varint too long')
    return r, p

def zigzag(v):
    return (v >> 1) ^ -(v & 1)

def iter_fields(d, p=0, end=None):
    """Yield (field_number, wire_type, value, raw_slice) tuples."""
    if end is None: end = len(d)
    while p < end:
        start = p
        key, p = read_varint(d, p)
        fn, wt = key >> 3, key & 7
        if fn == 0: raise ValueError('field 0')
        if wt == 0:
            v, p = read_varint(d, p)
        elif wt == 1:
            if p+8 > end: raise ValueError('trunc f64')
            v = d[p:p+8]; p += 8
        elif wt == 2:
            ln, p = read_varint(d, p)
            if p + ln > end: raise ValueError('len overflow')
            v = d[p:p+ln]; p += ln
        elif wt == 5:
            if p+4 > end: raise ValueError('trunc f32')
            v = d[p:p+4]; p += 4
        else:
            raise ValueError(f'wiretype {wt}')
        yield fn, wt, v, (start, p)

def looks_like_message(b):
    if not b: return False
    try:
        n = 0
        for fn, wt, v, _ in iter_fields(b):
            n += 1
        return n > 0
    except Exception:
        return False

def is_text(b):
    return bool(b) and all(32 <= c < 127 or c in (9,10,13) for c in b)

def dump(d, depth=0, path='', maxdepth=12, out=None, limit_repeat=3):
    lines = out if out is not None else []
    seen = {}
    try:
        fields = list(iter_fields(d))
    except Exception as e:
        return lines
    for fn, wt, v, _ in fields:
        seen[fn] = seen.get(fn, 0) + 1
        idx = seen[fn]
        p2 = f"{path}.{fn}" if path else str(fn)
        ind = '  '*depth
        if wt == 0:
            lines.append(f"{ind}{p2} varint {v}  (zz={zigzag(v)})")
        elif wt == 5:
            f32 = struct.unpack('<f', v)[0]
            i32 = struct.unpack('<i', v)[0]
            lines.append(f"{ind}{p2} f32 {f32:.6g}  (i32={i32})")
        elif wt == 1:
            f64 = struct.unpack('<d', v)[0]
            i64 = struct.unpack('<q', v)[0]
            lines.append(f"{ind}{p2} f64 {f64:.6g}  (i64={i64})")
        else:
            if idx > limit_repeat:
                if idx == limit_repeat+1:
                    lines.append(f"{ind}{p2} ... (more repeats)")
                continue
            if is_text(v) and len(v) > 0:
                lines.append(f"{ind}{p2} str[{len(v)}] {v.decode('latin1')!r}")
            elif depth < maxdepth and looks_like_message(v):
                lines.append(f"{ind}{p2} msg[{len(v)}] #{idx}")
                dump(v, depth+1, p2, maxdepth, lines, limit_repeat)
            else:
                lines.append(f"{ind}{p2} bytes[{len(v)}] {v[:32].hex()}")
    return lines

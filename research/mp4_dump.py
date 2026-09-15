import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import struct, sys

CONTAINERS = {b'moov',b'trak',b'mdia',b'minf',b'stbl',b'edts',b'dinf',b'udta',b'mvex',b'moof',b'traf',b'meta'}

def dump(f, start, end, depth=0, out=None):
    pos = start
    while pos < end - 7:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8: break
        size, typ = struct.unpack('>I4s', hdr)
        hsize = 8
        if size == 1:
            size = struct.unpack('>Q', f.read(8))[0]; hsize = 16
        elif size == 0:
            size = end - pos
        if size < hsize: break
        line = '  '*depth + f"{typ.decode('latin1')} size={size} off={pos}"
        print(line)
        if typ in CONTAINERS:
            off = hsize
            if typ == b'meta':
                off += 4
            dump(f, pos+off, pos+size, depth+1)
        pos += size

fn = sys.argv[1]
with open(fn,'rb') as f:
    f.seek(0,2); sz = f.tell()
    dump(f, 0, sz)

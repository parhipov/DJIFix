"""Minimal MP4 (ISO BMFF) reader: box walk + sample table extraction."""
import struct

CONTAINERS = {b'moov',b'trak',b'mdia',b'minf',b'stbl',b'edts',b'dinf',b'udta',b'mvex',b'moof',b'traf'}

class Box:
    __slots__ = ('type','offset','size','hsize','children','_f')
    def __init__(self, typ, offset, size, hsize):
        self.type, self.offset, self.size, self.hsize = typ, offset, size, hsize
        self.children = []
    @property
    def body(self):
        return (self.offset + self.hsize, self.size - self.hsize)
    def find(self, *path):
        node = self
        for want in path:
            nxt = None
            for c in node.children:
                if c.type == want:
                    nxt = c; break
            if nxt is None: return None
            node = nxt
        return node
    def findall(self, typ):
        return [c for c in self.children if c.type == typ]
    def __repr__(self):
        return f"<{self.type.decode('latin1')} @{self.offset} +{self.size}>"

def walk(f, start, end, depth=0):
    out = []
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
        if size < hsize or pos + size > end: break
        b = Box(typ, pos, size, hsize)
        if typ in CONTAINERS:
            b.children = walk(f, pos+hsize, pos+size, depth+1)
        elif typ == b'meta':
            b.children = walk(f, pos+hsize+4, pos+size, depth+1)
        out.append(b)
        pos += size
    return out

def read(f, box, extra=0, n=None):
    off, size = box.body
    f.seek(off+extra)
    return f.read(size-extra if n is None else n)

def u32a(data):
    return struct.unpack('>%dI' % (len(data)//4), data[:len(data)//4*4])

class Track:
    def __init__(self, f, trak):
        self.trak = trak
        tkhd = read(f, trak.find(b'tkhd'))
        ver = tkhd[0]
        if ver == 1:
            self.id = struct.unpack('>I', tkhd[20:24])[0]
        else:
            self.id = struct.unpack('>I', tkhd[12:16])[0]
        mdhd = read(f, trak.find(b'mdia', b'mdhd'))
        if mdhd[0] == 1:
            self.timescale, self.duration = struct.unpack('>IQ', mdhd[20:32])
        else:
            self.timescale, self.duration = struct.unpack('>II', mdhd[12:20])
        hdlr = read(f, trak.find(b'mdia', b'hdlr'))
        self.handler_type = hdlr[4:8]
        self.handler_name = hdlr[20:].split(b'\0')[0].decode('latin1', 'replace')
        stbl = trak.find(b'mdia', b'minf', b'stbl')
        self.stbl = stbl
        # stsd formats
        stsd = read(f, stbl.find(b'stsd'))
        self.formats = []
        n = struct.unpack('>I', stsd[4:8])[0]
        p = 8
        for _ in range(n):
            esz, efmt = struct.unpack('>I4s', stsd[p:p+8])
            self.formats.append(efmt)
            p += max(esz, 8)
        # sizes
        stsz = stbl.find(b'stsz')
        if stsz is not None:
            d = read(f, stsz)
            samp_size, count = struct.unpack('>II', d[4:12])
            if samp_size:
                self.sizes = [samp_size]*count
            else:
                self.sizes = list(u32a(d[12:12+4*count]))
        else:
            d = read(f, stbl.find(b'stz2'))
            raise NotImplementedError('stz2')
        # chunk offsets
        stco = stbl.find(b'stco')
        if stco is not None:
            d = read(f, stco)
            cnt = struct.unpack('>I', d[4:8])[0]
            self.chunks = list(u32a(d[8:8+4*cnt]))
        else:
            d = read(f, stbl.find(b'co64'))
            cnt = struct.unpack('>I', d[4:8])[0]
            self.chunks = list(struct.unpack('>%dQ' % cnt, d[8:8+8*cnt]))
        # sample-to-chunk
        d = read(f, stbl.find(b'stsc'))
        cnt = struct.unpack('>I', d[4:8])[0]
        self.stsc = [struct.unpack('>III', d[8+12*i:20+12*i]) for i in range(cnt)]
        # time-to-sample
        d = read(f, stbl.find(b'stts'))
        cnt = struct.unpack('>I', d[4:8])[0]
        self.stts = [struct.unpack('>II', d[8+8*i:16+8*i]) for i in range(cnt)]

    @property
    def sample_count(self):
        return len(self.sizes)

    def sample_offsets(self):
        """Yield (offset, size) per sample."""
        offs = []
        si = 0
        nsamp = len(self.sizes)
        entries = self.stsc
        for ei, (first_chunk, spc, _desc) in enumerate(entries):
            last_chunk = entries[ei+1][0]-1 if ei+1 < len(entries) else len(self.chunks)
            for ci in range(first_chunk, last_chunk+1):
                if ci-1 >= len(self.chunks) or si >= nsamp: break
                off = self.chunks[ci-1]
                for _ in range(spc):
                    if si >= nsamp: break
                    offs.append((off, self.sizes[si]))
                    off += self.sizes[si]
                    si += 1
            if si >= nsamp: break
        return offs

    def sample_times(self):
        """Yield decode time (in timescale units) per sample."""
        t = 0
        out = []
        for cnt, delta in self.stts:
            for _ in range(cnt):
                out.append(t); t += delta
        return out

    def samples(self, f):
        times = self.sample_times()
        for i, (off, size) in enumerate(self.sample_offsets()):
            f.seek(off)
            yield (times[i] if i < len(times) else None), off, f.read(size)

def open_mp4(fn):
    f = open(fn, 'rb')
    f.seek(0,2); sz = f.tell()
    boxes = walk(f, 0, sz)
    moov = next(b for b in boxes if b.type == b'moov')
    tracks = [Track(f, t) for t in moov.findall(b'trak')]
    return f, boxes, moov, tracks


def raw(f, box):
    """Whole box including its header."""
    f.seek(box.offset)
    return f.read(box.size)

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import sys
from mp4parse import open_mp4
import pb
fn, want = sys.argv[1], sys.argv[2]
nsamp = int(sys.argv[3]) if len(sys.argv)>3 else 1
rep = int(sys.argv[4]) if len(sys.argv)>4 else 3
f, boxes, moov, tracks = open_mp4(fn)
tr = next(t for t in tracks if t.formats[0].decode()==want)
for i,(t,off,data) in enumerate(tr.samples(f)):
    if i>=nsamp: break
    print(f"===== {want} sample {i} t={t} len={len(data)}")
    print('\n'.join(pb.dump(data, limit_repeat=rep)))

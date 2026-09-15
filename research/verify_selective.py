"""Check local sidecars against their control and original telemetry, no video decode."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import argparse
import json
from pathlib import Path

import numpy as np

from dji_o4 import read_telemetry
from mp4parse import open_mp4
from selective import window_gate
from sidecar import quat_record_offsets


def metadata_payloads(path):
    f, _, _, tracks = open_mp4(str(path))
    try:
        track = next(t for t in tracks if t.formats and t.formats[0] == b'djmd')
        return [bytes(data) for _, _, data in track.samples(f)]
    finally:
        f.close()


def verify(folder):
    folder = Path(folder)
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    source_payloads = metadata_payloads(report['source'])
    _, control, _ = read_telemetry(str(folder / '00_control.mp4'))
    ct = np.array([s.t_us for s in control])
    cq = np.array([s.q_dji for s in control])
    outside = window_gate(ct / 1e6, report['windows'], report['fade_s']) == 0
    results = {}
    for name in report['variants']:
        path = folder / (name + '.mp4')
        payloads = metadata_payloads(path)
        assert len(payloads) == len(source_payloads), 'metadata frame count changed'
        for original, edited in zip(source_payloads, payloads):
            assert len(original) == len(edited), 'metadata payload size changed'
            a, b = bytearray(original), bytearray(edited)
            for offset in quat_record_offsets(original):
                for component in range(4):
                    start = offset + 3 + 5 * component
                    a[start:start + 4] = b'\0' * 4
                    b[start:start + 4] = b'\0' * 4
            assert a == b, 'non-quaternion metadata changed'
        _, samples, _ = read_telemetry(str(path))
        t = np.array([s.t_us for s in samples])
        q = np.array([s.q_dji for s in samples])
        np.testing.assert_array_equal(t, ct)
        assert np.isfinite(q).all(), 'nonfinite quaternion'
        assert np.max(abs(np.linalg.norm(q, axis=1) - 1)) < 2e-7
        np.testing.assert_array_equal(q[outside], cq[outside])
        results[name] = dict(samples=len(samples), unchanged_outside_windows=True,
                             timestamps_unchanged=True, other_metadata_unchanged=True)
    (folder / 'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(folder, results, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folders', nargs='+')
    for folder in ap.parse_args().folders:
        verify(folder)

#!/usr/bin/env python3
"""
One decoding pass over a clip that measures the camera's real rotation two ways.

  * per frame pair f -> f+1: the affine flow decomposition (`imagerot.affine_flow`
    logic): roll from the curl, which parallax cannot fake, plus the shift terms;
  * per window f -> f+gap (default 5 frames): a plane homography and a pure
    rotation fit on the same chained tracks, both decomposed to a rotation vector.
    Between consecutive frames the baseline is ~1.5 px and rotation cannot be told
    from translation over the ground; over 5 frames it can. Both estimates are
    kept -- the consumer decides which to trust (see fix_pipeline.py).

Output is one .npz (saved every 200 frames, so a partial file is usable):

  roll_deg, div, shear, shift_x_deg, shift_y_deg, npts        per pair f->f+1, raw pixels
  roll_n_deg, div_n, shear_n, shift_xn_deg, shift_yn_deg      the same in undistorted coords
  kabsch1_deg (N,3)                                            pure rotation per pair f->f+1
  homog_deg, kabsch_deg (N,3), normal (N,3), npts_win,
  resid_h_px, resid_r_px                                       per window f->f+gap
  meta_json                                                    parameters

Rotation vectors are in OpenCV camera axes (x right, y down, z forward), in
degrees, describing the ray motion from the first to the second frame; the signed
axis map onto the telemetry frame is fitted by the consumer (`best_axis_map`).

The pass decodes the clip in a background thread and tracks on the CPU by
default (0.16 s per 4K frame on 24 threads, unchanged numbers). `--backend gpu`
tracks on the GPU through OpenCL at 0.075 s per frame, but it is not the same
measurement and it is not interchangeable with the CPU one: sub-pixel
differences in the tracked points leave the per-pair quantities alone (roll
agrees to ~3 % of its own rms) while the f->f+gap window estimates move by
~0.2 deg in kabsch_deg and ~1 deg in homog_deg. Measured on one 180 s clip, that
is enough to change which pitch/yaw events fix_pipeline confirms. The same shift
appears between a 21 px and a 31 px window on the CPU, so it is the estimator's
own reproducibility floor rather than anything the GPU does wrong -- but it is a
reason to keep one backend for a given clip.

    python src/measure_rotation.py "F:\\36\\clip.MP4" -o artifacts/main/clip/clip_image.npz
    python src/measure_rotation.py "F:\\36\\clip.MP4" -o out.npz --backend gpu
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imagerot import Lens, rotation_kabsch, _pick_decomposition  # noqa: E402


def rays_from_normalised(n):
    """(N,2) pinhole-normalised points -> (N,3) unit rays, without undistorting again."""
    r = np.column_stack([n[:, 0], n[:, 1], np.ones(len(n))])
    return r / np.linalg.norm(r, axis=1, keepdims=True)


AFFINE_KEYS = ('roll_deg', 'div', 'shear', 'shift_x_deg', 'shift_y_deg')
AFFINE_N_KEYS = ('roll_n_deg', 'div_n', 'shear_n', 'shift_xn_deg', 'shift_yn_deg')

CPU_WIN, GPU_WIN = 31, 21   # OpenCV's OpenCL LK silently falls back to the CPU above 21


def gpu_device():
    """The OpenCL GPU OpenCV would use, or None. OpenCL is what the stock
    opencv-python wheel ships; it has no CUDA, so cv2.cuda is never an option."""
    try:
        if not cv2.ocl.haveOpenCL():
            return None
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.useOpenCL():
            return None
        d = cv2.ocl.Device_getDefault()
        if d is None or not d.name() or d.type() == 2:   # 2 = CL_DEVICE_TYPE_CPU
            return None
        return d.name().strip()
    except Exception:  # noqa: BLE001
        return None


class Flow:
    """Corner detection and LK tracking, on the GPU when OpenCL has one.

    Two things matter for speed and both are here rather than at the call site:
    every LK call rebuilds the image pyramid, so all the live chains are tracked
    in one call instead of one call each (identical result, ~1.7x), and on the
    GPU the images stay in device memory across frames -- only the point lists
    cross the bus."""

    def __init__(self, gpu, win, levels, mask, max_pts, quality=0.01, min_dist=10,
                 block=7, fb_err=0.5):
        self.gpu, self.win, self.levels = gpu, (win, win), levels
        self.max_pts, self.quality, self.min_dist, self.block = max_pts, quality, min_dist, block
        self.fb_err = fb_err
        self.mask = cv2.UMat(mask) if gpu else mask
        self.crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)

    def upload(self, grey):
        return cv2.UMat(grey) if self.gpu else grey

    def corners(self, img):
        p = cv2.goodFeaturesToTrack(img, self.max_pts, self.quality, self.min_dist,
                                    mask=self.mask, blockSize=self.block)
        if p is None:
            return None
        if isinstance(p, cv2.UMat):
            p = p.get()
        return p.reshape(-1, 2).astype(np.float32)

    def track(self, a, b, pts):
        """pts (N,2) -> (p1 (N,2), good (N,) bool), forward-backward checked."""
        p0 = np.ascontiguousarray(pts.reshape(-1, 1, 2), dtype=np.float32)
        src = cv2.UMat(p0) if self.gpu else p0
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, src, None, winSize=self.win,
                                             maxLevel=self.levels, criteria=self.crit)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, winSize=self.win,
                                               maxLevel=self.levels, criteria=self.crit)
        if self.gpu:
            p1, st, p0b, st2 = p1.get(), st.get(), p0b.get(), st2.get()
        p1 = p1.reshape(-1, 2)
        good = (st.ravel() == 1) & (st2.ravel() == 1)
        good &= np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1) < self.fb_err
        return p1, good


class GreyReader:
    """Decoding thread: full frame -> scaled grey, handed over through a queue.

    Decoding one 4K frame costs about as much as everything else in the loop put
    together and OpenCV drops the GIL while it runs, so it overlaps with the
    tracking instead of adding to it. Hardware decoding was tried and lost:
    downloading the 4K surface costs more than the decode it saves."""

    def __init__(self, video, first, scale, prefetch=8):
        self.video, self.first, self.scale = video, first, scale
        self.q = queue.Queue(max(1, prefetch))
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.video)
            if not cap.isOpened():
                raise RuntimeError('cannot open %s' % self.video)
            if self.first:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.first)
                if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - self.first) > .1:
                    cap.release()
                    cap = cv2.VideoCapture(self.video)
                    for _ in range(self.first):
                        if not cap.grab():
                            raise RuntimeError('cannot seek to frame %d' % self.first)
            while not self.stop.is_set():
                ok, img = cap.read()
                if not ok:
                    break
                small = cv2.resize(img, None, fx=self.scale, fy=self.scale,
                                   interpolation=cv2.INTER_AREA)
                grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                while not self.stop.is_set():
                    try:
                        self.q.put(grey, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as e:  # noqa: BLE001
            self.error = e
        finally:
            if cap is not None:
                cap.release()
            try:
                self.q.put(None, timeout=1.0)
            except queue.Full:
                pass

    def read(self):
        """Next grey frame, or None at the end of the clip."""
        while True:
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                if not self.thread.is_alive() and self.q.empty():
                    if self.error:
                        raise self.error
                    return None
                continue
            if item is None and self.error:
                raise self.error
            return item

    def close(self):
        self.stop.set()
        try:
            while not self.q.empty():
                self.q.get_nowait()
        except Exception:  # noqa: BLE001
            pass
        self.thread.join(timeout=2.0)


def affine_fit(a, b, cx, cy, f_small):
    """v = A p + b on tracked points; returns (roll_deg, div, shear, sx_deg, sy_deg).

    The 2N x 6 design matrix of the stacked least-squares problem has no unknown
    shared between an x row and a y row, so it splits into two 3-parameter
    systems. Solving those two 3x3 normal equations per reweighting step gives
    the same numbers as one least-squares solve on the stacked matrix and costs
    about a tenth as much; the columns are scaled to O(1) first so the normal
    equations stay as well conditioned as the original SVD.
    """
    p = np.asarray(a, dtype=np.float64) - np.array([cx, cy], dtype=np.float64)
    v = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    s0 = max(float(np.abs(p).max()), 1e-9)
    X = np.empty((len(p), 3))
    X[:, 0] = p[:, 0] / s0
    X[:, 1] = p[:, 1] / s0
    X[:, 2] = 1.0
    vx, vy = np.ascontiguousarray(v[:, 0]), np.ascontiguousarray(v[:, 1])
    wx = np.ones(len(p))
    wy = np.ones(len(p))
    c1 = c2 = None
    for _ in range(6):
        c1 = _wls3(X, vx, wx)
        c2 = _wls3(X, vy, wy)
        rx = np.abs(X @ c1 - vx)
        ry = np.abs(X @ c2 - vy)
        s = max(1.4826 * np.median(np.concatenate((rx, ry))), 1e-9)
        wx = 1.0 / np.sqrt(1.0 + (rx / (2 * s)) ** 2)
        wy = 1.0 / np.sqrt(1.0 + (ry / (2 * s)) ** 2)
    a11, a12, bx = c1[0] / s0, c1[1] / s0, c1[2]
    a21, a22, by = c2[0] / s0, c2[1] / s0, c2[2]
    return (np.degrees(0.5 * (a21 - a12)), 0.5 * (a11 + a22), 0.5 * (a11 - a22),
            np.degrees(bx / f_small), np.degrees(by / f_small))


def _wls3(X, y, w):
    """Weighted least squares on three unknowns, through the normal equations."""
    w2 = w * w
    Xw = X * w2[:, None]
    A = X.T @ Xw
    A[np.diag_indices(3)] += 1e-12 * np.trace(A)
    return np.linalg.solve(A, Xw.T @ y)


def solve_window(a, b, lens, focal_px, f_small, prev_n, min_inliers):
    """Homography + pure-rotation estimates for one chained track set."""
    na, nb = lens.to_normalised(a), lens.to_normalised(b)
    Hm, inl = cv2.findHomography(na, nb, cv2.RANSAC, 1.5 / f_small, maxIters=500, confidence=0.995)
    if Hm is None or inl is None or inl.sum() < min_inliers:
        return None
    inl = inl.ravel().astype(bool)
    Hm = Hm / Hm[2, 2]
    proj = cv2.perspectiveTransform(na[inl].reshape(-1, 1, 2), Hm).reshape(-1, 2)
    res_h = float(np.median(np.linalg.norm(proj - nb[inl], axis=1)) * focal_px)
    _, Rs, _, ns = cv2.decomposeHomographyMat(Hm, np.eye(3))
    rvec, n = _pick_decomposition(Rs, ns, None, prev_n)
    R, res_r, _ = rotation_kabsch(lens.to_rays(a[inl]), lens.to_rays(b[inl]))
    kab = np.degrees(cv2.Rodrigues(R)[0].ravel())
    return dict(homog=rvec, normal=n, kabsch=kab, npts=int(inl.sum()),
                resid_h=res_h, resid_r=float(np.median(res_r) * focal_px))


def measure(video, out_path, scale=0.25, gap=5, max_pts=1500, radius_frac=0.6,
            first=0, count=None, focal_px=None, distortion=None, min_inliers=80,
            fb_err=0.5, save_every=200, progress=True, lens=None,
            backend='cpu', win=None, levels=4, prefetch=8):
    """lens: optional dict(f, cx, cy, D, name) overriding the clip's own lens model,
    e.g. from lenscal.load_gyroflow_lens (the profile you pick in Gyroflow).

    backend: 'cpu' (the default, and the reference measurement), 'gpu' for OpenCL,
    'auto' for the GPU when there is one. Nothing reaches the GPU unless it is
    asked for by name. win is the LK window; it defaults to 31 on the CPU and
    21 on the GPU, which is the largest window OpenCV's OpenCL LK accepts before
    it quietly falls back to the CPU. Backends are not interchangeable within a
    clip -- see the module docstring."""
    if backend not in ('auto', 'gpu', 'cpu'):
        raise ValueError('backend must be auto, gpu or cpu')
    dev = None if backend == 'cpu' else gpu_device()
    if backend == 'gpu' and dev is None:
        raise SystemExit('no OpenCL GPU available for --backend gpu')
    gpu = dev is not None
    cv2.ocl.setUseOpenCL(bool(gpu))
    if win is None:
        win = GPU_WIN if gpu else CPU_WIN
    if gpu and win > GPU_WIN:
        print('  LK window %d is above %d: OpenCV would fall back to the CPU, using the CPU'
              % (win, GPU_WIN), flush=True)
        gpu, dev = False, None
        cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit('cannot open %s' % video)
    total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if count is None:
        count = total - first - 1
    cx0, cy0 = W0 / 2.0, H0 / 2.0
    lens_source = 'clip metadata'
    if lens is not None:
        focal_px, distortion = float(lens['f']), list(lens['D'])
        cx0, cy0 = float(lens.get('cx', cx0)), float(lens.get('cy', cy0))
        lens_source = str(lens.get('name', 'profile'))
    if focal_px is None or distortion is None:
        from dji_o4 import read_telemetry
        clip, _, _ = read_telemetry(video)
        focal_px = focal_px or float(clip.focal_length)
        distortion = distortion if distortion is not None else list(clip.distortion_coeffs)
    lens = Lens(f=focal_px, cx=cx0, cy=cy0, D=distortion, scale=scale)
    f_small = focal_px * scale

    out = {k: np.full(count, np.nan) for k in AFFINE_KEYS + AFFINE_N_KEYS}
    out['npts'] = np.zeros(count, dtype=int)
    out['kabsch1_deg'] = np.full((count, 3), np.nan)
    out['homog_deg'] = np.full((count, 3), np.nan)
    out['kabsch_deg'] = np.full((count, 3), np.nan)
    out['normal'] = np.full((count, 3), np.nan)
    out['npts_win'] = np.zeros(count, dtype=int)
    out['resid_h_px'] = np.full(count, np.nan)
    out['resid_r_px'] = np.full(count, np.nan)
    meta = dict(video=os.path.abspath(video), first=first, count=count, gap=gap, scale=scale,
                focal_px=focal_px, distortion=list(map(float, distortion)), width=W0, height=H0,
                cx=cx0, cy=cy0, lens_source=lens_source,
                max_pts=max_pts, radius_frac=radius_frac, fb_err=fb_err,
                win=int(win), levels=int(levels), backend=('gpu' if gpu else 'cpu'),
                device=(dev or 'cpu'),
                axes='OpenCV camera: x right, y down, z forward; rotation of rays from frame f to f+gap')

    start_at = 0
    if os.path.exists(out_path):
        try:
            # closed before the first save(): Windows refuses to replace a file
            # that still has an open handle, and np.load keeps the zip open
            with np.load(out_path) as prev_d:
                pm = json.loads(str(prev_d['meta_json']))
                # a partial file written before the backend switch existed carries
                # no 'win': it came from the CPU path, which is what CPU_WIN means
                same = (pm.get('win', CPU_WIN) == meta['win']
                        # the same clip reached by a differently spelled path is
                        # still the same clip: drag-and-drop and a typed command
                        # disagree about the drive letter's case on Windows
                        and os.path.normcase(pm.get('video', '')) == os.path.normcase(meta['video'])
                        and all(pm.get(k) == meta[k]
                                for k in ('first', 'count', 'gap', 'scale', 'max_pts')))
                if same:
                    got = np.flatnonzero(np.isfinite(prev_d['roll_deg']))
                    if len(got):
                        start_at = max(0, int(got[-1]) - 2 * gap)
                        for k in out:
                            out[k][:start_at] = prev_d[k][:start_at]
                        print('resuming at frame %d' % (first + start_at), flush=True)
        except Exception as e:  # noqa: BLE001
            print('could not resume from %s: %s' % (out_path, e), flush=True)

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tmp = out_path + '.tmp.npz'
        np.savez_compressed(tmp, meta_json=json.dumps(meta), **out)
        os.replace(tmp, out_path)

    cap.release()
    reader = GreyReader(video, first + start_at, scale, prefetch=prefetch)
    prev_grey = reader.read()
    if prev_grey is None:
        reader.close()
        raise SystemExit('cannot read frame %d' % (first + start_at))
    h, w = prev_grey.shape
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < (radius_frac * min(w, h)) ** 2).astype(np.uint8) * 255
    flow = Flow(gpu, win, levels, mask, max_pts, fb_err=fb_err)
    if progress:
        print('  %s, LK window %d' % (('GPU (OpenCL): %s' % dev) if gpu else 'CPU', win), flush=True)

    prev = flow.upload(prev_grey)
    chains = deque()   # [start_index, start_pts, cur_pts]

    def seed(img, index):
        p = flow.corners(img)
        if p is not None and len(p) >= 60:
            chains.append([index, p.copy(), p.copy()])

    seed(prev, start_at)
    # the homography decomposition picks its branch against the previous window's
    # plane normal, so a resumed run has to start from the normal it left off at;
    # without this the first few windows after a resume pick the other branch
    prev_n = None
    if start_at:
        had = np.flatnonzero(np.isfinite(out['normal'][:start_at]).all(1))
        if len(had):
            prev_n = out['normal'][had[-1]]
    t0 = time.time()
    last_save = start_at
    i = start_at
    while True:
        i += 1
        grey = reader.read()
        if grey is None or i > count + gap:
            break
        cur = flow.upload(grey)

        # One LK call for every live chain at once. They all run prev -> cur, and
        # each call would otherwise rebuild the same two image pyramids.
        live = [ch for ch in chains if len(ch[2]) >= 60]
        for ch in chains:
            if len(ch[2]) < 60:
                ch[2] = ch[2][:0]
        if live:
            sizes = [len(ch[2]) for ch in live]
            p1, good = flow.track(prev, cur, np.concatenate([ch[2] for ch in live]))
            off = 0
            for ch, n in zip(live, sizes):
                g = good[off:off + n]
                ch[1] = ch[1][g]
                ch[2] = p1[off:off + n][g]
                off += n
        for ch in chains:
            if i - ch[0] == 1 and ch[0] < count and len(ch[1]) >= 60:
                vals = affine_fit(ch[1], ch[2], cx, cy, f_small)
                for k, v in zip(AFFINE_KEYS, vals):
                    out[k][ch[0]] = v
                out['npts'][ch[0]] = len(ch[1])
                # The same fit in undistorted pinhole coordinates. In raw fisheye
                # pixels a pan bends the flow and leaks into the curl (measured:
                # 30-40 % of the "roll" on this lens); in normalised coordinates
                # a pan is a uniform shift plus symmetric terms, so the curl is
                # roll alone, and the shift is an angle without a lens factor.
                na, nb = lens.to_normalised(ch[1]), lens.to_normalised(ch[2])
                vals_n = affine_fit(na, nb, 0.0, 0.0, 1.0)
                for k, v in zip(AFFINE_N_KEYS, vals_n):
                    out[k][ch[0]] = v
                R1, _, _ = rotation_kabsch(rays_from_normalised(na), rays_from_normalised(nb))
                out['kabsch1_deg'][ch[0]] = np.degrees(cv2.Rodrigues(R1)[0].ravel())
        while chains and i - chains[0][0] >= gap:
            start, a, b = chains.popleft()
            if start >= count or len(a) < min_inliers:
                continue
            sol = solve_window(a, b, lens, focal_px, f_small, prev_n, min_inliers)
            if sol is None:
                continue
            prev_n = sol['normal']
            out['homog_deg'][start] = sol['homog']
            out['kabsch_deg'][start] = sol['kabsch']
            out['normal'][start] = sol['normal']
            out['npts_win'][start] = sol['npts']
            out['resid_h_px'][start] = sol['resid_h']
            out['resid_r_px'][start] = sol['resid_r']
        if i < count:
            seed(cur, i)
        prev = cur
        if i - last_save >= save_every:
            save()
            last_save = i
            if progress:
                el = time.time() - t0
                rate = (i - start_at) / el
                print('  %d/%d frames, %.2f s/frame, ETA %.0f s' % (i, count, 1 / rate, (count - i) / rate), flush=True)
    reader.close()
    save()
    if progress:
        print('done: %d frames with affine, %d windows with rotation -> %s'
              % (np.isfinite(out['roll_deg']).sum(), np.isfinite(out['homog_deg']).all(1).sum(), out_path), flush=True)
    return out, meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--scale', type=float, default=0.25)
    ap.add_argument('--gap', type=int, default=5)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=None)
    ap.add_argument('--max-pts', type=int, default=1500)
    ap.add_argument('--lens', help='Gyroflow lens profile JSON to use instead of the clip metadata model')
    ap.add_argument('--backend', choices=('auto', 'gpu', 'cpu'), default='cpu',
                    help='cpu (default, the reference numbers); gpu: OpenCL, ~2x faster, '
                         'shifts the window estimates; auto: gpu when there is one')
    ap.add_argument('--win', type=int, default=None, help='LK window, default 31 on the CPU and 21 on the GPU')
    ap.add_argument('--prefetch', type=int, default=8, help='frames the decoding thread may run ahead')
    a = ap.parse_args()
    lens = None
    if a.lens:
        import cv2 as _cv2
        from lenscal import load_gyroflow_lens
        cap = _cv2.VideoCapture(a.video)
        w, h = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        lens = load_gyroflow_lens(a.lens, w, h)
    measure(a.video, a.out, scale=a.scale, gap=a.gap, first=a.first, count=a.count,
            max_pts=a.max_pts, lens=lens, backend=a.backend, win=a.win, prefetch=a.prefetch)

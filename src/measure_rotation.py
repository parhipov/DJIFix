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

    python src/measure_rotation.py "F:\\36\\clip.MP4" -o artifacts/main/clip/clip_image.npz
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imagerot import Lens, rotation_kabsch, _pick_decomposition  # noqa: E402

AFFINE_KEYS = ('roll_deg', 'div', 'shear', 'shift_x_deg', 'shift_y_deg')
AFFINE_N_KEYS = ('roll_n_deg', 'div_n', 'shear_n', 'shift_xn_deg', 'shift_yn_deg')


def affine_fit(a, b, cx, cy, f_small):
    """v = A p + b on tracked points; returns (roll_deg, div, shear, sx_deg, sy_deg)."""
    p = a - np.array([cx, cy])
    v = b - a
    M = np.zeros((2 * len(p), 6))
    M[0::2, 0] = p[:, 0]; M[0::2, 1] = p[:, 1]; M[0::2, 4] = 1
    M[1::2, 2] = p[:, 0]; M[1::2, 3] = p[:, 1]; M[1::2, 5] = 1
    y = v.reshape(-1)
    wt = np.ones(len(y))
    sol = None
    for _ in range(6):
        W = wt[:, None]
        sol, *_ = np.linalg.lstsq(M * W, y * wt, rcond=None)
        r = np.abs(M @ sol - y)
        s = max(1.4826 * np.median(r), 1e-9)
        wt = 1.0 / np.sqrt(1.0 + (r / (2 * s)) ** 2)
    a11, a12, a21, a22, bx, by = sol
    return (np.degrees(0.5 * (a21 - a12)), 0.5 * (a11 + a22), 0.5 * (a11 - a22),
            np.degrees(bx / f_small), np.degrees(by / f_small))


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
            fb_err=0.5, save_every=200, progress=True, lens=None):
    """lens: optional dict(f, cx, cy, D, name) overriding the clip's own lens model,
    e.g. from lenscal.load_gyroflow_lens (the profile you pick in Gyroflow)."""
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
                axes='OpenCV camera: x right, y down, z forward; rotation of rays from frame f to f+gap')

    start_at = 0
    if os.path.exists(out_path):
        try:
            prev_d = np.load(out_path)
            pm = json.loads(str(prev_d['meta_json']))
            same = all(pm.get(k) == meta[k] for k in ('video', 'first', 'count', 'gap', 'scale', 'max_pts'))
            if same:
                done = int(prev_d['homog_deg'].shape[0])
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

    cap.set(cv2.CAP_PROP_POS_FRAMES, first + start_at)
    if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (first + start_at)) > .1:
        cap.release()
        cap = cv2.VideoCapture(video)
        for _ in range(first + start_at):
            if not cap.grab():
                raise SystemExit('cannot seek')
    ok, img = cap.read()
    if not ok:
        raise SystemExit('cannot read frame %d' % (first + start_at))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    prev = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = prev.shape
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < (radius_frac * min(w, h)) ** 2).astype(np.uint8) * 255
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)

    chains = deque()   # [start_index, start_pts, cur_pts]

    def seed(grey, index):
        p = cv2.goodFeaturesToTrack(grey, max_pts, 0.01, 10, mask=mask, blockSize=7)
        if p is not None and len(p) >= 60:
            p = p.reshape(-1, 2).astype(np.float32)
            chains.append([index, p.copy(), p.copy()])

    seed(prev, start_at)
    prev_n = None
    t0 = time.time()
    last_save = start_at
    i = start_at
    while True:
        i += 1
        ok, img = cap.read()
        if not ok or i > count + gap:
            break
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cur = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        for ch in chains:
            p0 = ch[2]
            if len(p0) < 60:
                ch[2] = p0[:0]
                continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0.reshape(-1, 1, 2), None,
                                                 winSize=(31, 31), maxLevel=4, criteria=crit)
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None,
                                                   winSize=(31, 31), maxLevel=4, criteria=crit)
            g = (st.ravel() == 1) & (st2.ravel() == 1)
            g &= np.linalg.norm(p0 - p0b.reshape(-1, 2), axis=1) < fb_err
            ch[1] = ch[1][g]
            ch[2] = p1.reshape(-1, 2)[g]
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
                R1, _, _ = rotation_kabsch(lens.to_rays(ch[1]), lens.to_rays(ch[2]))
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
    cap.release()
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
    a = ap.parse_args()
    lens = None
    if a.lens:
        import cv2 as _cv2
        from lenscal import load_gyroflow_lens
        cap = _cv2.VideoCapture(a.video)
        w, h = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        lens = load_gyroflow_lens(a.lens, w, h)
    measure(a.video, a.out, scale=a.scale, gap=a.gap, first=a.first, count=a.count, max_pts=a.max_pts, lens=lens)

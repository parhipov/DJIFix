"""
Measure the camera's real inter-frame rotation from the image, so the telemetry
can be checked against ground truth instead of against itself.

Features are tracked with pyramidal Lucas-Kanade, lifted to unit rays through the
lens model DJI itself provides (equidistant fisheye, f = 1457.07 px, the four
distortion coefficients from the clip metadata -- the same model Gyroflow uses),
then the rotation is recovered from the essential matrix so that forward
translation and parallax are modelled rather than mistaken for rotation.

Only the central part of the frame is used: normalising a 155 deg fisheye onto a
pinhole plane blows up towards the edge.
"""
import numpy as np
import cv2


class Lens:
    def __init__(self, f=1457.07373046875, cx=1920.0, cy=1440.0,
                 D=(0.15513110160827637, 0.1371408998966217,
                    -0.0938614010810852, 0.0041704000905156136), scale=1.0):
        self.K = np.array([[f * scale, 0, cx * scale],
                           [0, f * scale, cy * scale],
                           [0, 0, 1.0]])
        self.D = np.array(D, dtype=np.float64).reshape(4, 1)

    def to_normalised(self, pts):
        """(N,2) pixel coords -> (N,2) pinhole-normalised coords."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.fisheye.undistortPoints(p, self.K, self.D)
        return out.reshape(-1, 2)

    def to_rays(self, pts):
        """(N,2) pixel coords -> (N,3) unit rays."""
        n = self.to_normalised(pts)
        r = np.column_stack([n[:, 0], n[:, 1], np.ones(len(n))])
        return r / np.linalg.norm(r, axis=1, keepdims=True)


def rotation_kabsch(a, b, iters=6, huber=None):
    """Rotation R with R @ a ~= b for unit-ray sets, robustly.

    An essential-matrix solve is degenerate here: the inter-frame baseline is
    about a pixel and a half, so triangulation cannot pick a solution. A pure
    rotation fit is well conditioned, and forward translation shows up as a
    radial flow field, which is close to orthogonal to the rotation basis over a
    symmetric field of view, so it mostly averages out. Outliers (moving
    objects, near ground with strong parallax) are handled by reweighting.
    """
    w = np.ones(len(a))
    R = np.eye(3)
    for it in range(iters):
        H = (b * w[:, None]).T @ (a * w[:, None])
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        res = np.linalg.norm((R @ a.T).T - b, axis=1)
        sc = huber if huber else max(1.4826 * np.median(res), 1e-9)
        w = 1.0 / np.sqrt(1.0 + (res / (2.0 * sc)) ** 2)
    return R, res, w


def frame_rotations(video, first, count, scale=0.5, max_pts=1200,
                    radius_frac=0.55, min_inliers=40, progress=False):
    """Rotation vectors (deg) between consecutive frames, OpenCV camera axes.

    Returns (frames, rvecs, inliers) where rvecs[i] takes frame first+i to
    first+i+1. NaN rows mean the estimate did not converge.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit('cannot open %s' % video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    lens = Lens(scale=scale)

    ok, img = cap.read()
    if not ok:
        raise SystemExit('cannot read frame %d' % first)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    prev = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = prev.shape
    cxs, cys = w / 2.0, h / 2.0
    rmax = radius_frac * min(cxs, cys) * 2

    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - cxs) ** 2 + (yy - cys) ** 2) < rmax ** 2).astype(np.uint8) * 255

    rvecs = np.full((count, 3), np.nan)
    inl = np.zeros(count, dtype=int)
    for i in range(count):
        ok, img = cap.read()
        if not ok:
            break
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cur = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(prev, max_pts, 0.01, 12, mask=mask, blockSize=7)
        if p0 is not None and len(p0) >= min_inliers:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev, cur, p0, None, winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(
                cur, prev, p1, None, winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
            good = (st.ravel() == 1) & (st2.ravel() == 1)
            good &= np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1) < 1.0
            a = p0.reshape(-1, 2)[good]
            b = p1.reshape(-1, 2)[good]
            if len(a) >= min_inliers:
                ra = lens.to_rays(a)
                rb = lens.to_rays(b)
                R, res, wts = rotation_kabsch(ra, rb)
                rvecs[i] = np.degrees(cv2.Rodrigues(R)[0].ravel())
                inl[i] = int((wts > 0.5).sum())
        prev = cur
        if progress and i % 25 == 0:
            print('   frame %d/%d' % (i, count), flush=True)
    cap.release()
    return np.arange(first, first + count), rvecs, inl


def _track(prev, cur, mask, max_pts, fb_err=0.5):
    """Forward-backward checked Lucas-Kanade tracks between two grey frames."""
    p0 = cv2.goodFeaturesToTrack(prev, max_pts, 0.01, 10, mask=mask, blockSize=7)
    if p0 is None or len(p0) < 60:
        return None, None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(31, 31), maxLevel=4, criteria=crit)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, winSize=(31, 31), maxLevel=4, criteria=crit)
    g = (st.ravel() == 1) & (st2.ravel() == 1)
    g &= np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1) < fb_err
    if g.sum() < 60:
        return None, None
    return p0.reshape(-1, 2)[g], p1.reshape(-1, 2)[g]


def _pick_decomposition(Rs, ns, prior_rvec=None, prev_n=None):
    """Choose one of decomposeHomographyMat's candidates.

    The two physically valid solutions differ in how much of the flow is called
    parallax and how much rotation. Temporal continuity of the plane normal picks
    the right one nearly always; a rotation prior (the telemetry) only breaks the
    tie when there is no history. The prior never enters the measured value.
    """
    best, best_cost = None, None
    for R, n in zip(Rs, ns):
        n = np.asarray(n).ravel()
        # the scene is in front of the camera: normal points towards it (z > 0 in OpenCV)
        if n[2] < 0:
            n = -n
        rvec = np.degrees(cv2.Rodrigues(np.asarray(R))[0].ravel())
        cost = 0.0
        if prev_n is not None:
            cost += 1.0 - float(np.dot(n, prev_n))
        if prior_rvec is not None:
            cost += 0.02 * float(np.linalg.norm(rvec - prior_rvec))
        if best is None or cost < best_cost:
            best, best_cost = (rvec, n), cost
    return best


def homography_rotations(video, first, count, scale=0.25, max_pts=1500, radius_frac=0.6,
                         focal_px=1457.07373046875, distortion=None, size=None,
                         prior_rvecs=None, progress=None, min_inliers=80):
    """Three-axis camera rotation between consecutive frames, from a plane homography.

    Flying low over the ground, the inter-frame flow is a homography
    H = R + t n^T / d of the ground plane. Decomposing it with the lens model
    separates the rotation R from the translation t (which a pure-rotation fit
    would partly read as pitch/yaw). Points are lifted through DJI's own fisheye
    model before the fit, so the homography lives in normalised coordinates.

    Returns dict of arrays (length `count`): rvec_deg (N,3) rotation from frame f
    to f+1 in OpenCV camera axes, npts (inliers), resid_px (median reprojection
    error in source pixels), kabsch_deg (N,3) pure-rotation fit for comparison,
    normal (N,3) chosen plane normal. NaN where no estimate.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit('cannot open %s' % video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - first) > .1:
        cap.release()
        cap = cv2.VideoCapture(video)
        for _ in range(first):
            if not cap.grab():
                raise SystemExit('cannot seek to frame %d' % first)
    ok, img = cap.read()
    if not ok:
        raise SystemExit('cannot read frame %d' % first)
    H0, W0 = img.shape[:2]
    if size is None:
        size = (W0, H0)
    lens = Lens(f=focal_px, cx=size[0] / 2.0, cy=size[1] / 2.0,
                D=distortion if distortion is not None else Lens().D.ravel(), scale=scale)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    prev = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = prev.shape
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < (radius_frac * min(w, h)) ** 2).astype(np.uint8) * 255
    f_small = focal_px * scale

    out = {'rvec_deg': np.full((count, 3), np.nan), 'kabsch_deg': np.full((count, 3), np.nan),
           'normal': np.full((count, 3), np.nan), 'npts': np.zeros(count, dtype=int),
           'resid_px': np.full(count, np.nan)}
    prev_n = None
    for i in range(count):
        ok, img = cap.read()
        if not ok:
            break
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cur = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        a, b = _track(prev, cur, mask, max_pts)
        if a is not None:
            na, nb = lens.to_normalised(a), lens.to_normalised(b)
            Hm, inl = cv2.findHomography(na, nb, cv2.RANSAC, 1.5 / f_small, maxIters=500, confidence=0.995)
            if Hm is not None and inl is not None and inl.sum() >= min_inliers:
                inl = inl.ravel().astype(bool)
                Hm = Hm / Hm[2, 2]
                proj = cv2.perspectiveTransform(na[inl].reshape(-1, 1, 2), Hm).reshape(-1, 2)
                resid = np.median(np.linalg.norm(proj - nb[inl], axis=1)) * focal_px
                nsol, Rs, ts, ns = cv2.decomposeHomographyMat(Hm, np.eye(3))
                prior = None if prior_rvecs is None else prior_rvecs[i]
                rvec, n = _pick_decomposition(Rs, ns, prior, prev_n)
                prev_n = n
                out['rvec_deg'][i] = rvec
                out['normal'][i] = n
                out['npts'][i] = int(inl.sum())
                out['resid_px'][i] = resid
                ra = lens.to_rays(a[inl])
                rb = lens.to_rays(b[inl])
                R, _, _ = rotation_kabsch(ra, rb)
                out['kabsch_deg'][i] = np.degrees(cv2.Rodrigues(R)[0].ravel())
        prev = cur
        if progress and i % 50 == 0:
            progress(i, count)
    cap.release()
    return out


def chain_rotations(video, first, count, gap=5, scale=0.25, max_pts=1500, radius_frac=0.6,
                    focal_px=1457.07373046875, distortion=None, size=None,
                    progress=None, min_inliers=80, fb_err=0.5, parallax_ratio=1.3):
    """Like `homography_rotations` but over `gap` frames, with tracks chained frame
    by frame so fast motion does not break the optical flow.

    Between consecutive frames the baseline is ~1.5 px and a homography cannot tell
    a small rotation from a small translation over the plane -- the normal jumps
    around and the split is noise. Over 5 frames the perspective terms are 5x
    larger and the decomposition settles. The result at index f is the rotation
    from frame f to frame f+gap; entries with no estimate are NaN.

    Returns dict: rvec_deg, kabsch_deg (N,3), normal (N,3), npts, resid_px.
    """
    from collections import deque
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit('cannot open %s' % video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - first) > .1:
        cap.release()
        cap = cv2.VideoCapture(video)
        for _ in range(first):
            if not cap.grab():
                raise SystemExit('cannot seek to frame %d' % first)
    ok, img = cap.read()
    if not ok:
        raise SystemExit('cannot read frame %d' % first)
    if size is None:
        size = (img.shape[1], img.shape[0])
    lens = Lens(f=focal_px, cx=size[0] / 2.0, cy=size[1] / 2.0,
                D=distortion if distortion is not None else Lens().D.ravel(), scale=scale)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    prev = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = prev.shape
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < (radius_frac * min(w, h)) ** 2).astype(np.uint8) * 255
    f_small = focal_px * scale
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)

    out = {'rvec_deg': np.full((count, 3), np.nan), 'kabsch_deg': np.full((count, 3), np.nan),
           'homog_deg': np.full((count, 3), np.nan),
           'normal': np.full((count, 3), np.nan), 'npts': np.zeros(count, dtype=int),
           'resid_px': np.full(count, np.nan), 'kabsch_resid_px': np.full(count, np.nan),
           'used_homography': np.zeros(count, dtype=bool)}
    chains = deque()   # each: [start_index, start_pts, cur_pts]
    prev_n = None

    def seed(grey, index):
        p = cv2.goodFeaturesToTrack(grey, max_pts, 0.01, 10, mask=mask, blockSize=7)
        if p is not None and len(p) >= 60:
            p = p.reshape(-1, 2).astype(np.float32)
            chains.append([index, p.copy(), p.copy()])

    seed(prev, 0)
    for i in range(1, count + gap + 1):
        ok, img = cap.read()
        if not ok:
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
        while chains and i - chains[0][0] >= gap:
            start, a, b = chains.popleft()
            if start >= count or len(a) < min_inliers:
                continue
            na, nb = lens.to_normalised(a), lens.to_normalised(b)
            Hm, inl = cv2.findHomography(na, nb, cv2.RANSAC, 1.5 / f_small, maxIters=500, confidence=0.995)
            if Hm is None or inl is None or inl.sum() < min_inliers:
                continue
            inl = inl.ravel().astype(bool)
            Hm = Hm / Hm[2, 2]
            proj = cv2.perspectiveTransform(na[inl].reshape(-1, 1, 2), Hm).reshape(-1, 2)
            res_h = np.median(np.linalg.norm(proj - nb[inl], axis=1)) * focal_px   # full-res px
            out['resid_px'][start] = res_h
            _, Rs, _, ns = cv2.decomposeHomographyMat(Hm, np.eye(3))
            rvec, n = _pick_decomposition(Rs, ns, None, prev_n)
            prev_n = n
            out['homog_deg'][start] = rvec
            out['normal'][start] = n
            out['npts'][start] = int(inl.sum())
            ra, rb = lens.to_rays(a[inl]), lens.to_rays(b[inl])
            R, res_r, _ = rotation_kabsch(ra, rb)
            kab = np.degrees(cv2.Rodrigues(R)[0].ravel())
            out['kabsch_deg'][start] = kab
            # pure-rotation residual, unit-ray distance ~ radians -> full-res px
            res_r_px = np.median(res_r) * focal_px
            out['kabsch_resid_px'][start] = res_r_px
            # Model selection: when rotation alone explains the flow to within the
            # tracking noise, the plane decomposition is ill-conditioned and the
            # pure-rotation fit is the better estimate; otherwise parallax is real
            # and only the homography separates it from the rotation.
            use_h = res_r_px > parallax_ratio * res_h
            out['rvec_deg'][start] = rvec if use_h else kab
            out['used_homography'][start] = use_h
        if i < count:
            seed(cur, i)
        prev = cur
        if progress and i % 50 == 0:
            progress(i, count)
    cap.release()
    return out


AXIS_MAPS = None


def best_axis_map(a, b):
    """Find the signed axis permutation mapping measurement `a` onto `b`.

    Both are (N,3) rotation-vector series; this avoids having to guess how
    OpenCV's camera axes line up with Gyroflow's.
    """
    import itertools
    ok = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
    a, b = a[ok], b[ok]
    best = None
    for perm in itertools.permutations(range(3)):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    s = np.array([sx, sy, sz], dtype=float)
                    c = a[:, perm] * s
                    err = np.linalg.norm(c - b, axis=1).mean()
                    if best is None or err < best[0]:
                        best = (err, perm, (sx, sy, sz))
    return best


def affine_flow(video, first, count, scale=0.5, max_pts=1500,
                radius_frac=0.6, progress=False, focal_px=1457.07373046875):
    """Per-frame-pair affine flow decomposition, in the image plane.

    Fitting  v = A p + b  and splitting A into divergence / curl / shear
    separates the camera's roll from forward translation: roll produces a purely
    tangential (curl) field, while flying forward over a scene at any depth
    produces a radial (divergence) field plus shear plus a shift. So the curl
    term is a roll measurement that parallax cannot fake, which matters here --
    the drone is low over grass and the parallax is large.

    Returns dict of arrays: roll_deg, div, shear, shift_x_deg, shift_y_deg, npts.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit('cannot open %s' % video)

    # Some Gyroflow exports have a broken/missing seek index: asking for frame N
    # can silently land on an earlier undecodable frame. Fall back to decoding
    # from the beginning; slow for a late interval, but deterministic.
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - first) > .1:
        cap.release()
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise SystemExit('cannot reopen %s' % video)
        for frame in range(first):
            if not cap.grab():
                cap.release()
                raise SystemExit('cannot seek to frame %d' % first)

    ok, img = cap.read()
    if not ok:
        raise SystemExit('cannot read frame %d' % first)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    prev = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = prev.shape
    cx, cy = w / 2.0, h / 2.0
    f = focal_px * scale
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < (radius_frac * min(w, h)) ** 2).astype(np.uint8) * 255

    out = {k: np.full(count, np.nan) for k in
           ('roll_deg', 'div', 'shear', 'shift_x_deg', 'shift_y_deg')}
    out['npts'] = np.zeros(count, dtype=int)

    for i in range(count):
        ok, img = cap.read()
        if not ok:
            break
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cur = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev, max_pts, 0.01, 10, mask=mask, blockSize=7)
        if p0 is not None and len(p0) >= 60:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                prev, cur, p0, None, winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(
                cur, prev, p1, None, winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
            g = (st.ravel() == 1) & (st2.ravel() == 1)
            g &= np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1) < 0.5
            a = p0.reshape(-1, 2)[g]
            b = p1.reshape(-1, 2)[g]
            if len(a) >= 60:
                p = a - np.array([cx, cy])
                v = b - a
                # v = [[a11,a12],[a21,a22]] p + [bx,by]
                M = np.zeros((2 * len(p), 6))
                M[0::2, 0] = p[:, 0]; M[0::2, 1] = p[:, 1]; M[0::2, 4] = 1
                M[1::2, 2] = p[:, 0]; M[1::2, 3] = p[:, 1]; M[1::2, 5] = 1
                y = v.reshape(-1)
                wt = np.ones(len(y))
                for _ in range(6):
                    W = wt[:, None]
                    sol, *_ = np.linalg.lstsq(M * W, y * wt, rcond=None)
                    r = np.abs(M @ sol - y)
                    s = max(1.4826 * np.median(r), 1e-9)
                    wt = 1.0 / np.sqrt(1.0 + (r / (2 * s)) ** 2)
                a11, a12, a21, a22, bx, by = sol
                out['roll_deg'][i] = np.degrees(0.5 * (a21 - a12))
                out['div'][i] = 0.5 * (a11 + a22)
                out['shear'][i] = 0.5 * (a11 - a22)
                out['shift_x_deg'][i] = np.degrees(bx / f)
                out['shift_y_deg'][i] = np.degrees(by / f)
                out['npts'][i] = len(a)
        prev = cur
        if progress and i % 50 == 0:
            print('   %d/%d' % (i, count), flush=True)
    cap.release()
    return out

"""Per-pair residual motion of a stabilized render: shift x/y (px, full res) and roll (deg)."""
import cv2, numpy as np, sys
def measure(path, scale=0.25):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS)
    ok, f = cap.read(); H, W = f.shape[:2]
    g0 = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g1 = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(g0, 1500, 0.01, 8)
        if p0 is None or len(p0) < 30:
            out.append((np.nan,) * 3); g0 = g1; continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(31, 31), maxLevel=4)
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(31, 31), maxLevel=4)
        good = (st.ravel() == 1) & (stb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.5)
        a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        if len(a) < 30:
            out.append((np.nan,) * 3); g0 = g1; continue
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if M is None: out.append((np.nan,) * 3)
        else:
            # rotation about the image centre: shift of the centre point
            c = np.array([W * scale / 2, H * scale / 2])
            cc = M[:, :2] @ c + M[:, 2] - c
            out.append((cc[0] / scale, cc[1] / scale, np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        g0 = g1
    return fps, W, H, np.array(out)
if __name__ == '__main__':
    for p in sys.argv[1:]:
        fps, W, H, r = measure(p)
        np.savez(p.replace('.mp4', '_resid.npz'), fps=fps, W=W, H=H, r=r)
        print(p, fps, W, H, len(r))

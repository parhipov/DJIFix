"""Direct f->f+k pure-rotation estimates (no chaining) for k in GAPS, on one clip."""
import sys, json, numpy as np, cv2
sys.path.insert(0, 'K:/Work/Python/DJI fix/src')
from imagerot import Lens, rotation_kabsch
video, npz, out = sys.argv[1:4]
meta = json.loads(str(np.load(npz)['meta_json']))
scale = 0.25
lens = Lens(f=meta['focal_px'], cx=meta.get('cx', meta['width'] / 2), cy=meta.get('cy', meta['height'] / 2), D=meta['distortion'], scale=scale)
GAPS = (1, 2, 3, 4, 6, 8)
cap = cv2.VideoCapture(video); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
n = len(frames); R = {k: np.full((n, 3), np.nan) for k in GAPS}; NP = {k: np.zeros(n) for k in GAPS}
h, w = frames[0].shape; mask = np.zeros((h, w), np.uint8); cv2.circle(mask, (w // 2, h // 2), int(0.6 * min(h, w)), 255, -1)
for i in range(n):
    p0 = cv2.goodFeaturesToTrack(frames[i], 1500, 0.01, 10, mask=mask)
    if p0 is None: continue
    for k in GAPS:
        if i + k >= n: continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(frames[i], frames[i + k], p0, None, winSize=(31, 31), maxLevel=5)
        pb, sb, _ = cv2.calcOpticalFlowPyrLK(frames[i + k], frames[i], p1, None, winSize=(31, 31), maxLevel=5)
        g = (st.ravel() == 1) & (sb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.5)
        if g.sum() < 60: continue
        a, b = lens.to_rays(p0.reshape(-1, 2)[g]), lens.to_rays(p1.reshape(-1, 2)[g])
        Rm, _, _ = rotation_kabsch(a, b)
        R[k][i] = np.degrees(cv2.Rodrigues(Rm)[0].ravel()); NP[k][i] = g.sum()
np.savez(out, **{'R%d' % k: R[k] for k in GAPS}, **{'N%d' % k: NP[k] for k in GAPS})
print('frames', n, {k: int(np.isfinite(R[k][:, 0]).sum()) for k in GAPS}, {k: int(np.median(NP[k][NP[k] > 0])) for k in GAPS})

import cv2, numpy as np, sys
sys.path.insert(0, 'scripts')
from residual import measure
# measure a time window of a render with two tracker settings
import residual
def seg(path, t0, t1, scale):
    cap = cv2.VideoCapture(path); fps = cap.get(5); cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    frames = []
    for _ in range(int((t1 - t0) * fps)):
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    tmp = 'full/_seg.mp4'
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    for f in frames: vw.write(f)
    vw.release()
    return measure(tmp.replace('.mp4', '.avi') if False else tmp, scale=scale)
for name in sys.argv[1:]:
    for sc in (0.5, 1.0):
        fps, W, H, r = seg('full/pro_%s_stab.mp4' % name, 52, 56, sc)
        r = np.nan_to_num(r); roll = np.radians(r[:, 2]) * 1920
        print(name, 'scale', sc, 'per-frame roll step rms px(4K)', round(float(np.diff(roll).std() * 2 / np.sqrt(2)), 2), ' x', round(float(np.diff(r[:, 0]).std() * 2 / np.sqrt(2)), 2))

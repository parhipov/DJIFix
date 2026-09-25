import cv2, sys, numpy as np
clip, t0, t1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
names = ('orig', 'fixed', 'R')
caps = [cv2.VideoCapture('full/%s_%s_stab.mp4' % (clip, n)) for n in names]
fps = caps[0].get(5)
for c in caps:
    for _ in range(int(t0 * fps)): c.grab()
W, H = 1280, 720
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H * 3 // 1 if False else W * 0 + W, ) if False else (W * 3 // 2, H * 3 // 2))
# layout: three panels of 640x360 side by side would be small; use a 2x2 grid with the centre crop magnified
vw.release()
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 1080))
for i in range(int((t1 - t0) * fps)):
    fr = []
    for c in caps:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    if len(fr) < 3: break
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    for k, (f, n) in enumerate(zip(fr, names)):
        full = cv2.resize(f, (960, 540))
        crop = f[270:810, 480:1440]                   # centre of the 1080p render, 1:1 pixels
        x0 = (k % 2) * 960; y0 = (k // 2) * 540
        panel = full if k < 3 else crop
        canvas[y0:y0 + 540, x0:x0 + 960] = panel
        cv2.putText(canvas, n, (x0 + 20, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    # fourth quadrant: difference fixed-orig magnitude is not useful; show the centre crop of R at 1:1
    canvas[540:1080, 960:1920] = fr[2][270:810, 480:1440]
    cv2.putText(canvas, 'R, centre 1:1', (980, 590), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(canvas, '%.2f s' % (t0 + i / fps), (1700, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    vw.write(canvas)
vw.release()
print(out)

# Example 2: DJI O4 Lite with the Flywoo O4 Wide lens, a left‑right yank

**English** | [Русский](README.ru.md)

`o4lite_53-58s.MP4` — 5.4 s (52.8–58.2 s) out of
`54-56-DJI_20260517170526_0020_D.MP4`, 3840×2880, 50 fps, HEVC 10‑bit, cut
without re‑encoding together with its telemetry. On this section the
Gyroflow‑stabilized video jerks sharply sideways at about 55.7 s (2.9 s into the
piece) even though the flight itself is smooth.

## Running it

The clip `o4lite_53-58s.MP4` (68 MB) is not in the repository but on the Releases
page; download it into this folder, then

```bat
run.bat
```

that is, `fix_telemetry.bat o4lite_53-58s.MP4 --lens flywoo --plots`. The lens
profile is mandatory for this camera: the telemetry describes the stock lens
while a Flywoo O4 Wide is fitted (the profile is in `lens_profiles/`, the same
one you would pick in Gyroflow). The first run measures the image (~1.5 min),
then assembles `o4lite_53-58s_telemetry_fixed.mp4`; load it into Gyroflow as
motion data.

## What was wrong and what was done

**Single‑frame yaw spikes in the telemetry.** In two frames DJI's fused attitude
reports a rotation the camera never made. The image (measured from the video)
confirms this. Values in degrees per frame; the times are those of the full clip
(subtract 52.8 s for the piece):

| time | telemetry | image | fixed |
|---|---|---|---|
| 55.70 | +0.05 | +0.31 | +0.17 |
| **55.72** | **+1.76** | +0.24 | +0.03 |
| 55.74 | −0.12 | +0.11 | +0.03 |
| 55.80 | −0.24 | +0.12 | +0.05 |
| **55.82** | **+0.68** | +0.17 | 0.00 |
| 55.84 | +0.03 | +0.19 | +0.08 |

A jump of +1.76° in one frame is 88 °/s that never happened; Gyroflow honestly
rotated the frame by those degrees, hence the yank. The spikes were found from
the telemetry itself (deviation from the local median), confirmed against the
image and returned to the median before all the filters. The neighbouring frames
are untouched.

**The 55.28–56.08 s event.** Around the spikes the telemetry disagrees with the
image in shape too (pitch of 0.9°/frame in the telemetry against 0.2–0.4 from the
image over 55.5–55.7 s). Inside the window the pitch/yaw shape was replaced by
the per‑frame shape from the image with a zero total over the window; up to 2.1°
was applied.

**Roll.** Not touched: the roll defect is not measured on this unit (its
telemetry-vs-image roll error above 4 Hz stays at the level of clean clips), so
the roll is left exactly as DJI wrote it; jitter 0.038 °/frame against an image
floor of 0.034.

The image's verdict for the whole piece: the pitch/yaw disagreement (99th
percentile) went 0.61 → 0.46 °/frame; the Gyroflow CLI reads the file exactly as
it was written (discrepancy 0.001°).

## What is not perfect

- The 55.28–56.08 s event is corrected "by shape": on this camera the image
  understates the magnitude of the rotation (parallax, the drone is low over the
  ground), so the total over the window is forced to zero and the replacement of
  the telemetry by the image inside the window is partial. The spikes themselves
  do not have this limitation.
- On the full clip 11 spikes are cut out (19.76, 25.78, 25.98, 26.10,
  37.24–37.34, 41.14, 55.72, 55.82, 91.54 s); two of them fall into this piece.
- On a short piece the calibrations (the timing constant, the Wiener curves, the
  witness channel for events) fall back to their defaults, which shows in the
  report as `corr nan` and `shift check: n/a`. On the full clip they are
  measured.

## Files

| file | what |
|---|---|
| `o4lite_53-58s.MP4` | the source piece (from Releases) |
| `run.bat` | the run, with the lens profile |
| `o4lite_53-58s_telemetry_fixed.mp4` | the result: fixed telemetry for Gyroflow |
| `o4lite_53-58s_00_control.mp4` | the same timing with the contents unchanged, for an A/B |
| `o4lite_53-58s_fix_report.json` | what was corrected and by how much, the spikes and the event with its verdict |
| `o4lite_53-58s_overview.png` | three panels of angular rates: telemetry, image, fixed; events shaded red, spikes marked by purple lines |
| `o4lite_53-58s_roll.png`, `_pitch_yaw.png`, `_correction.png` | plots: telemetry, image, fixed; the applied correction |

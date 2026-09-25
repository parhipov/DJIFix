# Example 1: DJI O4 Pro, stock lens, calm flight

**English** | [Русский](README.ru.md)

`o4pro_13-17s.MP4` — 4.8 s (12.6–17.4 s) out of `DJI_20260905181949_0005_D.MP4`,
3840×2880, 50 fps, HEVC 10‑bit, cut without re‑encoding together with its
telemetry (`src/trim_video.py`). This is a section where the Gyroflow‑stabilized
video shivers finely in roll even though the drone flies level.

## Running it

The clip `o4pro_13-17s.MP4` (73 MB) is not in the repository but on the Releases
page; download it into this folder, then

```bat
run.bat
```

that is, `fix_telemetry.bat o4pro_13-17s.MP4 --plots`. The first run measures the
image (~1.5 min for this piece), then assembles
`o4pro_13-17s_telemetry_fixed.mp4`. Load that into Gyroflow as the motion data
for the original clip.

## What was wrong and what was done

**Roll.** DJI's fused attitude trembles in roll from frame to frame while the
image is smooth. Measured on this piece (calm frames, <20 °/s, 157 frames):

| | roll jitter, °/frame | the same in px at the frame edge |
|---|---|---|
| DJI telemetry | 0.1105 | 4.6 |
| fixed telemetry | 0.0132 | 0.6 |
| real motion (from the image) | 0.0075 | 0.3 |

The roll correction: rms 0.068°, peak 0.30°. In the Gyroflow render the slow roll wobble (1–4 Hz) on the calm stretch drops from 3.7 to 2.3 px and the jitter above 4 Hz from 1.9 to 0.6 px (4K frame edge). Pitch and yaw: no telemetry spikes,
no events; the Wiener denoising above 4 Hz gave 0.023° rms. Timing: constant
shift 0 (4.8 s has too few fast frames to measure it; on the full clip it is
measured and is also 0), per‑frame exposure correction up to +1 ms.

Check: the Gyroflow CLI reads the file exactly as it was written (discrepancy
0.003°).

## What is not perfect

- The residual jitter is 0.0132 against the image floor of 0.0075. The 1–4 Hz
  part is taken from the image only on calm frames at least 1 s away from a
  manoeuvre, so the second after the fast tilt at the start keeps some of its
  wobble; below 1 Hz the telemetry is left as it is.
- On a piece this short the pipeline's calibrations (the timing constant, the
  Wiener curves, the roll axis, the witness channel) do not gather enough
  statistics and fall back to their defaults; the `o4pro_13-17s_fix_report.json`
  report shows this (`corr nan`, `default curve`). On the full clip they are
  measured.

## Files

| file | what |
|---|---|
| `o4pro_13-17s.MP4` | the source piece (from Releases) |
| `run.bat` | the run |
| `o4pro_13-17s_telemetry_fixed.mp4` | the result: fixed telemetry for Gyroflow |
| `o4pro_13-17s_00_control.mp4` | the same timing with the contents unchanged, for an A/B |
| `o4pro_13-17s_fix_report.json` | what was corrected and by how much |
| `o4pro_13-17s_roll.png` | roll: telemetry, image, fixed; the calm section magnified at the bottom |
| `o4pro_13-17s_pitch_yaw.png` | pitch/yaw: telemetry, image, fixed |
| `o4pro_13-17s_correction.png` | the applied correction per axis |
| `o4pro_13-17s_overview.png` | three panels of angular rates: telemetry, image, fixed; events shaded red, spikes marked by purple lines |

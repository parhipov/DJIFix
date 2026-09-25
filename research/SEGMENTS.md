# Problem-segment study

Two stages, kept apart on purpose:

1. **Find** every place where the stabilized result may be wrong, cut it out
   with context, and get the eye's verdict on it. Nothing is corrected.
2. **Fix**, judged on that base: a change must remove what the eye marked
   *visible* and must not touch what it marked *false alarm*, on bad and good
   clips alike.

Why: the first pipeline tuned five stages against the image as a reference.
A full-clip Gyroflow render A/B (2026-09-24, four clips incl. two the user
calls good) showed that only two of them help everywhere (roll from the image
above 4 Hz, image-confirmed spikes); the rest help on one clip and harm another.
The image is an excellent rotation sensor where the scene is rigid and lit, and
a bad one near foliage in wind, in the dark, and close to the ground. Tuning
against it tunes against its errors. The eye is the only ground truth we have.

## Data

`research/clips.json` is the registry: path, camera, drone, lens (and the
Gyroflow lens profile when the camera does not wear the lens its telemetry
describes), firmware, and what the user said about the Gyroflow result
(`eye`: bad / good / unknown). The build checks every file header against it.

## Detectors (`research/diagnose.py`)

All four are tuned for recall; a false alarm costs one 20 s clip and one click.

| type | what | threshold | why this threshold |
|---|---|---|---|
| `spike` | jump of the telemetry increment from its 5-frame median, any axis | 0.35 deg/frame, rate < 200 deg/s | the fix pipeline's value; the Lite jerk at 55.7 s is 1.7 |
| `tel_jitter` | rms of telemetry minus image above 4 Hz over 0.5 s, where the image is self-consistent | 0.05 deg | ~p90 of the good clip is 0.03 |
| `render_jerk` | rms of residual motion above 4 Hz over 0.5 s in a Gyroflow render of the original | max(4 px, 3x the clip's calm median), rate < 60 deg/s | above ~60 deg/s frames are blurred and the tracker disagrees with itself by 2x |
| `image_bad` | the image's two roll estimators (pure-rotation fit, affine curl) disagree above 4 Hz, or < 300 points | 0.03 deg | not a defect: marks where stage 2 must not trust the image |

Image gains per axis are fitted at 2-4 Hz, the band where the image reads the
rotation with gain 0.9-1.0 on every clip measured (below it parallax, above it
DJI's own jitter).

## Rules

- The first and last 5 s of every file are never looked at (take-off, landing).
- Segments: +-10 s around an event, overlapping windows merged, at most 30 s,
  inside the usable window on keyframes (a lossless cut is whole GOPs).
- Two event-free control segments per clip where the clip allows it.
- Event id = clip + source time + type, so verdicts survive a rebuild.

## Outputs

`artifacts/segments/` (not in git): per segment the lossless source cut with
its telemetry (opens in Gyroflow as is), a 1080p H.264 preview of the original
stabilized with Gyroflow's defaults, and a plot; `index.html` (time order),
`index_by_severity.html` (worst first: sum over events of severity divided by
the detector threshold, `image_bad` not counted), `index.csv`, `meta.json`
(parameters, code version, header checks). Expensive steps are cached in
`artifacts/cache/<clip>/`. `research/check_segments.py` verifies the base.

## Labelling

In either index: open the preview (or the source cut in Gyroflow), find the
event at its time in the segment, choose *видно* / *зря поймали* / *не уверен*,
optionally comment; "Export CSV" saves the verdicts.

# DJI O4 Pro telemetry / Gyroflow shake — handoff

## 0. State as of 2026-09-14 (read this first; everything below is history)

**Deliverable:** `python src/main.py <video>` (or `fix_telemetry.bat`) → `artifacts/main/<name>/<name>_telemetry_fixed.mp4`
plus `<name>_00_control.mp4` (same timing, untouched content, for A/B). Load under
Gyroflow → Motion data. Built and verified on both test clips:
`F:\DJI_20260905181949_0005_D.MP4` (O4 Pro) and `F:,-56-DJI_20260517170526_0020_D.MP4` (O4 Lite).
The user has **not yet looked at these files**; that is the next step.

Pipeline (`src/fix_pipeline.py`, fed by `src/measure_rotation.py`, judged by `src/verify_fix.py`):

1. Timing: constant measured on the clip against the image roll (+0.2 ms Pro, −1.2 ms Lite ⇒ **pts is right**),
   plus `(median exposure − exposure_f)/2` per frame. `--timing dbgi` reproduces the old alignment for A/B.
2. Roll: image roll (per-pair pure rotation in ray space, `kabsch1_deg`) replaces the telemetry roll above 4 Hz,
   rate-weighted (R0 40), axis z. Pro calm-frame jitter 0.0752 → 0.0432 deg/frame against a 0.0371 floor.
3. Pitch/yaw noise: Wiener keep-gain above 4 Hz only (the old curve's 0.93 at 1 Hz removed 7 % of real pans, up to 1.8°).
4. Pitch/yaw: (a) **telemetry spikes** — one-frame jumps of the increment away from its 5-frame median (≥ 0.35°/frame),
   confirmed by the image not showing the jump (`detect_spikes`); the frame is put back on its median **before**
   the Wiener and event stages see the telemetry (otherwise the Wiener smears the spike over its neighbours and the
   fix rings: seen on the Lite sample, ±0.4° on the frames around 55.72 s). Net kept, decays over 1.5 s. This is what the Lite 55 s jerk
   *was*: +1.77° of yaw in one frame at 55.72 s (88 deg/s that never happened), +0.65 at 55.82, +0.56 at 55.66;
   the render jerked by 0.6–1.3° on exactly those frames. Every smoothing/window approach smeared it.
   (b) longer events from the 5-frame windows (detail after a 2 s median, both estimators agree, ≥ 0.10°/frame and
   ≥ 50 % of the telemetry's own motion, ≥ 0.10 s), shaped per frame from the per-pair rotation, zero-net, edges of
   3 frames, neighbouring windows merged. Each event cross-checked against the affine shift channels; **only
   confirmed events are applied** (unclear dropped: on the Pro the unclear 9.06 s event added a ±0.4° wobble the
   user could see). No cap. User feedback 2026-09-15: Lite 55 s "стало наконец-то лучше", Pro confirmed-only "стало
   лучше". Variants tried and rejected: gain 1.5/2.0 (worse), straight-line bridge over the window (11.8° error:
   real motion inside), free net for image-shaped events (offsets piled up to 20°).

### Findings that overturn earlier sections

- **§5.3 is wrong as a timing reference.** The dbgi index sits +3.75 ms from the native read on the Pro but −5.67 ms
  on the Lite (offset field −1.6 vs +12.7), and the image puts the optimum at pts on both cameras. `01_ALIGNED`
  (+10.54 ms) was ~10 ms late, which is why it did not help. The bit-exact match in §5.3 only proved Gyroflow reads
  what we write, not that the instant was right.
- **"External data loses readout/2" described the export, not the render.** In gyroflow 1.6.3 the render samples the
  middle row at `pts` for both paths (`frame_transform.rs`: rows from pts−readout/2 to pts+readout/2); the type-3
  export stamps `pts + readout/2`. With CLI `-g` the video's own telemetry is never parsed, so `frame_readout_time`
  stays 0 (RS correction off in every CLI verification run); the GUI keeps the video's value if the video was loaded
  first. Sources: `external/gyroflow`, `external/telemetry-parser` (rev cbc61ab); telemetry-parser's DJI code has a
  commented-out `quat_ts − exposure_time` correction (1.3.0 used exposure/2).
- **The exposure-centre model is not supported either**: it predicts −3.0 ms (Pro) / +5.5 ms (Lite); measured
  +0.2 / −1.2. Only the per-frame exposure *variation* has weak support (163 Pro frames: +3.9 observed vs +4.5 predicted).
- **The Sep-12 session pipeline** (`create_telemetry_bugfix.py`) applied 1.6–10.7° context-curve edits during fast pans
  that its 0.12 s impulse metric could not see; its rates were integrated on a time grid 6.79 ms off. Both fixed
  (edits bounded, grid shifted, excursion metric added). It remains an analysis tool; the deliverable is `main.py`.

### Image measurement facts (both clips)

- Roll from the image: gain 0.89 (Pro) / 1.03 (Lite), corr 0.92 / 0.86 on fast frames; calm-frame jitter 0.040
  vs telemetry 0.074. About 25–30 % of telemetry yaw leaks into the measured roll (free-fit axis 15–19° from z); the
  pipeline regresses that leak out and corrects about z only.
- Pitch/yaw from the image **under-read by 25–50 %** (Lite x 0.52–0.64, y 0.42–0.75; Pro 0.7–0.95), differently per
  estimator and rate band. Not an axis rotation (Procrustes gains stay). Translation over the ground confounded with
  rotation. Hence the ratio rule and the 1° cap on events. Open problem.
- DJI files end with 21–37-sample blocks whose increments are thousands of deg/s; the pipeline masks them (`valid`).

### Layout (2026-09-15)

`src/` is the working code, `research/` the experiment scripts (with `research/session/` the Sep-12 render-based
pipeline), `external/` the Gyroflow sources (git-ignored, see its README). Paths in the older sections below
refer to the pre-reorganisation layout: `src/selective.py` → `research/selective.py`,
`create_telemetry_bugfix.py` → `research/session/...`, and so on. `main.py` now writes next to the video and
deletes intermediates unless `--keep`; `--artifacts` reproduces the old `artifacts/main/<name>/` behaviour.
The test Lite clip was shot with a Flywoo O4 Wide lens while the telemetry describes the stock lens. Tested
2026-09-15 (`src/lenscal.py`): with the Flywoo profile (f 1758 vs 1578, k1 0.15 vs 0.69) the image/telemetry
residual drops 0.73 → 0.56°/frame and the pitch/yaw gains rise 0.46 → 0.53 — better, but the under-read stays,
and the Pro with its stock lens shows the same (0.69/0.53). A focal-scale scan finds **no** scale that brings the
gains to 1 on either clip, so the under-read is parallax (translation over the ground confounded with rotation),
not the lens. Consequences: `--lens profile.json` exists and should be used when the lens differs;
`--fit-focal` is diagnostic only (applies the scale only if the gains reach 1; they never did here).

### Examples and lens profiles (2026-09-15)

`examples/o4pro_13-17s/` and `examples/o4lite_53-58s/`: 5 s cuts (trim_video.py, telemetry intact, ~75 MB each),
`run.bat`, the fixed telemetry, the report and three PNGs (`src/plot_report.py`, also `main.py --plots`), plus a
README with before/after numbers and what is not perfect. `lens_profiles/` holds the Flywoo O4 Wide profile
(CC0, from gyroflow/lens_profiles); `--lens flywoo` resolves by file-name substring, the image cache remembers
the lens it was measured with and re-measures if it differs.

### Files added 2026-09-14

`src/timing.py`, `src/measure_rotation.py`, `src/fix_pipeline.py`, `src/verify_fix.py`, `tests/test_fix_pipeline.py`;
`src/main.py` rewritten; `denoise.py`/`rollfix.py` take `timing=`; `imagerot.py` gained `homography_rotations`,
`chain_rotations`; `external/` holds the Gyroflow sources.

### Next steps

1. Both deliverables are user-approved as of 2026-09-15. Remaining on the Pro at 8–10 s is Gyroflow's adaptive-zoom
   breathing (fov 0.851→0.779→0.815 over 3 s) and the smoothed path following a turn — settings, not telemetry; the
   user chose not to touch zoom. Lite spikes below 0.35°/frame (25 frames at 0.25) are at the visibility limit;
   `--spike-deg 0.25` exists but adds 1–5° in fast pans with weak confirmation.
2. If pitch/yaw events under- or over-shoot: `--gain`, `--event-thr`, `--no-events`; the report lists every event with
   its shift-channel verdict.
3. The pitch/yaw scale problem is the ceiling of the image method today; fixing it needs translation modelling
   (structure) or a better parallax-robust estimator.

---


**Current delivered correction:** user authorized correcting the mapped excursion.
`src/bridge_wobble.py` writes `artifacts/experiments/lite_bridge/local_wobble_bridge.mp4` on top
of the best tested 8 Hz sidecar. Changes only 55.3–56.0 s: quintic endpoint
bridge of export yaw/roll, 0.1 s smooth edges, export pitch retained. Max changes
1.249/1.840 deg. No new timing/filter parameters. Outside records equal 8 Hz.
Gyroflow export using the same excerpt project shows output return reduced only
modestly: yaw .958->.878 deg, roll 1.348->1.165. Do not claim the wobble removed.
Visual test pending. See folder README/verification/output_comparison.

**Newest request/result:** user stopped image-flow work and asked to map the
output wobble back to original telemetry. `trace_wobble.py` completed this;
results in `artifacts/analysis/wobble_trace`. Raw yaw maximum mapped to video is
55.6364 s (original record t=55.6454801 s, block 2782 slot 24, zero-based);
raw roll minimum 55.6195 s (record t=55.6285160 s, block 2781 slot 30).
After 8 Hz extrema are ~55.64/55.60, output extrema 55.56/55.46. Inspect
55.4–56.0 with context 55–56.3. Existing 9.063943ms control mapping is used,
not re-estimated. 8 Hz barely removes this slower reversal: raw/frame yaw return
to 56 s 1.545 deg -> filtered 1.477; roll 3.197 -> 3.160. Output .958/1.348.
Do not interpret extrema displacement as a measured sync offset or declare all
motion in that interval false. No image motion analysis or new correction was
performed on this request. Match conventions/time validated against native export.

**Current experiment (explicitly authorized video render):** user proposed
stabilizing in Gyroflow and exporting the resulting motion data, hoping the
visible jerk would appear there. Completed `artifacts/analysis/gyroflow_output`:
full 4644-frame camera JSON/CSV for native, external control, 8 Hz, and render
project; a 320x240/50 fps/0.55 MB excerpt requested at 51–60 s (452 frames, 9.04s).
The full-file render was stopped as unnecessarily slow; all telemetry exports
still cover the full file. See that folder's README for results and caveats.
`stab_quat` is a calculated camera path, not new image-measured telemetry; output
MP4 has only avc1. Fast >4 Hz angular-rate RMS in calculated output at 54–56 s:
external control .809 deg/s, 8 Hz .490 deg/s. Some fast path changes remain;
not yet proven to correspond to the user-observed residual jitter.
The external CLI project has stabilization.frame_readout_time=0 despite metadata
18.128 ms. No claim about the user's GUI; deferred independent investigation.
Scripts: gyroflow_output_check.py and analyze_gyroflow_output.py.

**Newest feedback:** neither -6 nor +6 ms produced a visible change. The user
suspects a much larger timing error. Do not claim that this proves a large
offset. Next diagnostic sweep keeps 8 Hz / gain 1 / 53–58 s fixed, testing
local content offsets -100/+100 ms and -300/+300 ms (5 and 15 frames at 50 fps).
The internal-clock audit is `artifacts/experiments/timing_clock_audit.json`; header-clock
consistency cannot establish attitude fusion latency relative to the image.

**Current direction:** widening to 51–60 s made essentially no difference;
the user requests investigating precise timing. Freeze 8 Hz / gain 1.0 and the
original 53–58 s mask, test local content shifts -6 and +6 ms. On Lite 54–56 s,
dbgi matched-reference timing minus the existing readout/2 baseline has
p10/p50/p90 = -6.733/-5.677/-5.551 ms (corrected for clip.fps_ratio).
This supports testing -6 ms but does NOT prove exposure timing or real motion.
Positive shift convention: filtered(t + extra_ms/1000). Timestamps stay intact.
Outputs: `artifacts/experiments/lite_sync_minus6`, `artifacts/experiments/lite_sync_plus6`.

**Latest feedback:** gain 125/150% still leaves slight jitter. The user asks
whether the correction missed the event time. The old 53–58 s window is already
at full strength from 53.5 to 57.5 s, including the reported ~55 s. Test window
coverage independently with 51–60 s at the best tested 8 Hz / gain 1.0.
This is not a telemetry/video synchronization sweep and should not be presented
as proof of correct synchronization. Results: `artifacts/experiments/lite_wide`.

**Latest steering:** 6 Hz still leaves a small jerk. The user suspects additional
artifacts elsewhere but explicitly defers that issue; the intended final method
must fix local bugs, not globally smooth the video. The user requests testing
correction strength rather than continuing cutoff sweeps. Keep the 8 Hz filter
fixed and test gain 1.25 / 1.50. gain=1 already reaches the filtered orientation;
larger gain extrapolates the relative quaternion rotation beyond that target and
is a hypothesis test, not automatically a better repair. The shared control has
a global readout/2 content shift; outside-window equality is to control, not raw.

**Latest visual feedback:** full-rate Lite 8 Hz and 2 Hz both substantially
improved the image: the horizontal jerk almost disappeared but remains slightly.
The user prefers 8 Hz as smoother; 2 Hz may introduce nearby artifacts, but the
user is uncertain. Keep 8 Hz as the best visually tested reference, not a complete
repair. Test one intermediate 6 Hz candidate next; do not equate less telemetry
variation with better stabilization. Automatic detection remains pending.

**Previous visual feedback:** both first Lite variants still leave a horizontal
left/right displacement around 55 s (the user explicitly distinguished it from
roll around the image centre). Do not report those trials as successful.
`artifacts/experiments/lite_v2/README.md` describes the next trials from `src/fullrate_trial.py`:
all-axis quaternion smoothing at the full telemetry rate, 8 Hz and 2 Hz,
within 53–58 s. No speed gating or 1-degree cap; no video decoding/rendering.
The full-rate trials received the positive but qualified feedback above.

**Update 2026-09-11:** see `artifacts/experiments/README.md` and `src/selective.py` for the current
local A/B trials on both Lite (53–58 s) and Pro (1–5, 8–12, 13–17 s).
The user explicitly reports that the abrupt high rotation shown by telemetry at
54–56 s did NOT happen in the image. Do not treat raw or low-passed telemetry
speed as independent evidence of real maneuvers or use it to veto correction of
these labeled windows. The new trials have no rate weighting; their added
rotation is smoothly bounded to 1 degree. Visual confirmation is still pending.
The older claims below are prior experiments, not proof of the cause or a
validated universal Lite/Pro repair. Manual windows are validation examples for
an eventual automatic detector, which has not yet been implemented.

Everything a fresh session needs. Written 2026-09-07. Read this before touching
anything; `README.md` has the same facts organised by topic rather than by story.

---

## 1. The task, in the user's terms

Reference clip: `F:\36\DJI_20260905181949_0005_D.MP4` (DJI O4 Pro air unit,
3840×2880, 50 fps, HEVC 10-bit, 92.24 s, 1.44 GB).

Three steps were asked for, in order:

1. Python script to extract the telemetry — **done and verified**.
2. Change something in the telemetry — the goal turned out to be *fixing a
   defect*, see §5.
3. Save it in a form Gyroflow can stabilise with — **done and verified**.

Hard constraints the user stated:

- The changed telemetry must be a **separate file** loaded under Gyroflow's
  "Данные движения" / Motion data. Explicitly **not** written back into the video.
  (`dji_o4.py patch` can do that and works, but it is not the route they want.)
- They work with the extracted dump a lot, in Python. That is why the working
  format is `artifacts/main/*_telemetry.npz` + the `src/telemetry.py` loader, not CSV.
- They want the **mathematics**, not settings advice. They have already tried
  settings, including a low-pass filter at 8–50 Hz, and so have many other people.
- Timestamps they give are eyeballed, not exact: "я же человек, я тебе мс не выдам".
  Do not build anything that depends on their timestamps being precise.

The complaint: **the whole frame jerks after stabilization**, at roughly 0:03,
0:10, 0:15 in the source, "как будто по радиусу, +- по часовой стрелке туда сюда"
— i.e. a roll wobble about the image centre.

Ignore `F:\36\DJI_20260905181949_0005_D_stabilized.mp4` — it is the user's own
18.6 s trimmed render, they said twice it is irrelevant.

---

## 2. What is actually in the file

Three tracks:

| track | format | handler | content |
|---|---|---|---|
| 1 | `hvc1` | — | 3840×2880, 50 fps, HEVC 10-bit, 114 Mbps |
| 2 | `djmd` | `DJI meta` | telemetry, protobuf `dvtm_O4P.proto`, product proto 02.00.06 |
| 3 | `dbgi` | `DJI dbgi` | camera/ISP debug info; **also carries two attitudes per frame**, see §5.3 |

**DJI records no raw gyro or accelerometer.** The `djmd` track holds *fused
attitude quaternions* — the finished camera orientation. 40 per video frame
(`imu_sampling_rate` = 2000 Hz), but the fusion only updates at 1000 Hz so every
value appears twice. 4612 frames × 40 = 184427 samples.

Protobuf paths (field numbers, verified against `telemetry-parser`'s
`dvtm_wm169.proto`, which is the proto Gyroflow uses for this device):

```
1 clip_meta
  1.1  clip_meta_header   {1 proto_file_name, 2 library_proto_version,
                           3 product_proto_version, 5 product_sn,
                           6 product_firmware_version, 9 clip_timestamp,
                           10 product_name}
  1.3  distortion_coefficients {1 = packed float[4]}
  1.4  sensor_readout_time     {1 = varint, nanoseconds}
  1.5  sensor_read_direction
  1.8  digital_focal_length    {1 = f32, pixels}
  1.9  eis_status
  1.10 imu_sampling_rate       {1 = varint}
  1.11 sensor_fps              {1 = f32}
2 stream_meta
  2.3  video_stream_meta  {1 w, 2 h, 3 framerate, 4 is_bit_depth_valid,
                           5 bit_depth, 6 bit_format, 8 video_codec_type}
3 frame_meta
  3.1  frame_meta_header  {2 frame_timestamp, microseconds}
  3.2  camera_frame_meta  {2 exposure_index, 3 iso, 4 exposure_time (packed
                           num/den), 5 digital_zoom_ratio, 6 white_balance_cct,
                           12 temperature, 15 focal_length}
  3.3  imu_frame_meta
    3.3.2 imu_attitude_after_fusion  {1 timestamp, 2 vsync, 3 repeated
                                      Quaternion, 4 offset (f32)}
```

A Quaternion record is 4 fixed32 with tags 1..4 = **w, x, y, z**, i.e. 20 bytes
with fixed-size tags. That is why every rewrite in this project is byte-for-byte
in place: no box sizes, chunk offsets or sample tables ever change.

Values for this clip: readout 13.5811 ms top-to-bottom, focal 1457.07 px,
distortion `[0.15513, 0.13714, -0.09386, 0.0041704]`, sensor fps 50.00417,
container fps 50.0, exposure 19.6 ms (≈353° shutter), ISO 380–400, SN
`9F2KN8W01100R3`, firmware `01.00.06.00`.

Gyroflow's JSON export serialises quaternions as `[x, y, z, w]` (nalgebra coord
order). Internally this project uses `(w, x, y, z)` everywhere.

### The camera-frame transform

`telemetry-parser` turns DJI's quaternion into the one Gyroflow stabilises with:

```
q_cam = (0,0,1,0) ⊗ q_dji ⊗ (0.5,-0.5,-0.5,0.5)
```

plus a sign flip whenever consecutive values jump hemispheres (kept so the series
stays continuous). `quat.dji_to_gyroflow` / `gyroflow_to_dji` implement it, and
`dji_o4.read_telemetry` reproduces the sign-flip bookkeeping.

### The timeline

Also copied from `telemetry-parser`:

```
vsync_ms  = 1000 / sensor_fps
fps_ratio = fps / sensor_fps
t(frame f, slot i) = (frame_ts_f - frame_ts_0)/1000
                     + ((i - offset_f)/n_f) * vsync_ms,   then / fps_ratio
```

Reproduced exactly: all 184427 timestamps match Gyroflow's own export.

---

## 3. Toolchain

| file | what |
|---|---|
| `dji_o4.py` | CLI: `extract`, `dump`, `sidecar`, `gcsv`, `patch`, `verify`; also `reference_indices()` |
| `telemetry.py` | loads the `.npz`, vectorised quaternion helpers, writes a sidecar back |
| `sidecar.py` | builds the telemetry-only MP4; `time_shift_ms`, `offset_mode` |
| `align.py` | the frame↔telemetry alignment correction (§5.3) |
| `variants.py` | builds a labelled A/B set of telemetry variants |
| `imagerot.py` | image-based rotation measurement (OpenCV): `frame_rotations`, `affine_flow` |
| `measure_clip.py` | runs `affine_flow` over the whole clip, saving incrementally |
| `mp4parse.py` | minimal ISO-BMFF reader (box walk + sample tables) |
| `pb.py` | protobuf wire-format reader, no `.proto` needed |
| `quat.py` | scalar quaternion math + the DJI→Gyroflow frame conversion |
| `mp4_dump.py`, `pbdump.py` | diagnostics for other DJI models |

```bash
python src/dji_o4.py extract "F:\36\DJI_20260905181949_0005_D.MP4" -o artifacts/main   # CSV + JSON + npz
python src/dji_o4.py sidecar  <video> -o artifacts/main/x.mp4 [--from-csv edited.csv]
python src/align.py           <video> -o artifacts/main/x.mp4 [--extra-ms N] [--mean-only]
python src/dji_o4.py verify   <video> artifacts/main/native_parsed.json
```

```python
from telemetry import load, from_axis_angle
tel = load('artifacts/main/DJI_20260905181949_0005_D_telemetry.npz')
tel.q_cam, tel.q_dji, tel.t_us, tel.frame, tel.idx, tel.inverted, tel.file_offset
tel.frames          # 15 per-frame arrays: sample_offset, vsync, iso, exposure, ...
tel.rates(), tel.euler(), tel.unique(), tel.window(10,12), tel.frame_slice(100)
tel.rotate(from_axis_angle([0,1,0], 5.0))
tel.write_sidecar(video, out)
```

### Environment

- Gyroflow **1.6.3** (`main.py` looks for `Gyroflow.exe` in the usual install
  locations; pass `--gyroflow PATH` otherwise).
- No ffmpeg / ffprobe on this machine. **OpenCV 4.9 and scipy 1.12 are installed**
  and OpenCV decodes this 4K 10-bit HEVC fine at ~0.076 s/frame.
- Python 3.11.4, numpy 1.25.

### Gyroflow CLI, the part that makes claims checkable

```bash
Gyroflow.exe <video> --export-metadata "2:parsed.json" -f   # its parse of the input telemetry
Gyroflow.exe <video> --export-metadata "3:camera.json" -f   # per frame: org_quat, stab_quat,
                                                            # org_euler, stab_euler, fov_scale,
                                                            # minimal_fov_scale, timestamp_ms
Gyroflow.exe <video> -g <motion-data-file> ...              # external telemetry
Gyroflow.exe <video> --preset "{ 'stabilization': { ... } }" ...
```

**Traps, all hit and confirmed:**

- Running the CLI in a tight shell loop makes some runs exit after ~0.8 s having
  loaded nothing, and the export is a 515-byte stub. Runs are fine individually.
  Always check the export is bigger than a few hundred bytes.
- Redirecting to `/dev/null` in Git Bash makes it exit having loaded nothing.
  Redirect to a file instead.
- `-s/--sync-params` only writes the parameters into the project. It does **not**
  run autosync; `offsets` comes back `{}`. There is no CLI path to autosync.
- `--export-metadata-fields` applies to type 3, not type 2.
- `--export-project 2` embeds the telemetry as base91-encoded compressed bytes in
  `gyro_source.file_metadata` — not human-readable, not an editing route.
- Both `--export-project` and `--export-metadata` write a `.gyroflow` next to the
  video. Delete it afterwards.

---

## 4. Steps 1 and 3, verified

Verified against Gyroflow's own parse, not by assertion:

- extraction: 184427 samples, **every timestamp identical**, worst quaternion
  component difference **1.08e-07** — that is float32 storage precision;
- readout time, focal length, distortion coefficients: identical;
- the sidecar MP4 (a 5.1 MB file holding only the `djmd` track plus an empty video
  track so Gyroflow still builds the official DJI lens profile): quaternions
  **bit-identical**, difference exactly 0.0, same timestamps, same lens profile;
- a deliberate +5° edit came back out of Gyroflow as 5.0000°–5.0000° across all
  184427 samples;
- `patch` on an unmodified CSV reproduces a **byte-identical** file (md5 match).

**Two delivery routes, and why the sidecar wins.** A `.gcsv` (Gyroflow IMU LOG)
carries angular *velocity*, not orientation, so Gyroflow re-integrates it and the
absolute attitude drifts (~1.5°/s here, ~150° over the clip); it also drops the
13.58 ms readout time and needs an integration method chosen by hand. Confirmed
working (`tscale 1e-6`, `gscale 1`, values in rad/s — Gyroflow read them back as
exactly ×57.2958 deg/s, so `gscale` is a multiplier to rad/s and the docs are
right where the source summary was not). Use the sidecar; keep the gcsv as the
plain-text fallback.

---

## 5. The shake: what it is and everything it is not

### 5.1 Settings cannot fix it, algebraically

If the telemetry is `q_tel = q_true ⊗ e`, then what Gyroflow puts on screen is

```
output = S(q_tel) ⊗ conj(q_tel) ⊗ q_true = S(q_true ⊗ e) ⊗ conj(e)
```

With weak smoothing `S(x) ≈ x` the error cancels — and so does the stabilization.
With strong smoothing `S → const`, so `output → const ⊗ conj(e)`: the error shows
up in **full**. The better the stabilization, the more visible the defect. Turning
the smoothness knob cannot help, which is why nobody has fixed this with settings.

The user's own settings, from a screenshot, are exactly Gyroflow's defaults:
method Default, smoothness 50 %, dynamic zoom with a 4 s window, zoom limit 130 %,
lens correction 100 %, horizon lock off.

### 5.2 One real but separate finding: the adaptive zoom breathes

With those defaults `fov_scale` runs 0.778…1.126 over the clip (1.126 → 0.806 in
the first 4.5 s, back to 0.869 by 7 s, down to 0.779 by 9 s, up to 0.992 by 17 s).
Median zoom velocity 3.3 %/s, peak 14.1 %/s, and the three moments the user named
sit on the steepest ramp (9.6 %/s at 3 s) and on two reversals. Setting
`adaptive_zoom_window = -1` (static crop) pins it at a constant 0.7958 with zero
velocity, at the cost of a permanently tighter crop than the dynamic average
(0.796 vs 0.910). This is a genuine visible artefact but it is a *setting*, and the
user says settings are not the problem, so it was set aside. Worth revisiting if
the telemetry work ever bottoms out.

### 5.3 The frame↔telemetry misalignment — found, fixed, not the answer

DJI states the alignment outright: in `dbgi`, field `2.1.11.7.4` is a **bit-exact
copy** of one record in the *same* frame's `djmd` block. Found in 4611 of 4612
frames, at index 19 (2388×) or 20 (2215×) of 40 — the block centre.
`dji_o4.reference_indices()` extracts it.

Gyroflow cannot see that and infers the slot from `offset`, reading each frame at
`frame_ts + readout/2`, i.e. slot `offset + n*readout/(2*vsync)`. Confirmed by
measurement on the 600 frames whose block spans more than 2° (so the slot is
unambiguous): mean slot 11.97 with mean `offset` −1.600, and
`n*readout/(2*vsync)` = 13.582 → 11.97 = −1.600 + 13.572.

So the built-in path reads **3.77 ms too early**, wandering with std 2.02 ms.
**And loading external motion data makes it worse:** with `-g` Gyroflow logs "Not a
main video, clearing per-frame offsets" and frame timestamps become 0, 20, 40 ms
instead of 6.79, 26.79, 46.79 — the `readout/2` term is dropped, so `i_used =
offset` and the error grows to 10.54 ms. This is why reloading *unmodified*
telemetry looks worse than loading nothing, and it explains the user's report that
`offset_const` seemed worse.

`align.py` fixes it by shifting the **contents** of each block, not the timing
fields: slot `i` gets the attitude interpolated at its own nominal time plus
`(i_ref − i_used)/n * vsync`. Do not rewrite `offset` instead — a timestamp then
goes negative and Gyroflow re-bases the entire timeline, moving the frame times as
well (`timestamp_ms` of frame 0 goes 6.7906 → 0.0), which silently confounds the
experiment. That mistake cost an hour.

Verified, as the error of Gyroflow's per-frame attitude against DJI's own reference:

| telemetry source | p50 | p90 | p99 |
|---|---|---|---|
| built-in | 0.106° = 2.70 px | 0.559° = 14.2 px | 3.81° = 96.9 px |
| sidecar, unmodified | 0.293° = 7.46 px | 1.415° = 36.0 px | 8.95° = 228 px |
| sidecar, **aligned** | 0.0000° = 0.00 px | 0.033° = 0.85 px | 0.170° = 4.33 px |

Median is a bit-exact match. **The user tested it: the picture still shakes.** So
this was a real defect worth fixing, but not the cause of the complaint. Keep the
fix — it is strictly better telemetry and it removes a confound.

### 5.4 The actual cause: DJI's fusion error, measured against the image

OpenCV is installed, so the image can be used as ground truth. `imagerot.py`
tracks features with pyramidal Lucas-Kanade (1200–1500 points, forward-backward
error 0.004 px), lifts them to unit rays through DJI's own fisheye model, and
fits motion two ways:

- `frame_rotations` — full 3-axis rotation by robust Kabsch on the rays. An
  essential-matrix solve is useless here: the inter-frame baseline is ~1.5 px, so
  `recoverPose` returns 0 inliers. Do not retry that.
- `affine_flow` — fits `v = A p + b` and splits `A` into divergence / curl /
  shear. **Curl is a roll measurement that parallax cannot fake** for a
  fronto-parallel scene, which matters because the drone is low over grass. Caveat:
  translation over a *tilted* plane does produce some curl (the antisymmetric part
  of `t nᵀ`), so the absolute discrepancy may be partly measurement. The jitter
  comparison below is immune to that, because such a bias varies smoothly.

Roll, image versus telemetry, per frame:

| window | image rms | telemetry rms | correlation | residual rms |
|---|---|---|---|---|
| ~3 s | 0.1227° | 0.1205° | **0.04** | 0.188° = 7.9 px at the corner |
| ~15 s | 0.0640° | 0.1023° | 0.583 | 0.083° = 3.5 px |
| ~60 s | 0.1893° | 0.1274° | 0.904 | 0.092° = 3.9 px |
| ~10 s | 2.6199° | 2.7372° | 0.989 | 0.410° = 17.2 px |
| ~25 s | 3.2358° | 3.1133° | 0.993 | 0.402° = 16.8 px |

At 3 s the amplitudes match and the correlation is **zero**: the telemetry reports
roll that did not happen and misses the roll that did. Corner pixels use a 2400 px
lever arm (41.9 px/°); the focal-length figure for pan/tilt is 25.43 px/°.

Smoothness test — rms of the frame-to-frame change of the roll rate, with my own
noise bounded from the smoothness of my own signal (white noise σ would give
≥ σ√2):

| window | image | my noise ≤ | telemetry | ratio |
|---|---|---|---|---|
| ~3 s | 0.0363° | 0.0257° | 0.1196° | 3.3× |
| ~15 s | 0.0104° | **0.0074°** | 0.1167° | **16×** |
| ~60 s | 0.0451° | 0.0319° | 0.0480° | 1.06× |
| ~10 s | 0.9799° | — | 0.8418° | 0.86× |
| ~25 s | 0.5342° | — | 0.4789° | 0.90× |

The telemetry jitters by **0.117–0.120° per frame regardless of the real motion**.
Where the drone moves fast it is swamped (ratio ~0.9); where it is slow it *is*
the whole signal. Repeatability check: the same window at scale 1.0 versus 0.5
correlates 0.894, and against the telemetry the correlation is 0.04 either way.

Spectrum, expressed as what it is worth on screen (roll angle at the frame corner):

| band | ~3 s | ~15 s |
|---|---|---|
| 0.5–3 Hz | **3.61 px** | **2.55 px** |
| 3–10 Hz | 0.93 px | 0.72 px |
| 10–25 Hz | 0.77 px | 0.45 px |
| 25–100 Hz | 0.44 px | 0.36 px |

All of it is below 3 Hz. That is why the 8–50 Hz low-pass the user (and others)
tried could not touch it.

### 5.5 Pattern hunt — every hypothesis, all negative

The user asked directly whether it can be fixed by maths on the telemetry alone.
It cannot; here is what was tested.

| hypothesis | test | result |
|---|---|---|
| timing offset | τ sweep against the image, ±25 ms | flat 3.83–3.97 px, no minimum |
| value quantization | lattice of distinct quaternion values | 91703 distinct of 92227, step ×100 coarser than float32 eps — no lattice |
| staircase / fusion slower than 1 kHz | fraction of steps where the rate changes | 100.0 % on all three axes — nothing is held |
| block seam discontinuity | extrapolate block N onto the first sample of N+1 | 0.29 px versus 0.29 px for the same test inside a block |
| periodicity | correlation with periods 3…300 frames | all < 0.25, at chance level |
| metadata coupling | offset, vsync, exposure, iso, WB, temperature | all within chance given a 10-frame correlation time |
| wrong IMU→camera axes | regression on ω (3 par) | this *is* the baseline |
| fusion lag | + terms ∝ ω̇ (6 par) | no improvement: 3.5 → 3.6 px, worse at 3 s |
| second-order lag | + terms ∝ ω̈ (9 par) | 3.5 → 3.0 px on 3 extra parameters; cross-validation shows overfitting |
| non-physical rate spikes | 1 kHz series | none, max 1981 °/s |

The error is **not white** either: autocorrelation +0.693 at lag 1 frame decaying
to zero by lag 12, i.e. a ~0.2 s correlation time (1–3 Hz), and |error| correlates
+0.55 with |rate|. So it is neither filterable noise nor a deterministic artefact
that can be subtracted. It is the fusion's own estimation error — gyro bias random
walk plus accelerometer-based horizon correction fighting flight accelerations —
and it is not predictable from the telemetry.

**The information is not in the file.** The quantity needed is the difference
between DJI's estimate and reality; the estimate is in the file, reality is only
in the pixels.

### 5.6 Also ruled out, on the user's instruction or by measurement

- Exposure and rolling shutter (19.6 ms of a 20 ms frame, readout 13.58 ms,
  rotational blur up to 409 px at speed): the user said to drop it, and at 3/10/15 s
  the blur is 1–16 px and the RS skew 0.6–11 px, so it is negligible there anyway.
- Gyroflow's own frame timing: exactly uniform, 20.000 ms, zero jitter.
- Telemetry dropouts: 17 frames only — 39 quaternions instead of 40, every
  **6.00 s** exactly (5.74, 11.74, … 89.74), plus two 21-sample frames at the end.
  Not at the reported moments.
- `sample_offset` dither: mostly cycles between −2.94 / −3.31 / −3.62 (this is the
  50.0 vs 50.0042 fps sawtooth), ±0.35 ms, sub-pixel. Large excursions do exist at
  24, 28, 44, 46, 86 s. The user tested `offset_const` — no help.
- The `dbgi` second attitude (field 7.3 = 2.1.11.2): sits 0.3 ms from 7.4, i.e. an
  adjacent sample, not an independent estimate. Only 3 quaternion-shaped fields
  exist in the whole `dbgi` sample and two are duplicates.

---

## 6. State right now

Deliverables in `artifacts/main/variants/` (all ~5.1 MB, load under Motion data, leave every
other setting alone; `README.txt` there repeats this):

| file | what |
|---|---|
| `00_baseline.mp4` | unmodified telemetry, control for A/B |
| `01_ALIGNED.mp4` | §5.3 fix: +10.54 ms mean, per frame, std 1.68 ms |
| `02_ALIGNED_minus2ms.mp4`, `03_ALIGNED_plus2ms.mp4` | ±2 ms bracket on the constant term |
| `04_ALIGNED_meanonly.mp4` | constant part only, no per-frame correction |

Other outputs in `artifacts/main/`: the extraction (`*_telemetry.npz` 3.7 MB, `*_quaternions.csv`
28 MB, `*_frames.csv`, `*_metadata.json`), `native_parsed.json` (Gyroflow's own
parse, the reference for `verify`), `stab_camera.json` / `cam_g_baseline.json` /
`cam_aligned_g.json` (per-frame stabilization exports for built-in / raw sidecar /
aligned sidecar), `telemetry_sidecar.mp4`, `telemetry.gcsv`, `affine_windows.pkl`
(the five measured windows), `rimg_700.npy`, `C_frame.npy`, `seam.npy`.

**In progress when this was written:** `measure_clip.py` running in the background,
image roll for all 4612 frames into `artifacts/main/roll_clip.npz` (arrays `roll`, `shift_x`,
`shift_y`, `div`, `npts`; it saves every 400 frames, so a partial file is usable —
check with `(~np.isnan(d['roll'])).sum()`). It was at 2000/4612 after 432 s, so
about 17 minutes total. Restart with
`python src/measure_clip.py "F:\36\DJI_20260905181949_0005_D.MP4" artifacts/main/roll_clip.npz`.

Calibration constants already measured, reuse rather than refit:

- optical axis in the telemetry frame, for the roll component:
  `u = [0.0573, -0.2774, 1.0202]` (fitted on the ~10 s and ~25 s windows,
  correlation 0.990, gain 1.0000)
- image→telemetry axis map for `frame_rotations`: negate X
- corner lever arm 41.9 px/°, focal 25.43 px/°

---

## 7. Next step, and be honest about its ceiling

The planned fix, once `roll_clip.npz` is complete:

1. `d[f] = image_roll[f] − telemetry_roll[f]` per frame, with
   `telemetry_roll = (per-frame rotation vector) · u`.
2. Integrate `d` into a correction angle, band-limited to roughly 0.3–8 Hz — leave
   DC and the very low frequencies to the telemetry, because the integral of an
   image measurement drifts over long spans.
3. Apply it as an extra rotation about the optical axis to every telemetry sample
   and write a sidecar (reuse `align.py`'s machinery: build an `edits` dict of
   DJI-frame quaternions and hand it to `sidecar.build`).
4. Verify the way everything else here was verified: re-export type 3 from
   Gyroflow and check the per-frame attitude now tracks the image roll.

**This corrects roll only.** Pan and tilt cannot be measured the same way — the
shift term `b` mixes rotation with translation inseparably, and separating them
needs scene structure, which is a different order of work. Tell the user the
expected ceiling before they test, so a partial improvement is not read as a
failure.

If roll turns out not to be enough, the honest options are: full structure-from-
motion for 3-axis image-based correction, or accepting that this clip's telemetry
is what it is. Do not go back to settings, filters or timeline theories — §5.5 is
a closed list.

---

## 8. How the user works

They interrupt with corrections mid-task and expect them applied immediately. They
push back hard on anything that looks like a simple or generic answer, and they are
usually right to. They value measured numbers over explanation, and they will test
a file quickly and report qualitatively ("потрясывает", "по часовой стрелке туда
сюда") — those qualitative reports have been reliable and worth taking literally.
Answer their direct questions directly, including when the answer is "no".

---

## 9. The roll correction from the image (added after §7 was written)

`measure_clip.py` finished: image roll for 4570 of 4612 frames,
`artifacts/main/roll_clip.npz` (arrays `roll`, `shift_x`, `shift_y`, `div`, `npts`).
`rollfix.py` turns it into a correction. **Verified through Gyroflow**, as the
frame-to-frame jitter of the roll rate:

| section | aligned telemetry | ROLLFIX | image (floor) | gap closed |
|---|---|---|---|---|
| calm < 20 °/s | 0.07691 | **0.04809** | 0.04834 | **101 %** |
| 20–60 °/s | 0.12268 | 0.10394 | 0.08206 | 46 % |
| fast > 60 °/s | 0.78479 | 0.78454 | — | left alone |

At the user's three moments (image / before / after): ~3 s 0.037/0.112/0.048,
~10 s 0.056/0.118/0.053, ~15 s 0.008/0.125/0.017. The applied correction is only
0.037° rms, peak 19 px at the frame corner.

### What the artefact actually looks like

Frame by frame at 14.4–15.2 s, roll in degrees per frame:

```
image        -0.025 -0.026 -0.029 -0.035 -0.036 -0.030 -0.024 -0.019 -0.031 ...
telemetry    -0.002 -0.083 +0.041 +0.013 -0.072 -0.039 -0.374 -0.130 -0.136 ...
```

1200 tracked points, drone turning at 2–12 °/s. The telemetry alternates sign
almost every frame: the artefact is **per-frame, up at Nyquist**, plus isolated
single-frame spikes of 0.2–0.4° (10–15 px at the corner). It is *not* the 0.5–3 Hz
wander §5.4 inferred from the rate spectrum — that reading was misleading.

The user's own framing was the key: "баги возникают иногда, обычно при ровном
полете - оттого их и видно. В среднем все хорошо". The noise floor is constant, so
it is only visible when real motion does not swamp it. The rate weighting in
`rollfix.py` reproduces that automatically; an event detector is not needed (though
`rollfix.event_theta` exists). For the record, thresholding the discrepancy at 8 px
during calm flight finds 73 bursts covering 11 % of the clip, and **all three
moments the user found by eye have events within ±2 s** — 2.58/3.36/3.68/4.16/
4.32/4.60 s, 8.88/10.08/10.24 s, 14.52/15.12/15.78 s — none of them the largest in
the clip, matching "есть ещё, но в глаза не бросается".

### Three bugs found while building this — check for them if you touch it

1. **Low pass too low.** Cutting at 8 Hz threw away the component that matters and
   closed 4 % of the gap instead of 101 %. Keep `hi` near Nyquist (24.8 Hz).
2. **Off-by-one in the integral.** `theta[f]` must be the sum of corrections
   *before* f, because the increment between f and f+1 picks up
   `theta[f+1] - theta[f]`. An unshifted `cumsum` applies frame f+1's correction to
   frame f, which inverts the phase of a near-Nyquist signal.
3. **Wrong baseline.** The discrepancy must be measured against the telemetry
   *after* the alignment shift, since that is what ends up in the file. Measuring
   against the unaligned telemetry makes the correction fight a 0.242 °/frame
   baseline difference seven times its own size. This one produced a *worse* result
   that looked like a conceptual failure for two rounds.

Diagnose all three the same way: export type 3 for the new file and for
`01_ALIGNED`, take the difference of the per-frame roll, and correlate it against
the intended correction. Landing correctly gives correlation +1.0000 and gain ~1.05.

### Independent validation of the image measurement

`validate_events.py` re-measured every detected event at full resolution with a
denser feature set (`artifacts/main/roll_events_fullres.npz`, array `roll_hi`). **Finished:
1862 frames.** Correlation 0.9843 between the two independent passes, and the two
discrepancy-against-telemetry series agree with correlation 0.8747 and gain 0.9913.
Per event, **0 of 45 failed to reproduce** — including the three that looked
suspicious because the tracked-point count was low: 43.36-44.24 s (461 points)
40.8 -> 38.4 px, 24.62-25.32 s (534) 29.6 -> 29.1 px, 75.12-75.28 s (608)
45.2 -> 44.9 px. So the discrepancy is the telemetry, not the measurement; this is
settled and does not need redoing.

Other supporting checks: at events the tracked-point count is nearly normal (1012
vs 1095), so it is not tracking failure; and the telemetry's own within-block
non-linearity is 1.30× higher at events (0.0238° vs 0.0183°) with
corr(|discrepancy|, roughness) = +0.17 — weak, but independent, and in the right
direction.

### Honest limits

- **Roll only.** Pan and tilt cannot be measured this way: the shift term mixes
  rotation with translation inseparably. Curl is parallax-immune for a
  fronto-parallel scene; translation over a *tilted* plane does leak some curl (the
  antisymmetric part of `t nᵀ`), which is why the correction is band-limited above
  4 Hz — the leak is smooth and stays out of that band.
- **Fast sections are not corrected**, deliberately: above 60 °/s the image is the
  noisier source (jitter 1.447 against the telemetry's 0.785, from 19.6 ms of
  motion blur and 13.58 ms of rolling shutter), and the artefact is invisible there
  anyway. `R0` above ~80 starts leaking that noise into the fast frames.
- The 46 % at 20–60 °/s is the honest ceiling of this method, not a tuning failure.

---

## 10. Can it be found and fixed from telemetry alone, with no video?

The user asked this directly. **Yes to both**, and the no-video fix is close to as good.

### Detection

Telemetry-only features, scored against the image-derived discrepancy as ground
truth ("bad" = the worst 20 % of frames, so chance precision is 0.20):

| feature (telemetry only) | AUC | precision@200 |
|---|---|---|
| band-passed 4–24.8 Hz magnitude of the roll | **0.823** | 0.82 |
| \|Δ(roll rate)\| per frame | 0.816 | 0.81 |
| within-block non-linearity | 0.793 | 0.80 |
| \|Δ²(roll rate)\| | 0.766 | 0.80 |

It works because the telemetry's own high-frequency roll content *is* mostly the
error. No video needed to find the moments.

### The spectral statement of the bug

Ratio of the telemetry's roll energy to the camera's real roll energy, on calm
frames (two contiguous calm runs, 305 frames, Welch):

| band | ratio | | band | ratio |
|---|---|---|---|---|
| 0–2 Hz | 1.29× | | 8–12 Hz | 6.02× |
| 2–5 Hz | 1.73× | | 12–16 Hz | 9.89× |
| 5–8 Hz | 3.44× | | 16–20 Hz | 13.26× |
| | | | 20–25 Hz | **20.59×** |

Above 5 Hz the fused attitude is 3–20× larger than the motion that happened.

### The fix: `denoise.py`

Wiener keep-gain = P_real/P_telemetry = 1/ratio, applied as a gain curve on the
per-frame roll, rate-weighted, then integrated into a rotation about the optical
axis exactly as `rollfix.py` does. Verified through Gyroflow, judged by the image
measurement the filter never sees:

| | jit calm | jit med | jit fast | dist calm | dist med |
|---|---|---|---|---|---|
| image (judge) | 0.04834 | 0.08206 | 1.44740 | 0 | 0 |
| aligned only | 0.07691 | 0.12268 | 0.78479 | 0.07291 | 0.09495 |
| `05_ROLLFIX` (uses video) | 0.04809 | 0.10394 | 0.78454 | 0.04980 | 0.07754 |
| `07_DENOISE` (no video) | 0.03032 | 0.07267 | 0.75361 | 0.05721 | 0.07751 |

* calm sections: recovers **68 %** of what the image-based fix achieves
* 20–60 °/s: **100 %** (0.07751 against 0.07754)
* faster: improves them too, which the image-based method cannot

The 68 % rather than 100 % at calm is the expected limit of a statistical filter:
it knows the error's *spectrum* but not its *phase*, while optical flow knows both.

`--r0 150` scores marginally better (dist calm 0.05695 vs 0.05760) but pushes the
smoothness prior into fast motion, where real roll genuinely has high-frequency
content and the image cannot check it. R0 80 is the default for that reason; the
"jit fast" metric improving there proves nothing, since smoothing always lowers it.

### What is and is not telemetry-only about this

The gain curve was *calibrated* against the image on this clip, once. The prior
underneath it — a calm drone has no real 8–25 Hz camera roll — is physics, not
information from the file, and the image measurement confirmed it. So `denoise.py`
transfers to other clips from the same device and firmware without any video pass;
re-calibrate the curve if the device or firmware changes.

---

## 11. Entry point

`src/main.py` + `fix_telemetry.bat` wrap the whole thing: video in,
`artifacts/main/<name>/<name>_telemetry_fixed.mp4` out (alignment + three-axis de-noising),
then a Gyroflow cross-check that prints the extraction match, the alignment error
against DJI's own reference, and the roll jitter on calm frames. Switches:
`--roll-only`, `--gain`, `--r0`, `--extract`, `--variants`, `--no-verify`,
`--gyroflow PATH`, plus `-o DIR` and `--beside` (write next to the source
video rather than into `artifacts/main/<name>/`). Globs and several files at once work.

Three things to know if you touch these:

* **Keep the .bat ASCII only.** cmd.exe reads it in the OEM codepage, and
  multi-byte characters inside `rem` lines get parsed as commands — the first
  version printed `'one' is not recognized as an internal or external command`
  from a comment containing Cyrillic.
* **Every path handed to Gyroflow must be absolute.** It is launched with its own
  directory as cwd, so a relative `--export-metadata` path silently produces
  nothing (and the run still reports success).
* `gyroflow_export()` treats an export under 10 kB as a failure. That is the
  tight-loop stub described in §3, and it is the only reliable way to notice it.

The three-axis de-noising (`denoise.build3`) is the default because after roll is
fixed it stops being the dominant residual: on clip 0003 at 7–10 s, roll dropped
from 4.95 px of apparent jitter to 1.03 px while pitch and yaw sat untouched at
1.11 and 1.30 px. Adding them took the total residual against the image from
1.49 px to 1.33 px, and the apparent jitter to 1.52 px against 1.81 px of real
camera motion. The pan/tilt keep-gain curve is calibrated far more weakly than the
roll one — a single 210-frame window on one clip, validated against the frame
shift rather than the curl — so `--gain 0.7` exists for when it overshoots.

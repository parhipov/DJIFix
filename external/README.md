# External sources

Reference sources used to check how Gyroflow reads DJI telemetry. They are not
part of this repository; clone them here when needed:

```bash
git clone --depth 1 --branch v1.6.3 https://github.com/gyroflow/gyroflow.git external/gyroflow
git clone https://github.com/AdrianEddy/telemetry-parser.git external/telemetry-parser
cd external/telemetry-parser && git checkout cbc61aba9607b2450f014e5592ca606b8335d5fc   # rev pinned in Gyroflow's Cargo.lock
```

Places worth reading (see also `docs/HANDOFF.md` §0):

- `telemetry-parser/src/dji/mod.rs` — the DJI timeline `quat_ts = frame_ts + ((i - offset)/n) * vsync`;
  the exposure-time correction is commented out.
- `gyroflow/src/core/lib.rs` (~L200-255) — `frame_readout_time` is only set for the main video; external
  motion data (`-g`) leaves it at 0 in the CLI.
- `gyroflow/src/core/stabilization/frame_transform.rs` (~L183-215) — rows are sampled from `pts - readout/2`
  to `pts + readout/2`; the frame's reference attitude is read at `pts`.
- `gyroflow/src/core/gyro_export.rs` (~L121) — the type-3 metadata export stamps `pts + readout/2`.
- `gyroflow/src/rendering/render_queue.rs` (~L1333) — CLI `-g` never parses the video's own telemetry.

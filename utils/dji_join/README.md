# dji_join: join DJI O4 Pro split recordings

The camera splits a recording into ~4-minute files (`DJI_20261019135716_0020_D.MP4`,
`DJI_20261019140118_0021_D.MP4`, …). The script finds such chains and joins them
without re-encoding, keeping every track including the DJI telemetry (`djmd`, `dbgi`),
so the result stabilizes in Gyroflow as one clip.

```bat
join.bat                     :: FOLDER from join.py, with all its subfolders
join.bat "F:\clips"          :: another folder (a folder can be dropped onto join.bat)
python join.py "F:\clips" --dry-run   :: only show what would be joined
```

Output: `<first file name>_joined.mp4` next to the sources. After a successful join the
sources are renamed to `_<name>` (disable with `--keep-names`) so they are easy to review
and delete. An existing `_joined.mp4` is never overwritten.

## How it groups

In order of the sequence number: segment N+1 continues N when its start time (from the
file name) is within `MAX_GAP_SECONDS` (3 s) of the end of N (start + duration).
Real splits are 0–1.1 s apart (the file-name time has 1 s resolution, so up to 2 s is rounding); a stop and a new recording leaves a larger gap and such files
stay separate. Verified on the footage: the telemetry across a split is continuous
(exactly one frame period, consecutive vsync counters).

## What does the joining

[mp4-merge](https://github.com/gyroflow/mp4-merge) v0.1.11 — the very crate Gyroflow uses
to join split recordings in its GUI (Gyroflow's CLI does not expose it). `mp4_merge.exe`
lives next to the script; when it is missing, the script downloads it from GitHub Releases.
ffmpeg is not an option here: its mp4 muxer cannot write the DJI data tracks
(`Could not find tag for codec none`), so the telemetry would be lost.

Dependency: `pip install av` (file durations).

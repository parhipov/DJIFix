@echo off
rem  DJI O4 Lite with the Flywoo O4 Wide lens: the lens profile must be given,
rem  the telemetry only knows the stock lens. Result: o4lite_53-58s_telemetry_fixed.mp4
rem  (load in Gyroflow as motion data), the report json and three PNGs next to this file.
if not exist "%~dp0o4lite_53-58s.MP4" (
    echo   o4lite_53-58s.MP4 is not here. Download it from the GitHub Releases page
    echo   of this repository and put it next to this run.bat.
    pause
    exit /b 1
)
call "%~dp0..\..\fix_telemetry.bat" "%~dp0o4lite_53-58s.MP4" --lens flywoo --plots %*

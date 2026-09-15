@echo off
rem  DJI O4 Pro, stock lens: fix the telemetry of the sample clip and draw the plots.
rem  Result: o4pro_13-17s_telemetry_fixed.mp4 (load in Gyroflow as motion data),
rem  o4pro_13-17s_fix_report.json and three PNGs next to this file.
if not exist "%~dp0o4pro_13-17s.MP4" (
    echo   o4pro_13-17s.MP4 is not here. Download it from the GitHub Releases page
    echo   of this repository and put it next to this run.bat.
    pause
    exit /b 1
)
call "%~dp0..\..\fix_telemetry.bat" "%~dp0o4pro_13-17s.MP4" --plots %*

@echo off
rem ---------------------------------------------------------------------------
rem  DJI O4 telemetry fixer.
rem
rem  Drag one or more DJI .MP4 files onto this file, or run it from a prompt:
rem      fix_telemetry.bat "F:\36\DJI_20260905181503_0003_D.MP4"
rem
rem  It writes NAME_telemetry_fixed.mp4 next to the video -- load that file in
rem  Gyroflow under Motion data. The video itself is never modified. Everything
rem  else the run produces (image measurement cache, control telemetry, reports)
rem  is deleted at the end; add --keep to retain it (the cache saves the ~25 min
rem  video pass next time, the control file NAME_00_control.mp4 is for A/B).
rem
rem  The first run on a clip decodes the whole video once to measure the real
rem  camera rotation (about 0.35 s per 4K frame, so ~25 minutes for 90 s).
rem  --no-image skips that step and only fixes the timing plus the de-noising.
rem
rem  -o "D:\some\dir" writes into a directory of your choice (and keeps everything);
rem  --artifacts writes into artifacts\main\NAME\ for development.
rem
rem  Extra switches go straight through to src\main.py, for example:
rem      fix_telemetry.bat video.MP4 --lens flywoo     (O4 Lite with the Flywoo O4 Wide lens)
rem      fix_telemetry.bat video.MP4 --keep
rem      fix_telemetry.bat video.MP4 --gain 0.7
rem      fix_telemetry.bat video.MP4 --no-image
rem      fix_telemetry.bat video.MP4 --gyroflow "D:\Gyroflow\Gyroflow.exe"
rem      fix_telemetry.bat *.MP4 --no-verify
rem
rem  The final cross-check runs Gyroflow's CLI. It is found in the standard install
rem  folders; elsewhere pass --gyroflow PATH or set the GYROFLOW environment variable.
rem  Without Gyroflow the check is skipped, the result is the same.
rem
rem  Keep this file ASCII only: cmd.exe reads it in the OEM codepage and
rem  multi-byte characters inside rem lines get parsed as commands.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo.
    echo   Drag a DJI .MP4 file onto this .bat, or pass it as an argument:
    echo       fix_telemetry.bat "F:\36\DJI_20260905181503_0003_D.MP4"
    echo.
    pause
    exit /b 1
)

set "PY="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)
if not defined PY (
    echo   Python was not found. Install it, or put python.exe on PATH.
    pause
    exit /b 1
)

"%PY%" -c "import numpy, scipy, cv2, matplotlib" 2>nul
if errorlevel 1 (
    echo   Installing the dependencies: numpy, scipy, opencv-python, matplotlib
    "%PY%" -m pip install --quiet numpy scipy opencv-python matplotlib
)

"%PY%" src\main.py %*
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" echo   Finished with errors, exit code %RC%
rem  Pause only when double-clicked, not when run from an open console
echo %CMDCMDLINE% | find /i "/c" >nul && pause
exit /b %RC%

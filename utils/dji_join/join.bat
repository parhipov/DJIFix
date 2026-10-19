@echo off
rem Склейка сегментов DJI O4 Pro. Перетащите папку на этот файл,
rem или запустите без аргументов — возьмётся FOLDER из join.py.
setlocal
python "%~dp0join.py" %*
echo.
pause

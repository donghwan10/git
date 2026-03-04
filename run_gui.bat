@echo off
setlocal

cd /d "%~dp0"
set "SCRIPT=%~dp0app\gui.py"

if not exist "%SCRIPT%" (
    echo File not found: "%SCRIPT%"
    pause
    exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "%SCRIPT%"
    goto :after_run
)

where python >nul 2>nul
if %errorlevel%==0 (
    python "%SCRIPT%"
    goto :after_run
)

echo Python executable not found.
echo Install Python 3 and ensure it is on PATH.
pause
exit /b 1

:after_run
if errorlevel 1 (
    echo.
    echo App failed to start.
    echo Check the error message above.
    echo If tkinter is missing, install Python with Tk support.
    pause
    exit /b 1
)

endlocal

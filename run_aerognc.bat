@echo off
setlocal

title AeroGNC-Lab - Simulation Workbench
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo.
    echo [AeroGNC-Lab] Python virtual environment was not found.
    echo Expected: "%PYTHON_EXE%"
    echo.
    echo One-time setup from this folder:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

echo [AeroGNC-Lab] Opening the Simulation Workbench...
echo [AeroGNC-Lab] Start page: Rocket, Satellite Orbit, Aircraft Flight, or Planet Trip.
echo.

"%PYTHON_EXE%" -m aerognc.cli workbench %*
set "RUN_STATUS=%ERRORLEVEL%"

if not "%RUN_STATUS%"=="0" (
    echo.
    echo [AeroGNC-Lab] Workbench stopped with exit code %RUN_STATUS%.
    pause
)

exit /b %RUN_STATUS%

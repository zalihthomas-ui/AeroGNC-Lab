@echo off
setlocal

title AeroGNC-Lab - Aquila-X1 Flight Deck
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo.
    echo [AeroGNC-Lab] The Python environment is not installed yet.
    echo One-time setup from this folder:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

echo [AeroGNC-Lab] Opening the fictional civilian Aquila-X1 flight deck...
echo [AeroGNC-Lab] Click the 3D window, press Space to start, and H for controls.
echo [AeroGNC-Lab] F9 saves exact states plus a debrief; F10 opens replay.
echo.

"%PYTHON_EXE%" -m aerognc.cli fly-aircraft ^
  --config configs\aircraft_sandbox.yaml ^
  --mesh assets\models\aquila_x1.obj ^
  --control-mode stability_assisted ^
  --camera chase ^
  --trail fading ^
  --trail-duration 45 ^
  --training-task altitude_speed_hold ^
  --recording-directory results\aircraft_live %*

set "RUN_STATUS=%ERRORLEVEL%"
if not "%RUN_STATUS%"=="0" (
    echo.
    echo [AeroGNC-Lab] Aircraft flight stopped with exit code %RUN_STATUS%.
    pause
)

exit /b %RUN_STATUS%

@echo off
setlocal

title AeroGNC-Lab - Mission Designer
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "PLANETARY_CATALOG=%~dp0configs\fictional_planetary_system.yaml"
set "MISSION_CONFIG=%~dp0configs\interplanetary_gravity_assist.yaml"

if not exist "%PYTHON_EXE%" (
    echo.
    echo [AeroGNC-Lab] Python virtual environment was not found.
    echo Expected: "%PYTHON_EXE%"
    echo.
    echo Create it from this folder with:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

if not exist "%PLANETARY_CATALOG%" (
    echo [AeroGNC-Lab] Fictional planetary catalog was not found:
    echo "%PLANETARY_CATALOG%"
    pause
    exit /b 1
)

echo [AeroGNC-Lab] Opening the guided Mission Designer...
echo [AeroGNC-Lab] Enter route, timing, spacecraft and maneuver values in the form.
echo.

"%PYTHON_EXE%" -m aerognc.cli mission-designer --catalog "%PLANETARY_CATALOG%" --verified-config "%MISSION_CONFIG%" %*
set "RUN_STATUS=%ERRORLEVEL%"

if not "%RUN_STATUS%"=="0" (
    echo.
    echo [AeroGNC-Lab] Mission Designer stopped with exit code %RUN_STATUS%.
    pause
)

exit /b %RUN_STATUS%

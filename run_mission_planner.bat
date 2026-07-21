@echo off
setlocal

title AeroGNC-Lab - Waypoint Mission Planner
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

echo [AeroGNC-Lab] Opening the interactive Waypoint Mission Planner...
echo [AeroGNC-Lab] Left-click adds a waypoint, drag to move, double-click sets home,
echo [AeroGNC-Lab] right-click for the waypoint menu, mouse wheel zooms.
echo [AeroGNC-Lab] Use Import/Export/Validate/Run in the side panel.
echo.

"%PYTHON_EXE%" -m aerognc.cli mission-planner %*
set "RUN_STATUS=%ERRORLEVEL%"

if not "%RUN_STATUS%"=="0" (
    echo.
    echo [AeroGNC-Lab] Mission planner stopped with exit code %RUN_STATUS%.
    pause
)

exit /b %RUN_STATUS%

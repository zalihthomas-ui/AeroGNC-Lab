@echo off
setlocal

title AeroGNC-Lab - Environment Diagnostic
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" goto run_venv

where py >nul 2>nul
if not errorlevel 1 goto run_py

where python >nul 2>nul
if not errorlevel 1 goto run_python

echo [FAIL] No Python interpreter was found.
echo Install Python 3.12 or newer, then run this file again.
pause
exit /b 2

:run_venv
"%PYTHON_EXE%" scripts\diagnose_environment.py %*
set "RUN_STATUS=%ERRORLEVEL%"
goto diagnostic_finished

:run_py
py -3 scripts\diagnose_environment.py %*
set "RUN_STATUS=%ERRORLEVEL%"
goto diagnostic_finished

:run_python
python scripts\diagnose_environment.py %*
set "RUN_STATUS=%ERRORLEVEL%"

:diagnostic_finished
echo.
if "%RUN_STATUS%"=="0" (
    echo [AeroGNC-Lab] Environment is ready.
) else (
    echo [AeroGNC-Lab] Review the FAIL entries and their Next actions above.
)
pause
exit /b %RUN_STATUS%

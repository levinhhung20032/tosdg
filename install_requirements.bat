@echo off
rem Checks the Python version and the packages tosdg needs, and installs / upgrades what is missing.
rem Required: numpy, pillow, jsonschema. Optional: moderngl (GPU mesh renderer).
rem Usage: double-click, or "install_requirements.bat quiet" (no pause; used by start_studio.bat).
cd /d "%~dp0"
title tosdg requirements

where python >nul 2>nul && (set PY=python) || (where py >nul 2>nul && (set PY=py) || (set PY=))
if "%PY%"=="" (
  echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^).
  if /i not "%~1"=="quiet" pause
  exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  echo Python 3.11 or newer is required. Found:
  %PY% --version
  if /i not "%~1"=="quiet" pause
  exit /b 1
)


for %%P in (numpy pillow jsonschema moderngl) do call :check %%P
if defined MISSING_REQ (
  echo.
  echo Installing / upgrading:%MISSING_REQ%
  %PY% -m pip install --upgrade %MISSING_REQ%
  if errorlevel 1 (
    echo.
    echo pip failed. Check the internet connection and try: %PY% -m pip install %MISSING_REQ%
    if /i not "%~1"=="quiet" pause
    exit /b 1
  )
) else (
  echo All requirements are installed.
)
if /i not "%~1"=="quiet" pause
exit /b 0

:check
rem %1 = pip package name; keep the minimum versions in step with pyproject.toml
set "PKG=%1"
set "MINV="
if /i "%PKG%"=="numpy" set "MINV=1.26"
if /i "%PKG%"=="pillow" set "MINV=10"
if /i "%PKG%"=="jsonschema" set "MINV=4.20"
if /i "%PKG%"=="moderngl" set "MINV=5.8"
%PY% -c "import sys,importlib.metadata as m; v=m.version('%PKG%'); t=lambda s:tuple(int(x) for x in s.split('.')[:2] if x.isdigit()); sys.exit(0 if t(v)>=t('%MINV%') else 1)" >nul 2>nul
if errorlevel 1 (
  echo   missing or too old: %PKG% ^(needs ^>= %MINV%^)
  set "MISSING_REQ=%MISSING_REQ% "%PKG%>=%MINV%""
) else (
  echo   ok: %PKG%
)
exit /b 0

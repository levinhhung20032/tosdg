@echo off
rem Downloads the background photos listed in data\backgrounds\index.json that this copy does not have yet
rem (the photos are not in git), from Wikimedia Commons, and checks them against the index.
rem Usage: double-click, or "fetch_backgrounds.bat [--force] [--no-masks] [--dry-run] [quiet]".
cd /d "%~dp0"
title tosdg - fetch background photos

where python >nul 2>nul && (set PY=python) || (where py >nul 2>nul && (set PY=py) || (set PY=))
if "%PY%"=="" (
  echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^).
  if /i not "%~1"=="quiet" pause
  exit /b 1
)
%PY% -c "import numpy, PIL, jsonschema" 2>nul || call install_requirements.bat quiet

set ARGS=
for %%A in (%*) do if /i not "%%~A"=="quiet" call set ARGS=%%ARGS%% %%A
%PY% tools\fetch_backgrounds.py %ARGS%
set RC=%errorlevel%
echo.
if "%RC%"=="0" (echo Done.) else (echo Finished with problems: see the PROBLEM lines above.)
echo %* | find /i "quiet" >nul || pause
exit /b %RC%

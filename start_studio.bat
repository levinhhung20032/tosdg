@echo off
rem Double-click to launch the tosdg studio and open it in the browser.
rem A Studio that is already running is closed first (forced restart): two Studios on one port or one workspace would
rem overwrite each other's state. Any other program on port 8765 is closed too, and the Studio tab of the old run closes
rem itself when the new tab opens.
cd /d "%~dp0"
set PYTHONPATH=src
title tosdg studio

where python >nul 2>nul && (set PY=python) || (set PY=py)

rem Make sure the packages are there; if one is missing (or too old) install_requirements.bat installs it.
%PY% -c "import numpy, PIL, jsonschema, moderngl" 2>nul || call install_requirements.bat quiet

:start
rem Close a running Studio and whatever else holds port 8765; stops here (with the program's name) only if the port cannot be freed.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\stop_studio.ps1" -Port 8765
if errorlevel 1 (
  pause
  goto :eof
)
rem The studio server opens the browser itself and stops when the browser tab is closed.
%PY% -m tosdg studio --exit-on-close
if errorlevel 1 (
  echo.
  echo Studio exited with an error ^(see the message above^). If a package is missing, run install_requirements.bat
  pause
  goto :eof
)

rem Bring this window back to the front (the browser has the focus when its tab is closed); the script gets around Windows' focus-stealing block.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bring_to_front.ps1" >nul 2>nul
echo.
rem The tab was closed: shut down or restart. Counts down 5 s; each second the previous line is erased (ANSI: up one line + clear it) and the new one printed. Shuts down if there is no answer.
for /f %%e in ('echo prompt $E^| cmd') do set "ESC=%%e"
echo Studio stopped.
echo - S: Shutdown
echo - R: Restart
set /a left=5
echo Shutting down in %left% s...
:wait
choice /c SRX /t 1 /d X /n >nul
if errorlevel 3 goto tick
if errorlevel 2 goto restart
goto :eof

:tick
set /a left-=1
if %left% leq 0 goto :eof
echo %ESC%[1A%ESC%[2KShutting down in %left% s...
goto wait

:restart
echo.
goto start

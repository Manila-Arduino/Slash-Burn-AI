@echo off

REM Define the folder for the autostart script
set AUTOSTART_FOLDER=%appdata%\Microsoft\Windows\Start Menu\Programs\Startup
set SCRIPT_DIR=%~dp0
for %%i in ("%SCRIPT_DIR%..") do set PROJECT_PATH=%%~fi
set VENV_PATH=%PROJECT_PATH%\venv
set PYTHON_SCRIPT=%PROJECT_PATH%\app.py

REM Create the autostart script
echo @echo off > "%AUTOSTART_FOLDER%\ai.bat"
echo cd "%PROJECT_PATH%" >> "%AUTOSTART_FOLDER%\ai.bat"
echo call "%VENV_PATH%\Scripts\activate.bat" >> "%AUTOSTART_FOLDER%\ai.bat"
echo python "%PYTHON_SCRIPT%" >> "%AUTOSTART_FOLDER%\ai.bat"
echo pause >> "%AUTOSTART_FOLDER%\ai.bat"

echo Autostart script created at: %AUTOSTART_FOLDER%\ai.bat
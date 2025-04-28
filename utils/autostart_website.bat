@echo off
setlocal

:: Get Startup folder path
set "AUTOSTART_FOLDER=%appdata%\Microsoft\Windows\Start Menu\Programs\Startup"

:: Prompt for website URL
set /p "URL=Enter website URL: "

:: Create website.bat in Startup folder
(
  echo @echo off
  echo start "" chrome --kiosk "%URL%"
  echo exit
) > "%AUTOSTART_FOLDER%\website.bat"

echo Created startup script:
echo   %AUTOSTART_FOLDER%\website.bat
endlocal
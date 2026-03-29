@echo off
REM RTSP Configuration Helper
REM Quick tool to enable/disable RTSP mode

echo.
echo ================================================
echo   RTSP Configuration Helper
echo ================================================
echo.
echo Current Settings:
echo.

REM Read current config (simple check)
findstr "USE_RTSP = True" config.py >nul
if %errorlevel% == 0 (
    echo   Mode: RTSP ENABLED
    findstr "RTSP_URL" config.py | findstr /v "# RTSP_URL"
) else (
    echo   Mode: WEBCAM
    findstr "CAMERA_INDEX" config.py | findstr /v "# CAMERA"
)

echo.
echo ================================================
echo.
echo What would you like to do?
echo.
echo   1. Enable RTSP Mode
echo   2. Disable RTSP (Use Webcam)
echo   3. Test RTSP Connection
echo   4. Run Main Application
echo   5. Start go2rtc Relay
echo   6. Setup Phone Camera -> go2rtc
echo   7. Exit
echo.

set /p choice="Enter choice (1-7): "

if "%choice%"=="1" goto enable_rtsp
if "%choice%"=="2" goto disable_rtsp
if "%choice%"=="3" goto test_rtsp
if "%choice%"=="4" goto run_app
if "%choice%"=="5" goto start_go2rtc
if "%choice%"=="6" goto setup_phone
if "%choice%"=="7" goto end

echo Invalid choice!
pause
goto end

:start_go2rtc
echo.
echo Starting go2rtc relay...
call start_go2rtc.bat
goto end

:setup_phone
echo.
echo Configuring phone camera input for go2rtc...
call setup_phone_camera.bat
goto end

:enable_rtsp
echo.
echo Enabling RTSP Mode...
echo.
set /p rtsp_url="Enter RTSP URL [rtsp://127.0.0.1:8554/live]: "
if "%rtsp_url%"=="" set rtsp_url=rtsp://127.0.0.1:8554/cam

REM Backup config
copy config.py config.py.bak >nul

REM Update config using PowerShell
powershell -Command "(Get-Content config.py) -replace 'USE_RTSP = False', 'USE_RTSP = True' | Set-Content config.py"
powershell -Command "(Get-Content config.py) -replace 'RTSP_URL = \".*\"', 'RTSP_URL = \"%rtsp_url%\"' | Set-Content config.py"

echo.
echo ✓ RTSP Mode Enabled!
echo   URL: %rtsp_url%
echo.
echo Next steps:
echo   1. Make sure OBS is running and streaming
echo   2. Run 'python test_rtsp.py' to test connection
echo   3. Run 'python main.py' to start detection
echo.
pause
goto end

:disable_rtsp
echo.
echo Disabling RTSP (Switching to Webcam)...

REM Backup config
copy config.py config.py.bak >nul

REM Update config using PowerShell
powershell -Command "(Get-Content config.py) -replace 'USE_RTSP = True', 'USE_RTSP = False' | Set-Content config.py"

echo.
echo ✓ Webcam Mode Enabled!
echo.
echo Run 'python main.py' to start detection with webcam
echo.
pause
goto end

:test_rtsp
echo.
echo Testing RTSP Connection...
echo.
python test_rtsp.py
pause
goto end

:run_app
echo.
echo Starting Application...
echo.
python main.py
pause
goto end

:end
echo.
echo Goodbye!
timeout /t 2 >nul

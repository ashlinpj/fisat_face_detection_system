@echo off
setlocal

cd /d "%~dp0"

echo.
echo ================================================
echo   go2rtc Local Relay Starter
echo ================================================
echo.

netstat -ano | findstr ":8554" >nul
set "RTSP_ALREADY=%errorlevel%"
netstat -ano | findstr ":1984" >nul
set "API_ALREADY=%errorlevel%"

if "%RTSP_ALREADY%"=="0" if "%API_ALREADY%"=="0" (
    echo go2rtc is already running.
    echo RTSP relay: rtsp://127.0.0.1:8554/cam
    echo Web UI:     http://127.0.0.1:1984
    echo.
    echo If you need a restart, run:
    echo   taskkill /IM go2rtc.exe /F
    echo then run this script again.
    echo.
    pause
    exit /b 0
)

set "GO2RTC_BIN=%cd%\go2rtc.exe"
if not exist "%GO2RTC_BIN%" (
    if exist "%cd%\go2rtc" (
        set "GO2RTC_BIN=%cd%\go2rtc"
    ) else (
        echo [ERROR] go2rtc binary not found in project folder.
        echo.
        echo Download go2rtc Windows binary from:
        echo   https://github.com/AlexxIT/go2rtc/releases
        echo.
        echo Put either go2rtc.exe or go2rtc in this folder:
        echo   %cd%
        echo.
        pause
        exit /b 1
    )
)

if not exist "go2rtc.yaml" (
    echo [ERROR] go2rtc.yaml not found.
    echo Create it or copy from repository defaults.
    pause
    exit /b 1
)

echo Starting go2rtc with config go2rtc.yaml ...
start "go2rtc" /B "%GO2RTC_BIN%" -config "%cd%\go2rtc.yaml"

timeout /t 2 >nul

netstat -ano | findstr ":8554" >nul
set "RTSP_OK=%errorlevel%"
netstat -ano | findstr ":1984" >nul
set "API_OK=%errorlevel%"

if not "%RTSP_OK%"=="0" (
    echo.
    echo [ERROR] go2rtc did not start correctly. Port 8554 is not listening.
    echo Possible reasons:
    echo   1. Wrong binary for your Windows architecture
    echo   2. Invalid source URL in go2rtc.yaml
    echo   3. Port blocked or already in use
    echo.
    echo Check binary and config, then retry.
    pause
    exit /b 1
)

echo.
echo go2rtc started.
echo RTSP relay: rtsp://127.0.0.1:8554/cam
if "%API_OK%"=="0" (
    echo Web UI:     http://127.0.0.1:1984
) else (
    echo Web UI:     not listening on 1984
)

echo.
echo Press any key to exit this launcher.
pause >nul

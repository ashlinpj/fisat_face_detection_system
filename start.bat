@echo off
echo ============================================================
echo   College Canteen Face Detection System - Quick Start
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo Python found. Starting setup...
echo.

REM Run setup
python setup.py

echo.
echo ============================================================
echo   Setup complete! Starting GUI application...
echo ============================================================
echo.

REM Start GUI application
python gui_app.py

pause

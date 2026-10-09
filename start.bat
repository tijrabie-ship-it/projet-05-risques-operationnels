@echo off
cd /d "%~dp0"
chcp 65001 >nul
title Smart Murabaha Evaluation System
color 0A

echo ===================================================
echo   Welcome to the Smart Murabaha Evaluation System
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Python is not installed. Please install Python first.
    pause
    exit /b
)

echo [1/3] Checking and installing required libraries...
echo This may take a few minutes if this is the first time...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [Error] Problem installing libraries. Check your internet connection.
    pause
    exit /b
)
echo All libraries verified successfully!
echo.
echo [2/3] The browser will open automatically shortly...
timeout /t 3 >nul
start http://127.0.0.1:5000

echo.
echo [3/3] Starting the AI server...
echo Press Ctrl + C to stop the server at any time.
echo ---------------------------------------------------

python app.py
pause

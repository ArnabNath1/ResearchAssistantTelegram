@echo off
echo =========================================
echo  Research Assistant Bot - Setup Script
echo =========================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
echo OK.

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo OK.

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip
echo OK.

echo [4/5] Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install some packages.
    pause
    exit /b 1
)
echo OK.

echo [5/5] Creating logs directory...
if not exist logs mkdir logs
echo OK.

echo.
echo =========================================
echo  Setup COMPLETE!
echo =========================================
echo.
echo NEXT STEP: Add your Telegram Bot Token to .env
echo   TELEGRAM_BOT_TOKEN=your_token_here
echo.
echo To run the bot:
echo   venv\Scripts\activate
echo   python main.py
echo.
pause

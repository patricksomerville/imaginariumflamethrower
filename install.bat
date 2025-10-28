@echo off
REM Installation script for Windows

echo ==========================================
echo Screenplay Voice Assistant - Installation
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python is not installed!
    echo Please install Python 3.8 or higher from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Python found: %PYTHON_VERSION%
echo.

REM Install required packages
echo Installing required packages...
echo This may take a few minutes...
echo.

python -m pip install --upgrade pip
python -m pip install openai gradio python-dotenv

echo.
echo ==========================================
echo Installation Complete!
echo ==========================================
echo.

REM Check if .env file exists
if exist .env (
    echo [OK] .env file found
) else (
    echo [!] No .env file found
    echo.
    if exist .env.example (
        echo Creating .env file from template...
        copy .env.example .env
        echo.
        echo Please edit .env and add your OpenAI API key
        echo You can open it with: notepad .env
        echo.
    )
)

echo.
echo ==========================================
echo Ready to run!
echo ==========================================
echo.
echo Start the voice assistant with:
echo   python voice_chat_simple.py
echo.
echo Then open in your browser:
echo   http://localhost:7860
echo.
pause

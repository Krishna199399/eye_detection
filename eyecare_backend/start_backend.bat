@echo off
echo ========================================
echo        EyeCare AI Backend Server
echo ========================================
echo.

REM Change to backend directory
cd /d "%~dp0"

echo 🔍 Checking Python installation...
REM Try py command first (Windows Python Launcher)
py --version >nul 2>&1
if errorlevel 1 (
    REM Fall back to python command
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Python not found! Please install Python 3.8+ from python.org
        echo    Make sure to check "Add Python to PATH" during installation
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
) else (
    set PYTHON_CMD=py
)

echo ✅ Python found
echo.

echo 🔍 Checking dependencies...
%PYTHON_CMD% -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Dependencies not found. Installing...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed
)

echo ✅ Dependencies ready
echo.

echo 🚀 Starting EyeCare AI Backend Server...
echo    Backend will run on: http://localhost:8000
echo    API docs available at: http://localhost:8000/docs
echo.
echo ⚠️  Keep this window open while using the application
echo    Press Ctrl+C to stop the server
echo.

%PYTHON_CMD% main.py

echo.
echo 🛑 Backend server stopped
pause

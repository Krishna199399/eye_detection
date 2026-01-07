@echo off
echo ========================================
echo        EyeCare AI Frontend Client
echo ========================================
echo.

REM Change to frontend directory
cd /d "%~dp0"

echo 🔍 Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found! Please install Node.js 16+ from nodejs.org
    pause
    exit /b 1
)

echo ✅ Node.js found
echo.

echo 🔍 Checking npm installation...
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm not found! Please install Node.js from nodejs.org (includes npm)
    pause
    exit /b 1
)

echo ✅ npm found
echo.

echo 🔍 Checking dependencies...
if not exist "node_modules\" (
    echo ⚠️  Dependencies not found. Installing...
    npm install
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed
)

echo ✅ Dependencies ready
echo.

echo 🚀 Starting EyeCare AI Frontend Client...
echo    Frontend will run on: http://localhost:5173
echo    Make sure backend is running on: http://localhost:8000
echo.
echo ⚠️  Keep this window open while using the application
echo    Press Ctrl+C to stop the client
echo.

npm run start

echo.
echo 🛑 Frontend client stopped
pause

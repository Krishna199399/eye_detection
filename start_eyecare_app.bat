@echo off
echo ========================================
echo           EyeCare AI Application
echo        Complete Frontend + Backend
echo ========================================
echo.

REM Change to project root directory
cd /d "%~dp0"

echo 🚀 Starting EyeCare AI Application...
echo.
echo This will launch both:
echo   - Backend Server (http://localhost:8000)
echo   - Frontend Client (http://localhost:5173)
echo.
echo ⚠️  Two terminal windows will open. Keep both open while using the app.
echo.

echo 🔧 Starting Backend Server...
start "EyeCare AI Backend" cmd /c "cd eyecare_backend && start_backend.bat"

echo ⏳ Waiting for backend to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

echo 🔧 Starting Frontend Client...
start "EyeCare AI Frontend" cmd /c "cd eyecare_frontend && start_frontend.bat"

echo.
echo ✅ EyeCare AI Application is starting!
echo.
echo 📌 Important URLs:
echo    • Frontend App: http://localhost:5173
echo    • Backend API: http://localhost:8000
echo    • API Documentation: http://localhost:8000/docs
echo.
echo 📋 Next Steps:
echo    1. Wait for both servers to start (about 30 seconds)
echo    2. Open http://localhost:5173 in your browser
echo    3. Upload an eye image for AI analysis
echo.
echo ⚠️  To stop the application:
echo    - Close both terminal windows that opened
echo    - Or press Ctrl+C in each terminal
echo.

pause

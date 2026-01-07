@echo off
echo ========================================
echo     EyeCare AI Model Setup & Training
echo ========================================
echo.

REM Change to backend directory
cd /d "%~dp0"

echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ from python.org
    echo    Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found
echo.

echo 🔧 Setting up directories...
if not exist "models" mkdir models
if not exist "models\saved_models" mkdir models\saved_models
if not exist "uploads" mkdir uploads
if not exist "database" mkdir database
echo ✅ Directories created

echo.
echo 📦 Installing Python dependencies...
echo    This may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    echo    Please check your internet connection and try again
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully
echo.

echo 🔍 Checking for existing model...
if exist "models\saved_models\eye_disease_model.h5" (
    echo ✅ Model already exists: models\saved_models\eye_disease_model.h5
    echo.
    set /p choice="Do you want to retrain the model? (y/N): "
    if /i not "%choice%"=="y" (
        echo 📋 Skipping training, using existing model
        goto :start_server
    )
)

echo.
echo 🧠 Training AI Model...
echo    This will train a CNN model for eye disease detection
echo    Training time: ~15-30 minutes (depends on your hardware)
echo.

REM Check if training script exists
if exist "build_tf215_model.py" (
    echo 🚀 Starting model training with build_tf215_model.py...
    python build_tf215_model.py
) else (
    echo ⚠️  Training script not found. Checking for alternative...
    if exist "train_eye_disease_model.py" (
        echo 🚀 Starting model training with train_eye_disease_model.py...
        python train_eye_disease_model.py --epochs 25 --batch_size 16
    ) else (
        echo ❌ No training script found!
        echo    Available files:
        dir *.py /b
        echo.
        echo    Please ensure you have a training script in this directory
        pause
        exit /b 1
    )
)

if errorlevel 1 (
    echo ❌ Model training failed!
    echo    Please check the error messages above
    pause
    exit /b 1
)

echo ✅ Model training completed successfully!
echo    Model saved to: models\saved_models\eye_disease_model.h5
echo.

:start_server
echo 🎯 Setup complete! 
echo.
echo 📊 Your EyeCare AI system is ready with:
echo    ✅ Python dependencies installed
echo    ✅ Directory structure created  
echo    ✅ AI model trained and ready
echo    ✅ Database configured
echo.
echo 🚀 Starting backend server for testing...
echo    Backend will run on: http://localhost:8000
echo    Press Ctrl+C to stop the server when done testing
echo.

python main.py

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📋 Next steps:
echo    1. Keep the backend running
echo    2. Open a new terminal and navigate to the frontend directory
echo    3. Run: start_frontend.bat
echo    4. Open http://localhost:4028 in your browser
echo.
pause

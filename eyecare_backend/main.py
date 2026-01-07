#!/usr/bin/env python3
"""
EyeCare AI Backend - Main Entry Point
Requires ML model to be available for operation
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 EyeCare AI Backend - Initializing...")
print("=" * 60)

# Check if model exists and TensorFlow is available
model_path = "models/saved_models/eye_disease_model.h5"
model_exists = os.path.exists(model_path)

try:
    import tensorflow as tf
    tf_available = True
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    tf_available = False
    print("❌ TensorFlow not available")
    print("Please install TensorFlow: pip install tensorflow==2.15.0")
    sys.exit(1)

try:
    from models.cnn_model import EyeDiseaseModel
    from utils.image_preprocessing import preprocess_image
    model_classes_available = True
    print("✅ Model classes imported successfully")
except ImportError as e:
    model_classes_available = False
    print(f"❌ Model classes import failed: {e}")
    sys.exit(1)

# Verify all requirements are met
if not model_exists:
    print("❌ ML Model file not found!")
    print(f"   Expected location: {model_path}")
    print("   To train a model, run: python build_tf215_model.py")
    sys.exit(1)

if not (model_exists and tf_available and model_classes_available):
    print("❌ Requirements not met for ML backend")
    print(f"   Model file exists: {model_exists}")
    print(f"   TensorFlow available: {tf_available}")
    print(f"   Model classes available: {model_classes_available}")
    sys.exit(1)

print("🧠 ML Model detected - Starting with real AI predictions")
from ml_backend import app

print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting EyeCare AI Backend (ML mode)...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

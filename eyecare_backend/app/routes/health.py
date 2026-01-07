from fastapi import APIRouter
from datetime import datetime
import psutil
import os

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify API status and system resources
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "EyeCare AI Backend",
        "version": "1.0.0",
        "system": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
    }

@router.get("/model-status")
async def model_status():
    """
    Check if the CNN model is loaded and ready for predictions
    """
    try:
        from models.cnn_model import EyeDiseaseModel
        model_instance = EyeDiseaseModel()
        model_loaded = model_instance.is_model_loaded()
        
        return {
            "model_status": "loaded" if model_loaded else "not_loaded",
            "model_ready": model_loaded,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "model_status": "error",
            "model_ready": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

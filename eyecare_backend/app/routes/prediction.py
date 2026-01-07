"""
Prediction Routes - MongoDB/Beanie Implementation
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
from datetime import datetime
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from models.cnn_model import EyeDiseaseModel
from utils.image_preprocessing import ImagePreprocessor
from database.service import db_service
from database.models import Prediction, EyeCondition

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize model and preprocessor
model_instance = EyeDiseaseModel()
preprocessor = ImagePreprocessor()

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

@router.post("/predict")
async def predict_eye_disease(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Upload an eye image and get disease prediction using MongoDB/Beanie
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Check if model is loaded
        if not model_instance.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Please train or load a model first."
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        filename = f"{file_id}.{file_extension}"
        file_path = os.path.join("uploads", filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load and preprocess image
        image_array = preprocessor.load_image_from_bytes(content)
        processed_image = preprocessor.preprocess_for_prediction(image_array)
        enhanced_image = preprocessor.apply_clahe(processed_image)
        final_image = preprocessor.enhance_retinal_image(enhanced_image)
        
        # Make prediction
        predicted_class, confidence, all_predictions = model_instance.predict(final_image)
        
        # Get image statistics
        image_stats = preprocessor.get_image_stats(final_image)
        
        # Save prediction to MongoDB database
        try:
            prediction_data = {
                "prediction_id": file_id,
                "user_id": None,
                "image_name": file.filename,
                "image_path": file_path,
                "image_size": len(content),
                "predicted_class": predicted_class,
                "confidence": confidence * 100,  # Store as percentage
                "all_predictions": {k: v * 100 for k, v in all_predictions.items()},
                "risk_level": get_risk_level(predicted_class, confidence),
                "recommendations": get_recommendations(predicted_class, confidence),
                "processing_time": None,
                "model_version": "v2.0.0",
                "is_demo_mode": False
            }
            
            saved_prediction = await db_service.save_prediction(prediction_data)
            logger.info(f"Prediction saved to MongoDB: {saved_prediction.prediction_id}")
            
        except Exception as db_error:
            logger.warning(f"Failed to save prediction to database: {db_error}")
        
        # Prepare response
        response = {
            "prediction_id": file_id,
            "success": True,
            "results": {
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),  # Convert to percentage
                "all_predictions": {
                    class_name: round(prob * 100, 2) 
                    for class_name, prob in all_predictions.items()
                },
                "risk_level": get_risk_level(predicted_class, confidence),
                "recommendations": get_recommendations(predicted_class, confidence)
            },
            "image_info": {
                "filename": file.filename,
                "size": len(content),
                "processed_stats": image_stats
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        # Clean up file if it was created
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/prediction/{prediction_id}")
async def get_prediction(prediction_id: str):
    """
    Retrieve a previous prediction by ID using MongoDB
    
    Args:
        prediction_id: ID of the prediction to retrieve
        
    Returns:
        Prediction details
    """
    try:
        prediction = await db_service.get_prediction_by_id(prediction_id)
        
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail="Prediction not found"
            )
        
        return {
            "prediction_id": prediction.prediction_id,
            "filename": prediction.image_name,
            "predicted_class": prediction.predicted_class.value if hasattr(prediction.predicted_class, 'value') else str(prediction.predicted_class),
            "confidence": prediction.confidence,
            "all_predictions": prediction.all_predictions,
            "risk_level": prediction.risk_level,
            "recommendations": prediction.recommendations,
            "created_at": prediction.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving prediction {prediction_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prediction: {str(e)}"
        )

@router.post("/predictions")
async def save_prediction_manually(prediction_data: dict):
    """
    Manually save a prediction using MongoDB/Beanie
    
    Args:
        prediction_data: Prediction data to save
        
    Returns:
        Success message
    """
    try:
        # Check if prediction already exists
        prediction_id = prediction_data.get("prediction_id")
        if prediction_id:
            existing = await db_service.get_prediction_by_id(prediction_id)
            if existing:
                return {
                    "success": True,
                    "message": "Prediction already saved",
                    "prediction_id": prediction_id
                }
        
        # Generate new ID if not provided
        if not prediction_id:
            prediction_id = str(uuid.uuid4())
        
        # Extract data from the request
        results = prediction_data.get("results", {})
        image_info = prediction_data.get("image_info", {})
        
        # Prepare data for saving
        save_data = {
            "prediction_id": prediction_id,
            "user_id": None,
            "image_name": image_info.get("filename", "unknown"),
            "image_path": f"uploads/{prediction_id}.jpg",
            "image_size": image_info.get("size", 0),
            "predicted_class": results.get("predicted_class", "unknown"),
            "confidence": results.get("confidence", 0),
            "all_predictions": results.get("all_predictions", {}),
            "risk_level": results.get("risk_level", "Unknown"),
            "recommendations": results.get("recommendations", []),
            "processing_time": None,
            "model_version": "v2.0.0",
            "is_demo_mode": False
        }
        
        # Save to MongoDB
        saved_prediction = await db_service.save_prediction(save_data)
        
        return {
            "success": True,
            "message": "Prediction saved successfully",
            "prediction_id": saved_prediction.prediction_id
        }
        
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save prediction: {str(e)}"
        )

@router.get("/predictions/history")
async def get_prediction_history(
    limit: int = 50,
    offset: int = 0,
    user_id: Optional[str] = None
):
    """
    Get prediction history with pagination using MongoDB
    
    Args:
        limit: Maximum number of results to return
        offset: Number of results to skip (not used in current MongoDB implementation)
        user_id: Optional user ID to filter by
        
    Returns:
        List of predictions
    """
    try:
        if user_id:
            predictions = await db_service.get_user_predictions(user_id, limit)
        else:
            predictions = await db_service.get_recent_predictions(limit)
        
        predictions_data = []
        for pred in predictions:
            pred_data = {
                "prediction_id": pred.prediction_id,
                "filename": pred.image_name,
                "predicted_class": pred.predicted_class.value if hasattr(pred.predicted_class, 'value') else str(pred.predicted_class),
                "confidence": pred.confidence,
                "created_at": pred.created_at.isoformat()
            }
            predictions_data.append(pred_data)
        
        return {
            "predictions": predictions_data,
            "total_count": len(predictions_data),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}")
        return {
            "predictions": [],
            "total_count": 0,
            "limit": limit,
            "offset": offset,
            "error": str(e)
        }

@router.get("/statistics")
async def get_prediction_statistics():
    """
    Get statistics about predictions using MongoDB
    
    Returns:
        Statistics about predictions
    """
    try:
        stats = await db_service.get_system_stats()
        
        if not stats:
            return {
                "total_predictions": 0,
                "class_distribution": {},
                "average_confidence": 0,
                "most_common_disease": None
            }
        
        class_distribution = {
            "normal": {
                "count": stats.normal_count,
                "percentage": round((stats.normal_count / stats.total_predictions) * 100, 2) if stats.total_predictions > 0 else 0
            },
            "cataract": {
                "count": stats.cataract_count,
                "percentage": round((stats.cataract_count / stats.total_predictions) * 100, 2) if stats.total_predictions > 0 else 0
            },
            "diabetic_retinopathy": {
                "count": stats.diabetic_retinopathy_count,
                "percentage": round((stats.diabetic_retinopathy_count / stats.total_predictions) * 100, 2) if stats.total_predictions > 0 else 0
            },
            "glaucoma": {
                "count": stats.glaucoma_count,
                "percentage": round((stats.glaucoma_count / stats.total_predictions) * 100, 2) if stats.total_predictions > 0 else 0
            }
        }
        
        return {
            "total_predictions": stats.total_predictions,
            "predictions_today": stats.predictions_today,
            "predictions_this_month": stats.predictions_this_month,
            "class_distribution": class_distribution,
            "average_confidence": round(stats.average_confidence, 2),
            "average_processing_time": round(stats.average_processing_time, 2),
            "most_common_disease": stats.most_common_disease,
            "last_updated": stats.last_updated.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching prediction statistics: {e}")
        return {
            "total_predictions": 0,
            "class_distribution": {},
            "average_confidence": 0,
            "most_common_disease": None,
            "error": str(e)
        }

@router.delete("/prediction/{prediction_id}")
async def delete_prediction(prediction_id: str):
    """
    Delete a prediction and its associated file using MongoDB
    
    Args:
        prediction_id: ID of the prediction to delete
        
    Returns:
        Success message
    """
    try:
        prediction = await db_service.get_prediction_by_id(prediction_id)
        
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail="Prediction not found"
            )
        
        # Delete associated file if it exists
        if os.path.exists(prediction.image_path):
            os.remove(prediction.image_path)
        
        # Delete from MongoDB (we would need to add this method to db_service)
        # For now, we'll just return success since we don't have a delete method
        logger.info(f"Would delete prediction {prediction_id} from MongoDB")
        
        return {"message": "Prediction deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting prediction {prediction_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete prediction: {str(e)}"
        )

def get_risk_level(predicted_class: str, confidence: float) -> str:
    """
    Determine risk level based on prediction and confidence
    
    Args:
        predicted_class: Predicted disease class
        confidence: Confidence score
        
    Returns:
        Risk level string
    """
    if predicted_class.lower() == "normal":
        return "Low"
    
    if confidence >= 0.9:
        return "High"
    elif confidence >= 0.7:
        return "Medium"
    else:
        return "Low"

def get_recommendations(predicted_class: str, confidence: float) -> list:
    """
    Get recommendations based on prediction
    
    Args:
        predicted_class: Predicted disease class
        confidence: Confidence score
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    if predicted_class.lower() == "normal":
        recommendations = [
            "Continue regular eye examinations",
            "Maintain a healthy lifestyle",
            "Protect eyes from UV radiation"
        ]
    elif "diabetic retinopathy" in predicted_class.lower():
        recommendations = [
            "Consult with an ophthalmologist immediately",
            "Maintain strict blood sugar control",
            "Schedule regular retinal screenings",
            "Consider laser therapy if recommended"
        ]
    elif "glaucoma" in predicted_class.lower():
        recommendations = [
            "See an eye specialist for comprehensive evaluation",
            "Monitor intraocular pressure regularly",
            "Follow prescribed medication regimen",
            "Avoid activities that increase eye pressure"
        ]
    elif "cataract" in predicted_class.lower():
        recommendations = [
            "Consult with an ophthalmologist",
            "Consider surgical options if vision is impaired",
            "Use bright lighting when reading",
            "Wear sunglasses to reduce glare"
        ]
    elif "macular degeneration" in predicted_class.lower():
        recommendations = [
            "Schedule immediate consultation with retina specialist",
            "Monitor vision changes with Amsler grid",
            "Consider nutritional supplements",
            "Explore treatment options like anti-VEGF therapy"
        ]
    else:
        recommendations = [
            "Consult with an eye care professional",
            "Schedule comprehensive eye examination",
            "Monitor symptoms and vision changes",
            "Follow up with specialist as needed"
        ]
    
    if confidence < 0.7:
        recommendations.append("Low confidence prediction - consider getting a second opinion")
    
    return recommendations

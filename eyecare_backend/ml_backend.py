#!/usr/bin/env python3
"""
EyeCare AI Backend with Real ML Model
Uses your trained TensorFlow model for actual eye disease predictions
"""
import os
import sys
import uuid
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import uvicorn
import logging
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize logger
logger = logging.getLogger(__name__)

print("🚀 EyeCare AI Backend - Loading ML Model...")
print("=" * 60)

try:
    import tensorflow as tf
    from models.cnn_model import EyeDiseaseModel
    from utils.image_preprocessing import preprocess_image
    print(f"✅ TensorFlow version: {tf.__version__}")
    print("✅ Model classes imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please make sure all dependencies are installed")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="EyeCare AI Backend - ML Powered",
    description="Deep Learning API for Eye Disease Detection using trained CNN model",
    version="2.0.0"
)

# Authentication endpoints
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional
import hashlib
import jwt
from datetime import datetime, timedelta

# Import database models and service
from database.models import UserRole, Gender, UserCreate as DatabaseUserCreate
from database.service import db_service

# JWT Configuration
SECRET_KEY = "eyecare-ai-secret-key-2024"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Pydantic models for authentication
class UserRegistration(BaseModel):
    firstName: str
    lastName: str
    email: str
    phone: str
    password: str
    userType: str
    medicalLicense: Optional[str] = None
    facilityName: Optional[str] = None
    facilityAddress: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_data: dict

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers to the frontend
)

# Global model instance
ml_model = None

@app.on_event("startup")
async def startup_event():
    """Load the trained ML model on startup"""
    global ml_model
    
    print("🧠 Loading trained eye disease model...")
    
    # Initialize MongoDB connection
    try:
        from database.connection import init_database
        await init_database()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("Note: Database features may not work properly")
    
    try:
        # Initialize model
        ml_model = EyeDiseaseModel()
        
        # Check if model file exists
        model_path = "models/saved_models/eye_disease_model.h5"
        if os.path.exists(model_path):
            success = ml_model.load_model()
            if success:
                print(f"✅ Model loaded successfully from {model_path}")
                print(f"📊 Model classes: {ml_model.classes}")
                print("🚀 EyeCare AI Backend ready with real ML predictions!")
            else:
                print("❌ Failed to load model")
                ml_model = None
        else:
            print(f"❌ Model file not found: {model_path}")
            print("Please train the model first using: python build_tf215_model.py")
            ml_model = None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        ml_model = None
    
    print("🔗 CORS enabled for React frontend (port 4028)")
    print("=" * 60)

@app.post("/api/v1/auth/register")
async def register_user(user_data: UserRegistration):
    """Register a new user"""
    try:
        # Check if user already exists in database
        existing_user = await db_service.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # Validate user type
        if user_data.userType not in ["patient", "healthcare_professional"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid user type"
            )
        
        # Validate healthcare professional fields
        if user_data.userType == "healthcare_professional":
            if not all([user_data.medicalLicense, user_data.facilityName, user_data.facilityAddress]):
                raise HTTPException(
                    status_code=400,
                    detail="Medical license, facility name, and address are required for healthcare professionals"
                )
        
        # Map frontend userType to database UserRole
        role = UserRole.DOCTOR if user_data.userType == "healthcare_professional" else UserRole.PATIENT
        
        # Create database user model
        db_user_data = DatabaseUserCreate(
            email=user_data.email,
            password=user_data.password,
            full_name=f"{user_data.firstName} {user_data.lastName}",
            age=None,  # Not provided by frontend
            gender=None,  # Not provided by frontend
            role=role
        )
        
        # Create user in database
        created_user = await db_service.create_user(db_user_data)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_data.email, "user_id": str(created_user.id)},
            expires_delta=access_token_expires
        )
        
        # Prepare user data for response (keeping the frontend expected structure)
        user_response = {
            "user_id": str(created_user.id),
            "first_name": user_data.firstName,
            "last_name": user_data.lastName,
            "email": user_data.email,
            "phone": user_data.phone,
            "user_type": user_data.userType,
            "medical_license": user_data.medicalLicense,
            "facility_name": user_data.facilityName,
            "facility_address": user_data.facilityAddress,
            "is_verified": True,
            "full_name": created_user.full_name,
            "role": created_user.role.value,
            "created_at": created_user.created_at.isoformat()
        }
        
        print(f"✅ User registered successfully in MongoDB: {user_data.email} ({user_data.userType})")
        
        return {
            "success": True,
            "message": "User registered successfully",
            "token": access_token,
            "token_type": "bearer",
            "user": user_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Registration error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )

@app.post("/api/v1/auth/login")
async def login_user(credentials: UserLogin):
    """Login user and return access token"""
    try:
        # Check if user exists in database
        user = await db_service.get_user_by_email(credentials.email)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not db_service.verify_password(credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Update last login time
        await db_service.update_user_last_login(str(user.id))
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": credentials.email, "user_id": str(user.id)},
            expires_delta=access_token_expires
        )
        
        # Parse full name to get first and last name (for frontend compatibility)
        name_parts = user.full_name.split(" ", 1)
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        # Map database role to frontend user type
        user_type = "healthcare_professional" if user.role == UserRole.DOCTOR else "patient"
        
        # Prepare user data for response (keeping frontend expected structure)
        user_response = {
            "user_id": str(user.id),
            "first_name": first_name,
            "last_name": last_name,
            "email": user.email,
            "phone": "",  # Not stored in database model
            "user_type": user_type,
            "medical_license": None,  # Not stored in database model
            "facility_name": None,  # Not stored in database model
            "facility_address": None,  # Not stored in database model
            "is_verified": user.is_active,
            "full_name": user.full_name,
            "role": user.role.value,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        
        print(f"✅ User logged in successfully from MongoDB: {credentials.email}")
        
        return {
            "success": True,
            "message": "Login successful",
            "token": access_token,
            "token_type": "bearer",
            "user": user_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Login failed: {str(e)}"
        )

@app.post("/api/v1/auth/logout")
async def logout_user():
    """Logout user (client-side token removal)"""
    return {
        "success": True,
        "message": "Logged out successfully"
    }

@app.get("/api/v1/auth/check-user/{email}")
async def check_user_exists(email: str):
    """Debug endpoint to check if user exists in database"""
    try:
        user = await db_service.get_user_by_email(email)
        if user:
            return {
                "exists": True,
                "user_id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value,
                "created_at": user.created_at.isoformat(),
                "is_active": user.is_active
            }
        else:
            return {
                "exists": False,
                "message": "User not found in database"
            }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        from database.connection import close_database
        await close_database()
        print("👋 Disconnected from MongoDB")
    except Exception as e:
        print(f"Error during shutdown: {e}")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "EyeCare AI Backend",
        "version": "2.0.0",
        "mode": "ml_powered",
        "model_loaded": ml_model is not None and ml_model.is_model_loaded()
    }

@app.get("/api/v1/model-status")
async def model_status():
    """Check model status"""
    if ml_model is None or not ml_model.is_model_loaded():
        return {
            "model_status": "not_loaded",
            "model_ready": False,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "ML model not loaded. Please check server logs."
        }
    
    return {
        "model_status": "loaded",
        "model_ready": True,
        "timestamp": datetime.utcnow().isoformat(),
        "classes": ml_model.classes,
        "input_shape": f"{ml_model.img_height}x{ml_model.img_width}x{ml_model.channels}",
        "message": "ML model ready for predictions"
    }

@app.post("/api/v1/predict")
async def predict_eye_disease(file: UploadFile = File(...)):
    """
    Real ML prediction endpoint using trained model
    """
    if ml_model is None or not ml_model.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Please check server configuration."
        )
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Generate unique prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Read and preprocess image
        image_bytes = await file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Preprocess image for model
        processed_image = preprocess_image(
            pil_image, 
            target_size=(ml_model.img_height, ml_model.img_width)
        )
        
        # Make prediction
        predicted_class, confidence, all_predictions = ml_model.predict(processed_image)
        
        # Convert confidence to percentage
        confidence_percent = confidence * 100
        all_predictions_percent = {k: v * 100 for k, v in all_predictions.items()}
        
        # Determine risk level
        if confidence_percent >= 80:
            risk_level = "High"
        elif confidence_percent >= 60:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Get recommendations
        recommendations = get_recommendations(predicted_class, confidence_percent)
        
        # Save prediction to database (optional)
        prediction_saved = False
        try:
            from database.service import db_service
            
            # Create uploads directory
            os.makedirs("uploads", exist_ok=True)
            
            # Save uploaded file
            file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            filename = f"{prediction_id}.{file_extension}"
            file_path = os.path.join("uploads", filename)
            
            with open(file_path, "wb") as buffer:
                buffer.write(image_bytes)
            
            prediction_data = {
                "prediction_id": prediction_id,
                "user_id": None,  # TODO: Add user authentication
                "image_name": file.filename,
                "image_path": file_path,
                "image_size": len(image_bytes),
                "predicted_class": predicted_class,
                "confidence": round(confidence_percent, 2),
                "all_predictions": {k: round(v, 2) for k, v in all_predictions_percent.items()},
                "risk_level": risk_level,
                "recommendations": recommendations,
                "processing_time": None,  # TODO: Add timing
                "model_version": "ML-v2.0.0",
                "is_demo_mode": False
            }
            
            await db_service.save_prediction(prediction_data)
            prediction_saved = True
            print(f"✅ Prediction saved to database: {prediction_id}")
            
        except Exception as db_error:
            logger.warning(f"Failed to save prediction to database: {db_error}")
            print(f"⚠️ Database unavailable, prediction not saved: {db_error}")
            # Continue with response even if DB save fails
        
        # Prepare response
        response = {
            "prediction_id": prediction_id,
            "success": True,
            "results": {
                "predicted_class": predicted_class,
                "confidence": round(confidence_percent, 2),
                "all_predictions": {k: round(v, 2) for k, v in all_predictions_percent.items()},
                "risk_level": risk_level,
                "recommendations": recommendations
            },
            "image_info": {
                "filename": file.filename,
                "size": len(image_bytes),
                "processed_size": f"{ml_model.img_height}x{ml_model.img_width}"
            },
            "model_info": {
                "classes": ml_model.classes,
                "architecture": "Custom CNN"
            },
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "ml_prediction"
        }
        
        print(f"✅ Prediction completed: {predicted_class} ({confidence_percent:.2f}%)")
        return response
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Import the enhanced treatment recommendation system
try:
    from utils.treatment_recommendations import get_recommendations as get_simple_recommendations
    from utils.treatment_recommendations import TreatmentRecommendationSystem
    enhanced_recommendations_available = True
    print("✅ Enhanced treatment recommendations loaded")
except ImportError as e:
    print(f"⚠️ Enhanced recommendations not available: {e}")
    enhanced_recommendations_available = False

def get_recommendations(predicted_class: str, confidence: float) -> list:
    """Get medical recommendations based on prediction (backward compatibility)"""
    if enhanced_recommendations_available:
        return get_simple_recommendations(predicted_class, confidence)
    
    # Fallback to original recommendations if enhanced system not available
    base_recommendations = {
        "normal": [
            "Your eye examination appears normal",
            "Continue regular eye check-ups as recommended by your doctor",
            "Maintain a healthy lifestyle with proper nutrition",
            "Protect your eyes from UV radiation with sunglasses",
            "Follow the 20-20-20 rule when using screens"
        ],
        "diabetic_retinopathy": [
            "⚠️ URGENT: Consult an ophthalmologist immediately",
            "This condition requires immediate medical attention",
            "Maintain strict blood sugar control",
            "Schedule regular retinal screenings",
            "Follow your diabetes management plan carefully",
            "Consider laser therapy if recommended by your doctor"
        ],
        "glaucoma": [
            "⚠️ IMPORTANT: See an eye specialist for comprehensive evaluation",
            "Monitor intraocular pressure regularly",
            "Follow prescribed medication regimen strictly",
            "Avoid activities that increase eye pressure",
            "Regular follow-up appointments are crucial",
            "Early treatment can prevent vision loss"
        ],
        "cataract": [
            "Consult with an ophthalmologist for evaluation",
            "Consider surgical options if vision is significantly impaired",
            "Use bright lighting when reading or doing close work",
            "Wear sunglasses to reduce glare",
            "Update eyeglass prescription as needed",
            "Surgery is highly effective when recommended"
        ]
    }
    
    recommendations = base_recommendations.get(predicted_class, [
        "Consult with an eye care professional for proper evaluation",
        "Schedule a comprehensive eye examination"
    ])
    
    # Add confidence-based notes
    if confidence >= 80:
        if predicted_class != "normal":
            recommendations.insert(0, f"High confidence detection ({confidence:.1f}%) - Seek medical attention promptly")
    elif confidence >= 60:
        recommendations.append(f"Moderate confidence ({confidence:.1f}%) - Consider getting a second opinion")
    else:
        recommendations.append(f"Low confidence ({confidence:.1f}%) - Results inconclusive, professional examination recommended")
    
    return recommendations

# Global treatment recommendation system instance
treatment_system = None
if enhanced_recommendations_available:
    treatment_system = TreatmentRecommendationSystem()
    print("✅ Treatment recommendation system initialized")

@app.post("/api/v1/predict-comprehensive")
async def predict_comprehensive_recommendations(file: UploadFile = File(...)):
    """
    Enhanced prediction endpoint with comprehensive treatment recommendations
    """
    if ml_model is None or not ml_model.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Please check server configuration."
        )
    
    if not enhanced_recommendations_available:
        raise HTTPException(
            status_code=503,
            detail="Enhanced treatment recommendations not available. Use /api/v1/predict for basic recommendations."
        )
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Generate unique prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Read and preprocess image
        image_bytes = await file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Preprocess image for model
        processed_image = preprocess_image(
            pil_image, 
            target_size=(ml_model.img_height, ml_model.img_width)
        )
        
        # Make prediction
        predicted_class, confidence, all_predictions = ml_model.predict(processed_image)
        
        # Convert confidence to percentage
        confidence_percent = confidence * 100
        all_predictions_percent = {k: v * 100 for k, v in all_predictions.items()}
        
        # Get comprehensive recommendations
        comprehensive_recommendations = treatment_system.get_comprehensive_recommendations(
            predicted_class, 
            confidence_percent
        )
        
        # Determine risk level (for backward compatibility)
        if confidence_percent >= 80:
            risk_level = "High"
        elif confidence_percent >= 60:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Save prediction to database (optional)
        prediction_saved = False
        try:
            from database.service import db_service
            
            # Create uploads directory
            os.makedirs("uploads", exist_ok=True)
            
            # Save uploaded file
            file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            filename = f"{prediction_id}.{file_extension}"
            file_path = os.path.join("uploads", filename)
            
            with open(file_path, "wb") as buffer:
                buffer.write(image_bytes)
            
            # Enhanced prediction data for database
            prediction_data = {
                "prediction_id": prediction_id,
                "user_id": None,  # TODO: Add user authentication
                "image_name": file.filename,
                "image_path": file_path,
                "image_size": len(image_bytes),
                "predicted_class": predicted_class,
                "confidence": round(confidence_percent, 2),
                "all_predictions": {k: round(v, 2) for k, v in all_predictions_percent.items()},
                "risk_level": risk_level,
                "recommendations": comprehensive_recommendations.get("immediate_actions", [])[:5],  # Store first 5 for compatibility
                "processing_time": None,  # TODO: Add timing
                "model_version": "Enhanced-ML-v2.0.0",
                "is_demo_mode": False
            }
            
            await db_service.save_prediction(prediction_data)
            prediction_saved = True
            print(f"✅ Enhanced prediction saved to database: {prediction_id}")
            
        except Exception as db_error:
            logger.warning(f"Failed to save enhanced prediction to database: {db_error}")
            print(f"⚠️ Database unavailable, enhanced prediction not saved: {db_error}")
        
        # Prepare comprehensive response
        response = {
            "prediction_id": prediction_id,
            "success": True,
            "results": {
                "predicted_class": predicted_class,
                "confidence": round(confidence_percent, 2),
                "all_predictions": {k: round(v, 2) for k, v in all_predictions_percent.items()},
                "risk_level": risk_level,
                "recommendations": comprehensive_recommendations.get("immediate_actions", [])  # For backward compatibility
            },
            "comprehensive_recommendations": comprehensive_recommendations,
            "image_info": {
                "filename": file.filename,
                "size": len(image_bytes),
                "processed_size": f"{ml_model.img_height}x{ml_model.img_width}"
            },
            "model_info": {
                "classes": ml_model.classes,
                "architecture": "Enhanced CNN with Treatment Recommendations"
            },
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "comprehensive_ml_prediction"
        }
        
        print(f"✅ Comprehensive prediction completed: {predicted_class} ({confidence_percent:.2f}%)")
        return response
        
    except Exception as e:
        print(f"❌ Comprehensive prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Comprehensive prediction failed: {str(e)}"
        )

@app.get("/api/v1/treatment-recommendations/{condition}")
async def get_treatment_recommendations_by_condition(condition: str, confidence: float = 75.0):
    """
    Get comprehensive treatment recommendations for a specific condition
    
    Args:
        condition: Eye condition name (e.g., 'diabetic_retinopathy', 'glaucoma')
        confidence: Confidence level (0-100, default 75%)
    """
    if not enhanced_recommendations_available or not treatment_system:
        raise HTTPException(
            status_code=503,
            detail="Enhanced treatment recommendations not available."
        )
    
    try:
        # Validate confidence range
        if not (0 <= confidence <= 100):
            raise HTTPException(
                status_code=400,
                detail="Confidence must be between 0 and 100"
            )
        
        # Get comprehensive recommendations
        comprehensive_recommendations = treatment_system.get_comprehensive_recommendations(
            condition, 
            confidence
        )
        
        return {
            "success": True,
            "condition": condition,
            "confidence": confidence,
            "recommendations": comprehensive_recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Treatment recommendations failed for {condition}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get treatment recommendations: {str(e)}"
        )

@app.get("/api/v1/supported-conditions")
async def get_supported_conditions():
    """
    Get list of supported eye conditions for treatment recommendations
    """
    if not enhanced_recommendations_available or not treatment_system:
        # Return basic conditions if enhanced system not available
        return {
            "success": True,
            "supported_conditions": [
                {
                    "name": "normal",
                    "display_name": "Normal/Healthy Eyes",
                    "description": "No eye disease detected"
                },
                {
                    "name": "diabetic_retinopathy",
                    "display_name": "Diabetic Retinopathy",
                    "description": "Diabetes-related eye disease"
                },
                {
                    "name": "glaucoma",
                    "display_name": "Glaucoma",
                    "description": "Increased intraocular pressure"
                },
                {
                    "name": "cataract",
                    "display_name": "Cataract",
                    "description": "Clouding of the eye's lens"
                }
            ],
            "enhanced_recommendations": False,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Enhanced conditions list
    return {
        "success": True,
        "supported_conditions": [
            {
                "name": "normal",
                "display_name": "Normal/Healthy Eyes",
                "description": "No eye disease detected",
                "severity_levels": ["normal"]
            },
            {
                "name": "diabetic_retinopathy",
                "display_name": "Diabetic Retinopathy",
                "description": "Diabetes-related eye disease affecting the retina",
                "severity_levels": ["mild", "moderate", "severe"]
            },
            {
                "name": "glaucoma",
                "display_name": "Glaucoma",
                "description": "Eye condition with increased intraocular pressure",
                "severity_levels": ["mild", "moderate", "severe"]
            },
            {
                "name": "cataract",
                "display_name": "Cataract",
                "description": "Clouding of the eye's natural lens",
                "severity_levels": ["mild", "moderate", "severe"]
            },
            {
                "name": "age_related_macular_degeneration",
                "display_name": "Age-Related Macular Degeneration",
                "description": "Eye disease affecting the macula",
                "severity_levels": ["mild"]
            },
            {
                "name": "hypertensive_retinopathy",
                "display_name": "Hypertensive Retinopathy",
                "description": "High blood pressure affecting the retina",
                "severity_levels": ["mild"]
            }
        ],
        "enhanced_recommendations": True,
        "total_conditions": 6,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/model-info")
async def get_model_info():
    """Get detailed model information"""
    if ml_model is None or not ml_model.is_model_loaded():
        return {"error": "Model not loaded"}
    
    return {
        "model_path": ml_model.model_path,
        "classes": ml_model.classes,
        "input_shape": {
            "height": ml_model.img_height,
            "width": ml_model.img_width,
            "channels": ml_model.channels
        },
        "architecture": "Custom CNN with transfer learning",
        "framework": f"TensorFlow {tf.__version__}",
        "total_classes": len(ml_model.classes)
    }

@app.get("/api/v1/statistics")
async def get_statistics():
    """Get model statistics (placeholder - could be enhanced with actual usage stats)"""
    return {
        "model_version": "2.0.0",
        "supported_classes": ml_model.classes if ml_model else [],
        "input_requirements": {
            "format": "RGB image",
            "size": "224x224 pixels (auto-resized)",
            "supported_types": ["JPEG", "PNG", "BMP"]
        },
        "performance_metrics": {
            "overall_accuracy": "67.46%",
            "class_accuracy": {
                "normal": "91.63%",
                "diabetic_retinopathy": "80.91%",
                "cataract": "54.81%",
                "glaucoma": "40.10%"
            }
        },
        "last_updated": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/predictions")
async def save_prediction_manually(prediction_data: dict):
    """Save a prediction manually (for cases where user wants to save existing analysis)"""
    try:
        from database.service import db_service
        
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
            "user_id": None,  # TODO: Add user authentication
            "image_name": image_info.get("filename", "unknown"),
            "image_path": f"uploads/{prediction_id}.jpg",  # Default path
            "image_size": image_info.get("size", 0),
            "predicted_class": results.get("predicted_class", "unknown"),
            "confidence": results.get("confidence", 0),
            "all_predictions": results.get("all_predictions", {}),
            "risk_level": results.get("risk_level", "Unknown"),
            "recommendations": results.get("recommendations", []),
            "processing_time": None,
            "model_version": "ML-v2.0.0",
            "is_demo_mode": False
        }
        
        # Save to database
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

@app.get("/api/v1/predictions/history")
async def get_prediction_history(limit: int = 50, offset: int = 0, user_id: str = None):
    """Get prediction history with pagination"""
    try:
        # Import database service
        from database.service import db_service
        
        if user_id:
            # Get predictions for specific user
            predictions = await db_service.get_user_predictions(user_id, limit)
        else:
            # Get recent predictions (all users)
            predictions = await db_service.get_recent_predictions(limit)
        
        # Convert to response format expected by frontend
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
        logger.error(f"Error fetching predictions: {e}")
        return {
            "predictions": [],
            "total_count": 0,
            "limit": limit,
            "offset": offset,
            "error": str(e)
        }

@app.get("/api/v1/predictions")
async def get_predictions(limit: int = 50, user_id: str = None):
    """Get prediction history (legacy endpoint - redirects to /predictions/history)"""
    return await get_prediction_history(limit, 0, user_id)

@app.get("/api/v1/images/{prediction_id}")
async def serve_image(prediction_id: str):
    """Serve uploaded images by prediction ID"""
    try:
        # Construct possible image paths with different extensions
        upload_dir = "uploads"
        possible_extensions = ["jpg", "jpeg", "png", "bmp", "gif"]
        
        for ext in possible_extensions:
            image_path = os.path.join(upload_dir, f"{prediction_id}.{ext}")
            if os.path.exists(image_path):
                return FileResponse(
                    path=image_path,
                    media_type=f"image/{ext}",
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
        
        # If no image found, return 404
        raise HTTPException(
            status_code=404,
            detail=f"Image not found for prediction ID: {prediction_id}"
        )
        
    except Exception as e:
        logger.error(f"Error serving image {prediction_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error serving image: {str(e)}"
        )

@app.get("/api/v1/admin/overview")
async def get_admin_overview():
    """Get admin overview statistics for dashboard"""
    try:
        from database.service import db_service
        
        # Get total predictions count
        total_predictions = await db_service.get_prediction_count()
        
        # Get predictions by class statistics
        class_stats = await db_service.get_prediction_stats_by_class()
        
        # Calculate accuracy (mock calculation - you can implement real accuracy tracking)
        accuracy = 89.5  # This could be calculated from validation data
        
        # Get recent predictions for monthly data
        recent_predictions = await db_service.get_recent_predictions(100)
        
        # Group by month for monthly trends
        monthly_data = []
        from collections import defaultdict
        from datetime import datetime
        
        monthly_counts = defaultdict(int)
        for pred in recent_predictions:
            try:
                if hasattr(pred, 'created_at'):
                    month_key = pred.created_at.strftime("%b %Y")
                    monthly_counts[month_key] += 1
            except:
                continue
        
        # Get last 6 months of data
        import calendar
        from dateutil.relativedelta import relativedelta
        current_date = datetime.utcnow()
        for i in range(6):
            month_date = current_date - relativedelta(months=i)
            month_name = calendar.month_abbr[month_date.month]
            month_key = f"{month_name} {month_date.year}"
            monthly_data.append({
                "month": month_key,
                "analyses": monthly_counts.get(month_key, 0)
            })
        
        monthly_data.reverse()  # Show oldest to newest
        
        # Calculate disease distribution
        total_with_class = sum(class_stats.values())
        healthy_count = class_stats.get("normal", 0)
        cataract_count = class_stats.get("cataract", 0)
        glaucoma_count = class_stats.get("glaucoma", 0)
        dr_count = class_stats.get("diabetic_retinopathy", 0)
        
        # Mock patient count (you can implement user counting if needed)
        unique_patients = max(1, total_predictions // 3)  # Estimate based on predictions
        
        response = {
            "total_analyses": total_predictions,
            "accuracy": accuracy,
            "patients": unique_patients,
            "diseased_cases": total_predictions - healthy_count,
            "healthy": healthy_count,
            "cataracts": cataract_count,
            "glaucoma": glaucoma_count,
            "diabetic_retinopathy": dr_count,
            "monthly_data": monthly_data,
            "class_distribution": class_stats,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting admin overview: {e}")
        # Return default/mock data if database fails
        return {
            "total_analyses": 0,
            "accuracy": 89.5,
            "patients": 0,
            "diseased_cases": 0,
            "healthy": 0,
            "cataracts": 0,
            "glaucoma": 0,
            "diabetic_retinopathy": 0,
            "monthly_data": [],
            "class_distribution": {},
            "last_updated": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.post("/api/v1/export/history")
async def export_history_data(export_request: dict):
    """Export patient history data as CSV or Excel file"""
    try:
        import csv
        import json
        from io import StringIO, BytesIO
        
        # Extract data and format from request
        history_data = export_request.get("data", [])
        export_format = export_request.get("format", "csv").lower()
        
        if not history_data:
            raise HTTPException(
                status_code=400,
                detail="No data provided for export"
            )
        
        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "csv":
            # Create CSV content
            output = StringIO()
            fieldnames = [
                'Analysis ID', 'Date', 'Condition', 'Confidence (%)', 
                'Risk Level', 'Filename', 'Notes'
            ]
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in history_data:
                writer.writerow({
                    'Analysis ID': item.get('id', ''),
                    'Date': datetime.fromisoformat(item.get('date', '')).strftime('%Y-%m-%d %H:%M:%S') if item.get('date') else '',
                    'Condition': item.get('condition', ''),
                    'Confidence (%)': item.get('confidence', ''),
                    'Risk Level': item.get('riskLevel', ''),
                    'Filename': item.get('filename', ''),
                    'Notes': item.get('notes', '')
                })
            
            # Convert to bytes
            csv_content = output.getvalue().encode('utf-8')
            output.close()
            
            filename = f"patient_history_{timestamp}.csv"
            media_type = "text/csv"
            
            return StreamingResponse(
                io.BytesIO(csv_content),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Expose-Headers": "Content-Disposition"
                }
            )
            
        elif export_format == "excel":
            try:
                import pandas as pd
                
                # Prepare data for DataFrame
                df_data = []
                for item in history_data:
                    df_data.append({
                        'Analysis ID': item.get('id', ''),
                        'Date': datetime.fromisoformat(item.get('date', '')).strftime('%Y-%m-%d %H:%M:%S') if item.get('date') else '',
                        'Condition': item.get('condition', ''),
                        'Confidence (%)': item.get('confidence', ''),
                        'Risk Level': item.get('riskLevel', ''),
                        'Filename': item.get('filename', ''),
                        'Notes': item.get('notes', '')
                    })
                
                df = pd.DataFrame(df_data)
                
                # Create Excel file in memory
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Patient History', index=False)
                    
                    # Auto-adjust column widths
                    worksheet = writer.sheets['Patient History']
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                
                excel_buffer.seek(0)
                filename = f"patient_history_{timestamp}.xlsx"
                media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                
                return StreamingResponse(
                    excel_buffer,
                    media_type=media_type,
                    headers={
                        "Content-Disposition": f"attachment; filename={filename}",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Expose-Headers": "Content-Disposition"
                    }
                )
                
            except ImportError:
                # Fallback to CSV if pandas/openpyxl not available
                logger.warning("pandas/openpyxl not available, falling back to CSV export")
                return await export_history_data({"data": history_data, "format": "csv"})
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported format. Use 'csv' or 'excel'"
            )
            
    except Exception as e:
        logger.error(f"Error exporting history data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )

if __name__ == "__main__":
    print("🚀 Starting EyeCare AI Backend with ML Model...")
    uvicorn.run(
        "ml_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

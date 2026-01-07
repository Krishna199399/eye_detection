"""
Database Models for EyeCare AI Authentication System
"""
from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, EmailStr
from beanie import Document
from bson import ObjectId


class UserRole(str, Enum):
    """User roles enumeration"""
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"


class Gender(str, Enum):
    """Gender enumeration"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer-not-to-say"


class PredictionClass(str, Enum):
    """Eye disease prediction classes"""
    NORMAL = "normal"
    DIABETIC_RETINOPATHY = "diabetic_retinopathy"
    GLAUCOMA = "glaucoma"
    CATARACT = "cataract"


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"


# Database Models (Beanie Documents)
class User(Document):
    """User document model"""
    email: EmailStr = Field(..., unique=True, index=True)
    hashed_password: str
    full_name: str
    age: Optional[int] = None
    gender: Optional[Gender] = None
    role: UserRole = UserRole.PATIENT
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    class Settings:
        name = "users"
        indexes = [
            "email",
            "created_at",
        ]


class UserSession(Document):
    """User session document model"""
    user_id: str = Field(..., index=True)
    session_token: str = Field(..., unique=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    is_active: bool = True
    
    class Settings:
        name = "user_sessions"
        indexes = [
            "user_id",
            "session_token",
            "expires_at",
        ]


class Prediction(Document):
    """Prediction result document model"""
    prediction_id: str = Field(..., unique=True, index=True)
    user_id: Optional[str] = Field(None, index=True)
    image_name: str
    image_path: str
    image_size: int
    predicted_class: PredictionClass
    confidence: float
    all_predictions: dict = Field(default_factory=dict)
    risk_level: RiskLevel
    recommendations: List[str] = Field(default_factory=list)
    processing_time: Optional[float] = None
    model_version: str
    is_demo_mode: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "predictions"
        indexes = [
            "prediction_id",
            "user_id",
            "created_at",
            "predicted_class",
        ]


# Pydantic Models for API
class UserCreate(BaseModel):
    """User creation model"""
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)
    age: Optional[int] = Field(None, ge=1, le=120)
    gender: Optional[Gender] = None
    role: Optional[UserRole] = UserRole.PATIENT


class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response model"""
    id: str
    email: EmailStr
    full_name: str
    age: Optional[int] = None
    gender: Optional[Gender] = None
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class Token(BaseModel):
    """JWT token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class PredictionCreate(BaseModel):
    """Prediction creation model"""
    image_name: str
    predicted_class: PredictionClass
    confidence: float
    all_predictions: dict
    risk_level: RiskLevel
    recommendations: List[str]


class PredictionResponse(BaseModel):
    """Prediction response model"""
    prediction_id: str
    user_id: Optional[str] = None
    image_name: str
    predicted_class: PredictionClass
    confidence: float
    all_predictions: dict
    risk_level: RiskLevel
    recommendations: List[str]
    created_at: datetime
    model_version: str
    is_demo_mode: bool


# Error Response Models
class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationErrorResponse(BaseModel):
    """Validation error response model"""
    detail: List[dict]
    status_code: int = 422
    timestamp: datetime = Field(default_factory=datetime.utcnow)

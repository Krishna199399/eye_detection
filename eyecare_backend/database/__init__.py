"""
Database package for EyeCare AI Authentication System
"""
from .models import (
    User, UserSession, Prediction, 
    UserCreate, UserLogin, UserResponse, Token,
    UserRole, Gender, PredictionClass, RiskLevel
)
from .service import db_service
from .connection import init_database, close_database

__all__ = [
    "User", "UserSession", "Prediction",
    "UserCreate", "UserLogin", "UserResponse", "Token",
    "UserRole", "Gender", "PredictionClass", "RiskLevel",
    "db_service", "init_database", "close_database"
]

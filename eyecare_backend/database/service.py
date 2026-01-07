"""
Database Service for EyeCare AI Authentication System
"""
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Optional, List
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from database.models import (
    User, UserSession, Prediction, UserCreate, UserRole, 
    PredictionClass, RiskLevel, Gender
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DatabaseService:
    """Database service for user and prediction management"""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None

    async def connect_database(self, database_url: str = "mongodb://localhost:27017", database_name: str = "eyecare_ai"):
        """Connect to MongoDB database"""
        try:
            self.client = AsyncIOMotorClient(database_url)
            self.database = self.client[database_name]
            
            # Initialize Beanie
            await init_beanie(
                database=self.database,
                document_models=[User, UserSession, Prediction]
            )
            print(f"✅ Connected to MongoDB database: {database_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            return False

    async def close_database(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("👋 Disconnected from MongoDB")

    # Password utilities
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)

    # User management
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            return await User.find_one(User.email == email)
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            return await User.get(user_id)
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None

    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_data.email)
        if existing_user:
            raise ValueError("User with this email already exists")

        # Hash password
        hashed_password = self.get_password_hash(user_data.password)

        # Create user document
        user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            age=user_data.age,
            gender=user_data.gender,
            role=user_data.role or UserRole.PATIENT
        )

        # Save to database
        await user.create()
        return user

    async def update_user_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp"""
        try:
            user = await User.get(user_id)
            if user:
                user.last_login = datetime.utcnow()
                await user.save()
                return True
            return False
        except Exception as e:
            print(f"Error updating user last login: {e}")
            return False

    # Session management
    async def create_user_session(self, user_id: str, session_token: str, expires_in_seconds: int = 1800) -> UserSession:
        """Create a new user session"""
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in_seconds)
        )
        await session.create()
        return session

    async def get_active_session(self, session_token: str) -> Optional[UserSession]:
        """Get active session by token"""
        try:
            session = await UserSession.find_one(
                UserSession.session_token == session_token,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            )
            return session
        except Exception as e:
            print(f"Error getting active session: {e}")
            return None

    async def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a user session"""
        try:
            session = await UserSession.find_one(UserSession.session_token == session_token)
            if session:
                session.is_active = False
                await session.save()
                return True
            return False
        except Exception as e:
            print(f"Error invalidating session: {e}")
            return False

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            await UserSession.find(UserSession.expires_at < datetime.utcnow()).delete()
            print("🧹 Cleaned up expired sessions")
        except Exception as e:
            print(f"Error cleaning up sessions: {e}")

    # Prediction management
    async def save_prediction(self, prediction_data: dict) -> Prediction:
        """Save a prediction result"""
        prediction = Prediction(
            prediction_id=prediction_data.get("prediction_id", str(uuid.uuid4())),
            user_id=prediction_data.get("user_id"),
            image_name=prediction_data["image_name"],
            image_path=prediction_data["image_path"],
            image_size=prediction_data["image_size"],
            predicted_class=PredictionClass(prediction_data["predicted_class"]),
            confidence=prediction_data["confidence"],
            all_predictions=prediction_data.get("all_predictions", {}),
            risk_level=RiskLevel(prediction_data.get("risk_level", "Unknown")),
            recommendations=prediction_data.get("recommendations", []),
            processing_time=prediction_data.get("processing_time"),
            model_version=prediction_data.get("model_version", "unknown"),
            is_demo_mode=prediction_data.get("is_demo_mode", False)
        )
        await prediction.create()
        return prediction

    async def get_prediction_by_id(self, prediction_id: str) -> Optional[Prediction]:
        """Get prediction by ID"""
        try:
            return await Prediction.find_one(Prediction.prediction_id == prediction_id)
        except Exception as e:
            print(f"Error getting prediction by ID: {e}")
            return None

    async def get_user_predictions(self, user_id: str, limit: int = 50) -> List[Prediction]:
        """Get predictions for a specific user"""
        try:
            return await Prediction.find(
                Prediction.user_id == user_id
            ).sort(-Prediction.created_at).limit(limit).to_list()
        except Exception as e:
            print(f"Error getting user predictions: {e}")
            return []

    async def get_recent_predictions(self, limit: int = 50) -> List[Prediction]:
        """Get recent predictions (all users)"""
        try:
            return await Prediction.find().sort(-Prediction.created_at).limit(limit).to_list()
        except Exception as e:
            print(f"Error getting recent predictions: {e}")
            return []

    # Statistics
    async def get_user_count(self) -> int:
        """Get total number of users"""
        try:
            return await User.count()
        except Exception as e:
            print(f"Error getting user count: {e}")
            return 0

    async def get_prediction_count(self) -> int:
        """Get total number of predictions"""
        try:
            return await Prediction.count()
        except Exception as e:
            print(f"Error getting prediction count: {e}")
            return 0

    async def get_prediction_stats_by_class(self) -> dict:
        """Get prediction statistics by class"""
        try:
            pipeline = [
                {"$group": {"_id": "$predicted_class", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            # Use the motor collection directly for aggregation
            collection = self.database.get_collection("Prediction")
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            return {result["_id"]: result["count"] for result in results}
        except Exception as e:
            print(f"Error getting prediction stats: {e}")
            return {}

    # Demo data creation
    async def create_demo_user(self):
        """Create a demo user for testing"""
        demo_email = "demo@eyecare.ai"
        
        # Check if demo user already exists
        existing_user = await self.get_user_by_email(demo_email)
        if existing_user:
            print(f"Demo user already exists: {demo_email}")
            return existing_user

        # Create demo user
        demo_user_data = UserCreate(
            email=demo_email,
            password="demo123",
            full_name="Demo User",
            age=30,
            gender=Gender.PREFER_NOT_TO_SAY,
            role=UserRole.PATIENT
        )

        try:
            demo_user = await self.create_user(demo_user_data)
            print(f"✅ Created demo user: {demo_email} / demo123")
            return demo_user
        except Exception as e:
            print(f"❌ Failed to create demo user: {e}")
            return None


# Global database service instance
db_service = DatabaseService()

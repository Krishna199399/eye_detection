"""
Database Connection Management for EyeCare AI
"""
import os
from dotenv import load_dotenv
from database.service import db_service

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "eyecare_ai")


async def init_database():
    """Initialize database connection and create demo data"""
    print("🔗 Initializing database connection...")
    
    success = await db_service.connect_database(DATABASE_URL, DATABASE_NAME)
    
    if success:
        # Create demo user for testing
        await db_service.create_demo_user()
        
        # Clean up expired sessions
        await db_service.cleanup_expired_sessions()
        
        print("✅ Database initialization completed")
    else:
        print("❌ Database initialization failed")
        
    return success


async def close_database():
    """Close database connection"""
    await db_service.close_database()

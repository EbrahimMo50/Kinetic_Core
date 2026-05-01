import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.api.db import models
from src.api.db.models import Base

# Ensure the data directory exists
Path("data").mkdir(parents=True, exist_ok=True)

# Database URL
DATABASE_URL = "sqlite:///./data/kinetic.sqlite"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all database tables"""
    if SessionLocal is None:
        raise Exception("Database session factory is not initialized")

    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

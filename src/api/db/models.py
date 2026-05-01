from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.sqlite import JSON

Base = declarative_base()


class Session(Base):
    """Represents a user session with uploaded dataset"""
    __tablename__ = "sessions"
    
    session_id = Column(String(36), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Dataset(Base):
    """
    Represents an uploaded dataset
    Contains:
    - session_id: Session identifier
    - file_name: Original filename
    - file_path: Path where dataset was uploaded
    """
    __tablename__ = "datasets"
    
    dataset_id = Column(String(36), primary_key=True, index=True)
    session_id = Column(String(36), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)  # Path where uploaded
    file_size_bytes = Column(Integer, nullable=True)
    
    # Dataset metadata
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    columns_info = Column(JSON, nullable=True)  # {column_name: data_type}
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    class Config:
        """Example usage"""
        pass


class TrainingJob(Base):
    """Represents a training job for a session"""
    __tablename__ = "training_jobs"
    
    job_id = Column(String(36), primary_key=True, index=True)
    session_id = Column(String(36), nullable=False, index=True)
    dataset_id = Column(String(36), nullable=False, index=True)
    task_type = Column(String(20), nullable=False)  # "classification", "regression", "clustering"
    status = Column(String(20), default="pending", nullable=False)  # pending, training, completed, failed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)


class Model(Base):
    """Represents a trained model"""
    __tablename__ = "models"
    
    model_id = Column(String(36), primary_key=True, index=True)
    session_id = Column(String(36), nullable=False, index=True)
    dataset_id = Column(String(36), nullable=False, index=True)
    job_id = Column(String(36), nullable=False, index=True)
    task_type = Column(String(20), nullable=False)
    model_type = Column(String(50), nullable=False)  # e.g., "RandomForestClassifier"
    file_path = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=False)  # Store all metrics as JSON
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    training_time_seconds = Column(Float, nullable=True)

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import relationship

Base = declarative_base()


class Session(Base):
    """Represents a user session with uploaded dataset"""
    __tablename__ = "sessions"
    
    session_id = Column(String(36), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    result = relationship("Result", back_populates="session", uselist=False)


class ResultStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"


class Result(Base):
    __tablename__ = "results"

    session_id = Column(String(36), ForeignKey("sessions.session_id"), primary_key=True, index=True)
    report = Column(JSON, nullable=True)
    status = Column(SAEnum(ResultStatus, native_enum=False), default=ResultStatus.IN_PROGRESS.value, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    session = relationship("Session", back_populates="result")
    files = relationship("File", back_populates="result", cascade="all, delete-orphan", primaryjoin="Result.session_id==File.result_session_id")


class FileType(str, Enum):
    DATASET = "DATASET"
    MODEL = "MODEL"
    OTHER = "OTHER"


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), ForeignKey("sessions.session_id"), nullable=False, index=True)
    result_session_id = Column(String(36), ForeignKey("results.session_id"), nullable=True, index=True)
    artifact_type = Column(String(20), nullable=False)
    original_name = Column(String(255), nullable=False)
    path = Column(String(500), nullable=False)
    mime_type = Column(String(100), nullable=True)
    artifact_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    result = relationship("Result", back_populates="files")

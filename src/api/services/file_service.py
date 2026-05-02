from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

import joblib
import os
import pandas as pd
import pickle
from sqlalchemy.orm import Session as DBSession

from src.api.db.models import File, FileType


class FileService:
    """Service for file and dataset operations"""

    @staticmethod
    def set_dataset(
        db: DBSession,
        session_id: str,
        dataset: pd.DataFrame,
        file_name: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Store dataset in database and return metadata.
        
        Args:
            db: Database session
            session_id: Session ID
            dataset: DataFrame
            file_name: Original filename
            file_path: Path to file
        
        Returns:
            Dataset metadata with ID and statistics
        """
        dataset_id = str(uuid4())
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        file_record = File(
            session_id=session_id,
            result_session_id=None,
            artifact_type=FileType.DATASET.value,
            original_name=file_name,
            path=file_path,
            mime_type=None,
            artifact_metadata={
                "file_size_bytes": file_size,
                "row_count": len(dataset),
                "column_count": len(dataset.columns),
                "columns_info": {col: str(dataset[col].dtype) for col in dataset.columns},
            },
        )

        db.add(file_record)
        db.commit()
        db.refresh(file_record)

        return {
            "dataset_id": dataset_id,
            "session_id": session_id,
            "row_count": len(dataset),
            "column_count": len(dataset.columns),
            "columns_info": {col: str(dataset[col].dtype) for col in dataset.columns},
        }

    @staticmethod
    def get_dataset_file(db: DBSession, session_id: str) -> File:
        """
        Retrieve dataset file record for a session.
        
        Args:
            db: Database session
            session_id: Session ID
        
        Returns:
            File record for the dataset
        """
        return db.query(File).filter(
            File.session_id == session_id,
            File.artifact_type == FileType.DATASET.value
        ).first()

    @staticmethod
    def save_model_file(model_obj: Any, model_type: str, session_id: str, file_name: str) -> Dict[str, Dict[str, str]]:
        """
        Save trained model to file system in both pkl and joblib formats.
        
        Args:
            model_obj: Trained model object
            model_type: Type of model for filename
            session_id: Session ID for directory path
            file_name: Original dataset filename
        
        Returns:
            Saved file metadata keyed by format (pkl, joblib)
        """
        models_dir = Path("uploads") / session_id
        models_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"{file_name}_{model_type}"
        
        # Save pkl
        pkl_name = f"{base_name}.pkl"
        pkl_path = models_dir / pkl_name
        with open(pkl_path, "wb") as f:
            pickle.dump(model_obj, f)
            
        # Save joblib
        joblib_name = f"{base_name}.joblib"
        joblib_path = models_dir / joblib_name
        joblib.dump(model_obj, joblib_path)

        return {
            "pkl": {
                "file_path": str(pkl_path),
                "file_name": pkl_name,
                "mime_type": "application/octet-stream",
            },
            "joblib": {
                "file_path": str(joblib_path),
                "file_name": joblib_name,
                "mime_type": "application/octet-stream",
            },
        }

    @staticmethod
    def get_model_file(db: DBSession, file_id: int) -> Tuple[str, str]:
        """
        Get the saved model file path for downloading by file id.
        
        Args:
            db: Database session
            file_id: File ID
            
        Returns:
            Tuple of (file_path, file_name)
        """
        file_record = db.query(File).filter(File.id == file_id).first()
        if not file_record:
            raise ValueError(f"No file found for file_id {file_id}")
        
        file_path = Path(file_record.path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found at {file_path}")
            
        return str(file_path), file_record.original_name

    @staticmethod
    def save_model_records(
        db: DBSession,
        session_id: str,
        saved_files: Dict[str, Dict[str, str]],
        model_type: str,
        metrics: Dict[str, Any],
        training_time: float
    ) -> Tuple[int, list]:
        """
        Save model file records to database.
        
        Args:
            db: Database session
            session_id: Session ID
            saved_files: Dict of saved file metadata
            model_type: Model type name
            metrics: Model metrics/report
            training_time: Training duration in seconds
        
        Returns:
            Tuple of (default_file_id, file_records_list)
        """
        file_records = []
        default_file_id = None

        for file_kind, metadata in saved_files.items():
            file_obj = File(
                session_id=session_id,
                result_session_id=session_id,
                artifact_type=FileType.MODEL.value,
                original_name=metadata["file_name"],
                path=metadata["file_path"],
                mime_type=metadata["mime_type"],
                artifact_metadata={
                    "format": file_kind,
                    "model_type": model_type if file_kind == "pkl" else None,
                    "metrics": metrics if file_kind == "pkl" else None,
                    "training_time_seconds": training_time if file_kind == "pkl" else None,
                },
            )
            db.add(file_obj)
            db.flush()
            file_records.append({
                "file_id": file_obj.id,
                "type": file_obj.artifact_type,
                "file_name": file_obj.original_name,
                "path": file_obj.path,
            })
            if file_kind == "pkl":
                default_file_id = file_obj.id

        return default_file_id, file_records

from datetime import datetime
from typing import Any, Dict
from uuid import uuid4
import pandas as pd
import os
import time
from sqlalchemy.orm import Session as DBSession

from src.api.db.db_engine import SessionLocal
from src.api.db.models import Result, ResultStatus
from src.api.services.file_service import FileService
from src.core.main import (
    get_clustering_models,
    get_classification_models,
    get_regression_models,
)


class TrainingService:
    """Service for model training orchestration"""

    @staticmethod
    def train_model_background(
        session_id: str,
        dataset_id: str,
        task_type: str,
        target_column: str | None = None,
    ) -> None:
        """
        Run model training in a dedicated database session.
        
        Args:
            session_id: Session ID
            dataset_id: Dataset ID (for compatibility, not used internally)
            task_type: Type of task (classification, regression, clustering)
            target_column: Target column name for supervised tasks
        """
        db = SessionLocal()
        try:
            TrainingService.train_model(
                db=db,
                session_id=session_id,
                dataset_id=dataset_id,
                task_type=task_type,
                target_column=target_column,
            )
        finally:
            db.close()

    @staticmethod
    def train_model(
        db: DBSession,
        session_id: str,
        dataset_id: str,
        task_type: str,
        target_column: str | None = None
    ) -> Dict[str, Any]:
        """
        Train ML model based on task type.
        
        Args:
            db: Database session
            session_id: Session ID
            dataset_id: Dataset ID (for compatibility, not used internally)
            task_type: 'classification', 'regression', or 'clustering'
            target_column: Target column name (required for classification/regression, optional for clustering)
        
        Returns:
            Training results with best model info
        """
        # Import here to avoid circular dependency
        from src.api.services.session_service import SessionService
        
        # Get dataset file
        db_dataset = FileService.get_dataset_file(db, session_id)
        if not db_dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Load dataset from file
        if not os.path.exists(db_dataset.path):
            raise ValueError(f"Dataset file at {db_dataset.path} not found on disk.")

        file_ext = str(db_dataset.path).lower().split('.')[-1]
        if file_ext == 'csv':
            dataset = pd.read_csv(db_dataset.path, na_values=['?'])
        else:
            dataset = pd.read_excel(db_dataset.path, na_values=['?'])
        
        # Validate target column for supervised learning
        if task_type in ["classification", "regression"]:
            if not target_column:
                raise ValueError(f"target_column is required for {task_type}")
            if target_column not in dataset.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(dataset.columns)}")
            target_column = str(target_column)
        
        # Ensure result row exists
        result_row = SessionService.ensure_session_result(db, session_id)
        
        try:
            start_time = time.time()
            
            # Route to correct training function based on task type
            if task_type == "classification":
                models_results = get_classification_models(dataset, target_column)
            elif task_type == "regression":
                models_results = get_regression_models(dataset, target_column)
            elif task_type == "clustering":
                models_results = get_clustering_models(dataset)
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
            
            training_time = time.time() - start_time
            best_model_data = None
            
            # Save only the best model
            for result in models_results:
                if result.get("best_model", False):
                    model_obj = result["model"]
                    model_type = result["model_type"].__name__
                    metrics = result["report"]
                    
                    # Save model files
                    saved_files = FileService.save_model_file(
                        model_obj, model_type, session_id, db_dataset.original_name
                    )
                    
                    # Save model file records to DB
                    default_file_id, file_records = FileService.save_model_records(
                        db, session_id, saved_files, model_type, metrics, training_time
                    )
                    
                    best_model_data = {
                        "model_type": model_type,
                        "metrics": metrics,
                        "file_id": default_file_id,
                        "files": file_records,
                    }
                    break
            
            # Mark result as done
            result_row.status = ResultStatus.DONE.value
            result_row.report = {
                "task_type": task_type,
                "training_time_seconds": training_time,
                "best_model": best_model_data,
            }
            result_row.error_message = None
            db.commit()
            
            return {
                "task_type": task_type,
                "training_time_seconds": training_time,
                "best_model": best_model_data
            }
        
        except Exception as e:
            result_row.status = ResultStatus.FAILED.value
            result_row.error_message = str(e)
            db.commit()
            raise

import pandas as pd
import pickle
import time
from typing import Dict, Tuple, Any
from sqlalchemy.orm import Session as DBSession
from uuid import uuid4
import joblib
import os

from src.core.main import (
    get_classification_models,
    get_regression_models,
    get_clustering_models
)
from src.api.db.models import Dataset, TrainingJob, Model
from pathlib import Path


class MLService:
    """Service for ML training operations"""
    
    @staticmethod
    def set_dataset(
        db: DBSession,
        session_id: str,
        dataset: pd.DataFrame,
        file_name: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Store dataset in database
        
        Args:
            db: Database session
            session_id: Session ID
            dataset: DataFrame
            file_name: Original filename
            file_path: Path to file
        
        Returns:
            Dataset metadata
        """
        dataset_id = str(uuid4())
        
        # Get file size

        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        # Create dataset record
        db_dataset = Dataset(
            dataset_id=dataset_id,
            session_id=session_id,
            file_name=file_name,
            file_path=file_path,
            file_size_bytes=file_size,
            row_count=len(dataset),
            column_count=len(dataset.columns),
            columns_info={col: str(dataset[col].dtype) for col in dataset.columns}
        )
        
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        
        return {
            "dataset_id": dataset_id,
            "session_id": session_id,
            "row_count": len(dataset),
            "column_count": len(dataset.columns),
            "columns_info": {col: str(dataset[col].dtype) for col in dataset.columns}
        }
    
    @staticmethod
    def train_model(
        db: DBSession,
        session_id: str,
        dataset_id: str,
        task_type: str,
        target_column: str | None = None
    ) -> Dict[str, Any]:
        """
        Train ML model based on task type
        
        Args:
            db: Database session
            session_id: Session ID
            dataset_id: Dataset ID
            task_type: 'classification', 'regression', or 'clustering'
            target_column: Target column name (required for classification/regression, optional for clustering)
        
        Returns:
            Training results with best model info
        """
        # Get dataset
        db_dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        if not db_dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load dataset from file
        if not os.path.exists(db_dataset.file_path):
            raise ValueError(f"Dataset file at {db_dataset.file_path} not found on disk.")
            
        file_ext = str(db_dataset.file_path).lower().split('.')[-1]
        if file_ext == 'csv':
            dataset = pd.read_csv(db_dataset.file_path, na_values=['?'])
        else:
            dataset = pd.read_excel(db_dataset.file_path, na_values=['?'])
        
        # Validate target column for supervised learning
        if task_type in ["classification", "regression"]:
            if not target_column:
                raise ValueError(f"target_column is required for {task_type}")
            if target_column not in dataset.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(dataset.columns)}")
                
            # We are now guaranteed target_column is a string
            target_column = str(target_column)
        
        # Create training job
        job_id = str(uuid4())
        training_job = TrainingJob(
            job_id=job_id,
            session_id=session_id,
            dataset_id=dataset_id,
            task_type=task_type,
            status="training"
        )
        db.add(training_job)
        db.commit()
        
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
            
            # Save only the best model
            best_model_data = None
            
            for result in models_results:
                is_best = result.get("best_model", False)
                
                if is_best:
                    model_obj = result["model"]
                    model_type = result["model_type"].__name__
                    metrics = result["report"]
                    model_id = str(uuid4())
                    
                    file_path, file_name = MLService.save_model_file(model_obj, model_id, model_type)
                    
                    # Store in database
                    db_model = Model(
                        model_id=model_id,
                        session_id=session_id,
                        dataset_id=dataset_id,
                        job_id=job_id,
                        task_type=task_type,
                        model_type=model_type,
                        metrics=metrics,
                        training_time_seconds=training_time,
                        file_path=file_path,
                        file_name=file_name
                    )
                    db.add(db_model)
                    
                    best_model_data = {
                        "model_id": model_id,
                        "model_type": model_type,
                        "metrics": metrics
                    }
                    break  # Stop after saving best model
            
            # Update job status
            training_job.status = "completed"
            db.commit()
            
            return {
                "job_id": job_id,
                "task_type": task_type,
                "training_time_seconds": training_time,
                "best_model": best_model_data
            }
        
        except Exception as e:
            training_job.status = "failed"
            training_job.error_message = str(e)
            db.commit()
            raise
    
    @staticmethod
    def get_model_results(
        db: DBSession,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve model results for session
        
        Args:
            db: Database session
            session_id: Session ID
        
        Returns:
            model info and metrics
        """
        model = db.query(Model).filter(
            Model.session_id == session_id,
        ).first()
        
        if not model:
            raise ValueError(f"No trained model found for session {session_id}")
        
        return {
            "model_id": model.model_id,
            "model_type": model.model_type,
            "task_type": model.task_type,
            "metrics": model.metrics,
            "training_time_seconds": model.training_time_seconds,
            "file_path": model.file_path,
            "file_name": model.file_name
        }


    @staticmethod
    def save_model_file(model_obj: Any, model_id: str, model_type: str) -> Tuple[str, str]:
        """
        Save trained model to file system in both pkl and joblib formats.
        
        Args:
            model_obj: Trained model object
            model_id: Model ID for filename
            model_type: Type of model for filename
        
        Returns:
            Default (.pkl) file_path and file_name
        """
        models_dir = Path("data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"{model_id}_{model_type}"
        
        # Save pkl
        pkl_name = f"{base_name}.pkl"
        pkl_path = models_dir / pkl_name
        with open(pkl_path, "wb") as f:
            pickle.dump(model_obj, f)
            
        # Save joblib
        joblib_name = f"{base_name}.joblib"
        joblib_path = models_dir / joblib_name
        joblib.dump(model_obj, joblib_path)
        
        return str(pkl_path), pkl_name

    @staticmethod
    def get_model_file(db: DBSession, session_id: str, format_type: str = "pkl") -> Tuple[str, str]:
        """
        Get the saved model file path in the requested format for downloading.
        
        Args:
            db: Database session
            session_id: Session ID
            format_type: Format requested by user ('pkl' or 'joblib')
            
        Returns:
            Tuple of (file_path, file_name)
        """
        if format_type not in ["pkl", "joblib"]:
            raise ValueError(f"Unsupported format: {format_type}")

        model = db.query(Model).filter(Model.session_id == session_id).first()
        if not model:
            raise ValueError(f"No trained model found for session {session_id}")
            
        models_dir = Path("data/models")
        file_name = f"{model.model_id}_{model.model_type}.{format_type}"
        file_path = models_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found at {file_path}")
            
        return str(file_path), file_name

    @staticmethod
    def get_status_model_training(db: DBSession, job_id: str) -> Dict[str, Any]:
        """
        Get training job status
        
        Args:
            db: Database session
            job_id: Training job ID
        
        Returns:
            Job status and error message if failed
        """
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        return {
            "job_id": job.job_id,
            "status": job.status,
        }
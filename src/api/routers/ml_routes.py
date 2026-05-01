from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Optional
import pandas as pd
import io
import uuid

from src.api.db.db_engine import get_db
from src.api.services.ml_service import MLService

router = APIRouter()

@router.post("/dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a dataset for machine learning.
    """
    valid_extensions = ('.csv', '.xlsx', '.xls')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(status_code=400, detail=f"Only {', '.join(valid_extensions)} files are supported")
        
    if not session_id:
        session_id = str(uuid.uuid4())
        
    try:
        content = await file.read()
        
        file_ext = file.filename.lower().split('.')[-1]
        if file_ext == 'csv':
            df = pd.read_csv(io.BytesIO(content), na_values=['?'])
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(content))
            
        
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        local_file_path = str(upload_dir / f"{session_id}.{file_ext}")
        
        if file_ext == 'csv':
            df.to_csv(local_file_path, index=False)
        elif file_ext in ['xlsx', 'xls']:
            df.to_excel(local_file_path, index=False)
        
        result = MLService.set_dataset(
            db=db,
            session_id=session_id,
            dataset=df,
            file_name=file.filename,
            file_path=local_file_path
        )   
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def train_model(
    session_id: str = Form(...),
    dataset_id: str = Form(...),
    task_type: str = Form(...),
    target_column: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Train models according to the task_type and return the best model.
    """
    valid_tasks = ["classification", "regression", "clustering"]
    if task_type not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task_type. Must be one of {valid_tasks}")
        
    try:
        result = MLService.train_model(
            db=db,
            session_id=session_id,
            dataset_id=dataset_id,
            task_type=task_type,
            target_column=target_column
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{session_id}")
async def get_results(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve the results of the best model trained on the data.
    """
    try:
        result = MLService.get_model_results(db, session_id)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/{session_id}")
async def download_model(
    session_id: str,
    format_type: str = "pkl",
    db: Session = Depends(get_db)
):
    """
    Retrieve the saved model in either joblib or pkl format as requested.
    """
    if format_type not in ["pkl", "joblib"]:
        raise HTTPException(status_code=400, detail="format_type must be either 'pkl' or 'joblib'")
        
    try:
        file_path, file_name = MLService.get_model_file(
            db=db, 
            session_id=session_id, 
            format_type=format_type
        )
        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type="application/octet-stream"
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

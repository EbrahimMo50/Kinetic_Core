from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Optional
import pandas as pd
import io
import uuid

from src.api.db.db_engine import get_db
from src.api.services.file_service import FileService
from src.api.services.training_service import TrainingService
from src.api.services.session_service import SessionService

router = APIRouter()

@router.post("/train")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a dataset, create a session, and queue model training.
    """
    valid_extensions = ('.csv', '.xlsx', '.xls')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(status_code=400, detail=f"Only {', '.join(valid_extensions)} files are supported")

    valid_tasks = ["classification", "regression", "clustering"]
    if task_type not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task_type. Must be one of {valid_tasks}")
        
    session_id = str(uuid.uuid4())
        
    try:
        content = await file.read()
        
        file_ext = file.filename.lower().split('.')[-1]
        if file_ext == 'csv':
            df = pd.read_csv(io.BytesIO(content), na_values=['?'])
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(content))
            
        
        upload_dir = Path("uploads") / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        local_file_path = str(upload_dir / file.filename)
        
        if file_ext == 'csv':
            df.to_csv(local_file_path, index=False)
        elif file_ext in ['xlsx', 'xls']:
            df.to_excel(local_file_path, index=False)

        SessionService.ensure_session_result(db, session_id)
        
        result = FileService.set_dataset(
            db=db,
            session_id=session_id,
            dataset=df,
            file_name=file.filename,
            file_path=local_file_path
        )   

        background_tasks.add_task(
            TrainingService.train_model_background,
            session_id,
            result["dataset_id"],
            task_type,
            target_column,
        )

        return {
            "session_id": session_id,
            "dataset_id": result["dataset_id"],
            "status": "queued",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{session_id}")
async def get_results(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Poll the training result for a session.
    """
    try:
        result = SessionService.get_model_results(db, session_id)
        if result.get("status") == "in_progress":
            return JSONResponse(
                status_code=200,
                content={"session_id": session_id, "status": "in_progress"},
            )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{file_id}")
async def download_file(
    file_id: int,
    db: Session = Depends(get_db)
):
    """
    Download a trained model artifact by file id.
    """
    try:
        file_path, file_name = FileService.get_model_file(
            db=db,
            file_id=file_id,
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



from datetime import datetime
from typing import Any, Dict
from sqlalchemy.orm import Session as DBSession

from src.api.db.models import Result, ResultStatus, File, FileType, Session as SessionRecord


class SessionService:
    """Service for session and result management"""

    @staticmethod
    def ensure_session_result(db: DBSession, session_id: str) -> Result:
        """
        Create or reset the session/result rows for a new training request.
        
        Args:
            db: Database session
            session_id: Session ID
        
        Returns:
            Result record for the session
        """
        session_record = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if session_record is None:
            session_record = SessionRecord(session_id=session_id)
            db.add(session_record)
        else:
            session_record.last_accessed = datetime.utcnow()

        result = db.query(Result).filter(Result.session_id == session_id).first()
        if result is None:
            result = Result(
                session_id=session_id,
                status=ResultStatus.IN_PROGRESS.value,
                report=None,
                error_message=None,
            )
            db.add(result)
        else:
            result.status = ResultStatus.IN_PROGRESS.value
            result.report = None
            result.error_message = None

        db.commit()
        db.refresh(result)
        return result

    @staticmethod
    def get_model_results(db: DBSession, session_id: str) -> Dict[str, Any]:
        """
        Retrieve model results for a session.
        
        Args:
            db: Database session
            session_id: Session ID
        
        Returns:
            Result dict with status, report, and files
        """
        result = db.query(Result).filter(Result.session_id == session_id).first()
        if not result:
            raise ValueError(f"No result found for session {session_id}")

        status_value = str(result.status).lower()
        
        # Return in_progress status
        if status_value == ResultStatus.IN_PROGRESS.value.lower():
            return {
                "session_id": session_id,
                "status": "in_progress",
            }
        
        # Return failure with error message
        if status_value == ResultStatus.FAILED.value.lower():
            return {
                "session_id": session_id,
                "status": "failed",
                "error_message": result.error_message,
            }
        
        # Get model files for done status
        status_value = "done" if status_value == ResultStatus.DONE.value.lower() else status_value
        
        files = db.query(File).filter(
            File.result_session_id == session_id,
            File.artifact_type == FileType.MODEL.value
        ).order_by(File.id.asc()).all()
        
        files_array = []

        for file_record in files:
            extension = str(file_record.original_name).lower().split(".")[-1]
            if extension not in {"pkl", "joblib"}:
                extension = ""

            file_type = extension if extension else "other"
            files_array.append({
                "type": file_type,
                "id": file_record.id,
            })

        report = result.report or {}
        if isinstance(report, dict):
            best_model = report.get("best_model")
            if not isinstance(best_model, dict):
                best_model = {}
            best_model["files"] = files_array
            report["best_model"] = best_model

        return {
            "session_id": session_id,
            "status": status_value,
            "report": report,
        }

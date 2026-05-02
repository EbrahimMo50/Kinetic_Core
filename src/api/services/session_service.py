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
        
        file_payload = []
        file_id = None

        for file_record in files:
            record = {
                "file_id": file_record.id,
                "file_name": file_record.original_name,
                "type": file_record.artifact_type,
                "mime_type": file_record.mime_type,
            }
            file_payload.append(record)
            if file_id is None and str(file_record.original_name).lower().endswith(".pkl"):
                file_id = file_record.id

        if file_id is None and file_payload:
            file_id = file_payload[0]["file_id"]

        report = result.report or {}

        return {
            "session_id": session_id,
            "status": status_value,
            "report": report,
        }

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.deps import get_current_user, get_db
from app.models import Record, User
from app.schemas import RecordOut

router = APIRouter(prefix="/api", tags=["records"])


@router.get("/records", response_model=list[RecordOut])
def list_records(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 100,
) -> list[RecordOut]:
    rows = db.scalars(
        select(Record).where(Record.user_id == user.user_id).order_by(Record.record_id.desc()).limit(limit)
    ).all()
    return [RecordOut.model_validate(r) for r in rows]

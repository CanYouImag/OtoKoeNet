from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.deps import get_current_user, get_db
from app.ml.engine import Engine
from app.models import Record, User
from app.routers.evaluate import _get_engine, _save_upload
from app.schemas import RecognizeOut

router = APIRouter(prefix="/api", tags=["recognize"])


@router.post("/recognize", response_model=RecognizeOut)
def recognize(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RecognizeOut:
    engine: Engine = _get_engine(request)
    audio_path = _save_upload(file)
    t0 = time.perf_counter()
    feat = engine.featurize(audio_path)
    text = engine.recognize(feat)
    duration = time.perf_counter() - t0

    db.add(
        Record(
            user_id=user.user_id,
            test_type="recognize",
            ref_text="",
            result_text=text,
            score=None,
            audio_path=audio_path,
        )
    )
    db.commit()

    return RecognizeOut(recognized_text=text)

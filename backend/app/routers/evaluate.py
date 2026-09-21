from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.bank import text_to_mora
from app.config import settings
from app.deps import get_current_user, get_db
from app.ml.engine import Engine
from app.models import Record, User
from app.schemas import EvaluateOut, MoraScore

router = APIRouter(prefix="/api", tags=["evaluate"])

_ALLOWED_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".aac", ".wma", ".mp4", ".webm"}


def _get_engine(request: Request) -> Engine:
    return request.app.state.engine


def _save_upload(file: UploadFile) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅支持 wav / flac / ogg 音频")
    name = f"{uuid.uuid4().hex}{ext}"
    path = settings.upload_dir / name
    with open(path, "wb") as f:
        f.write(file.file.read())
    return str(path)


def _color(score: float) -> str:
    if score >= settings.score_green:
        return "green"
    if score >= settings.score_yellow:
        return "yellow"
    return "red"


@router.post("/evaluate", response_model=EvaluateOut)
def evaluate(
    request: Request,
    file: UploadFile = File(...),
    ref_text: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvaluateOut:
    engine: Engine = _get_engine(request)
    morae = text_to_mora(ref_text)
    mora_ids: list[int] = []
    for m in morae:
        if not engine.mora_vocab.has(m):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"参考文本含词表外音节: {m}")
        mora_ids.append(engine.mora_vocab.encode([m])[0])
    if not mora_ids:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "参考文本无法解析")

    audio_path = _save_upload(file)
    t0 = time.perf_counter()
    feat = engine.featurize(audio_path)
    scores, total = engine.evaluate(feat, mora_ids)
    duration = time.perf_counter() - t0

    details = [
        MoraScore(phoneme=m, score=round(s * 100, 1), color=_color(s))
        for m, s in zip(morae, scores)
    ]
    total_score = round(total * 100, 1)

    db.add(
        Record(
            user_id=user.user_id,
            test_type="evaluate",
            ref_text=ref_text,
            result_text="",
            score=total_score,
            audio_path=audio_path,
        )
    )
    db.commit()

    return EvaluateOut(ref_text=ref_text, total_score=total_score, duration=duration, details=details)

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from otokoenet.align import AlignmentError, suspect_long_vowels

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
    try:
        aln = engine.evaluate_detailed(feat, mora_ids)
    except AlignmentError as e:
        db.commit()
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            {
                "error": "align_failed",
                "reason": e.reason,
                "detail": e.detail,
            },
        ) from e
    duration = time.perf_counter() - t0

    # 長音规则：标定后的默认工作点（rel<1.0, score<0.5×句内中位数），只加标志、
    # 不改分数（正类标签不是学习者真值，见 schemas.MoraScore.suspect_long_vowel）。
    suspect = set(suspect_long_vowels(morae, aln))

    details = [
        MoraScore(
            phoneme=m,
            score=round(s * 100, 1),
            color=_color(s),
            index=i,
            suspect_long_vowel=i in suspect,
            # start/end 是 Viterbi 路径上该 token 的帧区间。训到收敛的模型逐帧过度
            # 自信，Viterbi 几乎只给每个 token 1 帧，所以这两个值只表示「最可能的那
            # 几毫秒在哪」，宽度不携带时长信息。
            start_ms=round(aln.spans[i][0] * aln.frame_ms, 1),
            end_ms=round(aln.spans[i][1] * aln.frame_ms, 1),
            # duration/rel_duration 走后验：Viterbi 时长恒等于 1 帧（实测 6781 个
            # mora 实例标准差全为 0），拿去做前端展示毫无意义；后验时长才有区分度。
            # 注意逐实例取的是该 token 自己的两个状态，重复 mora 不会重复计数。
            duration_ms=round(aln.post_duration_ms(i), 1),
            rel_duration=round(aln.post_rel_durations[i], 3),
        )
        for i, (m, s) in enumerate(zip(morae, aln.scores))
    ]
    total_score = round(aln.total * 100, 1)

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

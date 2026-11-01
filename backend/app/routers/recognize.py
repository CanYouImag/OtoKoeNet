from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from otokoenet.data import AudioDecodeError

from app.ml.engine import Engine
from app.routers.evaluate import _get_engine, _save_upload
from app.schemas import RecognizeOut

router = APIRouter(prefix="/api", tags=["recognize"])


@router.post("/recognize", response_model=RecognizeOut)
def recognize(
    request: Request,
    file: UploadFile = File(...),
) -> RecognizeOut:
    engine: Engine = _get_engine(request)
    audio_path = _save_upload(file)
    try:
        try:
            feat = engine.featurize(audio_path)
        except AudioDecodeError as e:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "音频无法处理：文件损坏、为空、太短或格式不受支持"
            ) from e
        text = engine.recognize(feat)
    finally:
        audio_path.unlink(missing_ok=True)

    return RecognizeOut(recognized_text=text)

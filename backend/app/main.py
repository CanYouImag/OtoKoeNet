from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.ml.engine import Engine
from app.routers import auth, evaluate, recognize, records, texts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("otokoenet")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("loading model %s", settings.ckpt_path)
    app.state.engine = Engine(settings)
    logger.info("model loaded: %s", json.dumps(app.state.engine.manifest(), ensure_ascii=False))
    yield


app = FastAPI(title="日语口语练习 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(evaluate.router)
app.include_router(recognize.router)
app.include_router(records.router)
app.include_router(texts.router)


@app.get("/api/health")
def health(request: Request) -> dict:
    """健康检查必须反映「模型真的能推理」，而不只是进程活着。"""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "模型未加载")
    return {"status": "ok", "model": engine.manifest()}

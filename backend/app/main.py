from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
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
    logger.info("model loaded")
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
def health() -> dict:
    return {"status": "ok"}

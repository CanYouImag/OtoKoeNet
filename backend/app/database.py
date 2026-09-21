from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

engine = create_engine(settings.db_url, connect_args={"check_same_thread": False} if settings.db_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

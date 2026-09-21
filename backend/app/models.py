from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Record(Base):
    __tablename__ = "records"

    record_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.user_id"), index=True, nullable=False)
    test_type: Mapped[str] = mapped_column(String(16), nullable=False)
    ref_text: Mapped[str] = mapped_column(String(256), default="")
    result_text: Mapped[str] = mapped_column(String(512), default="")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_path: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

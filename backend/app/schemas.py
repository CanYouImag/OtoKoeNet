from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RegisterIn(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class MoraScore(BaseModel):
    phoneme: str
    score: float
    color: str


class EvaluateOut(BaseModel):
    ref_text: str
    total_score: float
    duration: float
    details: list[MoraScore]


class RecognizeOut(BaseModel):
    recognized_text: str


class RecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    record_id: int
    test_type: str
    ref_text: str
    result_text: str
    score: float | None
    created_at: datetime


class BankItem(BaseModel):
    orig: str
    hira: str
    n_mora: int


class BankText(BaseModel):
    id: int
    text: str
    items: list[BankItem]

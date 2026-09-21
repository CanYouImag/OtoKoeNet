from __future__ import annotations

from fastapi import APIRouter

from app.bank import build_bank
from app.schemas import BankText

router = APIRouter(prefix="/api", tags=["texts"])


@router.get("/texts", response_model=list[BankText])
def list_texts() -> list[BankText]:
    return [BankText(**item) for item in build_bank()]

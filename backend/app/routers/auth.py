from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models import User
from app.schemas import LoginIn, RegisterIn, TokenOut
from app.security import create_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenOut)
def register(body: RegisterIn, db: Session = Depends(get_db)) -> TokenOut:
    if db.scalar(select(User).where(User.username == body.username)):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "用户名已存在")
    user = User(username=body.username, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenOut(access_token=create_token(user.user_id, user.username), username=user.username)


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)) -> TokenOut:
    user = db.scalar(select(User).where(User.username == body.username))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    return TokenOut(access_token=create_token(user.user_id, user.username), username=user.username)

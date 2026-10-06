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
    # 实例级信息：参考文本里每出现一次就是一条，重复 mora 各自独立
    index: int = 0
    # start/end 为 Viterbi 命中帧区间（模型最自信的那几毫秒，宽度不携带时长信息）
    start_ms: float = 0.0
    end_ms: float = 0.0
    # duration/rel_duration 为后验时长与其相对中位数的比值
    duration_ms: float = 0.0
    rel_duration: float = 1.0
    # 長音规则（otokoenet.align.suspect_long_vowels，零训练、可直接上）：
    # 该 mora 是「ー」、后验相对时长偏低、且分数远低于句内中位数 → 疑似漏读/短读長音。
    # 前端可据此提示，但**不自动改分**：规则的正类标签是「模型把它听成别的音」，
    # 不是「学习者真的发错」，真正校准需要学习者录音（见 log/stage17_plan.log §三）。
    suspect_long_vowel: bool = False


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

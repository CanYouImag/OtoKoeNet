from __future__ import annotations

import pykakasi

from otokoenet.text import kana_to_mora, normalize_text

_kks = pykakasi.kakasi()

_RAW_TEXTS = [
    "今日はいい天気ですね。",
    "猫が一匹、庭にいます。",
    "北海道の冬はとても寒いですね。",
    "友達と一緒に映画を見に行きました。",
    "日本語の勉強は好きです。",
    "コーヒーを一杯お願いします。",
    "学校に遅刻してしまいました。",
    "電車が混んでいて、座れませんでした。",
    "駅前のスーパーで買い物をしました。",
    "大きな木の下で休みましょう。",
    "先生に質問があります。",
    "新しい靴を買いました。",
]


def build_bank() -> list[dict]:
    bank = []
    for i, raw in enumerate(_RAW_TEXTS):
        items = []
        for item in _kks.convert(raw):
            kana = item["kana"]
            n_mora = len(kana_to_mora(kana))
            items.append({"orig": item["orig"], "hira": item["hira"], "n_mora": n_mora})
        bank.append({"id": i + 1, "text": raw, "items": items})
    return bank


def text_to_mora(text: str) -> list[str]:
    return kana_to_mora("".join(c["kana"] for c in _kks.convert(text)))


def normalize_bank_text(text: str) -> str:
    return normalize_text(text)

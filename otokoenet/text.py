from __future__ import annotations

import json
from dataclasses import dataclass


_STRIP_CHARS = set(
    "、。，．「」『』（）()【】・？！？!…—\"'‘’“”<>《》〈〉&;:／/\\"
)

_SMALL_Y = set("ャュョ")
_SMALL_A = set("ァィゥェォ")
_MORA_SPECIAL = ("ッ", "ン", "ー")

SOS = "<sos>"
EOS = "<eos>"


def normalize_text(text: str) -> str:
    return "".join(ch for ch in text if ch not in _STRIP_CHARS and not ch.isspace())


def kana_to_mora(kana: str) -> list[str]:
    morae: list[str] = []
    i, n = 0, len(kana)
    while i < n:
        ch = kana[i]
        if ch in _MORA_SPECIAL:
            morae.append(ch)
            i += 1
            continue
        if i + 1 < n and kana[i + 1] in _SMALL_Y:
            morae.append(ch + kana[i + 1])
            i += 2
            continue
        if i + 1 < n and kana[i + 1] in _SMALL_A:
            morae.append(ch + kana[i + 1])
            i += 2
            continue
        morae.append(ch)
        i += 1
    return morae


@dataclass
class Vocab:
    symbols: list[str]
    blank: str = "<blank>"

    def __post_init__(self) -> None:
        if self.symbols[0] != self.blank:
            self.symbols = [self.blank] + list(self.symbols)
        for sym in (SOS, EOS):
            if sym not in self.symbols:
                self.symbols.append(sym)
        self._id2sym = {i: s for i, s in enumerate(self.symbols)}
        self._sym2id = {s: i for i, s in enumerate(self.symbols)}

    @property
    def sos_id(self) -> int:
        return self._sym2id[SOS]

    @property
    def eos_id(self) -> int:
        return self._sym2id[EOS]

    @classmethod
    def from_corpus(cls, sequences: list[list[str]], blank: str = "<blank>") -> "Vocab":
        seen: list[str] = []
        seen_set: set[str] = set()
        for seq in sequences:
            for sym in seq:
                if sym not in seen_set:
                    seen_set.add(sym)
                    seen.append(sym)
        return cls([blank] + seen, blank=blank)

    def encode(self, symbols: list[str]) -> list[int]:
        return [self._sym2id[s] for s in symbols]

    def has(self, sym: str) -> bool:
        return sym in self._sym2id

    def decode(self, ids: list[int]) -> list[str]:
        return [self._id2sym[i] for i in ids]

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def blank_id(self) -> int:
        return 0

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"symbols": self.symbols, "blank": self.blank}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["symbols"], blank=data["blank"])

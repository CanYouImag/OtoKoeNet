from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import jaconv
import pykakasi
from sudachipy import dictionary

from otokoenet.text import normalize_text


def _read_kana(text: str) -> str:
    return "".join(c["kana"] for c in pykakasi.kakasi().convert(text))


_PUNCT = set("。、．！？・，々「」『』（）…")

_DUMMY = "__BOS__"


def build_table(transcript_path: Path, out_path: Path) -> None:
    """用 JSUT 语料训练假名→汉字转换表（pykakasi 假名键 + Sudachi 分词 + 语料语言模型）。"""
    rows = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, text = line.split(":", 1)
            rows.append((utt, normalize_text(text)))

    sud = dictionary.Dictionary().create()
    kks = pykakasi.kakasi()

    unigram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bigram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_words = 0

    for _, text in rows:
        seq: list[str] = []
        for m in sud.tokenize(text):
            surf = m.surface()
            if surf in _PUNCT:
                continue
            kana = "".join(c["kana"] for c in kks.convert(surf))
            key = jaconv.kata2hira(kana)
            if not key:
                continue
            seq.append(key)
            unigram[key][surf] += 1
            total_words += 1
        prev = _DUMMY
        for key in seq:
            bigram[prev][key] += 1
            prev = key
        bigram[prev][_DUMMY] += 1

    data = {
        "unigram": {k: dict(v) for k, v in unigram.items()},
        "bigram": {k: dict(v) for k, v in bigram.items()},
        "total_words": total_words,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"table: {total_words} words, {len(unigram)} readings -> {out_path}")


class Kana2Kanji:
    """假名(片假名/平假名)→汉字：Viterbi 分词 + 语料 unigram/bigram。"""

    MAX_WORD = 8

    def __init__(self, table_path: Path) -> None:
        with open(table_path, encoding="utf-8") as f:
            data = json.load(f)
        self.unigram: dict[str, dict[str, int]] = data["unigram"]
        self.dict: dict[str, dict[str, int]] = data.get("dict", {})
        total = data["total_words"]
        alpha = 1.0
        self.n_r: dict[str, int] = {}
        for r, surfaces in self.unigram.items():
            self.n_r[r] = sum(surfaces.values())
        v = len(self.n_r) + total
        self.logp_r = {
            r: math.log((n + alpha) / (total + alpha * v))
            for r, n in self.n_r.items()
        }
        self.oov = math.log(alpha / (total + alpha * v))
        self.bigram = data["bigram"]
        delta = 0.5
        self.logp_next: dict[str, dict[str, float]] = defaultdict(dict)
        for prev, nxts in self.bigram.items():
            n_prev = sum(nxts.values())
            for r, n in nxts.items():
                self.logp_next[prev][r] = math.log((n + delta) / (n_prev + delta * v))

    def _best_surface(self, reading: str) -> str:
        surfaces = self.unigram.get(reading)
        if surfaces:
            return max(surfaces, key=surfaces.get)
        surfaces = self.dict.get(reading)
        return max(surfaces, key=surfaces.get) if surfaces else reading

    def convert(self, kana: str) -> str:
        s = jaconv.kata2hira(kana)
        n = len(s)
        if n == 0:
            return ""
        neg = float("-inf")
        score = [neg] * (n + 1)
        back: list[tuple[int, str, str] | None] = [None] * (n + 1)
        score[0] = 0.0
        for i in range(n):
            if score[i] == neg:
                continue
            for L in range(1, min(self.MAX_WORD, n - i) + 1):
                sub = s[i : i + L]
                if sub in self.n_r:
                    cost = self.logp_r[sub]
                    if back[i] is not None:
                        prev_r = back[i][1]
                        cost += self.logp_next.get(prev_r, {}).get(sub, self.oov)
                    v = score[i] + cost
                    if v > score[i + L]:
                        score[i + L] = v
                        back[i + L] = (i, sub, self._best_surface(sub))
                elif sub in self.dict:
                    cost = self.oov
                    if back[i] is not None:
                        prev_r = back[i][1]
                        cost += self.logp_next.get(prev_r, {}).get(sub, self.oov)
                    v = score[i] + cost
                    if v > score[i + L]:
                        score[i + L] = v
                        back[i + L] = (i, sub, self._best_surface(sub))
            if s[i] in self.n_r or s[i] in self.dict:
                continue
            v = score[i] + self.oov
            if v > score[i + 1]:
                score[i + 1] = v
                back[i + 1] = (i, s[i], s[i])
        out: list[str] = []
        i = n
        while i > 0:
            b = back[i]
            if b is None:
                i -= 1
                continue
            _, _, surf = b
            out.append(surf)
            i = b[0]
        return "".join(reversed(out))
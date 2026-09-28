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

# 插值语言模型权重：p(word) = w3·p3|2gram + w2·p2|1gram + w1·p1
_LM_P3 = 0.6
_LM_P2 = 0.3
_LM_P1 = 0.1


def build_table(transcript_path: Path, out_path: Path, use_trigram: bool = True) -> None:
    """用 JSUT 语料训练假名→汉字转换表（pykakasi 假名键 + Sudachi 分词 + n-gram 语言模型）。"""
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
    trigram: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
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
        for idx, key in enumerate(seq):
            prev1 = seq[idx - 1] if idx >= 1 else _DUMMY
            prev2 = seq[idx - 2] if idx >= 2 else _DUMMY
            bigram[prev1][key] += 1
            trigram[(prev2, prev1)][key] += 1
        trigram[(seq[-1] if seq else _DUMMY, _DUMMY)][_DUMMY] += 1

    data = {
        "unigram": {k: dict(v) for k, v in unigram.items()},
        "bigram": {k: dict(v) for k, v in bigram.items()},
        "trigram": {f"{a}\u0001{b}": dict(v) for (a, b), v in trigram.items()},
        "total_words": total_words,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"table: {total_words} words, {len(unigram)} readings -> {out_path}")


def _log(*args) -> float:
    v = math.log(*args) if args else 0.0
    return v


class Kana2Kanji:
    """假名(片假名/平假名)→汉字：Viterbi 分词 + 语料 unigram/bigram/trigram。"""

    MAX_WORD = 8

    def __init__(self, table_path: Path, use_trigram: bool = True) -> None:
        with open(table_path, encoding="utf-8") as f:
            data = json.load(f)
        self.unigram: dict[str, dict[str, int]] = data["unigram"]
        self.dict: dict[str, dict[str, int]] = data.get("dict", {})
        self.bigram: dict[str, dict[str, int]] = data.get("bigram", {})
        self.tgram: dict[str, dict[str, int]] = data.get("trigram", {})
        self._tri = use_trigram
        total = data["total_words"]
        self._v = len(self.unigram) + total
        self._build_probs()

    def _build_probs(self) -> None:
        """预计算 unigram/bigram/trigram 的平滑对数概率（delta 平滑 + 插值）。"""
        unigram = self.unigram
        v = self._v
        alpha = 1.0
        self.p1: dict[str, float] = {}
        self.n_r: dict[str, int] = {}
        for r, surfaces in unigram.items():
            n = sum(surfaces.values())
            self.n_r[r] = n
            self.p1[r] = _log((n + alpha) / (v))
        self.oov1 = _log(alpha / v)

        delta = 0.5
        self.p2: dict[str, dict[str, float]] = defaultdict(dict)
        for prev, nxts in self.bigram.items():
            n_prev = sum(nxts.values())
            d = self.p2[prev]
            for r, n in nxts.items():
                d[r] = _log((n + delta) / (n_prev + delta * v))
        self.p3: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for key, nxts in self.tgram.items():
            a, _, b = key.partition("\u0001")
            pair = (a, b)
            n_prev = sum(nxts.values())
            d = self.p3[pair]
            for r, n in nxts.items():
                d[r] = _log((n + delta) / (n_prev + delta * v))

    def logp_word(self, a: str, b: str, r: str) -> float:
        """给定上文 (a, b)（a 为上上文、b 为上文），返回 r 的插值 log 概率。"""
        p1 = self.p1[r] if r in self.p1 else self.oov1
        if b == _DUMMY or b not in self.p2:
            return p1  # 句首退化到 unigram
        p2 = self.p2[b].get(r)
        if p2 is None:
            p2 = self.oov1
        tri = self.p3.get((a, b), {})
        p3 = tri.get(r)
        if p3 is None:
            p3 = self.oov1
        p3e, p2e, p1e = math.exp(p3), math.exp(p2), math.exp(p1)
        return _log(_LM_P3 * p3e + _LM_P2 * p2e + _LM_P1 * p1e)

    def _best_surface(self, reading: str) -> str:
        surfaces = self.unigram.get(reading)
        if surfaces:
            return max(surfaces, key=surfaces.get)
        surfaces = self.dict.get(reading)
        return max(surfaces, key=surfaces.get) if surfaces else reading

    def candidates(self, s: str, i: int) -> list[str]:
        """位置 i 起的所有候选读音。"""
        out: list[str] = []
        for L in range(1, min(self.MAX_WORD, len(s) - i) + 1):
            sub = s[i : i + L]
            if sub in self.n_r or sub in self.dict:
                out.append(sub)
        return out

    def convert(self, kana: str) -> str:
        s = jaconv.kata2hira(kana)
        n = len(s)
        if n == 0:
            return ""
        neg = float("-inf")

        # dp[i][(r1, r2)] = 末尾两个读音为 (r1, r2)、覆盖 [0, i) 的最大 log 得分
        dp: list[dict[tuple[str, str], float]] = [dict() for _ in range(n + 1)]
        back: list[dict[tuple[str, str], tuple[int, str, tuple[str, str]]]] = [dict() for _ in range(n + 1)]
        dp[0][(_DUMMY, _DUMMY)] = 0.0

        for i in range(n):
            for (r1, r2), sc in dp[i].items():
                for sub in self.candidates(s, i):
                    end = i + len(sub)
                    cost = self.logp_word(r1, r2, sub) if i > 0 else self.p1.get(sub, self.oov1)
                    nsc = sc + cost
                    key = (r2, sub)
                    if nsc > dp[end].get(key, neg):
                        dp[end][key] = nsc
                        back[end][key] = (i, sub, (r1, r2))
                # 单字符 OOV 兜底
                ch = s[i]
                if ch not in self.n_r and ch not in self.dict:
                    end = i + 1
                    nsc = sc + self.oov1
                    key = (r2, ch)
                    if nsc > dp[end].get(key, neg):
                        dp[end][key] = nsc
                        back[end][key] = (i, ch, (r1, r2))

        best_key = None
        best_sc = neg
        for key, sc in dp[n].items():
            if sc > best_sc:
                best_sc, best_key = sc, key

        if best_key is None:
            return s

        readings: list[str] = []
        i, key = n, best_key
        while i > 0:
            b = back[i].get(key)
            if b is None:
                break
            start, sub, prev_key = b
            readings.append(sub)
            i, key = start, prev_key
        readings.reverse()
        return "".join(self._best_surface(r) for r in readings)
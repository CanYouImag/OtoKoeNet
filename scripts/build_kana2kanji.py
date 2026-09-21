from __future__ import annotations

import argparse
import gzip
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jaconv
import pykakasi
from sudachipy import dictionary

from otokoenet.kana2kanji import _DUMMY, _PUNCT
from otokoenet.text import normalize_text

_COMMON_PRI = {"news1", "news2", "ichi1", "ichi2", "spec1", "spec2", "gai1", "gai2"}

_HIRA = set("ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷ"
            "へべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔ"
            "ゕゖー")


def _is_hiragana(s: str) -> bool:
    return all(c in _HIRA for c in s)


def _kana_key(surf: str) -> str:
    return jaconv.kata2hira("".join(c["kana"] for c in pykakasi.kakasi().convert(surf)))


def jmdict_weight(pri: list[str]) -> int:
    if any(p in _COMMON_PRI for p in pri):
        return 3
    if any(p.startswith("nf") for p in pri):
        return 2
    return 1


def load_jmdict(path: Path) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], int]:
    """解析 JMdict XML。

    返回:
        (kanji_readings, kana_only, n_pairs)
        kanji_readings: {读音(hira): {汉字写法: 权重}} 只收有汉字表记的词条。
        kana_only: {读音(hira): {原假名写法(片/平): 权重}} 无汉字表记的词条（外来语等）。
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    mode = "rt" if str(path).endswith(".gz") else "r"
    with opener(path, mode, encoding="utf-8") as f:
        root = ET.fromstring(f.read())

    kanji_readings: dict[str, dict[str, int]] = defaultdict(dict)
    kana_only: dict[str, dict[str, int]] = defaultdict(dict)
    n_pairs = 0
    for entry in root.findall("entry"):
        kebs = [k.findtext("keb") for k in entry.findall("k_ele")]
        kebs = [k for k in kebs if k and "・" not in k]
        for r in entry.findall("r_ele"):
            reb = r.findtext("reb")
            if not reb:
                continue
            key = jaconv.kata2hira(reb)
            if not _is_hiragana(key):
                continue
            pri = [p.text or "" for p in (r.findall("re_pri") + entry.findall(".//ke_pri"))]
            w = jmdict_weight(pri)
            if kebs:
                restr = {x.text for x in r.findall("re_restr")}
                cands = [k for k in kebs if k in restr] if restr else kebs
                if not cands:
                    continue
                for surf in cands:
                    kanji_readings[key][surf] = max(kanji_readings[key].get(surf, 0), w)
                n_pairs += len(cands)
            else:
                kana_only[key][reb] = max(kana_only[key].get(reb, 0), w)
    return kanji_readings, kana_only, n_pairs


def build_table(transcript_path: Path, jmdict_path: Path, out_path: Path, max_surfaces: int = 10) -> None:
    rows = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, text = line.split(":", 1)
            rows.append(normalize_text(text))

    sud = dictionary.Dictionary().create()
    unigram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bigram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_words = 0

    for text in rows:
        seq: list[str] = []
        for m in sud.tokenize(text):
            surf = m.surface()
            if surf in _PUNCT:
                continue
            key = _kana_key(surf)
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

    jmdict, kana_only, n_pairs = load_jmdict(jmdict_path)

    dic: dict[str, dict[str, int]] = {}
    for key, surfs in jmdict.items():
        if len(key) > 8:
            continue
        if key in unigram:
            extras = {s: w for s, w in surfs.items() if s not in unigram[key]}
            if extras:
                dic[key] = extras
        else:
            dic[key] = surfs
    for key, surfs in kana_only.items():
        if len(key) > 8:
            continue
        if key in unigram:
            extras = {s: w for s, w in surfs.items() if s not in unigram[key]}
            if extras:
                dic.setdefault(key, {}).update(extras)
        else:
            dic.setdefault(key, {}).update(surfs)
    dic = {k: dict(sorted(v.items(), key=lambda kv: kv[1], reverse=True)[:max_surfaces]) for k, v in dic.items()}

    data = {
        "unigram": {k: dict(v) for k, v in unigram.items()},
        "dict": dic,
        "bigram": {k: dict(v) for k, v in bigram.items()},
        "total_words": total_words,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"corpus readings={len(unigram)} total_words={total_words}")
    print(f"jmdict pairs={n_pairs} jmdict readings+extras+loanwords={len(dic)}")
    print(f"  -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", type=str, default="data/jsut_ver1.1/basic5000/transcript_utf8.txt")
    ap.add_argument("--jmdict", type=str, default="data/jmdict/JMdict_e.gz")
    ap.add_argument("--out", type=str, default="data/cache/basic5000/kana2kanji.json")
    ap.add_argument("--max-surfaces", type=int, default=10)
    args = ap.parse_args()
    build_table(Path(args.transcript), Path(args.jmdict), Path(args.out), args.max_surfaces)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import gzip
import json
import pickle
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


def _kana_key(kks: pykakasi.kakasi, surf: str) -> str:
    return jaconv.kata2hira("".join(c["kana"] for c in kks.convert(surf)))


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


def load_pykakasi_dict(path: Path, max_len: int = 8) -> dict[str, dict[str, int]]:
    """读取 pykakasi 内置 kanji→kana 词典（pickle），反转为 reading→surface 词表。

    词典结构为 {id: {surface: [(reading, None), ...]}}，reading 直接取用（无需 pykakasi.convert）。
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for surfaces in raw.values():
        for surf, readings in surfaces.items():
            if not isinstance(readings, (list, tuple)):
                readings = [(readings, None)]
            for r, _ in readings:
                if r is None:
                    continue
                rk = jaconv.kata2hira(r)
                if not rk or len(rk) > max_len or not _is_hiragana(rk):
                    continue
                out[rk][surf] = max(out[rk].get(surf, 0), 1)
    return out


def load_rows_from_manifest(path: Path) -> list[tuple[str, str]]:
    """从 cache manifest 读 (utt, text)。

    manifest 的 entries 就是划分后的 train 集合，`text` 已在 prepare.py 里
    规范化过（这里再跑一次 normalize_text 是幂等的）。**这是唯一安全的来源**：
    直接读 transcript_utf8.txt 会把 val 200 + test 200 句一起算进 unigram /
    bigram / trigram 统计，等于把验证测试集泄漏进转换表。
    """
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    return [(e["utt"], normalize_text(e["text"])) for e in manifest["entries"]]


def load_rows_from_transcript(path: Path) -> list[tuple[str, str]]:
    """从 transcript_utf8.txt 读 (utt, text)。**包含全部 5000 句，会泄漏。**"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, text = line.split(":", 1)
            rows.append((utt, normalize_text(text)))
    return rows


def check_no_eval_leakage(source_utts: set[str], cache_dir: Path) -> None:
    """断言转换表语料与 val/test manifest 零重叠。"""
    for split in ("val", "test"):
        p = cache_dir / f"{split}.json"
        if not p.exists():
            print(f"  [warn] {p} 不存在，跳过 {split} 泄漏检查")
            continue
        with open(p, encoding="utf-8") as f:
            man = json.load(f)
        overlap = source_utts & {e["utt"] for e in man["entries"]}
        if overlap:
            raise SystemExit(
                f"泄漏：转换表语料与 {split}.json 重叠 {len(overlap)} 句，例 {sorted(overlap)[:5]}"
            )
        print(f"  [ok] 与 {split}.json 零重叠（{len(man['entries'])} 句）")


def build_table(
    source_rows: list[tuple[str, str]],
    jmdict_path: Path | None,
    out_path: Path,
    max_surfaces: int = 10,
    use_trigram: bool = True,
    pykakasi_db: Path | None = None,
    cache_dir: Path | None = None,
) -> None:
    rows = [text for _, text in source_rows]
    if cache_dir is not None:
        check_no_eval_leakage({u for u, _ in source_rows}, cache_dir)

    sud = dictionary.Dictionary().create()
    kks = pykakasi.kakasi()
    unigram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bigram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    trigram: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_words = 0

    for text in rows:
        seq: list[str] = []
        for m in sud.tokenize(text):
            surf = m.surface()
            if surf in _PUNCT:
                continue
            key = _kana_key(kks, surf)
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

    dic: dict[str, dict[str, int]] = {}

    def merge(key: str, candidates: dict[str, int]) -> None:
        if len(key) > 8:
            return
        if key in unigram:
            extras = {s: w for s, w in candidates.items() if s not in unigram[key]}
            if extras:
                dic.setdefault(key, {})
                for s, w in extras.items():
                    dic[key][s] = max(dic[key].get(s, 0), w)
        elif key not in dic:
            dic[key] = dict(candidates)

    if jmdict_path is not None and jmdict_path.exists():
        jmdict, kana_only, n_pairs = load_jmdict(jmdict_path)
        for key, cands in jmdict.items():
            merge(key, cands)
        for key, cands in kana_only.items():
            merge(key, cands)
        print(f"jmdict pairs={n_pairs}")
    else:
        print(f"skip JMdict: {jmdict_path} 不存在")

    if pykakasi_db is not None and pykakasi_db.exists():
        pk = load_pykakasi_dict(pykakasi_db)
        for key, cands in pk.items():
            merge(key, cands)
        print(f"pykakasi readings={len(pk)}")

    dic = {k: dict(sorted(v.items(), key=lambda kv: kv[1], reverse=True)[:max_surfaces]) for k, v in dic.items()}

    data = {
        "unigram": {k: dict(v) for k, v in unigram.items()},
        "dict": dic,
        "bigram": {k: dict(v) for k, v in bigram.items()},
        "trigram": {f"{a}\u0001{b}": dict(v) for (a, b), v in trigram.items()},
        "total_words": total_words,
        "source_utts": len(source_rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"corpus readings={len(unigram)} total_words={total_words}")
    print(f"dict readings={len(dic)}")
    print(f"  -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="data/cache/basic5000_v2/train.json",
                    help="划分后的 train manifest（唯一安全来源：结构性排除 val/test）")
    ap.add_argument("--cache-dir", type=str, default="data/cache/basic5000_v2",
                    help="用于 val/test 泄漏检查；缺省取 manifest 所在目录")
    ap.add_argument("--transcript", type=str, default=None,
                    help="改用 transcript_utf8.txt（**含全部 5000 句，会泄漏 val/test**，仅供对照）")
    ap.add_argument("--jmdict", type=str, default="data/jmdict/JMdict_e.gz")
    ap.add_argument("--out", type=str, default="data/cache/basic5000_v2/kana2kanji.json")
    ap.add_argument("--max-surfaces", type=int, default=12)
    ap.add_argument("--pykakasi-db", type=str, default=None,
                    help="pykakasi kanwadict4.db 路径（合并其反向读音→汉字词表），缺省自动探测")
    args = ap.parse_args()

    if args.transcript:
        source_rows = load_rows_from_transcript(Path(args.transcript))
        print(f"! 用 transcript 构建：{len(source_rows)} 句（包含 val/test，指标不可信）")
    else:
        source_rows = load_rows_from_manifest(Path(args.manifest))
        print(f"用 train manifest 构建：{len(source_rows)} 句 <- {args.manifest}")
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.manifest).parent

    pyk_db = None
    if args.pykakasi_db:
        pyk_db = Path(args.pykakasi_db)
    else:
        import site
        try:
            for sp in site.getsitepackages() + [site.getusersitepackages()]:
                cand = Path(sp) / "pykakasi" / "data" / "kanwadict4.db"
                if cand.exists():
                    pyk_db = cand
                    break
        except Exception:
            pass
        if pyk_db is None:  # 兜底：从已安装 pykakasi 定位
            k = pykakasi.__file__ and Path(pykakasi.__file__).resolve().parent / "data" / "kanwadict4.db"
            if k and k.exists():
                pyk_db = k
    if pyk_db and pyk_db.exists():
        print(f"pykakasi lexicon: {pyk_db}")
    else:
        print("pykakasi kanwadict4.db 未找到，跳过词典融合")

    jmdict_path = None if args.jmdict in ("none", "None", "") else Path(args.jmdict)
    build_table(
        source_rows,
        jmdict_path,
        Path(args.out),
        args.max_surfaces,
        use_trigram=True,
        pykakasi_db=pyk_db,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()
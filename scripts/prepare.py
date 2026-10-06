from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pykakasi
from tqdm import tqdm

from otokoenet import augment
from otokoenet.data import (
    apply_cmvn,
    cache_cmvn_is_baked,
    compute_cmvn,
    extract_fbank,
    load_wav,
)
from otokoenet.text import Vocab, kana_to_mora, normalize_kana, normalize_text

# 语料根目录下的子集名。JSUT ver1.1 的官方目录名里 `onomatopee300` 是三个 e
# （JSUT 自己的拼写），不要"修正"成 onomatopoeia，否则找不到目录。
ALL_SUBSETS = (
    "basic5000",
    "countersuffix26",
    "loanword128",
    "onomatopee300",
    "precedent130",
    "repeat500",
    "travel1000",
    "utparaphrase512",
    "voiceactress100",
)


def parse_transcript(path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, text = line.split(":", 1)
            rows.append((utt, text))
    return rows


def collect_rows(root: Path, subsets: list[str]) -> list[tuple[str, str, Path]]:
    """读取多个子集的转写，返回 (subset, utt, text)。

    utt id 必须全局唯一：cache 以 `<utt>.npy` 为键，重名会被静默覆盖。
    """
    rows: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for sub in subsets:
        sub_dir = root / sub
        wav_dir = sub_dir / "wav"
        tpath = sub_dir / "transcript_utf8.txt"
        if not tpath.exists():
            raise SystemExit(
                f"子集 {sub} 缺少 {tpath}；可用子集："
                f"{[d.name for d in sorted(root.iterdir()) if (d / 'transcript_utf8.txt').exists()]}"
            )
        for utt, text in parse_transcript(tpath):
            if utt in seen:
                raise SystemExit(f"utt id 在子集间重复：{utt}（{sub}）；cache 会互相覆盖")
            if not (wav_dir / f"{utt}.wav").exists():
                raise SystemExit(f"{sub} 的 {utt} 有转写但缺 wav：{wav_dir / (utt + '.wav')}")
            seen.add(utt)
            rows.append((sub, utt, text))
    return rows


def _load_manifest_utts(cache_dir: Path, split: str) -> list[str]:
    with open(cache_dir / f"{split}.json", encoding="utf-8") as f:
        return [e["utt"] for e in json.load(f)["entries"]]


def utt_subset(utt: str, subsets: list[str]) -> str | None:
    """从 utt id 反查子集名（如 `REPEAT500_set1_001` -> `repeat500`）。

    按前缀匹配而不是 `rsplit`，因为 repeat500 的 id 是三段式。
    """
    up = utt.upper()
    for s in sorted(subsets, key=len, reverse=True):
        if up.startswith(s.upper() + "_"):
            return s
    return None


def build_split(
    rows: list[tuple[str, str, Path]],
    args: argparse.Namespace,
    norm_texts: dict[str, str],
) -> tuple[list[str], list[str], list[str], list[str], dict]:
    """返回 (train_utts, val_utts, test_utts, probe_utts, meta)。

    val/test 的顺序沿用来源（旧 cache 的原顺序 / 洗牌后的下标序），不用
    `set` 的迭代序——`str` 的 hash 随机化会让它逐进程变，而 eval 的
    `BucketedBatchSampler(shuffle=False)` 在等长样本上按 manifest 下标 tie-break，
    顺序一变 batch 组成就变，指标会出现 1e-3 量级的无意义抖动。

    两种模式：

    - `random`（默认，向后兼容）：按 `--seed` 洗牌后切 val/test，其余进 train。
    - `holdout-from`：**逐字复用** `--holdout-from` 指向的旧 cache 的 val/test，
      其余全部进 train。这让「同划分对比」成立：新 cache 与旧 cache 的 val/test
      是同一批句子，指标可以直接比；不这么做的话换了划分就只能说新模型更好。
    """
    all_utts = [utt for _, utt, _ in rows]
    rng = random.Random(args.seed)

    if args.holdout_from:
        holdout = Path(args.holdout_from)
        val_utts = _load_manifest_utts(holdout, "val")
        test_utts = _load_manifest_utts(holdout, "test")
        missing = (set(val_utts) | set(test_utts)) - set(all_utts)
        if missing:
            raise SystemExit(
                f"--holdout-from {holdout} 的 val/test 有 {len(missing)} 句不在当前语料中，"
                f"无法复用划分：{sorted(missing)[:5]}"
            )
        overlap = set(val_utts) & set(test_utts)
        if overlap:
            raise SystemExit(f"旧 cache 的 val 与 test 互相重叠 {len(overlap)} 句，拒绝复用")
        val_set, test_set = set(val_utts), set(test_utts)
        train_utts = [u for u in all_utts if u not in val_set and u not in test_set]
        mode = f"holdout-from:{holdout}"
    else:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        test_set = {all_utts[i] for i in indices[: args.test_size]}
        val_set = {all_utts[i] for i in indices[args.test_size : args.test_size + args.val_size]}
        val_utts = [all_utts[i] for i in indices[args.test_size : args.test_size + args.val_size]]
        test_utts = [all_utts[i] for i in indices[: args.test_size]]
        train_utts = [u for u in all_utts if u not in test_set and u not in val_set]
        mode = "random"

    # probe：从 train 侧再切出一个「域探针」集，只作诊断，永不参与选模。
    # 按**规范化文本分组**切，否则 repeat500（100 句 × 5 遍）会把同一句话的
    # 4 遍留在 train、1 遍放进 probe，直接泄漏。
    #
    # probe 只从**新增子集**里切，不碰 `--holdout-from` 那个 cache 的来源子集。
    # 两个原因：(1) 探针的用途是观察 basic5000 之外的域，从 basic5000 里切没有
    # 信息量；(2) 更要紧的是，从 basic5000 train 里切会抽走只出现在那里的稀有
    # 汉字（实测 44 个），让 `--vocab-from` 的前缀保持直接失败，分类头就没法
    # warm start。
    probe_utts: list[str] = []
    if args.probe_frac > 0:
        if args.probe_subsets:
            probe_subsets = [s.strip() for s in args.probe_subsets.split(",") if s.strip()]
            unknown = [s for s in probe_subsets if s not in ALL_SUBSETS]
            if unknown:
                raise SystemExit(f"--probe-subsets 含未知子集 {unknown}")
        else:
            excluded = set()
            if args.holdout_from:
                excluded = {
                    utt_subset(u, args.subsets_list) for u in list(val_utts) + list(test_utts)
                }
                excluded.discard(None)
            probe_subsets = [s for s in args.subsets_list if s not in excluded]
        by_sub: dict[str, list[str]] = defaultdict(list)
        for sub, utt, _ in rows:
            if utt in train_utts:
                by_sub[sub].append(utt)
        for sub in probe_subsets:
            pool = by_sub.get(sub, [])
            if not pool:
                continue
            groups: dict[str, list[str]] = defaultdict(list)
            for utt in pool:
                groups[norm_texts[utt]].append(utt)
            keys = sorted(groups)
            random.Random(args.seed + 7).shuffle(keys)
            budget = int(round(args.probe_frac * len(pool)))
            taken: list[str] = []
            for k in keys:
                if len(taken) >= budget:
                    break
                taken.extend(sorted(groups[k]))
            if not taken:
                continue
            probe_utts.extend(taken)
            probe_set = set(taken)
            train_utts = [u for u in train_utts if u not in probe_set]

    meta = {
        "mode": mode,
        "probe_frac": args.probe_frac,
        "probe": len(probe_utts),
        "probe_subsets": probe_subsets if args.probe_frac > 0 else [],
    }
    return train_utts, val_utts, test_utts, probe_utts, meta


def assert_no_text_leak(
    train_utts: list[str], held: dict[str, list[str]], norm_texts: dict[str, str]
) -> dict:
    """train 与任何留出集之间不允许出现同一句文本。

    按时长/CER 看不出来，但只要有一句 val 文本原样出现在 train，那个句子上的
    指标就是背诵结果。JSUT 各子集互不重文本（已实测），所以这是硬断言而不是警告。
    """
    train_texts = {norm_texts[u] for u in train_utts}
    report: dict[str, int] = {}
    for split, utts in held.items():
        inter = train_texts & {norm_texts[u] for u in utts}
        report[split] = len(inter)
        if inter:
            raise SystemExit(
                f"文本级泄漏：train 与 {split} 有 {len(inter)} 句相同文本，"
                f"例：{sorted(inter)[:3]}"
            )
    return report


def build_vocab(
    train_texts: list[str], train_morae: list[list[str]], args: argparse.Namespace
) -> tuple[Vocab, Vocab, dict]:
    """构建 train-only 词表。

    `--vocab-from` 指向旧 cache 时，符号顺序 = **旧顺序在前、新符号追加在后**。
    这是续训的前提：`train.py` 的 `pretrain_init` 只能对变长的分类头做
    「按行前缀拷贝」，只有旧词表是新词表的前缀时，逐行拷贝才对应同一个符号。
    按出现顺序重建词表（默认做法）不保证前缀关系，会静默把权重贴错类。
    """
    char_vocab = Vocab.from_corpus([list(t) for t in train_texts])
    mora_vocab = Vocab.from_corpus(train_morae)
    meta: dict = {"vocab_mode": "from-corpus"}
    if not args.vocab_from:
        return char_vocab, mora_vocab, meta

    base = Path(args.vocab_from)
    info = {"vocab_mode": "prefix-preserved", "vocab_from": str(base)}
    for name, built, fname in (
        ("char", char_vocab, "char_vocab.json"),
        ("mora", mora_vocab, "mora_vocab.json"),
    ):
        old = Vocab.load(str(base / fname))
        missing = [s for s in old.symbols if not built.has(s)]
        if missing:
            raise SystemExit(
                f"{base / fname} 的 {len(missing)} 个符号在新 train 里没出现"
                f"（例 {missing[:5]}）；前缀保持不成立，续训的分类头无法安全 warm start"
            )
        merged = list(old.symbols) + [s for s in built.symbols if not old.has(s)]
        rebuilt = Vocab(merged, blank=old.blank)
        if rebuilt.symbols[: len(old.symbols)] != old.symbols:
            raise SystemExit(f"{name} 词表前缀校验失败")
        info[f"{name}_vocab"] = {"old": len(old.symbols), "new": len(rebuilt.symbols)}
        if name == "char":
            char_vocab = rebuilt
        else:
            mora_vocab = rebuilt
    return char_vocab, mora_vocab, info


def _load_cached_feature(
    source_cache: Path | None,
    utt: str,
    n_mels: int,
    source_mean: np.ndarray | None,
    source_std: np.ndarray | None,
) -> np.ndarray | None:
    if source_cache is None:
        return None
    path = source_cache / f"{utt}.npy"
    if not path.exists():
        return None
    feat = np.load(path)
    if feat.ndim != 2 or feat.shape[1] != n_mels:
        return None
    if source_mean is not None and source_std is not None:
        feat = (feat - source_mean) / source_std
    return feat.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsut-root", type=str, default="data/jsut_ver1.1")
    ap.add_argument(
        "--subsets",
        type=str,
        default="basic5000",
        help="逗号分隔的子集名；`all` 表示语料根下全部子集",
    )
    ap.add_argument("--cache-dir", type=str, default="data/cache/basic5000")
    ap.add_argument("--reuse-cache-dir", type=str, default=None)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=80)
    ap.add_argument(
        "--holdout-from",
        type=str,
        default=None,
        help="复用该 cache 的 val/test（同一批句子），其余全部进 train",
    )
    ap.add_argument(
        "--vocab-from",
        type=str,
        default=None,
        help="旧 cache 目录；词表顺序保持为「旧符号在前，新符号追加」",
    )
    ap.add_argument(
        "--probe-frac",
        type=float,
        default=0.0,
        help="每个子集从 train 侧切出的域探针比例（按文本分组切，只作诊断）",
    )
    ap.add_argument(
        "--probe-subsets",
        type=str,
        default=None,
        help="逗号分隔；默认只从「非 holdout 来源」的子集切 probe",
    )
    ap.add_argument(
        "--source-cmvn-applied",
        action="store_true",
        help="显式确认对 --reuse-cache-dir 的源特征再次套用其 mean/std（源已烘焙时会双归一化）",
    )
    ap.add_argument(
        "--overwrite-cmvn",
        action="store_true",
        help="允许覆盖已烘焙 CMVN 的目标 cache（默认拒绝，避免二次归一化）",
    )
    ap.add_argument(
        "--noise-dir",
        type=str,
        default=None,
        help="MUSAN 等噪声根目录；只对 train 施加，val/test/probe 保持干净",
    )
    ap.add_argument("--rir-dir", type=str, default=None, help="RIR 根目录；只对 train 施加")
    ap.add_argument("--noise-prob", type=float, default=0.3, help="每条 train 音频加噪概率")
    ap.add_argument("--rir-prob", type=float, default=0.3, help="每条 train 音频卷积 RIR 概率")
    ap.add_argument(
        "--snr-db-range",
        type=str,
        default="5,20",
        help="加噪 SNR 区间（dB），在此区间内均匀采样",
    )
    ap.add_argument(
        "--augment-seed",
        type=int,
        default=1234,
        help="按 (seed,utt) 播种，保证同一 utt 的增强可复现",
    )
    args = ap.parse_args()

    if args.val_size < 0 or args.test_size < 0 or args.val_size + args.test_size < 1:
        raise ValueError("val-size and test-size must produce a non-empty split")
    if not 0.0 <= args.probe_frac < 1.0:
        raise ValueError("probe-frac must be in [0, 1)")
    if not 0.0 <= args.noise_prob <= 1.0 or not 0.0 <= args.rir_prob <= 1.0:
        raise ValueError("noise-prob / rir-prob 必须在 [0, 1]")
    snr_lo, snr_hi = (float(v) for v in args.snr_db_range.split(","))
    if snr_lo > snr_hi:
        raise ValueError(f"snr-db-range 需 low,high 且 low<=high，实际 {args.snr_db_range}")
    noise_files = augment.index_audio(args.noise_dir)
    rir_files = augment.index_audio(args.rir_dir)
    augment_on = bool(noise_files or rir_files)
    if args.noise_dir and not noise_files:
        raise SystemExit(f"--noise-dir {args.noise_dir} 下没有音频文件")
    if args.rir_dir and not rir_files:
        raise SystemExit(f"--rir-dir {args.rir_dir} 下没有音频文件")
    if augment_on:
        print(
            f"波形增强: noise={len(noise_files)} 文件(p={args.noise_prob}) "
            f"rir={len(rir_files)} 文件(p={args.rir_prob}) snr={args.snr_db_range}dB "
            f"→ 仅 train"
        )

    root = Path(args.jsut_root)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    available = [d.name for d in sorted(root.iterdir()) if (d.name in ALL_SUBSETS)]
    if args.subsets.strip() == "all":
        args.subsets_list = list(available)
    else:
        args.subsets_list = [s.strip() for s in args.subsets.split(",") if s.strip()]
    unknown = [s for s in args.subsets_list if s not in ALL_SUBSETS]
    if unknown:
        raise SystemExit(f"未知子集 {unknown}；已知子集：{list(ALL_SUBSETS)}")

    rows = collect_rows(root, args.subsets_list)
    wav_dirs = {sub: root / sub / "wav" for sub in args.subsets_list}
    print(f"subsets={args.subsets_list} utts={len(rows)}")

    # 目标 cache 若已烘焙旧 CMVN，重跑会对其特征二次归一化并污染 mean/std。
    # 判定只依赖目标目录本身，放在特征提取之前，避免白跑一遍。
    if any(cache.glob("*.npy")):
        target_baked = cache_cmvn_is_baked(cache, wav_dirs, args.n_mels, args.sample_rate, n_utt=3)
        if target_baked is True and not args.overwrite_cmvn:
            raise SystemExit(
                f"目标 cache {cache} 的特征已烘焙其现有 mean/std；重跑会二次归一化。"
                "请换一个 --cache-dir，或显式加 --overwrite-cmvn。"
            )

    source_cache = Path(args.reuse_cache_dir) if args.reuse_cache_dir else None
    source_mean = None
    source_std = None
    if source_cache is not None:
        mean_path = source_cache / "mean.npy"
        std_path = source_cache / "std.npy"
        if mean_path.exists() and std_path.exists():
            # 复用源特征前必须先判定源 cache 的 CMVN 是否已烘焙进 `<utt>.npy`：
            # 已烘焙时再套一次 `apply_cmvn` 会得到「双归一化」特征，随后
            # `compute_cmvn` 记录的是双归一化分布的统计量，与真正落盘的特征
            # 不对应——这会让在线推理（原始 fbank → CMVN）喂进偏移极大的输入。
            baked = cache_cmvn_is_baked(
                source_cache, wav_dirs, args.n_mels, args.sample_rate, n_utt=3
            )
            if baked is True and not args.source_cmvn_applied:
                raise SystemExit(
                    f"源 cache {source_cache} 的特征已烘焙其 mean/std；再归一化会产生"
                    "双归一化特征。确认要这么做请显式加 --source-cmvn-applied，"
                    "或用 --reuse-cache-dir 指向未归一化的源 cache。"
                )
            source_mean = np.load(mean_path)
            source_std = np.load(std_path)

    kks = pykakasi.kakasi()
    train_texts: list[str] = []
    char_seqs: list[list[str]] = []
    mora_seqs: list[list[str]] = []
    feat_train: list[np.ndarray] = []
    entries = []
    norm_texts: dict[str, str] = {}

    for sub, utt, text in tqdm(rows, desc="extract"):
        norm_text = normalize_text(text)
        kana = normalize_kana("".join(c["kana"] for c in kks.convert(text)))
        morae = kana_to_mora(kana)
        norm_texts[utt] = norm_text
        entries.append(
            {
                "utt": utt,
                "subset": sub,
                "wav": str(wav_dirs[sub] / f"{utt}.wav"),
                "text": norm_text,
                "kana": kana,
            }
        )

    train_utts, val_utts, test_utts, probe_utts, split_mode = build_split(rows, args, norm_texts)
    split_of = {u: "train" for u in train_utts}
    split_of.update({u: "val" for u in val_utts})
    split_of.update({u: "test" for u in test_utts})
    split_of.update({u: "probe" for u in probe_utts})
    leak_report = assert_no_text_leak(
        train_utts, {"val": val_utts, "test": test_utts, "probe": probe_utts}, norm_texts
    )

    for entry in entries:
        utt = entry["utt"]
        feat = _load_cached_feature(source_cache, utt, args.n_mels, source_mean, source_std)
        if feat is None:
            wav = load_wav(entry["wav"], args.sample_rate)
            feat = extract_fbank(wav, args.sample_rate, args.n_mels).numpy().astype(np.float32)
        np.save(cache / f"{utt}.npy", feat)
        entry["feat"] = str(cache / f"{utt}.npy")
        if split_of[utt] == "train":
            train_texts.append(entry["text"])
            char_seqs.append(list(entry["text"]))
            mora_seqs.append(kana_to_mora(entry["kana"]))
            feat_train.append(feat)

    mean, std = compute_cmvn(feat_train)
    np.save(cache / "mean.npy", mean)
    np.save(cache / "std.npy", std)

    char_vocab, mora_vocab, vocab_info = build_vocab(train_texts, mora_seqs, args)
    char_vocab.save(str(cache / "char_vocab.json"))
    mora_vocab.save(str(cache / "mora_vocab.json"))

    split_entries: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    if probe_utts:
        split_entries["probe"] = []
    oov_chars = 0
    oov_mora = 0
    n_aug = 0
    for entry in entries:
        utt = entry["utt"]
        char_ids = char_vocab.encode_with_unk(list(entry["text"]))
        mora_ids = mora_vocab.encode_with_unk(kana_to_mora(entry["kana"]))
        oov_chars += sum(t == char_vocab.unk_id for t in char_ids)
        oov_mora += sum(t == mora_vocab.unk_id for t in mora_ids)
        if augment_on and split_of[utt] == "train":
            # 波形级增强：从原始 wav 重新提特征，再用**干净 train** 的 mean/std 归一化。
            # CMVN 用干净统计量是有意的——val/test/probe 是干净音频，若用含噪统计量
            # 归一化，干净集会被平移，指标不可比。这条路径也不受 --reuse-cache-dir
            # 的双归一化影响（它直接读 wav，不读缓存特征）。
            wav = load_wav(entry["wav"], args.sample_rate)
            aug_wav = augment.augment_waveform(
                wav,
                args.sample_rate,
                noise_files,
                rir_files,
                augment.utt_rng(args.augment_seed, utt),
                noise_prob=args.noise_prob,
                rir_prob=args.rir_prob,
                snr_db_range=(snr_lo, snr_hi),
            )
            raw = extract_fbank(aug_wav, args.sample_rate, args.n_mels).numpy().astype(np.float32)
            np.save(cache / f"{utt}.npy", apply_cmvn(raw, mean, std))
            n_aug += 1
        else:
            np.save(
                cache / f"{utt}.npy",
                apply_cmvn(np.load(cache / f"{utt}.npy"), mean, std),
            )
        entry["char_ids"] = char_ids
        entry["mora_ids"] = mora_ids
        split_entries[split_of[utt]].append(entry)

    for split, split_data in split_entries.items():
        with open(cache / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump({"root": str(cache), "entries": split_data}, f, ensure_ascii=False)

    with open(cache / "train_transcript_utf8.txt", "w", encoding="utf-8") as f:
        for entry in split_entries["train"]:
            f.write(f"{entry['utt']}:{entry['text']}\n")

    subset_counts: dict[str, dict[str, int]] = {}
    for split, split_data in split_entries.items():
        for entry in split_data:
            subset_counts.setdefault(entry["subset"], {}).setdefault(split, 0)
            subset_counts[entry["subset"]][split] += 1

    split_meta = {
        "seed": args.seed,
        "subsets": args.subsets_list,
        "split_mode": split_mode["mode"],
        "probe_frac": split_mode["probe_frac"],
        "probe_subsets": split_mode["probe_subsets"],
        "train": len(split_entries["train"]),
        "val": len(split_entries["val"]),
        "test": len(split_entries["test"]),
        "probe": len(split_entries.get("probe", [])),
        "char_vocab": len(char_vocab),
        "mora_vocab": len(mora_vocab),
        "val_test_oov_chars": oov_chars,
        "val_test_oov_mora": oov_mora,
        "text_leak_with_train": leak_report,
        "augment": {
            "noise_dir": args.noise_dir,
            "rir_dir": args.rir_dir,
            "noise_prob": args.noise_prob,
            "rir_prob": args.rir_prob,
            "snr_db_range": [snr_lo, snr_hi],
            "seed": args.augment_seed,
            "n_train_augmented": n_aug,
            "cmvn_from": "clean-train",
        },
        "subset_counts": subset_counts,
        "sample_rate": args.sample_rate,
        "n_mels": args.n_mels,
        **vocab_info,
    }
    with open(cache / "split.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    print(
        f"train={len(split_entries['train'])} val={len(split_entries['val'])} "
        f"test={len(split_entries['test'])} probe={len(split_entries.get('probe', []))}"
    )
    print(f"char_vocab={len(char_vocab)} mora_vocab={len(mora_vocab)} ({vocab_info['vocab_mode']})")
    print(f"oov (val+test+probe): chars={oov_chars} mora={oov_mora}")
    print(f"text_leak_with_train={leak_report}")
    if augment_on:
        print(f"波形增强: train 中 {n_aug}/{len(split_entries['train'])} 条已加噪/混响（CMVN 用干净统计量）")
    for sub in args.subsets_list:
        c = subset_counts.get(sub, {})
        print(
            f"  {sub:18s} train={c.get('train', 0):5d} val={c.get('val', 0):3d} "
            f"test={c.get('test', 0):3d} probe={c.get('probe', 0):4d}"
        )
    print(f"cache={cache}")


if __name__ == "__main__":
    main()
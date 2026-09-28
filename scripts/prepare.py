from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pykakasi
from tqdm import tqdm

from otokoenet.data import (
    apply_cmvn,
    cache_cmvn_is_baked,
    compute_cmvn,
    extract_fbank,
    load_wav,
)
from otokoenet.text import Vocab, kana_to_mora, normalize_kana, normalize_text


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
    ap.add_argument("--cache-dir", type=str, default="data/cache/basic5000")
    ap.add_argument("--reuse-cache-dir", type=str, default=None)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=80)
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
    args = ap.parse_args()

    if args.val_size < 0 or args.test_size < 0 or args.val_size + args.test_size < 1:
        raise ValueError("val-size and test-size must produce a non-empty split")

    root = Path(args.jsut_root)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    transcript_path = root / "basic5000" / "transcript_utf8.txt"
    wav_dir = root / "basic5000" / "wav"

    # 目标 cache 若已烘焙旧 CMVN，重跑会对其特征二次归一化并污染 mean/std。
    # 判定只依赖目标目录本身，放在特征提取之前，避免白跑一遍。
    if any(cache.glob("*.npy")):
        target_baked = cache_cmvn_is_baked(cache, wav_dir, args.n_mels, args.sample_rate, n_utt=3)
        if target_baked is True and not args.overwrite_cmvn:
            raise SystemExit(
                f"目标 cache {cache} 的特征已烘焙其现有 mean/std；重跑会二次归一化。"
                "请换一个 --cache-dir，或显式加 --overwrite-cmvn。"
            )
    rows = parse_transcript(transcript_path)
    rng = random.Random(args.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    test_idx = set(indices[: args.test_size])
    val_idx = set(indices[args.test_size : args.test_size + args.val_size])
    train_idx = set(indices[args.test_size + args.val_size :])

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
                source_cache, wav_dir, args.n_mels, args.sample_rate, n_utt=3
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

    for rank, (utt, text) in enumerate(tqdm(rows, desc="extract")):
        norm_text = normalize_text(text)
        kana = normalize_kana("".join(c["kana"] for c in kks.convert(text)))
        morae = kana_to_mora(kana)
        feat = _load_cached_feature(source_cache, utt, args.n_mels, source_mean, source_std)
        if feat is None:
            wav_path = wav_dir / f"{utt}.wav"
            wav = load_wav(str(wav_path), args.sample_rate)
            feat = extract_fbank(wav, args.sample_rate, args.n_mels).numpy().astype(np.float32)
        feat_path = cache / f"{utt}.npy"
        np.save(feat_path, feat)
        entries.append(
            {
                "utt": utt,
                "wav": str(wav_dir / f"{utt}.wav"),
                "feat": str(feat_path),
                "text": norm_text,
                "kana": kana,
                "n_feat": int(feat.shape[0]),
            }
        )
        if rank in train_idx:
            train_texts.append(norm_text)
            char_seqs.append(list(norm_text))
            mora_seqs.append(morae)
            feat_train.append(feat)

    mean, std = compute_cmvn(feat_train)
    np.save(cache / "mean.npy", mean)
    np.save(cache / "std.npy", std)

    char_vocab = Vocab.from_corpus(char_seqs)
    mora_vocab = Vocab.from_corpus(mora_seqs)
    char_vocab.save(str(cache / "char_vocab.json"))
    mora_vocab.save(str(cache / "mora_vocab.json"))

    split_entries: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    oov_chars = 0
    oov_mora = 0
    for rank, entry in enumerate(entries):
        norm_text = entry["text"]
        kana = entry["kana"]
        char_ids = char_vocab.encode_with_unk(list(norm_text))
        mora_ids = mora_vocab.encode_with_unk(kana_to_mora(kana))
        oov_chars += sum(token == char_vocab.unk_id for token in char_ids)
        oov_mora += sum(token == mora_vocab.unk_id for token in mora_ids)
        feat = np.load(cache / f"{entry['utt']}.npy")
        np.save(cache / f"{entry['utt']}.npy", apply_cmvn(feat, mean, std))
        entry["char_ids"] = char_ids
        entry["mora_ids"] = mora_ids
        entry.pop("n_feat")
        if rank in test_idx:
            split_entries["test"].append(entry)
        elif rank in val_idx:
            split_entries["val"].append(entry)
        else:
            split_entries["train"].append(entry)

    for split, split_data in split_entries.items():
        with open(cache / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump({"root": str(cache), "entries": split_data}, f, ensure_ascii=False)

    with open(cache / "train_transcript_utf8.txt", "w", encoding="utf-8") as f:
        for entry in split_entries["train"]:
            f.write(f"{entry['utt']}:{entry['text']}\n")

    split_meta = {
        "seed": args.seed,
        "train": len(split_entries["train"]),
        "val": len(split_entries["val"]),
        "test": len(split_entries["test"]),
        "char_vocab": len(char_vocab),
        "mora_vocab": len(mora_vocab),
        "val_test_oov_chars": oov_chars,
        "val_test_oov_mora": oov_mora,
        "sample_rate": args.sample_rate,
        "n_mels": args.n_mels,
    }
    with open(cache / "split.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    print(
        f"train={len(split_entries['train'])} val={len(split_entries['val'])} "
        f"test={len(split_entries['test'])}"
    )
    print(f"char_vocab={len(char_vocab)} mora_vocab={len(mora_vocab)}")
    print(f"val/test oov: chars={oov_chars} mora={oov_mora}")
    print(f"cache={cache}")


if __name__ == "__main__":
    main()

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

from otokoenet.data import apply_cmvn, compute_cmvn, extract_fbank, load_wav
from otokoenet.text import Vocab, kana_to_mora, normalize_text


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsut-root", type=str, default="data/jsut_ver1.1")
    ap.add_argument("--cache-dir", type=str, default="data/cache/basic5000")
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=80)
    args = ap.parse_args()

    root = Path(args.jsut_root)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    transcript_path = root / "basic5000" / "transcript_utf8.txt"
    wav_dir = root / "basic5000" / "wav"
    rows = parse_transcript(transcript_path)
    rng = random.Random(args.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    test_idx = set(indices[: args.test_size])
    print(f"total={len(rows)} test={args.test_size}")

    kks = pykakasi.kakasi()
    mora_seqs = []
    char_seqs = []
    feat_train = []
    entries = []

    for rank, (utt, text) in enumerate(tqdm(rows, desc="extract")):
        norm_text = normalize_text(text)
        kana = "".join(c["kana"] for c in kks.convert(text))
        morae = kana_to_mora(kana)
        wav_path = wav_dir / f"{utt}.wav"
        wav = load_wav(str(wav_path), args.sample_rate)
        feat = extract_fbank(wav, args.sample_rate, args.n_mels).numpy()
        feat_path = cache / f"{utt}.npy"
        np.save(feat_path, feat)
        entries.append(
            {
                "utt": utt,
                "wav": str(wav_path),
                "feat": str(feat_path),
                "text": norm_text,
                "kana": kana,
                "n_feat": feat.shape[0],
                "is_test": rank in test_idx,
            }
        )
        char_seqs.append(list(norm_text))
        mora_seqs.append(morae)
        if rank not in test_idx:
            feat_train.append(feat)

    print("computing cmvn over train set...")
    mean, std = compute_cmvn(feat_train)
    np.save(cache / "mean.npy", mean)
    np.save(cache / "std.npy", std)

    char_vocab = Vocab.from_corpus(char_seqs)
    mora_vocab = Vocab.from_corpus(mora_seqs)
    char_vocab.save(str(cache / "char_vocab.json"))
    mora_vocab.save(str(cache / "mora_vocab.json"))
    print(f"char vocab={len(char_vocab)} mora vocab={len(mora_vocab)}")

    train_entries, test_entries = [], []
    for rank, entry in enumerate(entries):
        norm_text = entry["text"]
        kana = entry["kana"]
        entry["char_ids"] = char_vocab.encode(list(norm_text))
        entry["mora_ids"] = mora_vocab.encode(kana_to_mora(kana))
        feat = np.load(cache / f"{entry['utt']}.npy")
        feat = apply_cmvn(feat, mean, std)
        np.save(cache / f"{entry['utt']}.npy", feat)
        del entry["kana"], entry["text"], entry["n_feat"]
        if entry.pop("is_test"):
            test_entries.append(entry)
        else:
            train_entries.append(entry)

    train_manifest = {"root": str(cache), "entries": train_entries}
    test_manifest = {"root": str(cache), "entries": test_entries}
    with open(cache / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, ensure_ascii=False)
    with open(cache / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, ensure_ascii=False)
    print(f"train={len(train_entries)} test={len(test_entries)} -> {cache}")


if __name__ == "__main__":
    main()

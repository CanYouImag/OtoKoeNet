"""从 HuggingFace(HF-Mirror) 的 ja_asr.jsut_basic5000 parquet 重建 JSUT basic5000 语料布局。

产出与官网解压一致的目录：
  <jsut-root>/basic5000/wav/BASIC5000_####.wav
  <jsut-root>/basic5000/transcript_utf8.txt

用法：
  python scripts/prepare_hf.py \
    --parquet-dir data/hf_jsut \
    --jsut-root data/jsut_ver1.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=str, default="data/hf_jsut")
    ap.add_argument("--jsut-root", type=str, default="data/jsut_ver1.1")
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()

    parquet_dir = Path(args.parquet_dir)
    root = Path(args.jsut_root)
    wav_dir = root / "basic5000" / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    transcript_out = root / "basic5000" / "transcript_utf8.txt"

    with open(transcript_out, "w", encoding="utf-8") as f:
        for path in sorted(parquet_dir.glob("*.parquet")):
            table = pq.read_table(str(path), columns=["audio", "transcription"])
            audio_col = table.column("audio").to_pylist()
            text_col = table.column("transcription").to_pylist()
            for audio, text in tqdm(list(zip(audio_col, text_col)), desc=path.name):
                utt = Path(audio["path"]).stem
                (wav_dir / f"{utt}.wav").write_bytes(audio["bytes"])
                f.write(f"{utt}:{text}\n")

    n_wav = len(list(wav_dir.glob("*.wav")))
    print(f"wav={n_wav} -> {wav_dir}")

    n_lines = sum(1 for _ in open(transcript_out, encoding="utf-8"))
    print(f"transcript lines={n_lines} -> {transcript_out}")


if __name__ == "__main__":
    main()
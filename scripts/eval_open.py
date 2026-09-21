from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pykakasi
import torch

from otokoenet.data import Manifest, apply_cmvn, extract_fbank, load_wav
from otokoenet.decode import build_lexicon, ctc_collapse, edit_distance, nearest_top2
from otokoenet.kana2kanji import Kana2Kanji
from otokoenet.model import DualCTC
from otokoenet.text import Vocab, normalize_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, default="data/cache/basic5000")
    ap.add_argument("--ckpt", type=str, default="runs/basic5000/best.pt")
    ap.add_argument("--table", type=str, default="data/cache/basic5000/kana2kanji.json")
    ap.add_argument("--transcript", type=str, default="data/jsut_ver1.1/utparaphrase512/transcript_utf8.txt")
    ap.add_argument("--wav-dir", type=str, default="data/jsut_ver1.1/utparaphrase512/wav")
    ap.add_argument("--num-examples", type=int, default=6)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    char_vocab = Vocab.load(str(cache / "char_vocab.json"))
    mora_vocab = Vocab.load(str(cache / "mora_vocab.json"))
    mean = np.load(cache / "mean.npy")
    std = np.load(cache / "std.npy")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    model = DualCTC(
        in_dim=cfg["audio"]["n_mels"],
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        ffn_dim=mcfg["ffn_dim"],
        conv_kernel=mcfg["conv_kernel"],
        dropout=mcfg.get("dropout", 0.1),
        n_char=len(char_vocab),
        n_mora=len(mora_vocab),
    )
    state = ckpt.get("ema") or ckpt["model"]
    for name, dst in model.state_dict().items():
        if name not in state:
            continue
        src_t = state[name]
        if src_t.shape == dst.shape:
            dst.copy_(src_t)
        elif dst.dim() >= 2 and dst.shape[0] > src_t.shape[0] and dst.shape[1:] == src_t.shape[1:]:
            dst[: src_t.shape[0]].copy_(src_t)
    model.eval()

    test_manifest = Manifest.load(str(cache / "test.json"))
    lexicon = build_lexicon(Manifest.load(str(cache / "train.json")), test_manifest)
    converter = Kana2Kanji(Path(args.table))
    kks = pykakasi.kakasi()
    sample_rate = cfg["audio"]["sample_rate"]

    rows: list[tuple[str, str]] = []
    wav_dir = Path(args.wav_dir)
    for line in open(args.transcript, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        utt, text = line.split(":", 1)
        rows.append((utt, normalize_text(text)))

    def oracle_kana(text: str) -> str:
        return "".join(c["kana"] for c in kks.convert(text))

    @torch.no_grad()
    def asr_kana(wav_path: Path) -> str | None:
        if not wav_path.exists():
            return None
        wav = load_wav(str(wav_path), sample_rate)
        feat = extract_fbank(wav, sample_rate, cfg["audio"]["n_mels"]).numpy()
        feat = apply_cmvn(feat, mean, std)
        x = torch.from_numpy(feat).float().unsqueeze(0)
        feat_len = torch.tensor([feat.shape[0]], dtype=torch.long)
        char_logits, mora_logits, out_len = model(x, feat_len)
        mora_ids = ctc_collapse(mora_logits[0, : out_len[0]])
        char_ids, best_d, second_d = nearest_top2(mora_ids, lexicon)
        if second_d - best_d >= 1 and best_d <= max(4, len(mora_ids) // 4):
            return None  # 命中已知句（句子库路线）
        return "".join(mora_vocab.decode(mora_ids))

    def add(stat: dict, ref: str, hyp: str) -> None:
        stat["chars"] += len(ref)
        stat["err"] += edit_distance(list(ref), list(hyp))

    def ratio(stat: dict) -> str:
        c = stat["chars"]
        return f"{stat['err'] / c * 100:.2f}%" if c else "-"

    stat_oracle = {"chars": 0, "err": 0}
    stat_asr = {"chars": 0, "err": 0}
    n_gated = 0
    oracle_ex: list[tuple[str, str, str]] = []
    asr_ex: list[tuple[str, str, str]] = []

    for utt, ref in rows:
        kana = oracle_kana(ref)
        add(stat_oracle, ref, converter.convert(kana))
        if len(oracle_ex) < args.num_examples:
            oracle_ex.append((ref, kana, converter.convert(kana)))

        hyp_kana = asr_kana(wav_dir / f"{utt}.wav")
        if hyp_kana is None:
            n_gated += 1
            continue
        add(stat_asr, ref, converter.convert(hyp_kana))
        if len(asr_ex) < args.num_examples:
            asr_ex.append((ref, hyp_kana, converter.convert(hyp_kana)))

    print(f"oracle（完美假名 → 汉字转换）CER = {ratio(stat_oracle)}  chars={stat_oracle['chars']}")
    print(f"asr   （音频→mora→转换）   CER = {ratio(stat_asr)}   chars={stat_asr['chars']} gated={n_gated}")
    print("oracle examples:")
    for ref, kana, hyp in oracle_ex:
        print(f"  ref ={ref}\n  kana={kana}\n  hyp ={hyp}")
    print("asr examples:")
    for ref, kana, hyp in asr_ex:
        print(f"  ref ={ref}\n  kana={kana}\n  hyp ={hyp}")


if __name__ == "__main__":
    main()
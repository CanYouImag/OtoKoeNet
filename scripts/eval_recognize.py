from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from otokoenet.data import Manifest, apply_cmvn
from otokoenet.decode import NGramLM, build_lexicon, ctc_collapse, ctc_prefix_beam_search, edit_distance, nearest_top2
from otokoenet.kana2kanji import Kana2Kanji
from otokoenet.model import DualCTC
from otokoenet.text import Vocab


def build_lm(train_manifest: Manifest, mora_vocab: Vocab, order: int) -> NGramLM:
    seqs = [e["mora_ids"] for e in train_manifest.entries]
    return NGramLM(order=order).fit(seqs, vocab_size=len(mora_vocab))


def decode_mora(
    mora_logits: np.ndarray,
    *,
    decoder: str,
    beam_size: int,
    lm: NGramLM | None,
    lm_weight: float,
    length_penalty: float,
) -> list[int]:
    if decoder == "greedy":
        return ctc_collapse(mora_logits)
    seq, _ = ctc_prefix_beam_search(
        mora_logits,
        beam_size=beam_size,
        lm=lm if lm_weight > 0 else None,
        lm_weight=lm_weight,
        length_penalty=length_penalty,
    )[0]
    return seq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, default="data/cache/basic5000")
    ap.add_argument("--ckpt", type=str, default="runs/basic5000/best.pt")
    ap.add_argument("--table", type=str, default="data/cache/basic5000/kana2kanji.json")
    ap.add_argument("--decoder", type=str, choices=["greedy", "beam"], default="beam")
    # 默认值与 backend/app/config.py 的生产解码参数保持一致（阶段 5A/14 在 val 上
    # 选出来的点）。旧默认 beam=12 + lm_weight=1.0：lm 1.0 在 val 把 mora MER
    # 推到 12.66%，任何不带 --lm-weight 直接跑的验收数字都是坏的。
    ap.add_argument("--beam-size", type=int, default=24)
    ap.add_argument("--lm-order", type=int, default=4)
    ap.add_argument("--lm-weight", type=float, default=0.2)
    ap.add_argument("--length-penalty", type=float, default=0.0)
    ap.add_argument("--num-examples", type=int, default=8)
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

    train_manifest = Manifest.load(str(cache / "train.json"))
    test_manifest = Manifest.load(str(cache / "test.json"))
    lexicon = build_lexicon(train_manifest)
    converter = Kana2Kanji(Path(args.table))
    lm = build_lm(train_manifest, mora_vocab, args.lm_order)

    total_char = err = known_char = known_err = open_char = open_err = 0
    n_known = n_open = 0
    examples: list[tuple[str, str, int]] = []

    @torch.no_grad()
    def recognize(entry: dict) -> tuple[str, bool]:
        feat = np.load(entry["feat"])
        x = torch.from_numpy(feat).float().unsqueeze(0)
        feat_len = torch.tensor([feat.shape[0]], dtype=torch.long)
        char_logits, mora_logits, out_len = model(x, feat_len)
        mora_ids = decode_mora(
            mora_logits[0, : out_len[0]].detach().numpy(),
            decoder=args.decoder,
            beam_size=args.beam_size,
            lm=lm,
            lm_weight=args.lm_weight,
            length_penalty=args.length_penalty,
        )
        char_ids, best_d, second_d = nearest_top2(mora_ids, lexicon)
        if second_d - best_d >= 1 and best_d <= max(4, len(mora_ids) // 4):
            return "".join(char_vocab.decode(char_ids)), True
        kana = "".join(mora_vocab.decode(mora_ids))
        return converter.convert(kana), False

    for entry in test_manifest.entries:
        ref = "".join(char_vocab.decode(entry["char_ids"]))
        hyp, known = recognize(entry)
        d = edit_distance(list(ref), list(hyp))
        total_char += len(ref)
        err += d
        if known:
            known_char += len(ref)
            known_err += d
            n_known += 1
        else:
            open_char += len(ref)
            open_err += d
            n_open += 1
            if len(examples) < args.num_examples:
                examples.append((ref, hyp, d))

    def pct(e, c):
        return f"{e / c * 100:.2f}%" if c else "-"

    print(
        f"decoder={args.decoder} beam={args.beam_size} lm_order={args.lm_order} "
        f"lm_weight={args.lm_weight} length_penalty={args.length_penalty}"
    )
    print(f"total : n={n_known + n_open} CER={pct(err, total_char)}")
    print(f"known : n={n_known} CER={pct(known_err, known_char)}")
    print(f"open  : n={n_open} CER={pct(open_err, open_char)}")
    print("examples (open):")
    for ref, hyp, d in examples:
        print(f"  ref={ref}  hyp={hyp}  cer={d}")


if __name__ == "__main__":
    main()
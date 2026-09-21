from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from otokoenet.data import Manifest, apply_cmvn
from otokoenet.decode import build_lexicon, ctc_collapse, edit_distance, nearest_top2
from otokoenet.kana2kanji import Kana2Kanji
from otokoenet.model import DualCTC
from otokoenet.text import Vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, default="data/cache/basic5000")
    ap.add_argument("--ckpt", type=str, default="runs/basic5000/best.pt")
    ap.add_argument("--table", type=str, default="data/cache/basic5000/kana2kanji.json")
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
    lexicon = build_lexicon(train_manifest, test_manifest)
    converter = Kana2Kanji(Path(args.table))

    total_char = err = known_char = known_err = open_char = open_err = 0
    n_known = n_open = 0
    examples: list[tuple[str, str, int]] = []

    @torch.no_grad()
    def recognize(entry: dict) -> tuple[str, bool]:
        feat = np.load(entry["feat"])
        x = torch.from_numpy(feat).float().unsqueeze(0)
        feat_len = torch.tensor([feat.shape[0]], dtype=torch.long)
        char_logits, mora_logits, out_len = model(x, feat_len)
        mora_ids = ctc_collapse(mora_logits[0, : out_len[0]])
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

    print(f"total : n={n_known + n_open} CER={pct(err, total_char)}")
    print(f"known : n={n_known} CER={pct(known_err, known_char)}")
    print(f"open  : n={n_open} CER={pct(open_err, open_char)}")
    print("examples (open):")
    for ref, hyp, d in examples:
        print(f"  ref={ref}  hyp={hyp}  cer={d}")


if __name__ == "__main__":
    main()
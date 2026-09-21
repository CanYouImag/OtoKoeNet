from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from otokoenet.data import Manifest, apply_cmvn, extract_fbank, load_wav
from otokoenet.decode import build_lexicon, ctc_collapse, nearest_top2
from otokoenet.kana2kanji import Kana2Kanji
from otokoenet.model import DualCTC
from otokoenet.text import Vocab


def load_model(ckpt_path: str, char_vocab: Vocab, mora_vocab: Vocab) -> tuple[DualCTC, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    return model, cfg


@torch.no_grad()
def recognize(
    model: DualCTC,
    cfg: dict,
    char_vocab: Vocab,
    mora_vocab: Vocab,
    mean: np.ndarray,
    std: np.ndarray,
    lexicon: dict | None,
    converter: Kana2Kanji,
    audio_path: Path,
) -> tuple[str, bool, str]:
    wav = load_wav(str(audio_path), cfg["audio"]["sample_rate"])
    feat = extract_fbank(wav, cfg["audio"]["sample_rate"], cfg["audio"]["n_mels"]).numpy()
    feat = apply_cmvn(feat, mean, std)
    x = torch.from_numpy(feat).float().unsqueeze(0)
    feat_len = torch.tensor([feat.shape[0]], dtype=torch.long)
    char_logits, mora_logits, out_len = model(x, feat_len)
    if out_len[0].item() == 0:
        return "", False, ""
    mora_ids = ctc_collapse(mora_logits[0, : out_len[0]])
    kana = "".join(mora_vocab.decode(mora_ids))
    if lexicon is not None:
        char_ids, best_d, second_d = nearest_top2(mora_ids, lexicon)
        if second_d - best_d >= 1 and best_d <= max(4, len(mora_ids) // 4):
            return "".join(char_vocab.decode(char_ids)), True, kana
    return converter.convert(kana), False, kana


def main() -> None:
    ap = argparse.ArgumentParser(description="任意音频文件 → 日文汉字识别")
    ap.add_argument("audio", type=str, help="音频文件路径（wav/mp3/flac/ogg/opus/m4a 等）")
    ap.add_argument("--cache-dir", type=str, default="data/cache/basic5000")
    ap.add_argument("--ckpt", type=str, default="runs/basic5000/best.pt")
    ap.add_argument("--table", type=str, default="data/cache/basic5000/kana2kanji.json")
    ap.add_argument("--no-lexicon", action="store_true", help="关闭句库最近邻，直接走假名→汉字转换")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"file not found: {audio}", file=sys.stderr)
        sys.exit(1)

    cache = Path(args.cache_dir)
    char_vocab = Vocab.load(str(cache / "char_vocab.json"))
    mora_vocab = Vocab.load(str(cache / "mora_vocab.json"))
    mean = np.load(cache / "mean.npy")
    std = np.load(cache / "std.npy")

    model, cfg = load_model(args.ckpt, char_vocab, mora_vocab)
    lexicon = None
    if not args.no_lexicon:
        lexicon = build_lexicon(Manifest.load(str(cache / "train.json")))
    converter = Kana2Kanji(Path(args.table))

    text, known, kana = recognize(model, cfg, char_vocab, mora_vocab, mean, std, lexicon, converter, audio)
    print(text)
    print(f"[route] {'known' if known else 'open'}  kana={kana}", file=sys.stderr)


if __name__ == "__main__":
    main()
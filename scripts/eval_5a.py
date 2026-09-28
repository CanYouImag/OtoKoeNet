from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from otokoenet.data import Manifest
from otokoenet.decode import (
    NGramLM,
    build_lexicon,
    ctc_collapse,
    ctc_prefix_beam_search,
    edit_distance,
    nearest_top2,
)
from otokoenet.kana2kanji import Kana2Kanji
from otokoenet.model import DualCTC
from otokoenet.text import SOS, EOS, UNK, Vocab, kana_to_mora, normalize_kana, normalize_text


def load_model(ckpt_path: str, cache: Path) -> tuple[DualCTC, dict, Vocab, Vocab]:
    char_vocab = Vocab.load(str(cache / "char_vocab.json"))
    mora_vocab = Vocab.load(str(cache / "mora_vocab.json"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    model = DualCTC(
        in_dim=cfg["audio"]["n_mels"],
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        ffn_dim=m["ffn_dim"],
        conv_kernel=m["conv_kernel"],
        dropout=m.get("dropout", 0.1),
        n_char=len(char_vocab),
        n_mora=len(mora_vocab),
        ctc_gate=m.get("ctc_gate", False),
        gate_min_weight=m.get("gate_min_weight", 0.05),
    )
    state = ckpt.get("ema") or ckpt["model"]
    cur = model.state_dict()
    missing = [k for k in cur if k not in state and "num_batches_tracked" not in k]
    bad_shape = [k for k in cur if k in state and state[k].shape != cur[k].shape]
    if missing or bad_shape:
        raise RuntimeError(f"checkpoint 不兼容: missing={missing[:5]} bad_shape={bad_shape[:5]}")
    # EMA 影子权重按设计不含 num_batches_tracked（该 buffer 只在 momentum=None 时
    # 影响归一化），因此只能 strict=False；上面的 missing/bad_shape 断言负责
    # 保证真正的不兼容不会被静默放过。
    inc = model.load_state_dict({k: v for k, v in state.items() if k in cur}, strict=False)
    bad_missing = [k for k in inc.missing_keys if "num_batches_tracked" not in k]
    if bad_missing or inc.unexpected_keys:
        raise RuntimeError(
            f"checkpoint 不兼容: missing={bad_missing[:5]} unexpected={inc.unexpected_keys[:5]}"
        )
    model.eval()
    return model, cfg, char_vocab, mora_vocab


@torch.no_grad()
def forward_split(
    model: DualCTC, manifest: Manifest, batch_size: int, num_threads: int
) -> list[np.ndarray]:
    """对整个 split 做一次前向，返回逐句 mora 帧级 logits（T_out, n_mora）。

    v2 缓存的特征已在 prepare 阶段内置 CMVN（mean≈0/std≈1），此处不再重复归一化。
    """
    torch.set_num_threads(num_threads)
    order = sorted(range(len(manifest)), key=lambda i: manifest.entries[i]["feat"])
    out: dict[int, np.ndarray] = {}
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        feats = [np.load(manifest.entries[j]["feat"]) for j in idx]
        T = max(f.shape[0] for f in feats)
        x = torch.zeros(len(feats), T, feats[0].shape[1])
        for k, f in enumerate(feats):
            x[k, : f.shape[0]] = torch.from_numpy(f)
        feat_len = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        res = model(x, feat_len)
        mora_logits, out_len = res[1], res[2]
        for k, j in enumerate(idx):
            out[j] = mora_logits[k, : int(out_len[k])].detach().numpy().astype(np.float32)
    return [out[j] for j in range(len(manifest))]


def strip_special(ids: list[int], vocab: Vocab) -> list[int]:
    """去掉 CTC 输出里可能出现的未训练特殊符号（sos/eos）。"""
    bad = {vocab._sym2id[s] for s in (SOS, EOS) if s in vocab._sym2id}
    return [i for i in ids if i not in bad]


def decode_mora(
    logits: np.ndarray,
    mora_vocab: Vocab,
    *,
    decoder: str,
    beam_size: int,
    lm: NGramLM | None,
    lm_weight: float,
    length_penalty: float,
) -> list[int]:
    if decoder == "greedy":
        return strip_special(ctc_collapse(logits), mora_vocab)
    seq, _ = ctc_prefix_beam_search(
        logits,
        blank_id=mora_vocab.blank_id,
        beam_size=beam_size,
        lm=lm if lm_weight > 0 else None,
        lm_weight=lm_weight,
        length_penalty=length_penalty,
        max_len=int(logits.shape[0]),
    )[0]
    return strip_special(seq, mora_vocab)


class Stat:
    __slots__ = ("n", "chars", "err")

    def __init__(self) -> None:
        self.n = 0
        self.chars = 0
        self.err = 0

    def add(self, ref: str, hyp: str) -> int:
        d = edit_distance(list(ref), list(hyp))
        self.n += 1
        self.chars += len(ref)
        self.err += d
        return d

    def rate(self) -> float:
        return self.err / self.chars if self.chars else float("nan")

    def fmt(self) -> str:
        return f"{self.rate() * 100:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 5A 验收指标：kana CER / mora MER / kanji CER")
    ap.add_argument("--cache-dir", default="data/cache/basic5000_v2")
    ap.add_argument("--ckpt", default="runs/basic5000_stage05a/best.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--table", default="data/cache/basic5000_v2/kana2kanji.json")
    ap.add_argument("--lm-order", type=int, default=4)
    ap.add_argument("--beam-size", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-threads", type=int, default=12)
    ap.add_argument(
        "--decoder",
        default="greedy",
        help="greedy | beam（扫参时用 sweep）",
    )
    ap.add_argument("--lm-weight", type=float, default=0.0)
    ap.add_argument("--length-penalty", type=float, default=0.0)
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="在 split 上扫描 lm_weight / beam_size，按 mora MER 选最优（只在 val 上做）",
    )
    ap.add_argument("--num-examples", type=int, default=8)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    split_name = args.split
    model, cfg, char_vocab, mora_vocab = load_model(args.ckpt, cache)
    train_manifest = Manifest.load(str(cache / "train.json"))
    manifest = Manifest.load(str(cache / f"{split_name}.json"))

    print(f"ckpt       : {args.ckpt}")
    print(f"ckpt_step  : {torch.load(args.ckpt, map_location='cpu', weights_only=False).get('step')}")
    print(f"cache/split: {cache} / {split_name}  n={len(manifest)}")
    print(f"vocab      : n_char={len(char_vocab)} n_mora={len(mora_vocab)}")

    t0 = time.time()
    logits = forward_split(model, manifest, args.batch_size, args.num_threads)
    print(f"forward    : {time.time() - t0:.1f}s (cmvn baked in cache)")

    # 参考文本：kana 用 normalize_kana(manifest) 重新切 mora 再拼接，保证与假设同一分段口径
    refs = []
    for e in manifest.entries:
        kana = normalize_kana(e["kana"])
        refs.append(
            {
                "utt": e["utt"],
                "text": normalize_text(e["text"]),
                "kana": "".join(kana_to_mora(kana)),
                "mora_ids": e["mora_ids"],
            }
        )

    configs: list[dict]
    if args.sweep:
        configs = [
            {"decoder": "greedy", "beam_size": 0, "lm_weight": 0.0, "length_penalty": 0.0},
            *[
                {"decoder": "beam", "beam_size": b, "lm_weight": w, "length_penalty": lp}
                for b in (args.beam_size, 24)
                for w in (0.0, 0.05, 0.2, 0.5, 1.0)
                for lp in (0.0,)
            ],
        ]
    else:
        configs = [
            {
                "decoder": args.decoder,
                "beam_size": args.beam_size,
                "lm_weight": args.lm_weight,
                "length_penalty": args.length_penalty,
            }
        ]

    # 句库最近邻（只从 train 构建）
    lexicon = build_lexicon(train_manifest)
    lm = NGramLM(order=args.lm_order).fit(
        [e["mora_ids"] for e in train_manifest.entries], vocab_size=len(mora_vocab)
    )
    table = Path(args.table)
    converter = Kana2Kanji(table) if table.exists() else None
    if converter is None:
        print(f"[!] kana2kanji 表不存在 ({table})：kana→汉字路线与 oracle G2P CER 不可用")

    results = []
    for cfgd in configs:
        st_mer, st_kana = Stat(), Stat()
        st_lex, st_conv = Stat(), Stat()
        n_known = 0
        hyps = []
        for i, lg in enumerate(logits):
            hyp_ids = decode_mora(
                lg,
                mora_vocab,
                decoder=cfgd["decoder"],
                beam_size=cfgd["beam_size"] or args.beam_size,
                lm=lm,
                lm_weight=cfgd["lm_weight"],
                length_penalty=cfgd["length_penalty"],
            )
            hyp_kana = "".join(mora_vocab.decode(hyp_ids))
            st_mer.add("".join(str(t) for t in refs[i]["mora_ids"]), "".join(str(t) for t in hyp_ids))
            st_kana.add(refs[i]["kana"], hyp_kana)
            char_ids, best_d, second_d = nearest_top2(hyp_ids, lexicon)
            is_known = second_d - best_d >= 1 and best_d <= max(4, max(1, len(hyp_ids) // 4))
            n_known += int(is_known)
            hyps.append(
                {
                    "hyp_kana": hyp_kana,
                    "hyp_lexicon": "".join(char_vocab.decode(char_ids)),
                    "best_d": best_d,
                    "second_d": second_d,
                    "is_known": is_known,
                    "hyp_conv": converter.convert(hyp_kana) if converter else None,
                }
            )
            st_lex.add(refs[i]["text"], hyps[-1]["hyp_lexicon"])
            if converter:
                st_conv.add(refs[i]["text"], hyps[-1]["hyp_conv"])
        results.append(
            {
                **cfgd,
                "mora_mer": st_mer.rate(),
                "kana_cer": st_kana.rate(),
                "kanji_cer_lexicon": st_lex.rate(),
                "kanji_cer_converter": st_conv.rate() if converter else None,
                "known_rate": n_known / len(manifest),
            }
        )
        r = results[-1]
        print(
            f"[{split_name}] {cfgd['decoder']:6s} beam={cfgd['beam_size']:<3d} "
            f"lm_w={cfgd['lm_weight']:<5g} lp={cfgd['length_penalty']:<4g} | "
            f"mora MER {st_mer.fmt():>7s} | kana CER {st_kana.fmt():>7s} | "
            f"kanji CER(lexicon) {st_lex.fmt():>7s}"
            + (f" | kanji CER(conv) {st_conv.fmt():>7s}" if converter else "")
            + f" | known {r['known_rate'] * 100:.1f}%"
        )

    # char 头 greedy 假名外汉字输出（与解码配置无关，算一次）
    @torch.no_grad()
    def char_greedy() -> Stat:
        st = Stat()
        torch.set_num_threads(args.num_threads)
        order = sorted(range(len(manifest)), key=lambda i: manifest.entries[i]["feat"])
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            feats = [np.load(manifest.entries[j]["feat"]) for j in idx]
            T = max(f.shape[0] for f in feats)
            x = torch.zeros(len(feats), T, feats[0].shape[1])
            for k, f in enumerate(feats):
                x[k, : f.shape[0]] = torch.from_numpy(f)
            feat_len = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
            char_logits, _, out_len = model(x, feat_len)[:3]
            for k, j in enumerate(idx):
                ids = strip_special(ctc_collapse(char_logits[k, : int(out_len[k])].numpy()), char_vocab)
                st.add(refs[j]["text"], "".join(char_vocab.decode(ids)))
        return st

    st_char = char_greedy()
    print(f"[{split_name}] char_head greedy  kanji CER {st_char.fmt()}")

    if converter:
        st_oracle = Stat()
        for r in refs:
            st_oracle.add(r["text"], converter.convert(r["kana"]))
        print(
            f"[{split_name}] oracle G2P (perfect kana -> kanji) CER {st_oracle.fmt()}"
            f"  chars={st_oracle.chars}"
        )
    else:
        st_oracle = None

    if args.sweep:
        best = min(results, key=lambda r: r["mora_mer"])
        print(f"[{split_name}] BEST by mora MER: {json.dumps(best, ensure_ascii=False)}")

    print(f"--- {split_name} examples (worse first) ---")
    worse = sorted(
        range(len(refs)),
        key=lambda i: -edit_distance(list(refs[i]["kana"]), list(hyps[i]["hyp_kana"])),
    )
    for i in worse[: args.num_examples]:
        h = hyps[i]
        print(f"  ref_text={refs[i]['text']}")
        print(f"  ref_kana={refs[i]['kana']}")
        print(f"  hyp_kana={h['hyp_kana']}  d_lex={h['best_d']}/{h['second_d']} known={h['is_known']}")
        print(f"  hyp_lex ={h['hyp_lexicon']}")
        if h["hyp_conv"] is not None:
            print(f"  hyp_conv={h['hyp_conv']}")

    if args.json_out:
        payload = {
            "ckpt": args.ckpt,
            "cache_dir": args.cache_dir,
            "split": split_name,
            "n": len(manifest),
            "results": results,
            "char_head_greedy_kanji_cer": st_char.rate(),
            "oracle_g2p_cer": st_oracle.rate() if st_oracle else None,
            "open_set_cer": None,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

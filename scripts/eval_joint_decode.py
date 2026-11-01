"""阶段 19：Char-Mora 序列级联合解码评测（零训练，只动解码）。

对照三条路线，汉字 CER 都算在同一个 manifest 上：

  A. converter      ：mora 头 beam → kana → kana2kanji 转换表（当前生产路径）
  B. char greedy    ：char 头 argmax 折叠（当前生产完全没用）
  C. joint          ：A 与 B 的候选 + char 头 beam，统一按
                      `char_ll + β·mora_ll + γ·lm` 重新排序（见 otokoenet/joint_decode.py）

β/γ 只在 **val** 上扫，test/probe 用选定值，防止像 stage14 那样「两个口径结论相反」时
自欺。

    python scripts/eval_joint_decode.py \
        --ckpt runs/jsut_full_stage14/best.pt --cache-dir data/cache/jsut_full_v1 \
        --splits val test probe --out log/stage19_joint_decode.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pykakasi  # noqa: E402
import torch  # noqa: E402

from eval_5a import load_model  # noqa: E402
from otokoenet.data import Manifest  # noqa: E402
from otokoenet.decode import (  # noqa: E402
    NGramLM,
    ctc_collapse,
    ctc_prefix_beam_search,
    edit_distance,
)
from otokoenet.joint_decode import (  # noqa: E402
    NEG,
    char_ids as to_char_ids,
    ctc_log_likelihood,
    text_to_mora_ids,
)
from otokoenet.kana2kanji import Kana2Kanji  # noqa: E402
from otokoenet.text import normalize_text, SOS, EOS  # noqa: E402

BETAS = [0.0, 0.3, 0.5, 1.0]
GAMMAS = [0.0, 0.2, 0.5]


def strip_special(ids, vocab):
    bad = {vocab._sym2id[s] for s in (SOS, EOS) if s in vocab._sym2id}
    return [i for i in ids if i not in bad]


@torch.no_grad()
def forward_both(model, manifest, batch_size, num_threads, cache):
    """返回逐句 (char_logits, mora_logits)（已裁到 out_len）。"""
    torch.set_num_threads(num_threads)
    order = sorted(range(len(manifest)), key=lambda i: manifest.entries[i]["feat"])
    char_out: dict[int, np.ndarray] = {}
    mora_out: dict[int, np.ndarray] = {}
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        feats = [np.load(manifest.entries[j]["feat"]) for j in idx]
        T = max(f.shape[0] for f in feats)
        x = torch.zeros(len(feats), T, feats[0].shape[1])
        for k, f in enumerate(feats):
            x[k, : f.shape[0]] = torch.from_numpy(f)
        feat_len = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        res = model(x, feat_len)
        char_logits, mora_logits, out_len = res[0], res[1], res[2]
        for k, j in enumerate(idx):
            n = int(out_len[k])
            char_out[j] = char_logits[k, :n].numpy().astype(np.float32)
            mora_out[j] = mora_logits[k, :n].numpy().astype(np.float32)
    return [char_out[j] for j in range(len(manifest))], [mora_out[j] for j in range(len(manifest))]


def lm_score_of(ids, lm) -> float:
    if lm is None:
        return 0.0
    s, ctx = 0.0, []
    for tok in ids:
        s += lm.conditional_logp(ctx, tok)
        ctx.append(tok)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/jsut_full_stage14/best.pt")
    ap.add_argument("--cache-dir", default="data/cache/jsut_full_v1")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "probe"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--beam-size", type=int, default=24)
    ap.add_argument("--lm-weight", type=float, default=0.2)
    ap.add_argument("--n-best", type=int, default=8)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--out", default="log/stage19_joint_decode.json")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    model, cfg, char_vocab, mora_vocab = load_model(args.ckpt, cache)
    train_manifest = Manifest.load(str(cache / "train.json"))
    lm = NGramLM(order=4).fit(
        [e["mora_ids"] for e in train_manifest.entries], vocab_size=len(mora_vocab)
    )
    table = cache / "kana2kanji.json"
    converter = Kana2Kanji(table) if table.exists() else None
    kks = pykakasi.kakasi()

    report: dict = {
        "ckpt": args.ckpt,
        "cache_dir": args.cache_dir,
        "beam_size": args.beam_size,
        "lm_weight": args.lm_weight,
        "n_best": args.n_best,
        "betas": BETAS,
        "gammas": GAMMAS,
        "splits": {},
    }

    for split in args.splits:
        manifest = Manifest.load(str(cache / f"{split}.json"))
        entries = list(manifest.entries[: args.limit])
        refs = [normalize_text(e["text"]) for e in entries]
        print(f"\n=== {split} n={len(entries)} ===")
        t0 = time.time()
        char_logits, mora_logits = forward_both(
            model, manifest, 8, args.num_threads, cache
        )
        print(f"forward {time.time() - t0:.1f}s")

        st_conv = st_charg = 0
        st_joint = {(b, g): 0 for b in BETAS for g in GAMMAS}
        n_chars = 0
        for i, e in enumerate(entries):
            ref = refs[i]
            n_chars += len(ref)
            cl, ml = char_logits[i], mora_logits[i]

            # A. converter 路线（mora beam 的 top-1）
            mora_top, _ = ctc_prefix_beam_search(
                ml, beam_size=args.beam_size, lm=lm, lm_weight=args.lm_weight
            )[0]
            kana = "".join(mora_vocab.decode(strip_special(list(mora_top), mora_vocab)))
            conv_text = converter.convert(kana) if converter else kana
            st_conv += edit_distance(list(ref), list(conv_text))

            # B. char greedy
            cg_ids = strip_special(ctc_collapse(cl), char_vocab)
            st_charg += edit_distance(list(ref), list("".join(char_vocab.decode(cg_ids))))

            # 候选池
            mora_cands = ctc_prefix_beam_search(
                ml, beam_size=args.beam_size, lm=lm, lm_weight=args.lm_weight,
                n_best=args.n_best,
            )
            char_cands = ctc_prefix_beam_search(
                cl, beam_size=max(8, args.beam_size // 3), n_best=args.n_best
            )
            cand: dict[str, tuple[float, float, float]] = {}

            def add_text(text: str) -> None:
                text = normalize_text(text)
                if not text:
                    return
                cids = to_char_ids(text, char_vocab)
                mid = text_to_mora_ids(text, mora_vocab, kks)
                char_ll = ctc_log_likelihood(cl, cids) if cids else NEG
                mora_ll = ctc_log_likelihood(ml, mid) if mid else NEG
                lms = lm_score_of(mid, lm) if mid else NEG
                old = cand.get(text)
                if old is None or char_ll + mora_ll > old[0] + old[1]:
                    cand[text] = (char_ll, mora_ll, lms)

            for ids, _ in mora_cands:
                ka = "".join(mora_vocab.decode(strip_special(list(ids), mora_vocab)))
                add_text(converter.convert(ka) if converter else ka)
            for ids, _ in char_cands:
                add_text("".join(char_vocab.decode(strip_special(list(ids), char_vocab))))

            for b in BETAS:
                for g in GAMMAS:
                    best_t, best_s = None, NEG
                    for t, (cll, mll, lms) in cand.items():
                        s = cll + b * mll + g * lms
                        if s > best_s:
                            best_s, best_t = s, t
                    st_joint[(b, g)] += edit_distance(list(ref), list(best_t or ""))

        row = {
            "n": len(entries),
            "chars": n_chars,
            "converter_kanji_cer": st_conv / n_chars,
            "char_greedy_kanji_cer": st_charg / n_chars,
            "joint_kanji_cer": {
                f"b{b}_g{g}": st_joint[(b, g)] / n_chars for b in BETAS for g in GAMMAS
            },
        }
        report["splits"][split] = row
        print(
            f"  汉字 CER: converter {row['converter_kanji_cer']*100:.2f}% | "
            f"char greedy {row['char_greedy_kanji_cer']*100:.2f}%"
        )
        best = min(row["joint_kanji_cer"].items(), key=lambda kv: kv[1])
        print(f"  joint best {best[0]} {best[1]*100:.2f}%")

    if "val" in report["splits"]:
        best_cfg = min(
            report["splits"]["val"]["joint_kanji_cer"].items(), key=lambda kv: kv[1]
        )
        report["selected_on_val"] = {"cfg": best_cfg[0], "cer": best_cfg[1]}
        print(f"\nval 选出的 joint 配置: {best_cfg[0]} ({best_cfg[1]*100:.2f}%)")
        print("各 split 该配置下:")
        for s in args.splits:
            r = report["splits"][s]
            j = r["joint_kanji_cer"][best_cfg[0]]
            print(
                f"  [{s}] converter {r['converter_kanji_cer']*100:.2f}% | "
                f"char greedy {r['char_greedy_kanji_cer']*100:.2f}% | "
                f"joint {j*100:.2f}%"
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""長音规则后处理的标定与 A/B（方案四.3，零训练）。

`otokoenet.align.suspect_long_vowels` 的默认规则（后验相对时长 < 0.8 且分数低于
句内中位数）是拍出来的，`scripts/eval_engine_e2e.py` 在 val 上实测 FPR≈30%，
太高，会把读对的長音大量标红。本脚本把每个「ー」实例的信号与标签导出，扫阈值，
在**目标误报率**下选工作点，而不是用默认值。

标签口径（必须说清局限）：
  正类 = 该「ー」在**识别结果**里被替换/漏掉（`edit_ops`）。
  JSUT 是母语者正确朗读，所以：
    - 「标红」对**评估**而言是误报（用户其实读对了），用 FPR 衡量；
    - 正类标签量的是「模型把它听成了别的音」，是「模型不确定性」的代理，不是
      「学习者漏读」的标签。真正确认长音漏读需要学习者录音（与 Pearson 一样阻塞）。
    因此本脚本的结论是「规则在正确录音上的误报率可压到多少」，以及「它是否跟随
    识别错误」，不是「规则能抓多少学习者发音错误」。

    python scripts/eval_long_vowel.py --out log/stage17_long_vowel.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.config import Settings  # noqa: E402
from app.ml.engine import Engine  # noqa: E402
from otokoenet.align import AlignmentError, align_and_score  # noqa: E402
from otokoenet.data import Manifest  # noqa: E402
from scripts.eval_engine_e2e import edit_ops  # noqa: E402


def collect(engine: Engine, split: str, limit: int) -> list[dict]:
    manifest = Manifest.load(str(Path(engine.settings.cache_dir) / f"{split}.json"))
    rows: list[dict] = []
    n_utt = 0
    for e in list(manifest.entries)[:limit]:
        feat = np.load(e["feat"])
        target = np.asarray(e["mora_ids"], dtype=np.int64)
        logits = engine._encode(feat)[1]
        lp = torch.log_softmax(logits, dim=-1).numpy()
        try:
            # -99：规则分析要看每个实例，不能被整句阈值先拒掉
            aln = align_and_score(lp, target, frame_ms=engine.frame_ms, min_mean_logprob=-99.0)
        except AlignmentError:
            continue
        n_utt += 1
        ref_syms = [engine.mora_vocab.decode([target[j]])[0] for j in range(len(target))]
        hyp_mora = list(engine.mora_vocab.decode(engine.decode_mora(logits)))
        err_idx = {
            ri
            for op, ri, _ in edit_ops(ref_syms, hyp_mora)
            if op in ("sub", "del") and ref_syms[ri] == "ー"
        }
        for i, sym in enumerate(ref_syms):
            if sym != "ー":
                continue
            rows.append(
                {
                    "utt": e["utt"],
                    "rel": float(aln.post_rel_durations[i]),
                    "vrel": float(aln.rel_durations[i]),
                    "score": float(aln.scores[i]),
                    "gop": float(aln.gops[i]) if aln.gops else 0.0,
                    "sent_med_score": float(np.median(aln.scores)),
                    "sent_med_rel": float(np.median(aln.post_rel_durations)),
                    "is_err": int(i in err_idx),
                }
            )
    return rows, n_utt


def sweep(rows: list[dict], target_fp: float) -> dict:
    rel = np.array([r["rel"] for r in rows])
    score = np.array([r["score"] for r in rows])
    med_s = np.array([r["sent_med_score"] for r in rows])
    y = np.array([r["is_err"] for r in rows], dtype=int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    out: dict = {
        "n": len(rows),
        "n_pos_model_error": n_pos,
        "n_neg": n_neg,
        "base_rate": n_pos / len(rows) if rows else None,
        "grid": [],
    }
    best = None
    for rt in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for sk in (0.3, 0.5, 0.7, 0.9, 1.0):
            flag = (rel < rt) & (score < sk * med_s)
            fp = int(((flag == 1) & (y == 0)).sum())
            tp = int(((flag == 1) & (y == 1)).sum())
            fpr = fp / n_neg if n_neg else None
            rec = tp / n_pos if n_pos else None
            prec = tp / max(1, int(flag.sum())) if flag.sum() else None
            row = {
                "rel_threshold": rt,
                "score_k": sk,
                "flag_rate": float(flag.mean()),
                "fpr_on_correct": fpr,
                "recall_model_error": rec,
                "precision": prec,
            }
            out["grid"].append(row)
            if fpr is not None and fpr <= target_fp:
                rec_v = rec if rec is not None else -1.0
                if best is None or (rec_v, -fpr) > (
                    best["recall_model_error"] if best["recall_model_error"] is not None else -1.0,
                    -best["fpr_on_correct"],
                ):
                    best = row
    out["best_at_target_fpr"] = best
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/jsut_full_stage14/best.pt")
    ap.add_argument("--cache-dir", default="data/cache/jsut_full_v1")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "probe"])
    ap.add_argument("--limit", type=int, default=270)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--target-fp", type=float, default=0.05)
    ap.add_argument("--out", default="log/stage17_long_vowel.json")
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    s = Settings()
    s.ckpt_path = Path(args.ckpt).resolve()
    s.cache_dir = Path(args.cache_dir).resolve()
    engine = Engine(s)

    report: dict = {"target_fp": args.target_fp, "splits": {}}
    for split in args.splits:
        rows, n_utt = collect(engine, split, args.limit)
        res = sweep(rows, args.target_fp)
        res["n_utt"] = n_utt
        res["rows"] = rows
        report["splits"][split] = res
        b = res["best_at_target_fpr"]
        print(
            f"=== {split}: n_utt={n_utt} 長音实例={res['n']}（模型听错 {res['n_pos_model_error']}，"
            f"基率 {res['base_rate'] * 100:.1f}%）==="
        )
        if b:
            print(
                f"  目标 FPR<={args.target_fp * 100:.0f}% 的工作点: rel<{b['rel_threshold']} "
                f"且 score<{b['score_k']}×句内中位数 → 标红 {b['flag_rate'] * 100:.1f}%，"
                f"FPR {b['fpr_on_correct'] * 100:.1f}%，"
                f"跟随识别错误召回 {(b['recall_model_error'] or 0) * 100:.1f}%"
            )
        else:
            print("  在目标 FPR 下没有可用工作点（说明信号区分度不足）")
        # 默认规则的对照
        d = next(
            (r for r in res["grid"] if r["rel_threshold"] == 0.8 and r["score_k"] == 1.0), None
        )
        if d:
            print(
                f"  默认规则(rel<0.8, score<中位数): 标红 {d['flag_rate'] * 100:.1f}%，"
                f"FPR {(d['fpr_on_correct'] or 0) * 100:.1f}%，"
                f"召回 {(d['recall_model_error'] or 0) * 100:.1f}%"
            )

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

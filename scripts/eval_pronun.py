"""阶段 10：发音评估基线。

对 validation（绝不用 test 调参）逐句做 mora 级 CTC 强制对齐，产出：

1. 覆盖率与拒绝原因分布（不可行对齐不再静默变成 0 分）
2. 逐句分数、平均 log 概率、时长、每 mora 帧数
3. mora 级分数分布，按 促音 / 拗音 / 長音 / 普通 分组
4. 易混淆 mora 清单（低分率 × 出现次数）
5. 负对照：配错文本 / 加噪 / 截断，验证这个分数真的有区分度
6. 若给了人工评分文件，额外算 Pearson / Spearman / MAE / ECE

用法：
    python scripts/eval_pronun.py --limit 200 --out log/stage10_pronun_metrics.json
    python scripts/eval_pronun.py --human-scores data/human_scores.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from otokoenet.align import AlignmentError, align_and_score
from otokoenet.data import Manifest, cache_cmvn_is_baked
from otokoenet.text import Vocab
from scripts.eval_5a import load_model

FRAME_MS = 40.0  # fbank 10ms 经 4x 下采样
MORA_SPECIAL = ("っ", "ー")


def mora_group(sym: str) -> str:
    """分组：促音 / 拗音 / 長音 / 普通音節，四类互斥。

    促音（っ）与長音（ー）分开统计：两者的时长与发音难度不同（長音要持续一个
    完整音节，促音是闭音节爆破），混在一组会互相抵消掉各自的特征。
    """
    if sym == "っ":
        return "促音"
    if sym == "ー":
        return "長音"
    if sym and len(sym) >= 2 and sym[-1] in "ゃゅょャュョ":
        return "拗音"
    return "普通音節"


@torch.no_grad()
def log_probs_for(model, feat: np.ndarray) -> np.ndarray:
    """缓存特征已内置 CMVN（v2 cache 在 prepare 阶段烘焙），此处不得再归一化。

    重复应用 `mean.npy`/`std.npy` 会把特征二次归一化，整句平均 log 概率会从
    -0.21 掉到 -10.7，症状是「所有句子都被 low_score 拒绝」，很容易被误判成
    对齐阈值太严。start() 里用 cache_cmvn_is_baked() 断言这一点。
    """
    x = torch.from_numpy(feat).float().unsqueeze(0)
    _, mora_logits = model(x, torch.tensor([feat.shape[0]]))[:2]
    return torch.log_softmax(mora_logits[0], dim=-1).numpy()


def cohen_d(a: list[float], b: list[float]) -> float:
    """a 相对 b 的效应量；负值表示 a 更低。"""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt((va + vb) / 2)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    a, b = np.asarray(x, float), np.asarray(y, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra.tolist(), rb.tolist())


def ece(conf: list[float], correct: list[int], n_bins: int = 10) -> float:
    """期望校准误差。`correct` 为 0/1 标签；无标签时调用方不应调用本函数。"""
    c = np.asarray(conf, float)
    k = np.asarray(correct, float)
    if len(c) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (c > lo) & (c <= hi) if i else (c >= lo) & (c <= hi)
        if sel.sum() == 0:
            continue
        total += sel.mean() * abs(c[sel].mean() - k[sel].mean())
    return float(total)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    a = np.asarray(values, float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "p05": round(float(np.percentile(a, 5)), 4),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/basic5000_stage05a/best.pt")
    ap.add_argument("--cache-dir", default="data/cache/basic5000_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--min-mean-logprob", type=float, default=-3.0,
                    help="整句平均 log 概率下界；-3.0 来自 val 实测间隔（真实最差 -1.84 / 配错最好 -5.91）")
    ap.add_argument("--human-scores", default="", help="CSV: utt,human_score (0~100)")
    ap.add_argument("--out", default="log/stage10_pronun_metrics.json")
    ap.add_argument("--skip-negatives", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    cache = Path(args.cache_dir)
    model, _, _, mora_vocab = load_model(args.ckpt, cache)
    manifest = Manifest.load(str(cache / f"{args.split}.json"))
    n_use = min(args.limit, len(manifest))
    entries = list(manifest.entries[:n_use])

    # 缓存特征必须已烘焙 CMVN，否则上面的 log_probs_for 会二次归一化
    wav_dir = Path(entries[0]["wav"]).parent if entries and entries[0].get("wav") else Path(manifest.root)
    baked = cache_cmvn_is_baked(cache, wav_dir, 80, 16000, n_utt=3)
    if baked is False:
        raise SystemExit(
            f"{cache} 的特征没有烘焙它自己的 mean/std，不能直接喂给模型；"
            "请先跑 scripts/check_cmvn.py 或用 --source-cmvn-applied 重新 prepare。"
        )
    print(f"cache CMVN 已烘焙: {baked}")

    report: dict = {
        "ckpt": args.ckpt,
        "cache_dir": args.cache_dir,
        "split": args.split,
        "n_utt": n_use,
        "frame_ms": FRAME_MS,
        "min_mean_logprob": args.min_mean_logprob,
    }

    ok: list[dict] = []
    rejected: Counter = Counter()
    reject_examples: dict[str, str] = {}
    for e in entries:
        feat = np.load(e["feat"])
        target = np.asarray(e["mora_ids"], dtype=np.int64)
        lp = log_probs_for(model, feat)
        try:
            aln = align_and_score(
                lp, target, frame_ms=FRAME_MS, min_mean_logprob=args.min_mean_logprob
            )
        except AlignmentError as ex:
            rejected[ex.reason] += 1
            reject_examples.setdefault(ex.reason, f"{e['utt']}: {ex.detail}")
            continue
        syms = [mora_vocab.decode([i])[0] for i in target]
        ok.append(
            {
                "utt": e["utt"],
                "kana": e.get("kana", ""),
                "n_mora": aln.n_target,
                "n_frames": aln.n_frames,
                "total": aln.total,
                "mean_logprob": aln.mean_logprob,
                "blank_ratio": aln.blank_ratio,
                "utterance_ms": aln.utterance_ms,
                "frames_per_mora": aln.n_frames / aln.n_target,
                "scores": aln.scores,
                "durations": aln.durations,
                "rel_durations": aln.rel_durations,
                "post_durations": aln.post_durations,
                "post_rel_durations": aln.post_rel_durations,
                "syms": syms,
            }
        )

    report["accepted"] = len(ok)
    report["rejected"] = dict(rejected)
    report["reject_examples"] = reject_examples
    if not ok:
        print("没有任何句子通过对齐，阈值过严:", reject_examples)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    # ---- 句级 ----
    report["utterance"] = {
        "total_score": summarize([r["total"] for r in ok]),
        "mean_logprob": summarize([r["mean_logprob"] for r in ok]),
        "blank_ratio": summarize([r["blank_ratio"] for r in ok]),
        "frames_per_mora": summarize([r["frames_per_mora"] for r in ok]),
        "utterance_ms": summarize([r["utterance_ms"] for r in ok]),
        "n_mora": summarize([r["n_mora"] for r in ok]),
    }

    # ---- mora 级：分数 / 相对时长 / 时长，按组 ----
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"score": [], "rel_dur": [], "dur_ms": [], "post_rel": [], "post_dur_ms": []}
    )
    per_mora: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "sum": 0.0, "low": 0.0, "sum_rel": 0.0, "sum_dur": 0.0, "sum_post": 0.0}
    )
    all_scores: list[float] = []
    for r in ok:
        rows = zip(
            r["syms"],
            r["scores"],
            r["rel_durations"],
            r["durations"],
            r["post_rel_durations"],
            r["post_durations"],
        )
        for sym, sc, rel, dur, prel, pdur in rows:
            g = groups[mora_group(sym)]
            g["score"].append(sc)
            g["rel_dur"].append(rel)
            g["dur_ms"].append(dur * FRAME_MS)
            g["post_rel"].append(prel)
            g["post_dur_ms"].append(pdur * FRAME_MS)
            d = per_mora[sym]
            d["n"] += 1
            d["sum"] += sc
            d["sum_rel"] += rel
            d["sum_dur"] += dur * FRAME_MS
            d["sum_post"] += pdur * FRAME_MS
            if sc < 0.5:
                d["low"] += 1
            all_scores.append(sc)

    report["mora_level"] = {
        "n_instances": len(all_scores),
        "score": summarize(all_scores),
        "low_score_rate": round(sum(1 for s in all_scores if s < 0.5) / len(all_scores), 4),
        "by_group": {
            k: {
                "n": len(v["score"]),
                "score": summarize(v["score"]),
                "rel_duration": summarize(v["rel_dur"]),
                "duration_ms": summarize(v["dur_ms"]),
                "post_rel_duration": summarize(v["post_rel"]),
                "post_duration_ms": summarize(v["post_dur_ms"]),
            }
            for k, v in sorted(groups.items())
        },
    }
    weakest = sorted(
        (
            {
                "mora": sym,
                "n": int(d["n"]),
                "mean_score": round(d["sum"] / d["n"], 4),
                "low_rate": round(d["low"] / d["n"], 4),
                "mean_rel_dur": round(d["sum_rel"] / d["n"], 3),
                "mean_dur_ms": round(d["sum_dur"] / d["n"], 1),
                "mean_post_dur_ms": round(d["sum_post"] / d["n"], 1),
            }
            for sym, d in per_mora.items()
            if d["n"] >= 5
        ),
        key=lambda r: (r["mean_score"], -r["n"]),
    )
    report["weakest_mora"] = weakest[:20]
    report["weakest_mora_low_rate"] = sorted(weakest, key=lambda r: -r["low_rate"])[:20]

    # ---- 负对照：证明分数有区分度，而不是恒定高分 ----
    if not args.skip_negatives:
        neg: dict[str, list[float]] = defaultdict(list)
        pos: list[float] = []
        rng = np.random.default_rng(0)
        for idx, e in enumerate(entries):
            feat = np.load(e["feat"])
            target = np.asarray(e["mora_ids"], dtype=np.int64)
            try:
                a = align_and_score(
                    log_probs_for(model, feat),
                    target,
                    frame_ms=FRAME_MS,
                    min_mean_logprob=args.min_mean_logprob,
                )
            except AlignmentError:
                continue
            pos.append(a.total)
            # 1) 配错文本：取另一句的 mora 序列，长度相近更好，所以按长度就近配
            partner = entries[(idx + 1) % len(entries)]
            ptarget = np.asarray(partner["mora_ids"], dtype=np.int64)
            try:
                b = align_and_score(
                    log_probs_for(model, feat),
                    ptarget,
                    frame_ms=FRAME_MS,
                    min_mean_logprob=-99.0,
                )
                neg["mismatched_text"].append(b.total)
            except AlignmentError as ex:
                rejected[f"neg_{ex.reason}"] += 1
            # 2) 加噪：SNR≈5dB 的高斯噪声
            noise = rng.normal(0, feat.std() * 2, size=feat.shape).astype(np.float32)
            try:
                c = align_and_score(
                    log_probs_for(model, feat + noise),
                    target,
                    frame_ms=FRAME_MS,
                    min_mean_logprob=-99.0,
                )
                neg["noisy_snr5db"].append(c.total)
            except AlignmentError as ex:
                rejected[f"neg_{ex.reason}"] += 1
            # 3) 截断：只留前 60%，模拟「读得太短」
            cut = max(1, int(feat.shape[0] * 0.6))
            try:
                d = align_and_score(
                    log_probs_for(model, feat[:cut]),
                    target,
                    frame_ms=FRAME_MS,
                    min_mean_logprob=-99.0,
                )
                neg["truncated_60pct"].append(d.total)
            except AlignmentError as ex:
                rejected[f"neg_{ex.reason}"] += 1
        report["negative_controls"] = {
            "matched": summarize(pos),
            **{k: {**summarize(v), "cohen_d_vs_matched": round(cohen_d(v, pos), 3)} for k, v in neg.items()},
        }
        report["rejected_after_negatives"] = dict(rejected)

    # ---- 人工评分相关性（有数据才算，没有就明确写不可验证）----
    if args.human_scores:
        human: dict[str, float] = {}
        for line in Path(args.human_scores).read_text().splitlines()[1:]:
            if not line.strip():
                continue
            utt, score = line.split(",")[:2]
            human[utt.strip()] = float(score)
        pairs = [(r["utt"], r) for r in ok if r["utt"] in human]
        if not pairs:
            report["human"] = {"status": "不可验证", "reason": "无 utt 与人工评分对齐"}
        else:
            xs = [r["total"] for _, r in pairs]
            ys = [human[u] for u, _ in pairs]
            mae = float(np.mean(np.abs(np.asarray(xs) * 100 - np.asarray(ys))))
            report["human"] = {
                "status": "已计算",
                "n": len(pairs),
                "pearson": round(pearson(xs, ys), 4),
                "spearman": round(spearman(xs, ys), 4),
                "mae_points": round(mae, 3),
                "ece": round(
                    ece(
                        [r["total"] for _, r in pairs],
                        [int(abs(r["total"] * 100 - human[r["utt"]]) < 10) for _, r in pairs],
                    ),
                    4,
                ),
                "note": "ece 的伪标签为「模型分与人工分之差 <10 分」，仅作占位；真标签需人工逐音素标注",
            }
    else:
        report["human"] = {
            "status": "不可验证",
            "reason": "未提供人工评分数据（--human-scores）。develop.md 要求的相关性需项目方提供标注。",
        }

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"→ {args.out}")
    print(f"通过对齐 {len(ok)}/{n_use}，拒绝 {dict(rejected)}")
    u = report["utterance"]
    print(
        f"句级 total 均值 {u['total_score']['mean']:.4f}"
        f"（p05 {u['total_score']['p05']:.4f} / p95 {u['total_score']['p95']:.4f}）"
    )
    print(f"每 mora 帧数中位数 {u['frames_per_mora']['p50']:.2f}")
    m = report["mora_level"]
    print(f"mora 实例 {m['n_instances']}，低分率(<0.5) {m['low_score_rate']:.4f}")
    for g, v in m["by_group"].items():
        print(
            f"  {g:8s} n={v['n']:5d} 分数均值={v['score']['mean']:.4f}"
            f" Viterbi时长中位={v['duration_ms']['p50']:.0f}ms(std={v['duration_ms']['std']:.1f})"
            f" 后验时长中位={v['post_duration_ms']['p50']:.0f}ms(std={v['post_duration_ms']['std']:.1f})"
            f" 后验相对时长中位={v['post_rel_duration']['p50']:.3f}"
        )
    if "negative_controls" in report:
        nc = report["negative_controls"]
        print(f"负对照: matched={nc['matched']['mean']:.4f}")
        for k, v in nc.items():
            if k == "matched":
                continue
            print(f"  {k:18s} {v['mean']:.4f}  Cohen's d={v['cohen_d_vs_matched']}")
    print(f"人工评分: {report['human']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

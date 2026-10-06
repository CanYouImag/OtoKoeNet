"""用 matched / mismatched 两类样本标定「文本-音频是否相符」的阈值（零训练）。

背景（`log/README.md` §2.3/§2.4）：`align_and_score` 用整句平均 log 概率
`mean_logprob` 与固定阈值 `-3.0` 比较。这个阈值是阶段 10 在域内 val 上标的，
stage14 复核时域内仍成立，但域外 probe 上正确句与配错句的分布已经压到同一数量级
（probe 98 个 SNR0dB 样本里 96 个被误接受）。修法不是再拍一个固定数，而是：

1. 用 `scripts/eval_engine_e2e.py --out` 产出的 `verify_raw`（每句 matched /
   mismatched 的 `mean_logprob`，分别打标签 1 / 0）；
2. 在**拟合集**（默认 val，绝不使用 probe，probe 是域外诊断集）上拟合
   Platt scaling：P(matched | mlp) = sigmoid(a * mlp + b)；
3. 在**评估集**（test/probe）上按「目标误拒率」选工作点，报告接受/拒绝准确率、
   ECE，并与生产固定阈值 `-3.0` 对比。

局限：Platt 只校准**分数->概率**的映射，不改变模型本身。若域外分布与拟合集
不重叠（probe 上 matched/mismatched 已经分不开），再好的标定也只能在
「误拒正确句」与「误收配错句」之间做权衡，不能凭空产生区分度。脚本会把
AUC 一并报出：AUC 接近 0.5 说明该域上分数没有区分度，标定无意义。

    python scripts/fit_calibration.py \
        --report log/stage17_engine_e2e.json \
        --fit-split val --eval-splits test probe
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PROD_THRESHOLD = -3.0


SENTINEL_FLOOR = -30.0


def fit_platt(mlp: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """拟合 sigmoid(a*mlp + b) 的 MLE（带弱 L2 防可分数据发散）。

    先把特征标准化再优化：mean_logprob 的类间差只有 0.1~0.3（命中时接近 0），
    不标准化时梯度下降会被量纲拖住。用 scipy BFGS 而不是手写梯度步，收敛更可靠。
    """
    from scipy.optimize import minimize

    mu = float(mlp.mean())
    sd = float(mlp.std()) or 1.0
    xs = (mlp - mu) / sd

    def nll(theta: np.ndarray) -> float:
        a, b = theta
        z = a * xs + b
        # log(1+exp(z)) - y*z = softplus(z) - y*z
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + 1e-4 * (a * a + b * b))

    res = minimize(nll, np.zeros(2), method="BFGS")
    a_s, b_s = res.x
    # z = a_s*(mlp-mu)/sd + b_s = (a_s/sd)*mlp + (b_s - a_s*mu/sd)
    return float(a_s / sd), float(b_s - a_s * mu / sd)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        total += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(total)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC（正类分数应更高）。"""
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def acc_at_threshold(prob: np.ndarray, y: np.ndarray, thr: float) -> float:
    pred = (prob >= thr).astype(int)
    return float((pred == y).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="scripts/eval_engine_e2e.py 写出的 json")
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--eval-splits", nargs="+", default=["test", "probe"])
    ap.add_argument("--target-frr", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    splits = report["splits"]

    def arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
        raw = splits[split]["verify_raw"]
        mlp = np.asarray(raw["matched_mlp"] + raw["mismatched_mlp"], dtype=float)
        y = np.asarray([1] * len(raw["matched_mlp"]) + [0] * len(raw["mismatched_mlp"]))
        # `align_and_score` 对不可行对齐（多为配错文本）返回 AlignmentError，脚本
        # 记的 mean_logprob 是哨兵 -99。**不能丢**：这些正是「坚决拒绝」的负样本，
        # 丢掉后域内负类只剩 1 个（实测 199/200 配错文本因不可行被拒），拟合无意义。
        # 用一个有限地板值保留「这是一个极低分」的信息。
        return np.maximum(mlp, SENTINEL_FLOOR), y

    if args.fit_split not in splits:
        raise SystemExit(f"拟合集 {args.fit_split} 不在报告里：{list(splits)}")
    fit_mlp, fit_y = arrays(args.fit_split)
    a, b = fit_platt(fit_mlp, fit_y)
    print(f"Platt 拟合（{args.fit_split}, n={len(fit_mlp)}）: a={a:.4f} b={b:.4f}")
    print(f"  拟合集 AUC={auc(fit_mlp, fit_y):.4f}（固定阈值 {PROD_THRESHOLD} 准确率 "
          f"{acc_at_threshold(fit_mlp, fit_y, PROD_THRESHOLD) * 100:.2f}%）")

    out: dict = {"fit_split": args.fit_split, "platt": {"a": a, "b": b}, "splits": {}}
    for split in args.eval_splits:
        if split not in splits:
            continue
        mlp, y = arrays(split)
        prob = sigmoid(a * mlp + b)
        prod_acc = acc_at_threshold(mlp, y, PROD_THRESHOLD)
        row = {
            "n": int(len(y)),
            "n_matched": int((y == 1).sum()),
            "n_mismatched": int((y == 0).sum()),
            "auc": round(auc(mlp, y), 4),
            "prod_threshold_acc": round(prod_acc, 4),
            "platt_ece": round(ece(prob, y), 4),
            "platt_best_acc": round(max(acc_at_threshold(prob, y, t) for t in np.linspace(0, 1, 201)), 4),
            "operating_points": {},
        }
        # 目标误拒率：把 matched 的校准概率从小到大排序，取分位点当阈值
        matched_prob = np.sort(prob[y == 1])
        for frr in args.target_frr:
            if len(matched_prob) == 0:
                continue
            thr = float(matched_prob[min(len(matched_prob) - 1, int(frr * len(matched_prob)))])
            row["operating_points"][f"frr_{int(frr * 100)}pct"] = {
                "threshold_prob": round(thr, 4),
                "threshold_mlp": round(float((math.log(thr / (1 - thr)) - b) / a), 4)
                if 0 < thr < 1 and a != 0
                else None,
                "accuracy": round(acc_at_threshold(prob, y, thr), 4),
            }
        mscore = np.asarray(
            splits[split]["verify_raw"]["matched_total"]
            + splits[split]["verify_raw"]["mismatched_total"],
            dtype=float,
        )
        if len(mscore) == len(y):
            row["score_auc"] = round(auc(mscore, y), 4)
        out["splits"][split] = row
        ops = " ".join(
            f"FRR{k.split('_')[1]} acc={v['accuracy'] * 100:.1f}%"
            for k, v in row["operating_points"].items()
        )
        print(
            f"\n=== {split} n={row['n']} ===\n"
            f"  AUC(mean_logprob)={row['auc']:.4f}  AUC(total score)={row.get('score_auc')}\n"
            f"  固定阈值 {PROD_THRESHOLD} 准确率 {prod_acc * 100:.2f}% | "
            f"Platt 最优 {row['platt_best_acc'] * 100:.2f}% (ECE {row['platt_ece']:.4f})\n"
            f"  目标误拒率工作点: {ops}"
        )

    Path(args.report).with_name(Path(args.report).stem + "_calibration.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )
    print(f"\n→ {Path(args.report).with_name(Path(args.report).stem + '_calibration.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""阶段 19：OOD 检测 + 校准（零训练）。

背景（`log/README.md` §2.3/§2.7）：SNR0dB 下 mora 头识别 CER 约 39%，但 `/api/evaluate`
的「文本-音频是否相符」仍把 ~100% 的带噪音频判成相符。根因是模型在破坏输入上依然
过度自信（`mean_logprob` 仍 >-3.0），对 `mean_logprob` 做单调标定（stage17 的 Platt）
不改变排序，因此必须引入**新的、能反映输入质量的信号**。

本脚本把每条 (音频, 参考文本) 压成一组对齐统计特征，用一个标准化 + 逻辑回归的
detector 区分「干净的相符」与「不符 / OOD（噪声、截断）」，在 val 上按目标误拒率
选工作点，在 test/probe 上评。特征全部来自已有的 `MoraAlignment`：

    mean_logprob, gop_mean/std, score_mean/std, entropy_mean/std,
    rel_dur_skew, blank_ratio, mora_rate(=n_target/n_frames), is_rejected

标签：clean 匹配 = 1；clean 配错 + SNR0/5dB 匹配 + 截断 60% = 0。
（speed_1.1 不进训练：它是合法朗读，只是语速不同，用于观察 detector 是否误伤。）

与 `scripts/fit_calibration.py` 的区别：那个只把 `mean_logprob` 做 Platt；本脚本加入
后验熵等**对输入质量敏感**的特征，目标是真正分开 SNR0dB 与 clean，而不只是换个阈值。

    python scripts/train_ood.py --ckpt runs/jsut_full_stage14/best.pt \
        --cache-dir data/cache/jsut_full_v1 --splits val test probe \
        --out log/stage19_ood.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.config import Settings  # noqa: E402
from app.ml.engine import Engine  # noqa: E402
from otokoenet.align import AlignmentError  # noqa: E402
from otokoenet.data import Manifest  # noqa: E402
from otokoenet.ood import (  # noqa: E402
    FEATURE_NAMES,
    LogisticOOD,
    feature_vector,
    reject_vector,
)


def degrade(feat: np.ndarray, cond: str, rng: np.random.Generator) -> np.ndarray:
    """与 scripts/eval_engine_e2e.py 的 --degrade 完全同一套变换（特征域）。"""
    if cond == "clean":
        return feat
    if cond == "noise_snr5db":
        return (feat + rng.normal(0, feat.std() / 10 ** (5 / 20), feat.shape)).astype(np.float32)
    if cond == "noise_snr0db":
        return (feat + rng.normal(0, feat.std() / 10 ** (0 / 20), feat.shape)).astype(np.float32)
    if cond == "truncated_60pct":
        return feat[: max(1, int(feat.shape[0] * 0.6))]
    if cond == "speed_1.1":
        n = feat.shape[0]
        m = max(1, int(n / 1.1))
        xi = np.linspace(0, n - 1, m)
        return np.stack(
            [np.interp(xi, np.arange(n), feat[:, d]) for d in range(feat.shape[1])], axis=1
        ).astype(np.float32)
    raise ValueError(cond)


TRAIN_CONDS = ["noise_snr5db", "noise_snr0db", "truncated_60pct"]
EVAL_CONDS = ["clean", "noise_snr5db", "noise_snr0db", "truncated_60pct", "speed_1.1"]


def collect(
    engine: Engine, manifest: Manifest, min_mlp: float, limit: int, seed: int
) -> dict:
    entries = list(manifest.entries[:limit])
    # 配错文本伙伴：按 mora 长度就近取下一句（与 e2e 脚本一致）
    order = sorted(range(len(entries)), key=lambda i: len(entries[i]["mora_ids"]))
    partner: dict[int, int] = {}
    for pos, i in enumerate(order):
        partner[i] = order[min(pos + 1, len(order) - 1)]

    rng = np.random.default_rng(seed)
    # rows: (kind, cond) -> list[feature], 以及生产是否接受
    data: dict[str, dict[str, list]] = {}
    for i, e in enumerate(entries):
        clean = engine.featurize(e["wav"])
        target = list(e["mora_ids"])
        for cond in EVAL_CONDS:
            feat = degrade(clean, cond, rng)
            key = f"matched:{cond}"
            try:
                aln = engine.evaluate_detailed(feat, target)
                data.setdefault(key, []).append(
                    (feature_vector(aln), bool(aln.mean_logprob >= min_mlp))
                )
            except AlignmentError:
                data.setdefault(key, []).append((reject_vector(), False))
        # clean 配错
        neg = list(entries[partner[i]]["mora_ids"])
        try:
            aln = engine.evaluate_detailed(clean, neg)
            data.setdefault("mismatched:clean", []).append(
                (feature_vector(aln), bool(aln.mean_logprob >= min_mlp))
            )
        except AlignmentError:
            data.setdefault("mismatched:clean", []).append((reject_vector(), False))
    return data


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-2):
    """标准化 + 逻辑回归 MLE（scipy BFGS），返回 (mu, sd, coef, b)。"""
    from scipy.optimize import minimize

    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    xs = (x - mu) / sd

    def nll(theta: np.ndarray) -> float:
        w, b = theta[:-1], theta[-1]
        z = xs @ w + b
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + l2 * float(w @ w))

    res = minimize(nll, np.zeros(xs.shape[1] + 1), method="BFGS")
    w, b = res.x[:-1], res.x[-1]
    return mu, sd, w, b


def apply_logistic(x: np.ndarray, params) -> np.ndarray:
    mu, sd, w, b = params
    z = ((x - mu) / sd) @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/jsut_full_stage14/best.pt")
    ap.add_argument("--cache-dir", default="data/cache/jsut_full_v1")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "probe"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--target-frr", type=float, default=0.01, help="val clean 允许的误拒率")
    ap.add_argument("--out", default="log/stage19_ood.json")
    ap.add_argument(
        "--detector-out",
        default="data/cache/jsut_full_v1/ood_detector.json",
        help="把标准化+逻辑回归参数落盘，供 backend 的 OTOKOE_OOD=1 启用",
    )
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    settings = Settings()
    settings.ckpt_path = Path(args.ckpt).resolve()
    settings.cache_dir = Path(args.cache_dir).resolve()
    engine = Engine(settings)
    cache = Path(args.cache_dir)
    print(
        f"ckpt={settings.ckpt_path.name} step={engine.ckpt_step} "
        f"cache={cache.name} min_mean_logprob={settings.min_mean_logprob}"
    )

    t0 = time.time()
    got = {
        s: collect(engine, Manifest.load(str(cache / f"{s}.json")), settings.min_mean_logprob,
                   args.limit, seed=100 + n)
        for n, s in enumerate(args.splits)
    }
    print(f"特征提取完成: {time.time() - t0:.1f}s")

    if "val" not in got:
        raise SystemExit("必须在 --splits 里包含 val 作为拟合集")

    def vector(rows_by_key, keys) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for k, label in keys:
            for vec, _ in rows_by_key[k]:
                xs.append(vec)
                ys.append(label)
        return np.asarray(xs, float), np.asarray(ys, int)

    train_keys = [("matched:clean", 1)] + [(f"matched:{c}", 0) for c in TRAIN_CONDS] + [
        ("mismatched:clean", 0)
    ]
    x_tr, y_tr = vector(got["val"], train_keys)
    params = fit_logistic(x_tr, y_tr)
    print(
        "逻辑回归系数（标准化后）: "
        + " ".join(f"{n}={w:+.2f}" for n, w in zip(FEATURE_NAMES, params[2]))
        + f" b={params[3]:+.2f}"
    )

    # 在 val clean 匹配上按目标误拒率选阈值
    val_clean_prob = apply_logistic(
        np.asarray([v for v, _ in got["val"]["matched:clean"]], float), params
    )
    thr = float(np.quantile(val_clean_prob, args.target_frr))
    print(f"val clean n={len(val_clean_prob)} 目标 FRR={args.target_frr:.0%} → OOD 阈值 p>={thr:.4f}")
    if args.detector_out:
        LogisticOOD(params[0], params[1], params[2], params[3], thr).save(args.detector_out)
        print(f"detector → {args.detector_out}（backend 用 OTOKOE_OOD=1 启用）")

    report: dict = {
        "ckpt": str(settings.ckpt_path),
        "ckpt_step": engine.ckpt_step,
        "cache_dir": str(cache),
        "min_mean_logprob": settings.min_mean_logprob,
        "feature_names": FEATURE_NAMES,
        "logistic": {
            "mean": params[0].tolist(),
            "std": params[1].tolist(),
            "coef": params[2].tolist(),
            "b": params[3],
            "threshold": thr,
        },
        "splits": {},
    }

    for split, data in got.items():
        row: dict = {"n_per_key": {k: len(v) for k, v in data.items()}, "conditions": {}}
        # clean 正类 vs「全部负类（含 OOD）」的 AUC，供比较
        pos_x = np.asarray([v for v, _ in data["matched:clean"]], float)
        neg_rows = data["mismatched:clean"] + [r for c in TRAIN_CONDS for r in data[f"matched:{c}"]]
        all_x = np.concatenate([pos_x, np.asarray([v for v, _ in neg_rows], float)])
        all_y = np.concatenate([np.ones(len(pos_x)), np.zeros(len(neg_rows))])
        all_p = apply_logistic(all_x, params)
        row["auc_clean_vs_ood"] = round(auc(all_p, all_y), 4)
        for key in [f"matched:{c}" for c in EVAL_CONDS] + ["mismatched:clean"]:
            rows = data[key]
            x = np.asarray([v for v, _ in rows], float)
            prod_accept = float(np.mean([a for _, a in rows]))
            p = apply_logistic(x, params)
            row["conditions"][key] = {
                "n": len(rows),
                "prod_accept_rate": round(prod_accept, 4),
                "ood_accept_rate": round(float((p >= thr).mean()), 4),
            }
        # clean 误拒率
        clean = row["conditions"]["matched:clean"]
        clean["prod_frr"] = round(1 - clean["prod_accept_rate"], 4)
        clean["ood_frr"] = round(1 - clean["ood_accept_rate"], 4)
        report["splits"][split] = row

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 生产规则 vs OOD detector（接受率，越低越好=越能拒 OOD）===")
    hdr = f"{'condition':22s}" + "".join(f"{s:>12s}" for s in got)
    print(hdr)
    for key in [f"matched:{c}" for c in EVAL_CONDS] + ["mismatched:clean"]:
        line = f"{key:22s}"
        for s in got:
            row = report["splits"][s]["conditions"][key]
            line += f"  {row['prod_accept_rate']*100:4.0f}->{row['ood_accept_rate']*100:4.0f} "
        print(line)
    for s in got:
        row = report["splits"][s]
        print(f"  [{s}] clean 误拒 prod {row['conditions']['matched:clean']['prod_frr']*100:.1f}% "
              f"→ ood {row['conditions']['matched:clean']['ood_frr']*100:.1f}% ; "
              f"AUC={row['auc_clean_vs_ood']:.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

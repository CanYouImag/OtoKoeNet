"""OOD（输入不可信）检测器（阶段 19）。

把一条 `MoraAlignment` 压成一组对齐统计特征，用一个在 val 上标定好的
标准化 + 逻辑回归给出 `P(clean 相符)`。生产侧由 `settings.ood_enabled` 控制是否启用；
默认 **关闭**——见下方原因。

特征（顺序固定，必须与训练脚本一致）：
    mean_logprob, gop_mean/std, score_mean/std, entropy_mean/std,
    rel_dur_skew, blank_ratio, mora_rate(=n_target/n_frames)

实测（`log/stage19_ood.json`，stage14）：SNR0dB 假接受率 98%/98%/98% → 0%/0%/0%，
但 **probe clean 误拒 0% → 10.5%**（域外正确音频被误杀）。按「部署默认只在 val/test/probe
都不退化时才切换」的规矩，默认关闭；强噪声场景或配合域自适应再启用。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "mean_logprob",
    "gop_mean",
    "gop_std",
    "score_mean",
    "score_std",
    "entropy_mean",
    "entropy_std",
    "rel_dur_skew",
    "blank_ratio",
    "mora_rate",
    "is_rejected",
]

SENTINEL_MLP = -30.0


def _skew(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    a = np.asarray(values, dtype=float)
    sd = a.std()
    if sd < 1e-9:
        return 0.0
    return float(((a - a.mean()) ** 3).mean() / sd**3)


def feature_vector(aln) -> list[float]:
    gops = np.asarray(aln.gops, dtype=float) if aln.gops else np.zeros(1)
    scores = np.asarray(aln.scores, dtype=float) if aln.scores else np.zeros(1)
    return [
        float(aln.mean_logprob),
        float(gops.mean()),
        float(gops.std()),
        float(scores.mean()),
        float(scores.std()),
        float(aln.entropy_mean),
        float(aln.entropy_std),
        _skew(list(aln.post_rel_durations)),
        float(aln.blank_ratio),
        float(aln.n_target) / max(1, aln.n_frames),
        0.0,
    ]


def reject_vector() -> list[float]:
    """对齐不可行时的特征向量：保留「极低分」信息 + is_rejected=1。"""
    vec = [0.0] * len(FEATURE_NAMES)
    vec[FEATURE_NAMES.index("mean_logprob")] = SENTINEL_MLP
    vec[FEATURE_NAMES.index("blank_ratio")] = 1.0
    vec[FEATURE_NAMES.index("is_rejected")] = 1.0
    return vec


class LogisticOOD:
    def __init__(self, mean, std, coef, b, threshold) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.std[self.std < 1e-9] = 1.0
        self.coef = np.asarray(coef, dtype=float)
        self.b = float(b)
        self.threshold = float(threshold)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticOOD":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        lg = data["logistic"]
        return cls(lg["mean"], lg["std"], lg["coef"], lg["b"], lg["threshold"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "feature_names": FEATURE_NAMES,
                    "logistic": {
                        "mean": self.mean.tolist(),
                        "std": self.std.tolist(),
                        "coef": self.coef.tolist(),
                        "b": self.b,
                        "threshold": self.threshold,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def prob(self, aln) -> float:
        x = np.asarray(feature_vector(aln), dtype=float)
        z = float(((x - self.mean) / self.std) @ self.coef + self.b)
        return 1.0 / (1.0 + math.exp(-z))

    def is_ood(self, aln, threshold: float | None = None) -> bool:
        thr = self.threshold if threshold is None else threshold
        return self.prob(aln) < thr

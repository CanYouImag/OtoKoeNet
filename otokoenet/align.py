from __future__ import annotations

import math

import numpy as np

_NEG = -1e30


def forced_align(log_probs: np.ndarray, target: np.ndarray) -> list[int]:
    """CTC Viterbi 强制对齐。

    Args:
        log_probs: (T, C) 帧级 log 概率。
        target: (n,) 目标 token id 序列。

    Returns:
        path: 长度为 T 的状态序列，0 表示 blank，其余为目标 token id。
    """
    T, _ = log_probs.shape
    n = len(target)
    blank = 0

    dp_b = [_NEG] * (n + 1)
    dp_l = [_NEG] * (n + 1)
    dp_b[0] = 0.0

    trace = []
    for t in range(T):
        cur = log_probs[t]
        p_blank = cur[blank]
        nb = [_NEG] * (n + 1)
        nl = [_NEG] * (n + 1)
        b_back = [None] * (n + 1)
        l_back = [None] * (n + 1)
        for i in range(n + 1):
            best = dp_b[i]
            kind = 0
            if i >= 1 and dp_l[i] > best:
                best = dp_l[i]
                kind = 1
            nb[i] = best + p_blank
            b_back[i] = (kind, i)

            if i >= 1:
                best = dp_b[i - 1]
                kind = 0
                if i >= 2 and target[i - 2] != target[i - 1] and dp_l[i - 1] > best:
                    best = dp_l[i - 1]
                    kind = 1
                nl[i] = best + cur[target[i - 1]]
                l_back[i] = (kind, i - 1)
        dp_b, dp_l = nb, nl
        trace.append((b_back, l_back))

    if dp_l[n] >= dp_b[n]:
        kind, i = 1, n
    else:
        kind, i = 0, n

    path = [0] * T
    for t in range(T - 1, -1, -1):
        b_back, l_back = trace[t]
        if kind == 0:
            path[t] = 0
            kind, i = b_back[i]
        else:
            path[t] = int(target[i - 1])
            kind, i = l_back[i]
    return path


def score_alignment(log_probs: np.ndarray, target: np.ndarray, path: list[int]) -> tuple[list[float], float]:
    """按对齐结果对每个目标 token 打分。

    Args:
        log_probs: (T, C) log 概率。
        target: (n,) 目标序列。
        path: 对齐状态序列（长度 T）。

    Returns:
        scores: 每个 token 的置信度（0~1）。
        total: 全体平均置信度（0~1）。
    """
    T = len(path)
    scores = []
    for j, tok in enumerate(target):
        frames = [t for t in range(T) if path[t] == tok]
        if frames:
            scores.append(float(math.exp(float(np.mean(log_probs[frames, tok])))))
        else:
            scores.append(0.0)
    total = float(np.mean(scores)) if scores else 0.0
    return scores, total

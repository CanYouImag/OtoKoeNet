from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_NEG = -1e30

# 拒绝对齐的原因（evaluate 接口据此返回可读的错误，而不是一个静默的 0 分）
REASON_EMPTY_AUDIO = "empty_audio"
REASON_EMPTY_TARGET = "empty_target"
REASON_TOO_SHORT = "too_short"
REASON_RATIO = "implausible_mora_rate"
REASON_LOW_SCORE = "low_score"
REASON_SILENT = "silent_audio"


class AlignmentError(ValueError):
    """对齐不可行。`reason` 供 API 直接回传，`detail` 供日志定位。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def forced_align(log_probs: np.ndarray, target: np.ndarray) -> list[int]:
    """CTC Viterbi 强制对齐。

    Args:
        log_probs: (T, C) 帧级 log 概率。
        target: (n,) 目标 token id 序列。

    Returns:
        path: 长度为 T 的状态序列，0 表示 blank，其余为 1-based 目标 token 序号。
            存序号而非 token id，避免重复 token（如「のの」）的帧被合并。

    注意：调用方必须自行保证可行（见 `align_and_score`）。若 n > T，回溯会退化成
    全 blank 路径而不是报错，因此这里不作为对外入口。
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
            path[t] = i
            kind, i = l_back[i]
    return path


def score_alignment(log_probs: np.ndarray, target: np.ndarray, path: list[int]) -> tuple[list[float], float]:
    """按对齐结果对每个目标 token 打分。

    保留旧签名（Engine/旧脚本依赖），只做逐实例打分，不做可行性检查。
    """
    T = len(path)
    scores = []
    for j, tok in enumerate(target):
        frames = [t for t in range(T) if path[t] == j + 1]
        if frames:
            scores.append(float(math.exp(float(np.mean(log_probs[frames, tok])))))
        else:
            scores.append(0.0)
    total = float(np.mean(scores)) if scores else 0.0
    return scores, total


@dataclass
class MoraAlignment:
    """一次成功的逐 mora 对齐结果。

    `spans` / `durations` / `rel_durations` 是「每一次出现」的实例级信息：
    参考文本里重复出现的同一个 mora 各自有独立边界，便于前端标出具体位置。
    """

    path: list[int]
    spans: list[tuple[int, int]]
    scores: list[float]
    logprobs: list[float]
    durations: list[int]
    rel_durations: list[float]
    post_durations: list[float]
    post_rel_durations: list[float]
    total: float
    mean_logprob: float
    n_frames: int
    n_target: int
    frame_ms: float
    utterance_ms: float
    blank_ratio: float
    unaligned: list[int] = field(default_factory=list)

    @property
    def frame_rate(self) -> float:
        return 1000.0 / self.frame_ms

    def duration_ms(self, i: int) -> float:
        return self.durations[i] * self.frame_ms

    def post_duration_ms(self, i: int) -> float:
        return self.post_durations[i] * self.frame_ms

    def as_dict(self) -> dict:
        return {
            "scores": [round(s, 6) for s in self.scores],
            "logprobs": [round(v, 6) for v in self.logprobs],
            "spans": [list(s) for s in self.spans],
            "durations": self.durations,
            "rel_durations": [round(v, 4) for v in self.rel_durations],
            "post_durations": [round(v, 4) for v in self.post_durations],
            "post_rel_durations": [round(v, 4) for v in self.post_rel_durations],
            "total": round(self.total, 6),
            "mean_logprob": round(self.mean_logprob, 6),
            "n_frames": self.n_frames,
            "n_target": self.n_target,
            "frame_ms": self.frame_ms,
            "utterance_ms": round(self.utterance_ms, 2),
            "blank_ratio": round(self.blank_ratio, 6),
            "unaligned": self.unaligned,
        }


def _logaddexp3(a: np.ndarray, b: np.ndarray, c: np.ndarray | None) -> np.ndarray:
    out = np.logaddexp(a, b)
    return out if c is None else np.logaddexp(out, c)


def ctc_state_posteriors(log_probs: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CTC 前向后向，算每个帧上每个**扩展状态**的边缘后验。

    Returns:
        (post_s, labels)：post_s 形状 (T, S)，S = 2n+2；labels[s] 是状态 s 发射的
        标签 id（blank = 0）。状态 2j+1 和 2j+2 都发射 target[j]，但它们是
        **不同 token 的两个不同状态**，不能合并。

    为什么必须按状态而不是按标签：同一个 target 里重复出现同一个标签时
    （日语里「っ」「ー」和叠词都很常见），按标签聚合的后验是那几次出现**共享**
    的一列总和，逐实例取 `post[:, target[j]]` 会把整列复制给每一次出现，导致
    时长重复计数、且各实例拿到完全相同的值。逐实例时长必须取
    `Σ_t post_s[:, 2j+1] + post_s[:, 2j+2]`。
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    T, C = log_probs.shape
    n = int(len(target))
    S = 2 * n + 2
    neg = -1e30

    # labels: [blank, a0, a0, a1, a1, ..., a_{n-1}, a_{n-1}, blank]
    labels = np.empty(S, dtype=np.int64)
    labels[0] = 0
    labels[-1] = 0
    for j in range(n):
        labels[2 * j + 1] = target[j]
        labels[2 * j + 2] = target[j]

    lp = log_probs[:, labels]  # (T, S)
    # s -> s+2 是否允许：仅当 label(s) != label(s+2) 或 label(s) 为 blank
    skip_ok = np.ones(S, dtype=bool)
    if S > 2:
        same = labels[:-2] == labels[2:]
        skip_ok[2:] = ~(same & (labels[2:] != 0))

    alpha = np.full((T, S), neg)
    alpha[0, 0] = lp[0, 0]
    if S > 1:
        alpha[0, 1] = lp[0, 1]
    for t in range(1, T):
        prev = alpha[t - 1]
        cur = np.full(S, neg)
        cur[0] = prev[0] + lp[t, 0]
        stay = prev + lp[t]
        enter = np.full(S, neg)
        enter[1:] = prev[:-1] + lp[t, 1:]
        skip = np.full(S, neg)
        skip[2:] = np.where(skip_ok[2:], prev[:-2] + lp[t, 2:], neg)
        cur[1:] = _logaddexp3(stay[1:], enter[1:], skip[1:])
        alpha[t] = cur

    # 合法终止状态是 S-2（停在最后一个 label）和 S-1（末尾 blank），
    # 即「n 个 label 全部消耗完」的所有状态。状态 0 意味着一个 label 都没出，
    # n>0 时不能作为终止状态。
    log_z = float(np.logaddexp(alpha[T - 1, S - 2], alpha[T - 1, S - 1]))

    beta = np.full((T, S), neg)
    beta[T - 1, S - 2] = 0.0
    beta[T - 1, S - 1] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = beta[t + 1]
        nxt_lp = lp[t + 1]
        stay = nxt_lp + nxt
        # s -> s+1 / s -> s+2：下一状态的 beta 也要取 s+1 / s+2
        leave1 = np.full(S, neg)
        leave1[:-1] = nxt_lp[1:] + nxt[1:]
        leave2 = np.full(S, neg)
        ok = np.zeros(S, dtype=bool)
        ok[:-2] = skip_ok[2:]
        leave2[:-2] = np.where(ok[:-2], nxt_lp[2:] + nxt[2:], neg)
        b = _logaddexp3(stay, leave1, leave2)
        # 状态 0 可以前进到 1（进入第一个 label），所以不能用 stay 覆盖；
        # 末状态 S-1 只能 stay。
        b[S - 1] = stay[S - 1]
        beta[t] = b

    joint = alpha + beta - log_z  # (T, S)
    post_s = np.exp(joint)
    return post_s, labels


def ctc_posteriors(log_probs: np.ndarray, target: np.ndarray) -> np.ndarray:
    """CTC 前向后向，算每个帧上每个**标签**的边缘后验（按状态后验聚合）。

    为什么需要它：本项目 fbank 10ms 经 4 倍下采样后每帧 40ms，一个 mora 只有
    3~4 帧；而训到收敛的 CTC 模型逐帧极度自信（实测每帧 argmax 平均 logprob
    -0.03），于是 Viterbi 路径会给每个 token 只分配 1 帧、其余全给 blank。
    实测 6781 个 mora 实例的时长标准差全为 0，时长特征完全失效。
    后验是平滑的，`Σ_t P(标签=j | x_t)` 才是可用的时长估计。

    **注意**：重复标签的逐实例时长不能用这个函数的结果，见
    `ctc_state_posteriors` 的 docstring。要逐实例时长请用状态后验。

    Args:
        log_probs: (T, C) 帧级 log 概率。
        target: (n,) 目标 token id 序列。

    Returns:
        (T, C) 概率（非 log），每行和为 1。

    实现采用 Graves 的扩展状态空间 s ∈ [0, 2n+1]，label(0)=label(2n+1)=blank，
    转移 s → s, s+1, s+2，其中 s → s+2 在 label(s)==label(s+2) 且非 blank 时禁止
    （这就是重复 token 不能被合并的原因）。
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T, C = log_probs.shape
    post_s, labels = ctc_state_posteriors(log_probs, target)
    post = np.zeros((T, C), dtype=np.float64)
    for s in range(len(labels)):
        post[:, labels[s]] += post_s[:, s]
    return post


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def align_and_score(
    log_probs: np.ndarray,
    target: np.ndarray,
    *,
    frame_ms: float = 40.0,
    min_mora_frames: float = 1.0,
    max_mora_frames: float = 40.0,
    min_mean_logprob: float = -3.0,
    energy_floor: float = 1e-4,
) -> MoraAlignment:
    """逐 mora 强制对齐 + 打分，并对不可行的情况显式拒绝。

    之前 `Engine.evaluate` 直接调 `forced_align`，在 n > T（学习者说得比参考文本短，
    实际很常见）时回溯退化成全 blank 路径，接口安静地返回 0 分，用户无法区分
    「读错」和「读得太短」。这里在入口处把这类输入挑出来。

    Args:
        log_probs: (T, C) 帧级 log 概率。
        target: (n,) 目标 token id 序列。
        frame_ms: 每帧毫秒数，用于时长特征。
        min_mora_frames / max_mora_frames: 每个 mora 的合理帧数区间（可行性）。
        min_mean_logprob: 整句平均 log 概率下界，低于则判为「整体不可信」。
        energy_floor: log_probs 在全 blank 位置上的能量下界，用于识别静音输入。

    Raises:
        AlignmentError: 对齐不可行，`reason` 见模块常量。
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs 必须是 (T, C)，实际 {log_probs.shape}")
    T, C = log_probs.shape
    n = int(len(target))

    if T == 0:
        raise AlignmentError(REASON_EMPTY_AUDIO, f"T=0, C={C}")
    if n == 0:
        raise AlignmentError(REASON_EMPTY_TARGET, f"T={T}")
    if np.isfinite(log_probs).sum() < T * C * 0.5:
        raise AlignmentError(REASON_SILENT, f"log_probs 有 {T * C - int(np.isfinite(log_probs).sum())} 个非有限值")
    # 静音/常量输入：blank 概率处处占绝对优势，对齐结果没有信息量
    blank_lp = log_probs[:, 0]
    if float(blank_lp.mean()) < math.log(energy_floor) and float(np.abs(log_probs).max()) < energy_floor:
        raise AlignmentError(REASON_SILENT, f"blank 平均 logprob={blank_lp.mean():.3f}")
    if T < n:
        raise AlignmentError(
            REASON_TOO_SHORT,
            f"{n} 个 mora 需要至少 {n} 帧（实际 {T} 帧 = {T * frame_ms:.0f}ms）",
        )
    per_mora = T / n
    if not (min_mora_frames <= per_mora <= max_mora_frames):
        raise AlignmentError(
            REASON_RATIO,
            f"每 mora {per_mora:.2f} 帧超出 [{min_mora_frames}, {max_mora_frames}]",
        )
    if int(target.min()) < 0 or int(target.max()) >= C:
        raise AlignmentError(REASON_EMPTY_TARGET, f"token id 越界 [0, {C - 1}]")

    path = forced_align(log_probs, target)

    # 逐实例边界。path 存的是 1-based 序号，重复 token 天然分开。
    frames_of: list[list[int]] = [[] for _ in range(n)]
    for t, state in enumerate(path):
        if state:
            frames_of[state - 1].append(t)
    spans: list[tuple[int, int]] = []
    durations: list[int] = []
    scores: list[float] = []
    logprobs: list[float] = []
    unaligned: list[int] = []
    for j, tok in enumerate(target):
        fr = frames_of[j]
        if not fr:
            unaligned.append(j)
            spans.append((-1, -1))
            durations.append(0)
            scores.append(0.0)
            logprobs.append(_NEG / 100.0)
            continue
        spans.append((fr[0], fr[-1] + 1))
        durations.append(len(fr))
        lp = float(np.mean(log_probs[fr, int(tok)]))
        logprobs.append(lp)
        scores.append(math.exp(lp))

    # 未对齐实例理论上不该出现（T >= n 已保证存在可行路径），出现即视为不可信
    if unaligned:
        raise AlignmentError(
            REASON_TOO_SHORT,
            f"索引 {unaligned[:5]} 未分配到任何帧（T={T}, n={n}）",
        )

    med = _median([float(d) for d in durations])
    rel = [float(d) / med if med > 0 else 0.0 for d in durations]

    # 后验时长：Viterbi 路径在本项目里每个 token 只占 1 帧（模型逐帧过度自信），
    # 所以时长特征必须用后验，否则标准差恒为 0。
    # 逐实例取该 token 自己的两个状态 2j+1 / 2j+2，**不能**按标签聚合后取列：
    # target 里重复的标签（っ、ー、叠词）会共享同一列聚合值，重复计数。
    post_s, _ = ctc_state_posteriors(log_probs, target)
    post_durations = [
        float(post_s[:, 2 * j + 1].sum() + post_s[:, 2 * j + 2].sum()) for j in range(len(target))
    ]
    pmed = _median(post_durations)
    post_rel = [d / pmed if pmed > 0 else 0.0 for d in post_durations]
    total = float(np.mean(scores))
    mean_logprob = float(np.mean(logprobs))
    if mean_logprob < min_mean_logprob:
        raise AlignmentError(
            REASON_LOW_SCORE,
            f"整句平均 logprob={mean_logprob:.3f} < {min_mean_logprob}",
        )
    n_blank = sum(1 for s in path if s == 0)
    return MoraAlignment(
        path=path,
        spans=spans,
        scores=scores,
        logprobs=logprobs,
        durations=durations,
        rel_durations=rel,
        post_durations=post_durations,
        post_rel_durations=post_rel,
        total=total,
        mean_logprob=mean_logprob,
        n_frames=T,
        n_target=n,
        frame_ms=frame_ms,
        utterance_ms=T * frame_ms,
        blank_ratio=n_blank / T,
    )

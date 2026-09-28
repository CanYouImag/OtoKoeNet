"""`otokoenet/align.py` 的正确性回归。

重点是 `ctc_posteriors`：它是手写的 log 域前向后向，一旦索引错位不会抛异常，
只会静默给出错的时长（这类 bug 极难靠指标发现）。所以这里用一个逐状态、
非向量化的参考实现做逐元素比对，容差取机器精度量级。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from otokoenet.align import (
    REASON_RATIO,
    REASON_TOO_SHORT,
    AlignmentError,
    align_and_score,
    ctc_posteriors,
    forced_align,
)

NEG = -math.inf
FAILED: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def _ref_fb(lp: np.ndarray, target: np.ndarray) -> np.ndarray:
    """逐状态的朴素前向后向，作为 `ctc_posteriors` 的黄金参考。"""
    T, C = lp.shape
    n = len(target)
    S = 2 * n + 2
    labels = [0]
    for tok in target:
        labels += [int(tok), int(tok)]
    labels += [0]

    def allowed(a: int, b: int) -> bool:
        if b < a:
            return False
        if b - a == 2 and not (labels[a] != labels[b] or labels[a] == 0):
            return False
        return True

    alpha = np.full((T, S), NEG)
    alpha[0, 0] = lp[0][labels[0]]
    alpha[0, 1] = lp[0][labels[1]]
    for t in range(1, T):
        for s in range(S):
            vals = []
            for ps in (s, s - 1, s - 2):
                if ps < 0 or ps > s:
                    continue
                if ps == s - 2 and not allowed(ps, s):
                    continue
                if alpha[t - 1, ps] > NEG / 2:
                    vals.append(alpha[t - 1, ps] + lp[t][labels[s]])
            if vals:
                m = max(vals)
                alpha[t, s] = m + math.log(sum(math.exp(x - m) for x in vals))
    log_z = float(np.logaddexp(alpha[T - 1, S - 2], alpha[T - 1, S - 1]))

    beta = np.full((T, S), NEG)
    beta[T - 1, S - 2] = 0.0
    beta[T - 1, S - 1] = 0.0
    for t in range(T - 2, -1, -1):
        for s in range(S):
            vals = []
            for ns in (s, s + 1, s + 2):
                if ns > S - 1 or not allowed(s, ns):
                    continue
                if beta[t + 1, ns] > NEG / 2:
                    vals.append(lp[t + 1][labels[ns]] + beta[t + 1, ns])
            if vals:
                m = max(vals)
                beta[t, s] = m + math.log(sum(math.exp(x - m) for x in vals))

    post = np.zeros((T, C))
    for t in range(T):
        for s in range(S):
            post[t, labels[s]] += math.exp(alpha[t, s] + beta[t, s] - log_z)
    return post


def test_posteriors() -> None:
    print("== 1. ctc_posteriors 与逐状态参考实现一致 ==")
    rng = np.random.default_rng(7)
    cases = [
        (4, 3, [1, 2]),
        (6, 4, [1, 1, 2]),
        (5, 3, [2]),
        (7, 5, [3, 1, 3, 1]),
        (8, 6, [4, 4, 4]),
        (9, 4, [1, 2, 3, 1]),
        (12, 5, [2, 2, 2, 2]),
    ]
    worst = 0.0
    for T, C, tgt in cases:
        p = rng.random((T, C))
        p /= p.sum(axis=1, keepdims=True)
        lp = np.log(p)
        ref = _ref_fb(lp, np.asarray(tgt))
        got = ctc_posteriors(lp, np.asarray(tgt))
        worst = max(worst, float(np.abs(ref - got).max()))
    check(worst < 1e-12, f"{len(cases)} 组随机用例", f"max|参考-实现| = {worst:.3e}")


def test_posteriors_invariants() -> None:
    print("== 2. 后验不变量 ==")
    rng = np.random.default_rng(11)
    T, C, tgt = 30, 8, [1, 1, 5, 3, 3, 7]
    p = rng.random((T, C)) * 0.3 + 0.05
    p /= p.sum(axis=1, keepdims=True)
    post = ctc_posteriors(np.log(p), np.asarray(tgt))
    check(bool(np.all(post >= -1e-15)), "非负", f"min = {post.min():.3e}")
    check(float(np.abs(post.sum(1) - 1).max()) < 1e-12, "逐帧归一化", f"max|Σ-1| = {np.abs(post.sum(1) - 1).max():.3e}")
    total = post[:, np.asarray(tgt)].sum(0)
    check(bool(np.all(total > 0)), "每个目标 token 都有后验质量", f"min = {total.min():.3e}")
    # 注意：tgt 含重复 token，post[:, tgt].sum() 会把同一标签重复计入，
    # 所以这个恒等式必须按**唯一标签**求和。
    uniq = np.unique(np.asarray(tgt))
    mass = post[:, 0].sum() + post[:, uniq].sum()
    check(
        abs(float(mass) - T) < 1e-9,
        "blank 帧数 + 唯一标签帧数 = T",
        f"{post[:, 0].sum():.3f} + {post[:, uniq].sum():.3f} = {mass:.3f} vs T={T}",
    )


def test_post_duration_recovers_known_layout() -> None:
    print("== 3. 后验时长能还原已知注入位置 ==")
    T, C = 40, 6
    tgt = np.array([1, 3, 5, 2])
    places = [5, 13, 24, 33]
    # 背景概率压得很低，才能看出后验是否收敛到注入位置；
    # 若背景接近均匀（1/C），后验会按先验摊开，时长特征自然变糊。
    # 背景必须由 blank 主导。注意 log_probs 逐帧归一化，所以「把非 blank 概率
    # 设成 1e-4」并不能得到稀疏背景——归一化后它就变成均匀的 1/C，后验会合理地
    # 摊成 T/C（实测 6.58 ≈ 40/6）。要构造可分辨的场景，得让 blank 本身占 ~0.99。
    eps = 1e-3
    p = np.full((T, C), eps)
    p[:, 0] = 1.0 - eps * (C - 1)
    for t, tok in zip(places, tgt):
        p[t, 0] = 0.5
        p[t, tok] = 0.5
    p = p / p.sum(1, keepdims=True)
    post = ctc_posteriors(np.log(p), tgt)
    dur = post[:, tgt].sum(0)
    check(
        bool(np.all((dur > 0.9) & (dur < 2.0))),
        "后验时长收敛到 1 帧附近（blank 主导背景）",
        f"{np.round(dur, 2).tolist()}",
    )
    for j, (c, tok) in enumerate(zip(places, tgt)):
        w = post[:, tok]
        # 注意：注入帧只有一帧、背景极稀疏时后验总质量≈1 且摊得很薄，
        # 质心会落在 T/2 附近，没有意义；要检查的是后验的**峰**位置。
        check(int(w.argmax()) == c, f"token {tok} 的后验峰落在注入位置 {c}", f"argmax = {int(w.argmax())}")
    ratio = [post[t, tok] / max(post[:, tok].max(), 1e-12) for t, tok in zip(places, tgt)]
    check(bool(np.all(np.asarray(ratio) > 0.99)), "注入帧即后验峰值", f"min ratio = {min(ratio):.3f}")


def test_viterbi_matches_posterior_peak() -> None:
    print("== 4. Viterbi 路径落在后验峰上 ==")
    rng = np.random.default_rng(3)
    T, C, tgt = 50, 10, [2, 5, 2, 8, 1, 1]
    # 同样要让背景 blank 主导，否则均匀背景下 Viterbi 的最优解不唯一
    centres = [6, 14, 22, 30, 38, 46]
    eps = 1e-3
    p = np.full((T, C), eps)
    p[:, 0] = 1.0 - eps * (C - 1)
    for c, tok in zip(centres, tgt):
        for t in range(max(0, c - 2), min(T, c + 3)):
            p[t, 0] = 0.3
            p[t, tok] = 0.7
    p = p / p.sum(1, keepdims=True)
    lp = np.log(p)
    path = forced_align(lp, np.asarray(tgt))
    post = ctc_posteriors(lp, np.asarray(tgt))
    # 全部 6 个 token 都必须被发射，且每个 token 都要落在某个同标签的峰附近。
    # 注意 tgt 里 label 2 出现两次、label 1 出现两次（相邻重复），最优解允许把
    # 两个重复 token 都放进同一个峰，所以按「同标签的任一峰」判定而不是指定峰。
    emitted = [(t, v) for t, v in enumerate(path) if v]
    check(len(emitted) == len(tgt), "Viterbi 发射了全部 token", f"{len(emitted)}/{len(tgt)}")
    peaks = {tok: [c for c, tk in zip(centres, tgt) if tk == tok] for tok in set(tgt)}
    bad = [
        (t, tgt[v - 1], peaks[tgt[v - 1]])
        for t, v in emitted
        if not any(abs(t - c) <= 3 for c in peaks[tgt[v - 1]])
    ]
    check(not bad, "每个 token 都落在同标签的峰附近", f"异常={bad}")
    # 后验时长的排序应与注入的间隔一致（这里间隔均匀，只检查总量）
    uniq = np.unique(np.asarray(tgt))
    total = post[:, uniq].sum()
    check(
        abs(float(post[:, 0].sum() + total.sum()) - T) < 1e-9,
        "后验时长自洽",
        f"{post[:, 0].sum():.3f} + {total.sum():.3f} vs T={T}",
    )


def test_alignment_rejections() -> None:
    print("== 5. 不可行对齐被显式拒绝 ==")
    rng = np.random.default_rng(5)
    C = 10
    p = rng.random((30, C))
    p /= p.sum(1, keepdims=True)
    lp = np.log(p)

    try:
        align_and_score(lp[:3], np.arange(1, 9))
        check(False, "帧数 < token 数应拒绝", "却返回了结果")
    except AlignmentError as e:
        check(e.reason == REASON_TOO_SHORT, "帧数 < token 数", f"reason={e.reason} detail={e.detail}")

    try:
        align_and_score(np.zeros((0, C)), np.arange(1, 4))
        check(False, "空音频应拒绝", "却返回了结果")
    except AlignmentError as e:
        check(e.reason == "empty_audio", "空音频", f"reason={e.reason}")

    try:
        align_and_score(lp, np.array([]))
        check(False, "空目标应拒绝", "却返回了结果")
    except AlignmentError as e:
        check(e.reason == "empty_target", "空目标", f"reason={e.reason}")

    # 每 mora 帧数离谱（T=300, n=3 → 100 帧/mora > 40）
    try:
        align_and_score(np.log(np.full((300, C), 1 / C)), np.array([1, 2, 3]))
        check(False, "每 mora 帧数超界应拒绝", "却返回了结果")
    except AlignmentError as e:
        check(e.reason == REASON_RATIO, "每 mora 帧数超界", f"reason={e.reason} detail={e.detail}")


def test_mora_duration_is_not_degenerate() -> None:
    print("== 6. 逐 mora 时长特征有区分度 ==")
    rng = np.random.default_rng(13)
    C, T = 12, 120
    tgt = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # blank 主导的背景 + 每个 token 不同宽度，用来看后验时长能否还原宽度
    eps = 1e-3
    p = np.full((T, C), eps)
    p[:, 0] = 1.0 - eps * (C - 1)
    widths = [12, 8, 20, 6, 14, 10, 9, 11]
    centres = np.cumsum([0] + widths) + 4
    for c, w, tok in zip(centres, widths, tgt):
        for t in range(max(0, c - w // 2), min(T, c + w // 2)):
            p[t, 0] = 0.25
            p[t, tok] = 0.75
    p = p / p.sum(1, keepdims=True)
    aln = align_and_score(np.log(p), tgt, frame_ms=40.0)
    post = np.asarray(aln.post_durations)
    viterbi = np.asarray(aln.durations, dtype=float)
    check(float(post.std()) > 0.5, "后验时长有标准差", f"std = {post.std():.3f}, 值 = {np.round(post, 2).tolist()}")
    check(
        float(np.corrcoef(post, widths[: len(post)])[0, 1]) > 0.8,
        "后验时长与注入宽度相关",
        f"corr = {np.corrcoef(post, widths[: len(post)])[0, 1]:.3f}",
    )
    check(
        float(viterbi.std()) <= post.std(),
        "后验时长不劣于 Viterbi 时长",
        f"viterbi std={viterbi.std():.3f} (值 {viterbi.tolist()}) vs post std={post.std():.3f}",
    )
    check(
        bool(np.all(np.asarray(aln.post_rel_durations) > 0)),
        "相对后验时长为正",
    )


def test_repeated_token_durations() -> None:
    print("== 7. 重复 mora 的逐实例后验时长 ==")
    rng = np.random.default_rng(7)
    C, T = 10, 90
    # 大量重复标签：模拟日语的叠词、促音/长音连用
    tgt = np.array([1, 3, 1, 2, 3, 1, 4, 3, 5, 1, 3])
    lp = np.log(rng.dirichlet(np.ones(C) * 0.35, size=T))
    aln = align_and_score(lp, tgt, frame_ms=40.0)
    post = np.asarray(aln.post_durations)

    check(bool((post > 0).all()), "各实例后验时长为正", f"min={post.min():.4f}")
    check(float(post.max()) < T, "单实例后验时长不超过总帧数", f"max={post.max():.2f}, T={T}")

    # 回归点：同一标签的各实例不能拿到完全相同的值（旧实现按标签聚合取列）
    for tok in sorted(set(int(t) for t in tgt)):
        idx = [j for j in range(len(tgt)) if int(tgt[j]) == tok]
        if len(idx) < 2:
            continue
        vals = [round(float(post[j]), 6) for j in idx]
        check(len(set(vals)) == len(vals), f"token {tok} 的 {len(idx)} 个实例时长互不相同", f"{vals}")

    # 精确不变量：逐实例分解必须恰好划分该标签的总后验质量（不能重复计数）
    label_post = ctc_posteriors(lp, tgt)
    worst = 0.0
    for tok in set(int(t) for t in tgt):
        lhs = sum(float(post[j]) for j in range(len(tgt)) if int(tgt[j]) == tok)
        worst = max(worst, abs(lhs - float(label_post[:, tok].sum())))
    check(worst < 1e-9, "Σ实例时长 == 标签总后验（无重复计数）", f"最大偏差={worst:.2e}")

    # 反证：旧的按标签聚合写法会把整列复制给每个实例
    over = max(
        float(label_post[:, tok].sum()) * sum(1 for t in tgt if int(t) == tok) for tok in {1, 3}
    )
    check(
        float(post.sum()) < over,
        "总后验质量未被重复计数（旧写法会高估）",
        f"新实现 sum={post.sum():.2f} < 旧写法上界 {over:.2f}",
    )


def main() -> int:
    test_posteriors()
    test_posteriors_invariants()
    test_post_duration_recovers_known_layout()
    test_viterbi_matches_posterior_peak()
    test_alignment_rejections()
    test_mora_duration_is_not_degenerate()
    test_repeated_token_durations()
    print()
    if FAILED:
        print("FAIL:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("PASS: 前向后后验与逐状态参考一致；时长/相对时长特征可用；重复 mora 逐实例时长不重复计数；不可行对齐被显式拒绝")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

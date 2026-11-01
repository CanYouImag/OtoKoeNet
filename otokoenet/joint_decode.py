"""Char-Mora 序列级联合解码（阶段 19，零训练）。

为什么要序列级而不是用户方案里的「帧级 `score(t,c)=α·char + β·mora[proj(c)]`」：
`proj: char→单一 mora` 不成立 —— char 词表里 2545 个汉字是多音的（「日」= ひ/にち/じつ），
一个 char 无法映射到唯一 mora。因此改为在**候选序列**层面融合：

1. mora 头 beam 给出若干 kana 候选（强项，kana CER ~7%）；
2. char 头 beam 给出若干字符候选（当前生产完全没用到 char 头，greedy 汉字 CER ~31%）；
3. 每个字符/文本候选 `w` 用 pykakasi 读出假名→切 mora→`u`，对**两个头**分别算
   CTC 前向对数似然 `log P(w | char_head)` 与 `log P(u | mora_head)`；
4. converter 路线（mora beam → kana → kana2kanji）也作为候选进同一个池；
5. 统一按 `char_ll + β·mora_ll + γ·lm` 排序取最优。

`ctc_log_likelihood` 是标准 CTC forward（求和所有对齐），与 `forced_align` 的 Viterbi
不同：这里要的是「这个候选在音频下的总似然」，不是某条最优路径。
"""

from __future__ import annotations

import numpy as np

from otokoenet.decode import _log_softmax
from otokoenet.text import kana_to_mora

NEG = -1e30


def ctc_log_likelihood(logits: np.ndarray, target: list[int], blank_id: int = 0) -> float:
    """log P(target | logits)（CTC forward，所有对齐求和）。

    Args:
        logits: (T, V) 帧级 logits。
        target: 目标 token id（不含 blank）。
    Returns:
        对数似然；T < len(target) 时返回 `NEG`（不可行）。
    """
    if len(target) == 0:
        return NEG
    logp = _log_softmax(np.asarray(logits, dtype=np.float64))
    T, _ = logp.shape
    n = len(target)
    if T < n:
        return NEG

    S = 2 * n + 1
    labels = np.zeros(S, dtype=np.int64)
    for j, tok in enumerate(target):
        labels[2 * j + 1] = int(tok)

    # s -> s+2 只允许跳到**非 blank 且与 s-2 标签不同**的状态（Graves CTC 递归）。
    allow_skip = np.zeros(S, dtype=bool)
    if S > 2:
        allow_skip[2:] = (labels[2:] != blank_id) & (labels[:-2] != labels[2:])

    alpha = np.full(S, NEG)
    alpha[0] = logp[0, blank_id]
    alpha[1] = logp[0, int(target[0])]
    for t in range(1, T):
        prev = alpha
        cand = prev.copy()
        cand[1:] = np.logaddexp(cand[1:], prev[:-1])
        if S > 2:
            skip = np.where(allow_skip[2:], prev[:-2], NEG)
            cand[2:] = np.logaddexp(cand[2:], skip)
        alpha = cand + logp[t, labels]

    # 合法终止状态：最后一个 label（S-2=2n-1）或其后 blank（S-1=2n）
    return float(np.logaddexp(alpha[S - 2], alpha[S - 1]))


def text_to_mora_ids(text: str, mora_vocab, pykakasi_conv) -> list[int] | None:
    """文本 → pykakasi 假名 → mora id。含词表外 mora 时返回 None。"""
    kana = "".join(c["kana"] for c in pykakasi_conv.convert(text))
    ids: list[int] = []
    for m in kana_to_mora(kana):
        if not mora_vocab.has(m):
            return None
        ids.append(mora_vocab._sym2id[m])
    return ids or None


def char_ids(text: str, char_vocab) -> list[int] | None:
    """文本 → char id；词表外字符用 UNK（Vocab 保证有 <unk>）。"""
    try:
        return char_vocab.encode_with_unk(list(text)) or None
    except ValueError:
        return None

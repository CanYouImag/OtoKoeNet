from __future__ import annotations

import math

import numpy as np
import torch

from otokoenet.text import Vocab

_NEG = float("-inf")


def _logadd(a: float, b: float) -> float:
    """log(exp(a) + exp(b))，处理 -inf。"""
    if a == _NEG:
        return b
    if b == _NEG:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True) + 1e-12)


class NGramLM:
    """token id 级 n-gram 语言模型（backoff + add-δ 平滑），用于浅层融合。"""

    def __init__(self, order: int = 4, delta: float = 0.5) -> None:
        self.order = order
        self.delta = delta
        self.vocab_size = 0
        self.counts: list[dict[tuple[int, ...], int]] = [dict() for _ in range(order)]
        self._cache: dict[tuple[tuple, int], float] = {}

    def fit(self, sequences: list[list[int]], vocab_size: int) -> "NGramLM":
        self.vocab_size = vocab_size
        for seq in sequences:
            n = len(seq)
            for o in range(1, self.order + 1):
                cnt = self.counts[o - 1]
                for i in range(n - o + 1):
                    key = tuple(seq[i : i + o])
                    cnt[key] = cnt.get(key, 0) + 1
        self._cache.clear()
        return self

    def _prob_cond(self, hist: tuple[int, ...], token: int) -> float:
        for o in range(self.order, 1, -1):
            if len(hist) < o - 1:
                continue
            ctx = hist[-(o - 1) :] + (token,)
            cnt = self.counts[o - 1].get(ctx, 0)
            n_cntxt = self.counts[o - 2].get(hist[-(o - 1) :], 0)
            if n_cntxt > 0:
                return (cnt + self.delta) / (n_cntxt + self.delta * self.vocab_size)
        cnt = self.counts[0]
        c = cnt.get(token, 0)
        n_total = sum(cnt.values())
        return (c + self.delta) / (n_total + self.delta * self.vocab_size)

    def conditional_logp(self, context: list[int], token: int) -> float:
        key = (tuple(context), token)
        if key not in self._cache:
            v = math.log(self._prob_cond(tuple(context), token))
            if len(self._cache) > 200_000:
                self._cache.clear()
            self._cache[key] = v
        return self._cache[key]


def ctc_prefix_beam_search(
    logits: np.ndarray,
    blank_id: int = 0,
    beam_size: int = 12,
    lm: NGramLM | None = None,
    lm_weight: float = 1.0,
    length_penalty: float = 0.0,
    max_len: int = 200,
    topk: int | None = None,
    n_best: int = 1,
) -> list[tuple[list[int], float]]:
    """CTC 前缀束搜索（log 空间，支持 n-gram LM 浅层融合）。

    Args:
        logits: (T, V) 帧级 logits。blank 约定为 0。
        beam_size: 束宽。
        lm: n-gram 语言模型（作用于非 blank token 序列）。
        lm_weight: LM 融合权重。
        length_penalty: 每输出一个 token 附加的分数（可与 lm_weight 配合调长度）。
        max_len: 输出 token 数上限，防止束无限增长。
        topk: 每帧候选字符数（不含 blank），默认 max(4, min(V-1, beam_size*3))。
        n_best: 返回前 n 个结果。

    Returns:
        [(token_ids, score), ...]，score 已含 lm_weight / length_penalty。
    """
    T, V = logits.shape
    if T == 0:
        return [([], _NEG)] * max(1, min(n_best, 1))
    logp = _log_softmax(np.asarray(logits, dtype=np.float64))
    if topk is None:
        topk = max(4, min(V - 1, beam_size * 3))
    cands: list[list[int]] = []
    for t in range(T):
        idx = np.argsort(logp[t])[::-1]
        idx = idx[idx != blank_id][:topk]
        cands.append(idx.tolist())

    pb: dict[tuple, float] = {(): _NEG}
    pn: dict[tuple, float] = {(): 0.0}
    keys: list[tuple] = [()]

    for t in range(T):
        p_blank = logp[t, blank_id]
        new_pb: dict[tuple, float] = {}
        new_pn: dict[tuple, float] = {}
        for prefix in keys:
            b = pb[prefix]
            n = pn[prefix]
            total = _logadd(b, n)
            new_pb[prefix] = _logadd(new_pb.get(prefix, _NEG), total + p_blank)
            for c in cands[t]:
                p_c = logp[t, c]
                l_new = prefix + (c,)
                if prefix and c == prefix[-1]:
                    new_pn[prefix] = _logadd(new_pn.get(prefix, _NEG), n + p_c)
                    src = b
                else:
                    src = total
                if len(l_new) > max_len:
                    continue
                inc = 0.0
                if lm is not None and src > _NEG:
                    inc = lm_weight * lm.conditional_logp(list(prefix), c)
                new_pn[l_new] = _logadd(new_pn.get(l_new, _NEG), src + p_c + inc)
        scored = sorted(
            set(new_pb) | set(new_pn),
            key=lambda p: _logadd(new_pb.get(p, _NEG), new_pn.get(p, _NEG)) + length_penalty * len(p),
            reverse=True,
        )[:beam_size]
        if not scored:
            break
        pb = {p: new_pb.get(p, _NEG) for p in scored}
        pn = {p: new_pn.get(p, _NEG) for p in scored}
        keys = scored

    ranked = sorted(
        keys,
        key=lambda p: _logadd(pb.get(p, _NEG), pn.get(p, _NEG)) + length_penalty * len(p),
        reverse=True,
    )[:n_best]
    return [
        (list(p), _logadd(pb.get(p, _NEG), pn.get(p, _NEG)) + length_penalty * len(p))
        for p in ranked
    ]


def ctc_collapse(logits: torch.Tensor) -> list[int]:
    ids = logits.argmax(dim=-1).tolist()
    out: list[int] = []
    prev = -1
    for i in ids:
        if i != 0 and i != prev:
            out.append(i)
        prev = i
    return out


def build_lexicon(*manifests) -> dict[int, list[dict]]:
    """按 mora 长度分桶的语料词典：mora_ids -> char_ids。"""
    index: dict[int, list[dict]] = {}
    seen = set()
    for m in manifests:
        for e in m.entries:
            key = tuple(e["mora_ids"])
            if key in seen:
                continue
            seen.add(key)
            index.setdefault(len(e["mora_ids"]), []).append(
                {"mora_ids": list(e["mora_ids"]), "char_ids": list(e["char_ids"])}
            )
    return index


def nearest_kanji(hyp_mora: list[int], lexicon: dict[int, list[dict]], max_len_diff: int = 3) -> list[int]:
    """mora 序列 → 语料词典最近邻 → 汉字序列（FST 融合思路）。"""
    best_ids, _ = nearest_match(hyp_mora, lexicon, max_len_diff)
    return best_ids


def nearest_match(
    hyp_mora: list[int], lexicon: dict[int, list[dict]], max_len_diff: int = 3
) -> tuple[list[int], int]:
    """最近邻 + 编辑距离。供门控判断是否命中已知句子。"""
    best_ids, best_d, _ = nearest_top2(hyp_mora, lexicon, max_len_diff)
    return best_ids, best_d


def nearest_top2(
    hyp_mora: list[int], lexicon: dict[int, list[dict]], max_len_diff: int = 3
) -> tuple[list[int], int, int]:
    """返回最近邻 (char_ids, best_d, second_d)。second_d=最近邻与次近邻的距离。"""
    best_ids: list[int] = []
    best_d = 1 << 30
    second_d = 1 << 30
    l = len(hyp_mora)
    for l2 in range(max(0, l - max_len_diff), l + max_len_diff + 1):
        for e in lexicon.get(l2, []):
            d = edit_distance(hyp_mora, e["mora_ids"])
            if d < best_d:
                second_d = best_d
                best_d = d
                best_ids = e["char_ids"]
            elif d < second_d:
                second_d = d
    return best_ids, best_d, second_d


def edit_distance(ref: list[int], hyp: list[int]) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    lexicon: list[dict] | None = None,
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    total_char = 0
    err_char_ctc = 0
    err_char_fst = 0
    total_mora = 0
    err_mora = 0
    for feat_pad, feat_len, char_pad, char_len, mora_pad, mora_len in loader:
        feat_pad = feat_pad.to(device)
        feat_len = feat_len.to(device)
        out = model(feat_pad, feat_len)
        char_logits, mora_logits, out_len = out[0], out[1], out[2]
        for b in range(feat_pad.size(0)):
            ref_char = char_pad[b, : char_len[b]].tolist()
            total_char += len(ref_char)
            hyp_char_ctc = ctc_collapse(char_logits[b, : out_len[b]])
            err_char_ctc += edit_distance(ref_char, hyp_char_ctc)
            hyp_mora = ctc_collapse(mora_logits[b, : out_len[b]])
            if lexicon is not None:
                hyp_char_fst = nearest_kanji(hyp_mora, lexicon)
                err_char_fst += edit_distance(ref_char, hyp_char_fst)
            ref_mora = mora_pad[b, : mora_len[b]].tolist()
            total_mora += len(ref_mora)
            err_mora += edit_distance(ref_mora, hyp_mora)
    return {
        "cer_ctc": err_char_ctc / total_char if total_char else float("nan"),
        "cer_fst": err_char_fst / total_char if total_char else float("nan"),
        "mer": err_mora / total_mora if total_mora else float("nan"),
    }

from __future__ import annotations

import torch

from otokoenet.text import Vocab


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

"""端到端验收：走**后端生产代码路径**（`backend/app/ml/engine.py` 的 `Engine`），
回答两个产品问题：

1. 给定文本 + 音频，能否判断「音频是否就是这段文本」，并指出哪个 mora 不对？
   - `Engine.evaluate()` = 逐 mora CTC 强制对齐 + 后验分数，与 `/api/evaluate` 同一条路。
   - 判据：配错整句时的拒绝/判对率（Cohen's d）、**mora 定位准确率**（把参考里
     第 k 个 mora 换成近音 mora 后，分数最低的实例是否就是 k；随机基线 = 1/n_mora）。
2. 任意音频能否识别出说了什么？
   - `Engine.recognize()` = featurize(在线 fbank+CMVN) + beam 解码 + 句库最近邻/
     kana2kanji，与 `/api/recognize` 同一条路。
   - 判据：假名 CER、汉字 CER、已知句命中率；另加退化条件（噪声/截断/变速）下的
     假名 CER，用来回答「任意音频」而不是「JSUT 干净录音」。

与 `scripts/eval_5a.py`（离线批量口径，用缓存特征 + 自己解码）的区别：这里**不
复用 eval_5a 的解码**，而是调用 `Engine` 本身，因此测的是线上真实行为，包含
在线特征管线、部署默认解码参数（beam24/lm0.2）和 `min_mean_logprob` 阈值。

注意：mora 定位实验是「标签被替换」的代理实验，不等于人工发音数据；与人工评分的
相关性仍然不可验证（阶段 10 结论，见 log/stage10_pronun_metrics.json）。

    python scripts/eval_engine_e2e.py \
        --ckpt runs/jsut_full_stage14/best.pt \
        --cache-dir data/cache/jsut_full_v1 \
        --splits test probe --degrade --out log/stage14_engine_e2e.json
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
from otokoenet.align import AlignmentError, suspect_long_vowels  # noqa: E402
from otokoenet.data import Manifest  # noqa: E402
from otokoenet.decode import edit_distance  # noqa: E402
from otokoenet.text import kana_to_mora, normalize_kana, normalize_text  # noqa: E402

MORA_SPECIAL = ("っ", "ン", "ー")


def mora_group(sym: str) -> str:
    """与 scripts/eval_pronun.py 同一套分组口径（促音/拗音/長音/普通音節）。"""
    if sym == "っ":
        return "促音"
    if sym == "ー":
        return "長音"
    if sym and len(sym) >= 2 and sym[-1] in "ゃゅょャュョ":
        return "拗音"
    return "普通音節"


def edit_ops(ref: list[str], hyp: list[str]) -> list[tuple[str, int, int]]:
    """token 级 Levenshtein 回溯，返回 ('sub'|'del', ref_i, hyp_j) / ('ins', -1, hyp_j)。

    用来把识别错误归因到**参考侧**的具体 mora（develop.md 要求单独看促音/拗音/
    长音）。插入没有参考侧归属，单独统计。
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                if cost:
                    ops.append(("sub", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))
            i -= 1
            continue
        ops.append(("ins", -1, j - 1))
        j -= 1
    return ops


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


class Rate:
    __slots__ = ("n", "chars", "err")

    def __init__(self) -> None:
        self.n = 0
        self.chars = 0
        self.err = 0

    def add(self, ref: str, hyp: str) -> int:
        d = edit_distance(list(ref), list(hyp))
        self.n += 1
        self.chars += len(ref)
        self.err += d
        return d

    def rate(self) -> float:
        return self.err / self.chars if self.chars else float("nan")

    def fmt(self) -> str:
        return f"{self.rate() * 100:.2f}%"


def sub_mora(sym: str, vocab_syms: list[str]) -> str | None:
    """把一个 mora 换成「近音」的另一个 mora（确定性，不引入随机数）。

    分组规则与 log/README.md 阶段 10 一致：促音/长音各自成组；拗音按尾音
    （きょ→しょ，同一类韵尾、不同声母）；普通音节按首假名（か→き）。
    换不到就返回 None（该句跳过）。
    """
    if sym in MORA_SPECIAL:
        pool = [s for s in vocab_syms if s in MORA_SPECIAL and s != sym]
    elif len(sym) >= 2 and sym[-1] in "ゃゅょ":
        pool = [s for s in vocab_syms if len(s) >= 2 and s[-1] == sym[-1] and s != sym]
    else:
        pool = [s for s in vocab_syms if s[0] == sym[0] and s != sym]
    pool = sorted(set(pool))
    return pool[0] if pool else None


def make_settings(ckpt: str, cache: str) -> Settings:
    s = Settings()
    s.ckpt_path = Path(ckpt).resolve()
    s.cache_dir = Path(cache).resolve()
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/jsut_full_stage14/best.pt")
    ap.add_argument("--cache-dir", default="data/cache/jsut_full_v1")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument(
        "--degrade",
        action="store_true",
        help="额外跑退化条件（噪声/截断/变速）下的识别与打分",
    )
    ap.add_argument("--num-examples", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    settings = make_settings(args.ckpt, args.cache_dir)
    engine = Engine(settings)
    cache = Path(args.cache_dir)

    print(f"ckpt={settings.ckpt_path.name} step={engine.ckpt_step} cache={cache.name}")
    print(
        f"decoder={settings.decoder} beam={settings.beam_size} lm_weight={settings.lm_weight} "
        f"min_mean_logprob={settings.min_mean_logprob}"
    )
    print(f"n_char={len(engine.char_vocab)} n_mora={len(engine.mora_vocab)}")
    print(f"[!] 端到端延迟为单线程量级（{args.num_threads} 线程），不是生产并发数")

    consistency = engine.verify_cache_consistency(n_utt=5)
    print(
        f"cache 一致性: max|cache - featurize| = {consistency['max_abs_diff']:.3e} "
        f"({'PASS' if consistency['ok'] else 'FAIL'})"
    )

    vocab_syms = [engine.mora_vocab.decode([i])[0] for i in range(len(engine.mora_vocab))]
    report: dict = {
        "ckpt": str(settings.ckpt_path),
        "ckpt_step": engine.ckpt_step,
        "cache_dir": str(cache),
        "engine": engine.manifest(),
        "cache_consistency": consistency,
        "settings": {
            "decoder": settings.decoder,
            "beam_size": settings.beam_size,
            "lm_weight": settings.lm_weight,
            "min_mean_logprob": settings.min_mean_logprob,
            "num_threads": args.num_threads,
        },
        "splits": {},
    }

    for split in args.splits:
        manifest = Manifest.load(str(cache / f"{split}.json"))
        entries = list(manifest.entries[: args.limit])
        print(f"\n=== {split} n={len(entries)} ===")

        refs = [
            {
                "text": normalize_text(e["text"]),
                "kana": "".join(kana_to_mora(normalize_kana(e["kana"]))),
            }
            for e in entries
        ]
        # 配错文本的负样本：按 mora 长度就近取另一句（长度不同会把拒绝对齐的难度
        # 混进分数里，量到的主要是长度而不是「是否这段话」）
        by_len = sorted(range(len(entries)), key=lambda i: len(entries[i]["mora_ids"]))
        partner: dict[int, int] = {}
        for pos, i in enumerate(by_len):
            partner[i] = by_len[min(pos + 1, len(by_len) - 1)]

        st_kana, st_kanji = Rate(), Rate()
        matched_score, mismatched_score = [], []
        matched_mlp, mismatched_mlp = [], []
        reject_matched = reject_mismatch = 0
        loc_hit = loc_n = 0
        loc_hit_gop = 0
        loc_rand = 0.0
        sub_gap: list[float] = []
        sub_gap_gop: list[float] = []
        # 長音规则 A/B：用识别结果里「ー」是否被替换/漏掉当标签，量规则的召回与误报。
        long_n = long_err = long_flag = long_tp = long_fp = 0
        t_feat: list[float] = []
        t_rec: list[float] = []
        worst: list[tuple[int, str, str]] = []
        examples: list[dict] = []
        group_stat: dict[str, dict[str, float]] = {}
        confusions: dict[str, int] = {}
        n_ins = 0
        n_ref_mora = 0

        for i, e in enumerate(entries):
            feat = engine.featurize(e["wav"])
            t0 = time.time()
            hyp_text = engine.recognize(feat)  # /api/recognize 同一条路
            t_rec.append(time.time() - t0)
            _, mora_logits = engine._encode(feat)
            t_feat.append(time.time() - t0)
            hyp_ids = engine.decode_mora(mora_logits)
            hyp_kana = "".join(engine.mora_vocab.decode(hyp_ids))
            d_kana = st_kana.add(refs[i]["kana"], hyp_kana)
            st_kanji.add(refs[i]["text"], hyp_text)
            worst.append((d_kana, refs[i]["kana"], hyp_kana))

            # 识别错误归因到参考侧 mora（develop.md：促音/拗音/长音要单独看）
            ref_mora = kana_to_mora(normalize_kana(e["kana"]))
            hyp_mora = list(engine.mora_vocab.decode(hyp_ids))
            for sym in ref_mora:
                group_stat.setdefault(mora_group(sym), {"n": 0.0, "sub": 0.0, "del": 0.0})
                group_stat[mora_group(sym)]["n"] += 1
            n_ref_mora += len(ref_mora)
            for op, ri, hj in edit_ops(ref_mora, hyp_mora):
                if op == "ins":
                    n_ins += 1
                    continue
                d = group_stat[mora_group(ref_mora[ri])]
                d[op] += 1
                if op == "sub":
                    key = f"{ref_mora[ri]}->{hyp_mora[hj]}"
                    confusions[key] = confusions.get(key, 0) + 1
            if len(examples) < args.num_examples:
                examples.append(
                    {
                        "utt": e["utt"],
                        "ref_text": refs[i]["text"],
                        "ref_kana": refs[i]["kana"],
                        "hyp_kana": hyp_kana,
                        "hyp_text": hyp_text,
                        "d_kana": d_kana,
                    }
                )

            # ---- 任务 1：指定文本 vs 音频 ----
            target = list(e["mora_ids"])
            try:
                aln = engine.evaluate_detailed(feat, target)
                matched_score.append(aln.total)
                matched_mlp.append(aln.mean_logprob)

                # 長音规则：参考侧符号取自 target（与 aln 的下标一一对应）；
                # 标签「这个 ー 是否真的发错」用识别结果里它是否被 sub/del 近似。
                ref_syms_aln = [engine.mora_vocab.decode([target[j]])[0] for j in range(len(target))]
                long_err_idx = {
                    ri
                    for op, ri, _ in edit_ops(ref_syms_aln, hyp_mora)
                    if op in ("sub", "del") and ref_syms_aln[ri] == "ー"
                }
                flagged = set(suspect_long_vowels(ref_syms_aln, aln))
                for mi, sym in enumerate(ref_syms_aln):
                    if sym != "ー":
                        continue
                    long_n += 1
                    is_err = mi in long_err_idx
                    is_flag = mi in flagged
                    long_err += int(is_err)
                    long_flag += int(is_flag)
                    long_tp += int(is_err and is_flag)
                    long_fp += int((not is_err) and is_flag)
            except AlignmentError as ex:
                reject_matched += 1
                matched_score.append(0.0)
                matched_mlp.append(float("-99"))
                examples.append({"utt": e["utt"], "rejected_matched": ex.reason})
            neg = list(entries[partner[i]]["mora_ids"])
            try:
                neg_aln = engine.evaluate_detailed(feat, neg)
                mismatched_score.append(neg_aln.total)
                mismatched_mlp.append(neg_aln.mean_logprob)
            except AlignmentError:
                reject_mismatch += 1
                mismatched_score.append(0.0)
                mismatched_mlp.append(float("-99"))

            # mora 定位：把第 k 个 mora 换成近音 mora，看最低分是否落在 k
            if len(target) >= 8:
                k = (i * 7 + 3) % len(target)
                sym = engine.mora_vocab.decode([target[k]])[0]
                alt = sub_mora(sym, vocab_syms)
                if alt is not None and alt in vocab_syms:
                    sub_target = list(target)
                    sub_target[k] = vocab_syms.index(alt)
                    try:
                        saln = engine.evaluate_detailed(feat, sub_target)
                        hit = int(np.argmin(saln.scores)) == k
                        loc_hit += int(hit)
                        loc_n += 1
                        loc_rand += 1.0 / len(target)
                        others = [
                            s for j, s in enumerate(saln.scores) if j != k
                        ]
                        sub_gap.append(float(np.mean(others) - saln.scores[k]))
                        # GOP A/B：同一个注入实验，改用 GOP 排序看 top-1 是否更好。
                        # GOP 越高越像发对，所以同样取 argmin。
                        if saln.gops:
                            loc_hit_gop += int(np.argmin(saln.gops) == k)
                            others_g = [g for j, g in enumerate(saln.gops) if j != k]
                            sub_gap_gop.append(float(np.mean(others_g) - saln.gops[k]))
                    except AlignmentError:
                        pass

        def acc_at(thr: float, pos: list[float], neg: list[float]) -> float:
            """把 pos/neg 判成二分类（pos 高于 thr 即判「相符」）的准确率。"""
            ok = sum(1 for v in pos if v >= thr) + sum(1 for v in neg if v < thr)
            return ok / max(1, len(pos) + len(neg))

        def best_acc(pos: list[float], neg: list[float]) -> tuple[float, float]:
            cands = sorted(set([round(v, 4) for v in pos + neg]))
            best, bthr = 0.0, float("nan")
            for thr in cands:
                a = acc_at(thr, pos, neg)
                if a > best:
                    best, bthr = a, thr
            return best, bthr

        b_acc, b_thr = best_acc(matched_score, mismatched_score)
        # 生产规则：align_and_score 用整句平均 log 概率与 min_mean_logprob 比，
        # 低于阈值直接判「与参考文本明显不符」（/api/evaluate 返回 4xx）。
        thr_mlp = settings.min_mean_logprob
        prod_acc = (
            sum(1 for v in matched_mlp if v >= thr_mlp)
            + sum(1 for v in mismatched_mlp if v < thr_mlp)
        ) / max(1, len(matched_mlp) + len(mismatched_mlp))

        s = {
            "n": len(entries),
            "recognize": {
                "kana_cer": st_kana.rate(),
                "kanji_cer": st_kanji.rate(),
                "latency_recognize_p50": pct(t_rec, 0.5),
                "latency_recognize_p95": pct(t_rec, 0.95),
                "latency_featurize_encode_p95": pct(t_feat, 0.95),
            },
            "verify": {
                "matched_total_mean": float(np.mean(matched_score)) if matched_score else None,
                "matched_total_p05": pct(matched_score, 0.05),
                "matched_mlp_p05": pct(matched_mlp, 0.05),
                "matched_mlp_max": max(matched_mlp) if matched_mlp else None,
                "mismatched_total_mean": (
                    float(np.mean(mismatched_score)) if mismatched_score else None
                ),
                "mismatched_total_p95": pct(mismatched_score, 0.95),
                "mismatched_mlp_max": max(mismatched_mlp) if mismatched_mlp else None,
                "reject_matched": reject_matched,
                "reject_mismatched": reject_mismatch,
                "acc_optimal_score": b_acc,
                "acc_optimal_threshold": b_thr,
                "acc_production_rule": prod_acc,
                "production_threshold_mean_logprob": thr_mlp,
            },
            "mora_localization": {
                "n": loc_n,
                "top1_acc": loc_hit / loc_n if loc_n else None,
                "top1_acc_gop": loc_hit_gop / loc_n if loc_n else None,
                "random_baseline": loc_rand / loc_n if loc_n else None,
                "mean_score_gap": float(np.mean(sub_gap)) if sub_gap else None,
                "mean_gop_gap": float(np.mean(sub_gap_gop)) if sub_gap_gop else None,
            },
            "long_vowel_rule": {
                # 标签来自识别结果（参考侧「ー」是否被 sub/del），规则只用对齐分数与
                # 后验时长。召回 = 真发错的 ー 里被判出的比例；误报 = 发对的 ー 里被标红
                # 的比例。域内 JSUT 是正确录音，长音天然读对，这个标签量的是
                # 「模型没听成长音的实例」，是「漏读長音」的可用代理。
                "n": long_n,
                "n_labeled_error": long_err,
                "n_flagged": long_flag,
                "recall": long_tp / long_err if long_err else None,
                "precision": long_tp / long_flag if long_flag else None,
                "false_positive_rate": long_fp / (long_n - long_err) if long_n - long_err else None,
            },
            "verify_raw": {
                # 供 scripts/fit_calibration.py 拟合 Platt scaling：固定阈值 -3.0
                # 在域外已失效，需要按目标拒识率重新标定工作点。
                "matched_mlp": [round(float(v), 5) for v in matched_mlp],
                "mismatched_mlp": [round(float(v), 5) for v in mismatched_mlp],
                "matched_total": [round(float(v), 5) for v in matched_score],
                "mismatched_total": [round(float(v), 5) for v in mismatched_score],
            },
            "mora_error_by_group": {
                g: {
                    "n_ref": int(v["n"]),
                    "n_sub": int(v["sub"]),
                    "n_del": int(v["del"]),
                    "err_rate": (v["sub"] + v["del"]) / v["n"] if v["n"] else None,
                }
                for g, v in sorted(group_stat.items())
            },
            "n_ref_mora": n_ref_mora,
            "n_insertion": n_ins,
            "top_confusions": dict(
                sorted(confusions.items(), key=lambda kv: -kv[1])[:20]
            ),
            "examples": examples,
        }
        print(f"  识别: kana CER {st_kana.fmt()} | 汉字 CER {st_kanji.fmt()}")
        print(
            f"  延迟: recognize p50 {pct(t_rec, 0.5) * 1000:.0f}ms "
            f"p95 {pct(t_rec, 0.95) * 1000:.0f}ms"
        )
        print(
            f"  指定文本: 匹配分均值 {s['verify']['matched_total_mean']:.3f} / "
            f"配错 {(s['verify']['mismatched_total_mean'] or 0):.3f}；"
            f"最优阈值准确率 {b_acc * 100:.2f}%，生产规则 {prod_acc * 100:.2f}%"
        )
        print(
            f"  拒绝: 正确文本被拒 {reject_matched}/{len(entries)}，"
            f"配错文本被识别出来 {reject_mismatch}/{len(entries)}"
        )
        if loc_n:
            ml = s["mora_localization"]
            print(
                f"  mora 定位: score top1 {ml['top1_acc'] * 100:.1f}% | "
                f"GOP top1 {ml['top1_acc_gop'] * 100:.1f}% "
                f"（随机基线 {ml['random_baseline'] * 100:.1f}%），"
                f"均分差 {ml['mean_score_gap']:.3f} / GOP {ml['mean_gop_gap']:.3f}"
            )
        lv = s["long_vowel_rule"]
        if lv["n"]:
            rec = f"{lv['recall'] * 100:.1f}%" if lv["recall"] is not None else "n/a"
            fpr = (
                f"{lv['false_positive_rate'] * 100:.1f}%"
                if lv["false_positive_rate"] is not None
                else "n/a"
            )
            print(
                f"  長音规则: 标红 {lv['n_flagged']}/{lv['n']}，"
                f"召回 {rec}（真错 {lv['n_labeled_error']}）误报 {fpr}"
            )
        print(
            "  识别错误按参考 mora 分组: "
            + " ".join(
                f"{g} {v['err_rate'] * 100:.2f}%(n={v['n_ref']})"
                for g, v in s["mora_error_by_group"].items()
            )
            + f" | 插入 {n_ins}"
        )

        if args.degrade:
            deg = {}
            rng = np.random.default_rng(0)
            for cond in ("clean", "noise_snr5db", "noise_snr0db", "truncated_60pct", "speed_1.1"):
                st = Rate()
                n_ok = 0
                rej = 0
                for e in entries[: min(len(entries), 100)]:
                    feat = engine.featurize(e["wav"])
                    if cond == "noise_snr5db":
                        feat = feat + rng.normal(
                            0, feat.std() / 10 ** (5 / 20), feat.shape
                        ).astype(np.float32)
                    elif cond == "noise_snr0db":
                        feat = feat + rng.normal(
                            0, feat.std() / 10 ** (0 / 20), feat.shape
                        ).astype(np.float32)
                    elif cond == "truncated_60pct":
                        feat = feat[: max(1, int(feat.shape[0] * 0.6))]
                    elif cond == "speed_1.1":
                        # 线性插值重采样 = 近似变速（不改变特征维度语义）
                        n = feat.shape[0]
                        m = max(1, int(n / 1.1))
                        xi = np.linspace(0, n - 1, m)
                        feat = np.stack(
                            [np.interp(xi, np.arange(n), feat[:, d]) for d in range(feat.shape[1])],
                            axis=1,
                        ).astype(np.float32)
                    _, ml = engine._encode(feat)
                    hyp = "".join(engine.mora_vocab.decode(engine.decode_mora(ml)))
                    ref = "".join(kana_to_mora(normalize_kana(e["kana"])))
                    st.add(ref, hyp)
                    try:
                        aln = engine.evaluate_detailed(feat, list(e["mora_ids"]))
                        n_ok += int(aln.mean_logprob >= settings.min_mean_logprob)
                    except AlignmentError:
                        rej += 1
                n_deg = min(len(entries), 100)
                deg[cond] = {
                    "n": n_deg,
                    "kana_cer": st.rate(),
                    "verify_accept_rate": n_ok / n_deg,
                    "verify_rejected": rej,
                }
                print(
                    f"  [退化] {cond:16s} kana CER {deg[cond]['kana_cer'] * 100:6.2f}% "
                    f"正确文本仍被接受 {deg[cond]['verify_accept_rate'] * 100:6.2f}% "
                    f"对齐拒绝 {rej}/{n_deg}"
                )
            s["degrade"] = deg

        s["worst_kana"] = [
            {"d": d, "ref": r, "hyp": h} for d, r, h in sorted(worst, reverse=True)[:5]
        ]
        report["splits"][split] = s

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
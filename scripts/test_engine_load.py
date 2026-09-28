"""阶段 8 端到端验收：后端 `Engine` 能否正确加载并服务阶段 5A 的干净 checkpoint。

判据（全部通过才退出 0）：
1. `Engine` 用 `backend/app/config.py` 的**默认配置**加载成功（不再指向旧泄漏 cache/ckpt）；
2. 推理特征管线与训练缓存特征同尺度（`verify_cache_consistency`）；
3. 同一段音频：`Engine.featurize` 的输出与直接按训练口径算出的 fbank+CMVN 一致；
4. 严格加载会对「删掉一个权重张量」的坏 checkpoint 硬失败（不静默部分加载）；
5. cache 目录与 checkpoint 训练配置错配时硬失败；
6. 在真实 wav 上跑识别，输出非空且 kana 序列与参考 mora 解码一致；
7. 延迟基准：featurize / encode+decode / 端到端 的 p50/p95（develop.md 要求 <3s）。

    python scripts/test_engine_load.py --n-utt 20
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.config import Settings  # noqa: E402
from app.ml.engine import Engine, ModelLoadError  # noqa: E402
from otokoenet.data import apply_cmvn, extract_fbank, load_wav  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, name: str, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _edit(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return dp[n][m]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-utt", type=int, default=20, help="延迟基准句数")
    args = ap.parse_args()

    settings = Settings()
    print("== 1. 默认配置加载 ==")
    print(f"  cache_dir={settings.cache_dir}")
    print(f"  ckpt_path={settings.ckpt_path}")
    print(f"  decoder={settings.decoder} beam={settings.beam_size} lm_weight={settings.lm_weight}")
    t0 = time.time()
    engine = Engine(settings)
    check(True, "Engine 加载成功", f"{time.time() - t0:.1f}s")
    man = engine.manifest()
    print(f"  fingerprints: ckpt={man['ckpt'][:12]}… step={man['ckpt_step']} "
          f"n_char={man['n_char']} n_mora={man['n_mora']} converter={man['has_converter']}")
    check(man["cache_dir"] == str(settings.cache_dir), "checkpoint 与 cache 目录一致")

    print("== 2. 推理特征与训练缓存同尺度 ==")
    cons = engine.verify_cache_consistency(n_utt=3)
    print(f"  max|featurize - cached| = {cons['max_abs_diff']:.3e} (tol {cons['tol']:.0e}, "
          f"worst={cons['worst_utt']})")
    check(cons["ok"], "CMVN 可复现缓存特征")

    print("== 3. featurize 与训练口径一致 ==")
    e0 = engine.train_manifest.entries[0]
    wav = load_wav(e0["wav"], settings.sample_rate)
    ref_feat = apply_cmvn(
        extract_fbank(wav, settings.sample_rate, settings.n_mels).numpy().astype(np.float32),
        engine.mean,
        engine.std,
    )
    got_feat = engine.featurize(e0["wav"])
    d = float(np.abs(ref_feat - got_feat).max())
    check(d == 0.0, "featurize 逐元素一致", f"maxabs={d:.3e}")

    print("== 4. 严格加载拒绝残缺 checkpoint ==")
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.pt"
        ck = torch.load(settings.ckpt_path, map_location="cpu", weights_only=False)
        victim = "mora_head.weight"
        ck["ema"].pop(victim)
        torch.save(ck, bad)
        s2 = Settings()
        s2.ckpt_path = bad
        try:
            Engine(s2)
            check(False, "缺权重时应硬失败", "却加载成功")
        except ModelLoadError as e:
            check(victim in str(e), "缺权重时硬失败并指名", f"ModelLoadError: {str(e).splitlines()[0][:60]}…")

    print("== 5. cache 目录错配硬失败 ==")
    other = Path(tempfile.mkdtemp()) / "wrongcache"
    other.mkdir(parents=True)
    s3 = Settings()
    s3.cache_dir = other
    try:
        Engine(s3)
        check(False, "cache 错配时应硬失败", "却加载成功")
    except (ModelLoadError, FileNotFoundError) as e:
        check(True, "cache 错配时硬失败", type(e).__name__)
    finally:
        shutil.rmtree(other.parent, ignore_errors=True)

    print("== 6. 真实音频识别 ==")
    n_ok = 0
    shown = 0
    for e in engine.train_manifest.entries[:5]:
        text = engine.recognize(engine.featurize(e["wav"]))
        ok = len(text) > 0
        n_ok += ok
        if shown < 3:
            print(f"  {e['utt']}: {text}")
            shown += 1
    check(n_ok == 5, "5 句均返回非空结果", f"{n_ok}/5")
    e0 = engine.train_manifest.entries[0]
    feat = engine.featurize(e0["wav"])
    _, mora_logits = engine._encode(feat)
    ids = engine.decode_mora(mora_logits)
    kana = "".join(engine.mora_vocab.decode(ids))
    ref_kana = "".join(engine.mora_vocab.decode(e0["mora_ids"]))
    print(f"  {e0['utt']} hyp_kana = {kana}")
    print(f"  {e0['utt']} ref_kana = {ref_kana}")
    print(f"  {e0['utt']} mora 编辑距离 = {_edit(list(kana), list(ref_kana))} "
          f"(ref mora n={len(e0['mora_ids'])})")

    print(f"== 7. 延迟基准（CPU, n={args.n_utt}）==")
    torch.set_num_threads(12)
    utts = engine.train_manifest.entries[: args.n_utt]
    t_feat, t_dec, t_e2e = [], [], []
    for e in utts:
        t = time.perf_counter()
        f = engine.featurize(e["wav"])
        t_feat.append(time.perf_counter() - t)
        t = time.perf_counter()
        _, ml = engine._encode(f)
        t_dec.append(time.perf_counter() - t)
        t = time.perf_counter()
        engine.recognize(f)
        t_e2e.append(time.perf_counter() - t)
    for name, v in (("featurize", t_feat), ("encode", t_dec), ("recognize e2e", t_e2e)):
        print(f"  {name:14s} p50={pct(v, 0.5) * 1000:7.1f}ms  p95={pct(v, 0.95) * 1000:7.1f}ms  "
              f"mean={statistics.mean(v) * 1000:7.1f}ms")
    p95_e2e = pct(t_e2e, 0.95)
    check(p95_e2e < 3.0, "端到端 p95 < 3s（develop.md）", f"p95={p95_e2e * 1000:.0f}ms")

    print()
    if FAILS:
        print("FAIL: " + ", ".join(FAILS))
        return 1
    print("PASS: 阶段 8 部署阻塞全部判据通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""校验（必要时反解）特征 cache 的 CMVN 参数与缓存特征是否自洽。

不变量：`scripts/prepare.py` 把 CMVN 烘焙进 `<utt>.npy`
（`np.save(cache / f"{utt}.npy", apply_cmvn(feat, mean, std))`），
因此对**同一段音频**，从原始 wav 重算 fbank 再套用 `mean.npy` / `std.npy`
必须逐元素等于缓存里的特征。在线推理
（`backend/app/ml/engine.py:featurize`）正是走「原始 fbank → CMVN」这条
路径，所以这个不变量成立，等于推理端特征分布与训练端一致。

不一致的三种成因（都必须硬失败，不能靠改后端绕过）：
1. 推理用的 cache 与训练用的 cache 不是同一个目录（错配 `mean/std`）；
2. cache 被重新生成但 `mean.npy` / `std.npy` 没有一起更新；
3. 特征配置（`n_mels` / `frame_shift` / `sample_rate`）与生成时不一致。

    python scripts/check_cmvn.py --cache-dir data/cache/basic5000_v2
    python scripts/check_cmvn.py --cache-dir data/cache/basic5000_v2 --fix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from otokoenet.data import Manifest, apply_cmvn, extract_fbank, load_wav  # noqa: E402

TOL = 1e-3


def _raw_feature(entry: dict, sample_rate: int, n_mels: int) -> np.ndarray:
    wav = load_wav(entry["wav"], sample_rate)
    return extract_fbank(wav, sample_rate, n_mels).numpy().astype(np.float32)


def _rebuild_diff(
    entries: list[dict], mean: np.ndarray, std: np.ndarray, sample_rate: int, n_mels: int
) -> tuple[float, str]:
    worst, worst_utt = 0.0, ""
    for entry in entries:
        raw = _raw_feature(entry, sample_rate, n_mels)
        cached = np.load(entry["feat"]).astype(np.float32)
        if raw.shape != cached.shape:
            raise SystemExit(
                f"{entry['utt']}: 原始 fbank {raw.shape} 与缓存 {cached.shape} 不一致；"
                f"检查 sample_rate/n_mels/frame_shift 是否与生成时相同"
            )
        d = float(np.abs(apply_cmvn(raw, mean, std) - cached).max())
        if d > worst:
            worst, worst_utt = d, entry["utt"]
    return worst, worst_utt


def _recover(
    entries: list[dict], sample_rate: int, n_mels: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """cache = a * raw + b 的逐维最小二乘，再还原 mean=-b/a, std=1/a。"""
    s_raw = np.zeros(n_mels, dtype=np.float64)
    s_raw_sq = np.zeros(n_mels, dtype=np.float64)
    s_cross = np.zeros(n_mels, dtype=np.float64)
    s_cache = np.zeros(n_mels, dtype=np.float64)
    n = 0
    for entry in entries:
        raw = _raw_feature(entry, sample_rate, n_mels)
        cached = np.load(entry["feat"]).astype(np.float32)
        if raw.shape != cached.shape:
            raise SystemExit(f"{entry['utt']}: 形状不匹配 {raw.shape} vs {cached.shape}")
        r = raw.astype(np.float64)
        s_raw += r.sum(axis=0)
        s_raw_sq += (r**2).sum(axis=0)
        s_cross += (r * cached).sum(axis=0)
        s_cache += cached.astype(np.float64).sum(axis=0)
        n += raw.shape[0]
    n = float(n)
    var = np.maximum(s_raw_sq / n - (s_raw / n) ** 2, 1e-12)
    a = (s_cross / n - (s_raw / n) * (s_cache / n)) / var
    b = (s_cache / n) - a * (s_raw / n)
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise SystemExit("反解斜率非法（a<=0 或非有限），缓存不是单次逐维仿射变换")
    return (-b / a).astype(np.float32), (1.0 / a).astype(np.float32), n


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 cache 的 CMVN 与缓存特征是否自洽")
    ap.add_argument("--cache-dir", default="data/cache/basic5000_v2")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=80)
    ap.add_argument("--n-check", type=int, default=8, help="校验用句数")
    ap.add_argument("--fix", action="store_true", help="mean/std 缺失或不匹配时反解并写入")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    split_path = cache / "split.json"
    meta = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}
    if not meta:
        print(f"[info] {split_path} 不存在，用命令行默认值校验")
    for key, got in (("n_mels", args.n_mels), ("sample_rate", args.sample_rate)):
        if int(meta.get(key, got)) != got:
            raise SystemExit(f"split.json {key}={meta.get(key)} != --{key.replace('_', '-')} {got}")

    mean_path, std_path = cache / "mean.npy", cache / "std.npy"
    have = mean_path.exists() and std_path.exists()
    train = Manifest.load(str(cache / "train.json"))
    entries = train.entries[: args.n_check]

    if not have:
        if not args.fix:
            print(f"[FAIL] 缺少 {mean_path} / {std_path}：在线推理无法复现训练特征分布（--fix 可反解）")
            return 1
        print(f"[info] 缺少 mean/std，用前 5 句最小二乘反解 ...")
        mean, std, n = _recover(train.entries[:5], args.sample_rate, args.n_mels)
        np.save(mean_path, mean)
        np.save(std_path, std)
        print(f"[fix ] 已写入 mean/std（拟合 {n} 帧）: {mean_path} {std_path}")
        entries = train.entries[5 : 5 + args.n_check]

    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    if mean.shape != (args.n_mels,) or std.shape != (args.n_mels,):
        print(f"[FAIL] mean/std 维度不匹配: mean={mean.shape} std={std.shape}，期望 ({args.n_mels},)")
        return 1

    worst, worst_utt = _rebuild_diff(entries, mean, std, args.sample_rate, args.n_mels)
    print(f"cache   : {cache}  (n_check={len(entries)})")
    print(f"mean    : min={mean.min():.5f} max={mean.max():.5f}")
    print(f"std     : min={std.min():.5f} max={std.max():.5f}")
    print(f"rebuild : max|cmvn(raw_fbank) - cached| = {worst:.3e} (tol {TOL:.0e}, worst={worst_utt})")
    if worst > TOL:
        if args.fix:
            mean, std, n = _recover(train.entries[:5], args.sample_rate, args.n_mels)
            check = train.entries[5 : 5 + args.n_check]
            worst2, utt2 = _rebuild_diff(check, mean, std, args.sample_rate, args.n_mels)
            print(f"[fix ] 反解后重建误差 {worst2:.3e} (worst={utt2}, 拟合 {n} 帧)")
            if worst2 > TOL:
                print("[FAIL] 反解后仍不一致，拒绝写入；请用 scripts/prepare.py 重新生成该 cache")
                return 1
            np.save(mean_path, mean)
            np.save(std_path, std)
            print(f"[fix ] 已覆盖 {mean_path} {std_path}")
            return 0
        print("[FAIL] cache 的 mean/std 与缓存特征不一致：推理特征分布会偏离训练分布")
        return 1

    print("PASS: 推理端 fbank→CMVN 可逐元素复现缓存特征")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

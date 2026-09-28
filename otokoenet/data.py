from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_wav(path: str, sample_rate: int) -> torch.Tensor:
    try:
        wav, sr = sf.read(path, dtype="float32")
    except Exception:
        wav, sr = _decode_ffmpeg(path, sample_rate)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav)
    if sr != sample_rate:
        wav_t = torchaudio.functional.resample(wav_t, sr, sample_rate)
    return wav_t


def _decode_ffmpeg(path: str, sample_rate: int) -> tuple[np.ndarray, int]:
    """ffmpeg 回退解码：任意音频格式 → float32 单声道。"""
    import subprocess

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"cannot decode {path}: soundfile 不支持该格式且未找到 ffmpeg，请安装 ffmpeg"
        ) from None
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"audio decode failed for {path}: {err}")
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        raise RuntimeError(f"audio decode produced no samples for {path}")
    return wav, sample_rate


def extract_fbank(wav: torch.Tensor, sample_rate: int, n_mels: int) -> torch.Tensor:
    feat = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        num_mel_bins=n_mels,
        frame_length=25.0,
        frame_shift=10.0,
        sample_frequency=sample_rate,
    )
    return feat


def compute_cmvn(feat_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(feat_list[0].shape[1], dtype=np.float64)
    sq = np.zeros_like(sums)
    n = 0.0
    for f in feat_list:
        sums += f.sum(axis=0)
        sq += (f**2).sum(axis=0)
        n += f.shape[0]
    mean = sums / n
    var = sq / n - mean**2
    var = np.maximum(var, 1e-5)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def apply_cmvn(feat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (feat - mean) / std


def cmvn_reproduces(
    raw: np.ndarray, cached: np.ndarray, mean: np.ndarray, std: np.ndarray, tol: float = 1e-3
) -> bool:
    """`mean`/`std` 是否能逐元素复现缓存特征（即该 cache 是否已烘焙这组 CMVN）。"""
    if raw.shape != cached.shape:
        return False
    return bool(np.abs(apply_cmvn(raw, mean, std) - cached).max() <= tol)


def cache_cmvn_is_baked(
    cache_dir: Path, wav_dir: Path, n_mels: int, sample_rate: int, n_utt: int = 3
) -> bool | None:
    """探测某个特征 cache 的 `<utt>.npy` 是否已烘焙它自己的 `mean.npy`/`std.npy`。

    Returns:
        True  已烘焙（`mean.npy`/`std.npy` 可逐元素复现缓存特征）
        False 未烘焙（`mean.npy`/`std.npy` 存在但与缓存特征不一致）
        None  无法判定（缺 mean/std、缺 wav 或形状不一致）

    用于阻止 `scripts/prepare.py` 的复用路径把「已归一化」的特征再归一化一次。
    """
    mean_path, std_path = cache_dir / "mean.npy", cache_dir / "std.npy"
    if not (mean_path.exists() and std_path.exists()):
        return None
    mean = np.load(mean_path)
    std = np.load(std_path)
    checked = 0
    for path in sorted(cache_dir.glob("*.npy")):
        if path.name in ("mean.npy", "std.npy"):
            continue
        wav = wav_dir / f"{path.stem}.wav"
        if not wav.exists():
            continue
        cached = np.load(path)
        if cached.ndim != 2 or cached.shape[1] != n_mels:
            continue
        raw = extract_fbank(load_wav(str(wav), sample_rate), sample_rate, n_mels).numpy()
        if not cmvn_reproduces(raw.astype(np.float32), cached.astype(np.float32), mean, std):
            return False
        checked += 1
        if checked >= n_utt:
            return True
    return None


class SpecAugment:
    def __init__(
        self,
        n_time_masks: int = 2,
        max_time_width: int = 30,
        n_freq_masks: int = 1,
        max_freq_width: int = 27,
    ) -> None:
        self.n_time_masks = n_time_masks
        self.max_time_width = max_time_width
        self.n_freq_masks = n_freq_masks
        self.max_freq_width = max_freq_width

    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        x = feat
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.max_freq_width)
            f0 = random.randint(0, max(0, x.size(2) - f))
            x[..., f0 : f0 + f] = 0.0
        for _ in range(self.n_time_masks):
            t = random.randint(0, self.max_time_width)
            t0 = random.randint(0, max(0, x.size(1) - t))
            x[:, t0 : t0 + t, :] = 0.0
        return x


class Manifest:
    def __init__(self, entries: list[dict], root: str) -> None:
        self.entries = entries
        self.root = Path(root)

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["entries"], data["root"])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"root": str(self.root), "entries": self.entries}, f, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        return self.entries[idx]


def load_entry(entry: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feat = np.load(entry["feat"])
    char_ids = torch.tensor(entry["char_ids"], dtype=torch.long)
    mora_ids = torch.tensor(entry["mora_ids"], dtype=torch.long)
    return torch.from_numpy(feat), char_ids, mora_ids


def speed_perturb_feature(feat: torch.Tensor, factor: float) -> torch.Tensor:
    """特征域速度扰动：在时间轴上按 1/factor 线性重采样，近似「语速×factor」。"""
    t = feat.size(0)
    new_t = int(round(t / factor))
    if new_t <= 0 or abs(new_t - t) < 2:
        return feat
    f = feat.permute(1, 0).unsqueeze(0)  # (1, C, T)
    f = torch.nn.functional.interpolate(f, size=new_t, mode="linear", align_corners=False)
    return f.squeeze(0).permute(1, 0)


class CollateFn:
    def __init__(self, speed_perturb: bool = False, speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1)):
        self.speed_perturb = speed_perturb
        self.speed_factors = speed_factors

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        feats, chars, moras = zip(*batch)
        if self.speed_perturb:
            feats = [
                speed_perturb_feature(f, random.choice(self.speed_factors)) if self.speed_factors else f
                for f in feats
            ]
        T = max(f.size(0) for f in feats)
        C = feats[0].size(1)
        feat_pad = torch.zeros(len(feats), T, C)
        for i, f in enumerate(feats):
            feat_pad[i, : f.size(0)] = f
        feat_len = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
        char_len = torch.tensor([c.size(0) for c in chars], dtype=torch.long)
        mora_len = torch.tensor([m.size(0) for m in moras], dtype=torch.long)
        char_pad = torch.zeros(len(chars), int(char_len.max()), dtype=torch.long)
        mora_pad = torch.zeros(len(moras), int(mora_len.max()), dtype=torch.long)
        for i, c in enumerate(chars):
            char_pad[i, : c.size(0)] = c
        for i, m in enumerate(moras):
            mora_pad[i, : m.size(0)] = m
        return feat_pad, feat_len, char_pad, char_len, mora_pad, mora_len


class BucketedBatchSampler:
    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True, seed: int = 0) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

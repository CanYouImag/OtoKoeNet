"""波形级数据增强：MUSAN 噪声混合 + RIR 混响（方案二.4 的真版本）。

为什么必须在**波形级**：训练读的是 `scripts/prepare.py` 预先烘焙的缓存特征，
`otokoenet/data.py:CollateFn` 只能做特征域操作（SpecAugment / speed_perturb /
特征域加噪）。真实噪声与混响是在原始波形上发生的（加性噪声的谱形状、RIR 的卷积
畸变），必须在 `extract_fbank` 之前施加，这也是它们能改善「任意音频」的前提。

本模块只提供纯函数，`prepare.py` 负责遍历、按 utt 播种与落盘。所有函数接受/返回
`torch.Tensor`（单声道、float32、指定采样率）。

设计约束：
  - **只用 train**：val/test/probe 必须保持干净，否则验收数字不可比（`prepare.py`
    里由调用方保证）。
  - **可复现**：不用全局 `random`/`numpy` 随机源，调用方传入 `random.Random`；
    `prepare.py` 用 `sha256(augment_seed:utt)` 当种子，同一 utt 每次得到同一增强。
  - **不引入新依赖**：只用 soundfile / torchaudio / torch（仓库已依赖）。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}


def index_audio(root: str | Path | None) -> list[Path]:
    """递归收集目录下的音频文件（MUSAN/RIR 都可能有嵌套子目录）。"""
    if not root:
        return []
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"增强目录不存在: {p}")
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt().clamp_min(1e-8)


def crop_or_tile(x: torch.Tensor, n: int, rng: random.Random) -> torch.Tensor:
    """把噪声裁到/平铺到长度 n（随机起点）。"""
    if x.numel() >= n:
        start = rng.randrange(0, x.numel() - n + 1)
        return x[start : start + n]
    reps = n // max(1, x.numel()) + 1
    return x.repeat(reps)[:n]


def read_wav(path: str | Path, sample_rate: int) -> torch.Tensor:
    """整段读取（RIR 用；RIR 很短，不需要分段）。"""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    t = torch.from_numpy(np.asarray(wav).mean(axis=1))
    if sr != sample_rate:
        t = torchaudio.functional.resample(t, sr, sample_rate)
    return t


def read_noise_segment(
    path: str | Path, sample_rate: int, n: int, rng: random.Random
) -> torch.Tensor:
    """只读噪声文件的一段（随机偏移），避免把几分钟的音乐整段载入内存。

    MUSAN 里不少 music/speech 文件是数分钟的，7026 句 × 整段载入会反复分配大块内存。
    这里用 soundfile 的 seek 直接读到目标偏移。
    """
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        need = int(round(n * sr / sample_rate)) + 1
        if f.frames > need:
            f.seek(rng.randrange(0, f.frames - need + 1))
        else:
            f.seek(0)
        data = f.read(need if f.frames > need else -1, dtype="float32", always_2d=True)
    t = torch.from_numpy(np.asarray(data).mean(axis=1))
    if sr != sample_rate:
        t = torchaudio.functional.resample(t, sr, sample_rate)
    return crop_or_tile(t, n, rng)


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """按目标 SNR 混合：20*log10(rms(clean)/rms(noise_scaled)) = snr_db。

    用 RMS 而不是峰值，与 `scripts/eval_engine_e2e.py` 的退化实验口径一致。
    """
    target = rms(clean) / (10.0 ** (snr_db / 20.0))
    mixed = clean + noise * (target / rms(noise))
    return mixed


def apply_rir(wav: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """RIR 卷积 + 截回原长。

    归一化用 **单位能量（L2, ||h||=1）**：这是 ESPnet/SpeechBrain 的标准做法，卷积前后
    平均能量守恒（Parseval），语音不会被整体削弱或放大。峰值归一化会让长 RIR 的多个
    抽头把输出放大 ~sum|h| 倍（实测合成 RIR 把 0.1 级输入放大到 37）；L1 归一化则相反，
    会把直达声稀释到近乎静音。输出若仍越界，`augment_waveform` 末尾有峰值保护。
    """
    rir = rir - rir.mean()
    energy = rir.pow(2).sum().sqrt()
    if float(energy) < 1e-8:
        return wav
    rir = rir / energy
    y = torch.nn.functional.conv1d(
        wav.view(1, 1, -1), rir.view(1, 1, -1), padding=rir.numel() - 1
    )
    return y.view(-1)[: wav.numel()]


def _peak_normalize(x: torch.Tensor, ceiling: float = 0.99) -> torch.Tensor:
    peak = x.abs().max()
    if float(peak) > ceiling:
        x = x * (ceiling / peak)
    return x


def augment_waveform(
    wav: torch.Tensor,
    sample_rate: int,
    noise_files: list[Path],
    rir_files: list[Path],
    rng: random.Random,
    *,
    noise_prob: float = 0.3,
    rir_prob: float = 0.3,
    snr_db_range: tuple[float, float] = (5.0, 20.0),
) -> torch.Tensor:
    """对单条波形施加 RIR（可选）再叠加噪声（可选）。

    顺序：先混响、后加噪。真实声学里混响是房间对**干净源**的响应，背景噪声在传声器
    处叠加；反过来（先加噪再卷积）会把噪声也加上混响，偏离 MUSAN+RIR 的标准做法。
    """
    y = wav
    if rir_files and rng.random() < rir_prob:
        rir = read_wav(rng.choice(rir_files), sample_rate)
        if rir.numel() > 1:
            y = apply_rir(y, rir)
    if noise_files and rng.random() < noise_prob:
        noise = read_noise_segment(rng.choice(noise_files), sample_rate, y.numel(), rng)
        if noise.numel() > 0:
            y = mix_at_snr(y, noise, rng.uniform(*snr_db_range))
    return _peak_normalize(y)


def utt_rng(seed: int, utt: str) -> random.Random:
    """按 (seed, utt) 播种的确定性 RNG。不用内置 hash（有进程级盐，不可复现）。"""
    import hashlib

    h = hashlib.sha256(f"{seed}:{utt}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:8], "big"))

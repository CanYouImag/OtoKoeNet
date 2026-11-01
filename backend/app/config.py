from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Settings:
    repo_root: Path = REPO_ROOT
    # 生产默认 = 阶段 14（JSUT 全语料 40 epoch，val/test 选模，train-only 词表/句库/CMVN）。
    # 指纹：ckpt_step=27249, n_char=2701, n_mora=189, 7 个文件 sha 见
    # log/README.md §3「当前基线」。切换后 /api/health 会回报这组指纹，可用 curl 核对。
    # 旧值 basic5000_v2 / basic5000_stage05a 是阶段 1–4 的历史基线，只作对照组。
    cache_dir: Path = REPO_ROOT / "data" / "cache" / "jsut_full_v1"
    ckpt_path: Path = REPO_ROOT / "runs" / "jsut_full_stage14" / "best.pt"
    sample_rate: int = 16000
    n_mels: int = 80

    score_green: float = 0.75
    score_yellow: float = 0.45

    # 解码配置。lm_weight/beam_size 取自阶段 5A 在 **validation** 上的扫描结果
    # （log/stage05a_acceptance.log: beam=24, lm_weight=0.2 → mora MER 6.40%），
    # 阶段 14 在自己的 val 上复核一致（beam=24, lm=0.2 → mora MER 6.67%，见
    # log/stage16_eval5a_val.log）。旧默认 lm_weight=1.0 在 val 上把 mora MER
    # 恶化到 12.66%，不得再作为默认值；lm 在本数据上的全部可用增益只有
    # +0.03pp（6.70% → 6.67%），LM 不是瓶颈，不要再花时间重建 n-gram LM。
    decoder: str = "beam"             # greedy | beam
    beam_size: int = 24
    lm_order: int = 4
    lm_weight: float = 0.2
    length_penalty: float = 0.0

    # Char-Mora 联合解码（阶段 19，零训练）。生产 recognize() 用 mora 头 beam + converter
    # 时汉字 CER test 32.02%；把 char 头候选与 mora 头候选放进同一池、按
    # `char_ll + β·mora_ll + γ·lm` 重排后 test 27.62% / probe 26.00%（β/γ 只在 val 上选）。
    # 需要 pykakasi 读汉字候选的假名；缺失时自动回退到 converter 路线并打印一次警告。
    # 旧路径保留在 recognize() 的 except 分支里，便于 OTOKOE_JOINT_DECODE=0 回退对照。
    joint_decode: bool = os.environ.get("OTOKOE_JOINT_DECODE", "1") != "0"
    joint_beta: float = 1.0   # mora 头对数似然权重（val 选出）
    joint_gamma: float = 0.0  # mora n-gram LM 权重（val 选出）
    joint_n_best: int = 8
    joint_char_beam: int = 8

    # checkpoint 严格加载：任何缺失 / 形状不符 / cache 目录与 checkpoint 训练配置不符
    # 都在启动时硬失败。设 OTOKOE_ALLOW_CACHE_MISMATCH=1 可临时放行 cache 目录错配。
    # 逐 mora 对齐的时间分辨率：fbank frame_shift=10ms 经 4x 下采样 = 40ms/帧。
    # 与 otokoenet/model.py 的 ConvSubsampling 保持一致，改这里必须同时改模型。
    frame_ms: float = 40.0
    # 整句平均 log 概率下界：低于此值判为「与参考文本明显不符」，显式拒绝而非返回 0 分。
    # -3.0 来自 stage 10 在 val 的实测间隔：真实录音最差 -1.84，配错文本最好 -5.91，
    # 取对数中点 -3.3 并偏向宽松侧，避免把「学得好但读得不流利」误判成读错。
    # 阶段 14 复核（log/stage14_engine_e2e.json）：test 正确句最高 -0.0056、
    # test 配错句最高 -0.2795，-3.0 仍然成立；但域外 probe 上正确句最低到
    # -0.0100、配错句最高到 -0.0008，两侧已经压到同一数量级，阈值在域外
    # 基本失效（probe 98 个 SNR0dB 样本里 96 个被误判为相符）。
    min_mean_logprob: float = float(os.environ.get("OTOKOE_MIN_MEAN_LOGPROB", "-3.0"))

    strict_load: bool = os.environ.get("OTOKOE_STRICT_LOAD", "1") != "0"
    allow_cache_mismatch: bool = os.environ.get("OTOKOE_ALLOW_CACHE_MISMATCH", "0") == "1"

    # OOD（输入不可信）检测（阶段 19）。默认 **关闭**：detector 能把 SNR0dB 假接受
    # 98%→0%，但 probe clean 误拒会涨到 10.5%（域外正确音频被误杀），按「probe 不退化
    # 才切换」的规矩不设为默认。用 OTOKOE_OOD=1 开启，detector 参数由
    # scripts/train_ood.py 落盘。
    ood_enabled: bool = os.environ.get("OTOKOE_OOD", "0") == "1"
    ood_detector_path: str = os.environ.get(
        "OTOKOE_OOD_DETECTOR",
        str(REPO_ROOT / "data" / "cache" / "jsut_full_v1" / "ood_detector.json"),
    )


settings = Settings()

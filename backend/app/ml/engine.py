from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from otokoenet.align import AlignmentError, MoraAlignment, align_and_score
from otokoenet.data import Manifest, apply_cmvn, extract_fbank, load_wav
from otokoenet.decode import NGramLM, build_lexicon, ctc_collapse, ctc_prefix_beam_search, nearest_top2
from otokoenet.kana2kanji import Kana2Kanji
from otokoenet.model import DualCTC
from otokoenet.text import Vocab


class ModelLoadError(RuntimeError):
    """checkpoint / cache / 词表不满足部署前提。启动时必须直接失败。"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Engine:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.ckpt_path = Path(settings.ckpt_path)
        if not self.ckpt_path.exists():
            raise ModelLoadError(f"checkpoint 不存在: {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.ckpt = ckpt
        self.ckpt_step = int(ckpt.get("step", -1))
        cfg = ckpt["config"]

        # cache 必须与 checkpoint 的训练 cache 一致：CMVN 错配会让推理特征分布
        # 与训练分布不一致，且错配在指标上不可见（评测走缓存特征）。
        ckpt_cache = Path(cfg.get("data", {}).get("cache_dir", ""))
        cache = Path(settings.cache_dir)
        if ckpt_cache.name and ckpt_cache.resolve() != cache.resolve():
            msg = (
                f"cache 目录与 checkpoint 训练配置不一致: ckpt={ckpt_cache} "
                f"settings={cache}。CMVN/词表/句库都会错配。"
            )
            if getattr(settings, "allow_cache_mismatch", False):
                print(f"[warn] {msg} 已由 OTOKOE_ALLOW_CACHE_MISMATCH=1 放行")
            else:
                raise ModelLoadError(
                    msg + " 请改 settings.cache_dir，或设 OTOKOE_ALLOW_CACHE_MISMATCH=1 临时放行。"
                )
        self.cache_dir = cache
        self.frame_ms = float(getattr(settings, "frame_ms", 40.0))

        self.char_vocab = Vocab.load(str(cache / "char_vocab.json"))
        self.mora_vocab = Vocab.load(str(cache / "mora_vocab.json"))
        self.mean = np.load(cache / "mean.npy")
        self.std = np.load(cache / "std.npy")
        if self.mean.shape != (settings.n_mels,) or self.std.shape != (settings.n_mels,):
            raise ModelLoadError(
                f"CMVN 维度不匹配: mean={self.mean.shape} std={self.std.shape} "
                f"n_mels={settings.n_mels}"
            )
        ckpt_mels = int(cfg.get("audio", {}).get("n_mels", settings.n_mels))
        if ckpt_mels != settings.n_mels:
            raise ModelLoadError(f"n_mels 不一致: ckpt={ckpt_mels} settings={settings.n_mels}")

        mcfg = cfg["model"]
        self.model = DualCTC(
            in_dim=settings.n_mels,
            d_model=mcfg["d_model"],
            n_layers=mcfg["n_layers"],
            n_heads=mcfg["n_heads"],
            ffn_dim=mcfg["ffn_dim"],
            conv_kernel=mcfg["conv_kernel"],
            dropout=mcfg.get("dropout", 0.1),
            n_char=len(self.char_vocab),
            n_mora=len(self.mora_vocab),
            ctc_gate=mcfg.get("ctc_gate", False),
            gate_min_weight=mcfg.get("gate_min_weight", 0.05),
        )
        state = ckpt.get("ema") or ckpt["model"]
        self.load_report = self._load_state(state, strict=getattr(settings, "strict_load", True))
        self.model.eval()

        self.train_manifest = Manifest.load(str(cache / "train.json"))
        self.lexicon = build_lexicon(self.train_manifest)
        table_path = cache / "kana2kanji.json"
        self.converter = Kana2Kanji(table_path) if table_path.exists() else None
        if self.converter is None:
            print(
                f"[warn] {table_path} 不存在：汉字输出不可用，识别只返回假名"
                "（用 scripts/build_kana2kanji.py 基于 train 转写构建）"
            )
        self.lm: NGramLM | None = None
        if settings.lm_weight > 0:
            seqs = [e["mora_ids"] for e in self.train_manifest.entries]
            self.lm = NGramLM(order=settings.lm_order).fit(seqs, vocab_size=len(self.mora_vocab))

        self.fingerprints = {
            "ckpt": _sha256(self.ckpt_path),
            "ckpt_step": self.ckpt_step,
            "char_vocab": _sha256(cache / "char_vocab.json"),
            "mora_vocab": _sha256(cache / "mora_vocab.json"),
            "mean": _sha256(cache / "mean.npy"),
            "std": _sha256(cache / "std.npy"),
            "cache_dir": str(cache),
            "n_char": len(self.char_vocab),
            "n_mora": len(self.mora_vocab),
            "has_converter": self.converter is not None,
            "decoder": settings.decoder,
            "beam_size": settings.beam_size,
            "lm_weight": settings.lm_weight,
            "loaded_tensors": self.load_report["loaded"],
        }

    # ---------- 加载与一致性 ----------

    def _load_state(self, state: dict, strict: bool) -> dict:
        """严格加载权重。

        旧实现对 shape 不匹配的键静默跳过（`engine.py` 原 37-44 行），这会让
        「部分加载」的模型静默产出无意义 logits。这里改为：缺失键、形状不符、
        多余的非 BN 统计键都硬失败并打印明细。
        """
        own = self.model.state_dict()
        missing, mismatched, skipped = [], [], []
        for name, dst in own.items():
            if "num_batches_tracked" in name:
                # EMA.update 不保存该键（train.py:EMA.update），恒用模型自身取值
                skipped.append(name)
                continue
            if name not in state:
                missing.append(name)
                continue
            src = state[name]
            if src.shape != dst.shape:
                mismatched.append(f"{name}: ckpt{tuple(src.shape)} vs model{tuple(dst.shape)}")
                continue
            dst.copy_(src)
        extra = [k for k in state if k not in own]
        report = {
            "loaded": len(own) - len(missing) - len(mismatched) - len(skipped),
            "missing": missing,
            "mismatched": mismatched,
            "skipped": skipped,
            "extra": extra,
        }
        if strict and (missing or mismatched):
            detail = [f"  missing: {m}" for m in missing[:8]]
            detail += [f"  shape : {m}" for m in mismatched[:8]]
            if len(missing) > 8 or len(mismatched) > 8:
                detail.append(f"  ... 共 missing={len(missing)} shape={len(mismatched)}")
            raise ModelLoadError(
                "checkpoint 与模型结构不匹配，拒绝静默部分加载：\n" + "\n".join(detail)
            )
        if report["skipped"] or report["extra"]:
            print(
                f"[info] 加载明细: loaded={report['loaded']} skipped(num_batches_tracked)="
                f"{len(report['skipped'])} extra={len(report['extra'])}"
            )
        return report

    def verify_cache_consistency(self, n_utt: int = 3, tol: float = 1e-3) -> dict:
        """校验推理特征管线与训练缓存特征同尺度（等价于 scripts/check_cmvn.py）。

        返回 `max_abs_diff`：`featurize()` 的输出与训练缓存 `<utt>.npy` 的最大逐元素差。
        该值远大于 0 说明 CMVN 与训练用的不是同一组，在线指标会与缓存指标严重脱节。
        """
        worst, worst_utt = 0.0, ""
        checked = 0
        for entry in self.train_manifest.entries:
            raw = extract_fbank(
                load_wav(entry["wav"], self.settings.sample_rate),
                self.settings.sample_rate,
                self.settings.n_mels,
            ).numpy().astype(np.float32)
            cached = np.load(entry["feat"]).astype(np.float32)
            d = float(np.abs(apply_cmvn(raw, self.mean, self.std) - cached).max())
            if d > worst:
                worst, worst_utt = d, entry["utt"]
            checked += 1
            if checked >= n_utt:
                break
        return {
            "max_abs_diff": worst,
            "worst_utt": worst_utt,
            "n_checked": checked,
            "tol": tol,
            "ok": worst <= tol,
        }

    def manifest(self) -> dict:
        """部署身份：checkpoint / 词表 / CMVN / 解码配置指纹（/api/health 用）。"""
        return dict(self.fingerprints)

    # ---------- 推理 ----------

    def featurize(self, wav_path: str) -> np.ndarray:
        wav = load_wav(wav_path, self.settings.sample_rate)
        feat = extract_fbank(wav, self.settings.sample_rate, self.settings.n_mels).numpy()
        return apply_cmvn(feat, self.mean, self.std)

    @torch.no_grad()
    def _encode(self, feat: np.ndarray):
        x = torch.from_numpy(feat).float().unsqueeze(0)
        feat_len = torch.tensor([feat.shape[0]], dtype=torch.long)
        char_logits, mora_logits, out_len = self.model(x, feat_len)[:3]
        return char_logits[0, : out_len[0]], mora_logits[0, : out_len[0]]

    def decode_mora(self, mora_logits) -> list[int]:
        if self.settings.decoder == "beam":
            mora_ids, _ = ctc_prefix_beam_search(
                mora_logits.detach().numpy(),
                beam_size=self.settings.beam_size,
                lm=self.lm if self.settings.lm_weight > 0 else None,
                lm_weight=self.settings.lm_weight,
                length_penalty=self.settings.length_penalty,
            )[0]
            return list(mora_ids)
        return ctc_collapse(mora_logits)

    def recognize(self, feat: np.ndarray) -> str:
        _, mora_logits = self._encode(feat)
        mora_ids = self.decode_mora(mora_logits)
        kana = "".join(self.mora_vocab.decode(mora_ids))
        char_ids, best_d, second_d = nearest_top2(mora_ids, self.lexicon)
        # 已知句判定：最近句明显胜出（margin>=1）且绝对距离不大
        if second_d - best_d >= 1 and best_d <= max(4, len(mora_ids) // 4):
            return "".join(self.char_vocab.decode(char_ids))
        if self.converter is not None:
            return self.converter.convert(kana)
        # 无转换表时返回假名：旧实现在此返回句库最近邻的汉字，离散度极高
        # （val kanji CER 86%），比直接给假名更糟。
        return kana

    def evaluate_detailed(self, feat: np.ndarray, mora_ids: list[int]) -> MoraAlignment:
        """逐 mora 强制对齐，返回实例级边界/分数/时长。

        Raises:
            AlignmentError: 对齐不可行（音频太短、静音、与参考文本明显不符等）。
                调用方应把它转成 4xx，而不是让它冒成 500。
        """
        _, mora_logits = self._encode(feat)
        lp = torch.log_softmax(mora_logits, dim=-1).numpy()
        target = np.asarray(mora_ids, dtype=np.int64)
        return align_and_score(
            lp,
            target,
            frame_ms=self.frame_ms,
            min_mean_logprob=self.settings.min_mean_logprob,
        )

    def evaluate(self, feat: np.ndarray, mora_ids: list[int]) -> tuple[list[float], float]:
        aln = self.evaluate_detailed(feat, mora_ids)
        return aln.scores, aln.total

from __future__ import annotations

import numpy as np
import torch

from otokoenet.align import forced_align, score_alignment
from otokoenet.data import Manifest, apply_cmvn, extract_fbank, load_wav
from otokoenet.decode import build_lexicon, ctc_collapse, nearest_top2
from otokoenet.kana2kanji import Kana2Kanji
from otokoenet.model import DualCTC
from otokoenet.text import Vocab


class Engine:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.char_vocab = Vocab.load(str(settings.cache_dir / "char_vocab.json"))
        self.mora_vocab = Vocab.load(str(settings.cache_dir / "mora_vocab.json"))
        self.mean = np.load(settings.cache_dir / "mean.npy")
        self.std = np.load(settings.cache_dir / "std.npy")

        ckpt = torch.load(settings.ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
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
        )
        state = ckpt.get("ema") or ckpt["model"]
        for name, dst in self.model.state_dict().items():
            if name not in state:
                continue
            src_t = state[name]
            if src_t.shape == dst.shape:
                dst.copy_(src_t)
            elif dst.dim() >= 2 and dst.shape[0] > src_t.shape[0] and dst.shape[1:] == src_t.shape[1:]:
                dst[: src_t.shape[0]].copy_(src_t)
        self.model.eval()

        cache = settings.cache_dir
        self.lexicon = build_lexicon(
            Manifest.load(str(cache / "train.json")),
            Manifest.load(str(cache / "test.json")),
        )
        table_path = cache / "kana2kanji.json"
        self.converter = Kana2Kanji(table_path) if table_path.exists() else None

    def featurize(self, wav_path: str) -> np.ndarray:
        wav = load_wav(wav_path, self.settings.sample_rate)
        feat = extract_fbank(wav, self.settings.sample_rate, self.settings.n_mels).numpy()
        return apply_cmvn(feat, self.mean, self.std)

    @torch.no_grad()
    def _encode(self, feat: np.ndarray):
        x = torch.from_numpy(feat).float().unsqueeze(0)
        feat_len = torch.tensor([feat.shape[0]], dtype=torch.long)
        char_logits, mora_logits, out_len = self.model(x, feat_len)
        return char_logits[0, : out_len[0]], mora_logits[0, : out_len[0]]

    def recognize(self, feat: np.ndarray) -> str:
        _, mora_logits = self._encode(feat)
        mora_ids = ctc_collapse(mora_logits)
        char_ids, best_d, second_d = nearest_top2(mora_ids, self.lexicon)
        # 已知句判定：最近句明显胜出（margin>=1）且绝对距离不大
        if second_d - best_d >= 1 and best_d <= max(4, len(mora_ids) // 4):
            return "".join(self.char_vocab.decode(char_ids))
        if self.converter is not None:
            kana = "".join(self.mora_vocab.decode(mora_ids))
            return self.converter.convert(kana)
        return "".join(self.char_vocab.decode(char_ids))

    def evaluate(self, feat: np.ndarray, mora_ids: list[int]) -> tuple[list[float], float]:
        _, mora_logits = self._encode(feat)
        lp = torch.log_softmax(mora_logits, dim=-1).numpy()
        target = np.asarray(mora_ids, dtype=np.int64)
        path = forced_align(lp, target)
        return score_alignment(lp, target, path)

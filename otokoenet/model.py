from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _mask_time(x: torch.Tensor, lengths: torch.Tensor, dim: int) -> torch.Tensor:
    t = x.size(dim)
    m = torch.arange(t, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
    shape = [1] * x.dim()
    shape[0] = x.size(0)
    shape[dim] = t
    return x.masked_fill(m.view(shape), 0.0)


class ConvSubsampling(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.proj = nn.Linear(64 * (in_dim // 4), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        lengths = None
        if mask is not None:
            lengths = (~mask).sum(dim=1)
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        if lengths is not None:
            x = _mask_time(x, torch.div(lengths + 1, 2, rounding_mode="floor"), dim=2)
        x = torch.relu(self.conv2(x))
        if lengths is not None:
            x = _mask_time(x, subsampled_lengths(lengths), dim=2)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, t, c * f)
        x = self.dropout(self.proj(x))
        if lengths is not None:
            x = _mask_time(x, subsampled_lengths(lengths), dim=1)
        return x


def subsampled_lengths(lengths: torch.Tensor) -> torch.Tensor:
    l1 = torch.div(lengths + 1, 2, rounding_mode="floor")
    return torch.div(l1 + 1, 2, rounding_mode="floor")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """只统计有效帧的 BatchNorm1d（Conformer 卷积模块用）。

    与 `nn.BatchNorm1d` 的差别：给定 `(B, C, T)` 与有效长度时，mean/var 只在
    有效帧上统计，且归一化后把 padding 位置重新置零。

    为什么必须这样做：卷积模块位于时间轴上，紧邻 padding 的**有效**帧的
    depthwise conv 会读到 padding 位置。若 padding 在归一化后变成非零（普通
    BatchNorm 会把 0 映射成 `-mean/std`），有效帧就会随 batch 里其它样本的
    长度变化——实测训练模式下同一段音频的有效帧 logits 随伙伴长度变化最大
    0.88（`log/stage09_bn_mask_fix.log`），而 eval 模式恒为 0。
    同时 `running_mean`/`running_var` 也会被 padding 帧污染，使训练态与推理态
    统计量不一致。

    **必须继承 `nn.BatchNorm1d` 而不是裸 `nn.Module`**：`_BatchNorm._load_from_state_dict`
    有一段特判，state_dict 里缺 `num_batches_tracked` 时会自动补上当前值
    （train.py 的 EMA 影子权重与 ckpt["ema"] 都不含这个 buffer）。裸 nn.Module
    没有这段逻辑，EMA.apply()/eval_5a.load_model() 的 `strict=True` 加载会直接
    抛 `Missing key(s) in state_dict`。继承同时也保住了 `isinstance(m, nn.BatchNorm1d)`
    的兼容性。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.affine = True
        self.track_running_stats = True

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            return super().forward(x)
        mask = build_padding_mask(lengths, x.size(2))  # (B, T) True = padding
        keep = (~mask).to(x.dtype).unsqueeze(1)  # (B, 1, T)
        n = keep.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B, 1, 1)
        if self.training:
            mean = (x * keep).sum(dim=2, keepdim=True) / n
            var = (((x - mean) * keep) ** 2).sum(dim=2, keepdim=True) / n
            with torch.no_grad():
                # running 统计同样只用有效帧：逐样本 (n_b, mean_b, var_b) 汇总成
                # 全局 (N, mean, var)，其中 var_b + mean_b² 是二阶原点矩。
                n_b = n.reshape(-1)  # (B,)
                m_b = mean.squeeze(2)  # (B, C)
                v_b = var.squeeze(2)  # (B, C)
                n_tot = n_b.sum()
                gmean = (n_b.unsqueeze(1) * m_b).sum(dim=0) / n_tot
                gvar = (n_b.unsqueeze(1) * (v_b + m_b**2)).sum(dim=0) / n_tot - gmean**2
                self.num_batches_tracked += 1
                m = self.momentum
                self.running_mean.mul_(1 - m).add_(gmean, alpha=m)
                self.running_var.mul_(1 - m).add_(gvar.clamp(min=0), alpha=m)
        else:
            mean = self.running_mean.view(1, -1, 1)
            var = self.running_var.view(1, -1, 1)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x.masked_fill(mask.unsqueeze(1), 0.0)


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int, dropout: float = 0.1, expand: int = 2) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, d_model * expand * 2, 1)
        self.dw = nn.Conv1d(d_model * expand, d_model * expand, kernel, padding=kernel // 2, groups=d_model * expand)
        self.bn = MaskedBatchNorm1d(d_model * expand)
        self.pw2 = nn.Conv1d(d_model * expand, d_model, 1)
        self.act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ln(x)
        lengths = None
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = x.size(1) - mask.sum(dim=1)
        x = x.transpose(1, 2)
        x = self.pw1(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        if lengths is not None:
            x = _mask_time(x, lengths, dim=2)
        x = self.dw(x)
        # 归一化后必须重新置零：普通 BatchNorm 会把 padding 的 0 映射成非零值，
        # 紧邻 padding 的有效帧会读到它（见 MaskedBatchNorm1d docstring）。
        x = self.bn(x, lengths)
        x = self.act(x)
        x = self.pw2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, conv_kernel: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn1 = FeedForward(d_model, ffn_dim, dropout)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForward(d_model, ffn_dim, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        if mask is None:
            x = x + self.attn(x, x, x)[0]
        else:
            x = x + self.attn(x, x, x, key_padding_mask=mask)[0]
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        conv_kernel: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.subsample = ConvSubsampling(in_dim, d_model, dropout)
        self.pe = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, n_heads, ffn_dim, conv_kernel, dropout) for _ in range(n_layers)]
        )

    def forward(
        self, x: torch.Tensor, feat_len: torch.Tensor, capture_layer: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """`capture_layer=k>0` 时额外返回第 k 个 ConformerBlock 之后的输出（辅助 CTC 用）。

        中间层 CTC（方案二.2）只给浅层更短的梯度通路，推理不用、不增加推理参数。
        第 k 层之后的时间分辨率与最终输出一致，所以可以直接复用 `out_len` 做 CTC。
        """
        feat_mask = build_padding_mask(feat_len, x.size(1))
        x = self.subsample(x, feat_mask)
        out_len = subsampled_lengths(feat_len)
        mask = build_padding_mask(out_len.clamp(min=1, max=x.size(1)), x.size(1))
        x = self.pe(x)
        aux: torch.Tensor | None = None
        for i, block in enumerate(self.blocks):
            x = block(x, mask)
            if capture_layer and i == capture_layer - 1:
                aux = x
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        if aux is not None:
            aux = aux.masked_fill(mask.unsqueeze(-1), 0.0)
        return x, out_len, aux


def build_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


class SoftGate(nn.Module):
    """LA-Dual-CTC 式动态门控：按 utterance 的 encoder 池化特征调节双任务权重。

    输出 (B, 2) 的权重，softmax 归一化且带最小值下限，防止单任务坍缩。
    """

    def __init__(self, d_model: int, min_weight: float = 0.05) -> None:
        super().__init__()
        self.min_weight = min_weight
        self.pool_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 2)

    def forward(self, enc: torch.Tensor, out_len: torch.Tensor) -> torch.Tensor:
        t = enc.size(1)
        lengths = out_len.clamp(max=t)
        mask = torch.arange(t, device=enc.device).unsqueeze(0) >= lengths.unsqueeze(1)
        pooled = enc.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1)
        h = torch.relu(self.pool_proj(pooled))
        w = torch.softmax(self.gate(h), dim=-1)
        if self.min_weight > 0:
            span = 1.0 - 2.0 * self.min_weight
            w = self.min_weight + span * w
        return w


class DualCTC(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        conv_kernel: int,
        dropout: float,
        n_char: int,
        n_mora: int,
        ctc_gate: bool = False,
        gate_min_weight: float = 0.05,
        aux_ctc_layer: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(in_dim, d_model, n_layers, n_heads, ffn_dim, conv_kernel, dropout)
        self.char_head = nn.Linear(d_model, n_char)
        self.mora_head = nn.Linear(d_model, n_mora)
        self.ctc_gate = ctc_gate
        self.aux_ctc_layer = int(aux_ctc_layer) if aux_ctc_layer else 0
        if ctc_gate:
            self.gate = SoftGate(d_model, gate_min_weight)
        if self.aux_ctc_layer:
            # 训练期辅助 mora CTC 头（推理不用、不加载）。加在中间层之后，
            # 给 mora 目标一条更短的反传路径，缓解深层收敛慢。
            self.aux_mora_head = nn.Linear(d_model, n_mora)

    def forward(self, x: torch.Tensor, feat_len: torch.Tensor):
        enc, out_len, aux = self.encoder(x, feat_len, self.aux_ctc_layer)
        char_logits = self.char_head(enc)
        mora_logits = self.mora_head(enc)
        out: list[torch.Tensor] = [char_logits, mora_logits, out_len]
        if self.ctc_gate:
            out.append(self.gate(enc, out_len))
        if self.aux_ctc_layer:
            out.append(self.aux_mora_head(aux))
        return tuple(out)

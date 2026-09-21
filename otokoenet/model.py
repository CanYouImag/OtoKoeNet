from __future__ import annotations

import math

import torch
from torch import nn


class ConvSubsampling(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.proj = nn.Linear(64 * (in_dim // 4), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        b, c, t, f = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, t, c * f)
        return self.dropout(self.proj(x))


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


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int, dropout: float = 0.1, expand: int = 2) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, d_model * expand * 2, 1)
        self.dw = nn.Conv1d(d_model * expand, d_model * expand, kernel, padding=kernel // 2, groups=d_model * expand)
        self.bn = nn.BatchNorm1d(d_model * expand)
        self.pw2 = nn.Conv1d(d_model * expand, d_model, 1)
        self.act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x.transpose(1, 2)
        x = self.pw1(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


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
        x = x + self.conv(x)
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

    def forward(self, x: torch.Tensor, feat_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.subsample(x)
        out_len = subsampled_lengths(feat_len)
        max_len = out_len.max()
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= out_len.unsqueeze(1)
        mask = mask[:, : x.size(1)]
        x = self.pe(x)
        for block in self.blocks:
            x = block(x, mask)
        return x, out_len


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
    ) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(in_dim, d_model, n_layers, n_heads, ffn_dim, conv_kernel, dropout)
        self.char_head = nn.Linear(d_model, n_char)
        self.mora_head = nn.Linear(d_model, n_mora)
        self.ctc_gate = ctc_gate
        if ctc_gate:
            self.gate = SoftGate(d_model, gate_min_weight)

    def forward(self, x: torch.Tensor, feat_len: torch.Tensor):
        enc, out_len = self.encoder(x, feat_len)
        char_logits = self.char_head(enc)
        mora_logits = self.mora_head(enc)
        if self.ctc_gate:
            return char_logits, mora_logits, out_len, self.gate(enc, out_len)
        return char_logits, mora_logits, out_len

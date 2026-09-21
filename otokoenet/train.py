from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from otokoenet.data import BucketedBatchSampler, CollateFn, Manifest, SpecAugment, load_entry
from otokoenet.decode import build_lexicon, evaluate
from otokoenet.model import DualCTC
from otokoenet.text import Vocab


class CacheDataset(Dataset):
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return load_entry(self.manifest[idx])


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = model
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if "num_batches_tracked" not in k
        }
        self.backup: dict | None = None

    @torch.no_grad()
    def update(self) -> None:
        for k, v in self.model.state_dict().items():
            if "num_batches_tracked" in k:
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self) -> None:
        self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    @torch.no_grad()
    def restore(self) -> None:
        if self.backup is not None:
            self.model.load_state_dict(self.backup)


class WarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, total: int, base_lr: float) -> None:
        self.opt = optimizer
        self.warmup = max(warmup, 1)
        self.total = total
        self.base_lr = base_lr

    def step(self, step: int) -> float:
        if step < self.warmup:
            lr = self.base_lr * (step + 1) / self.warmup
        else:
            ratio = (step - self.warmup) / max(1, self.total - self.warmup)
            lr = self.base_lr * 0.01 + 0.5 * self.base_lr * (1 - 0.01) * (1 + math.cos(math.pi * ratio))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def _flatten(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.flatten() for g in grads])


def grad_pair(loss_char: torch.Tensor, loss_mora: torch.Tensor, named: list, trunk_names: list[str]):
    """两个任务在全部参数上的梯度（allow_unused），返回对齐 named 的两组梯度与索引。"""
    params = [p for _, p in named]
    idx = {n: i for i, (n, _) in enumerate(named)}
    g_char = torch.autograd.grad(loss_char, params, retain_graph=True, allow_unused=True)
    g_mora = torch.autograd.grad(loss_mora, params, retain_graph=True, allow_unused=True)
    return g_char, g_mora, idx


def trunk_stats(g_char: list[torch.Tensor], g_mora: list[torch.Tensor], idx: dict, trunk_names: list[str]):
    """共享 encoder 参数上的任务梯度余弦 + 范数比（A2 梯度冲突分析）。"""
    f1 = torch.cat([g_char[idx[n]].flatten() for n in trunk_names])
    f2 = torch.cat([g_mora[idx[n]].flatten() for n in trunk_names])
    cos = torch.dot(f1, f2) / (f1.norm() * f2.norm() + 1e-8)
    rnorm = f1.norm() / (f2.norm() + 1e-8)
    return cos.item(), rnorm.item()


def pcgrad(g_char: list[torch.Tensor], g_mora: list[torch.Tensor]):
    """PCGrad 投影：冲突时把各任务梯度投影到另一任务的零空间。"""
    f1, f2 = _flatten(g_char), _flatten(g_mora)
    d = torch.dot(f1, f2)
    if d < 0:
        g_char = [t - (d / (f2.norm().square() + 1e-12)) * u for t, u in zip(g_char, g_mora)]
        g_mora = [t - (d / (f1.norm().square() + 1e-12)) * u for t, u in zip(g_mora, g_char)]
    return g_char, g_mora


def build_model(cfg: dict, char_vocab: Vocab, mora_vocab: Vocab) -> DualCTC:
    m = cfg["model"]
    return DualCTC(
        in_dim=cfg["audio"]["n_mels"],
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        ffn_dim=m["ffn_dim"],
        conv_kernel=m["conv_kernel"],
        dropout=m["dropout"],
        n_char=len(char_vocab),
        n_mora=len(mora_vocab),
        ctc_gate=m.get("ctc_gate", False),
        gate_min_weight=m.get("gate_min_weight", 0.05),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/basic5000.yaml")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto/cpu/cuda（auto: CUDA 可用则用 GPU，否则 CPU）",
    )
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(cfg["train"].get("num_threads", 12))
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda={torch.cuda.get_device_name(0)}")

    cache = Path(cfg["data"]["cache_dir"])
    train_manifest = Manifest.load(str(cache / "train.json"))
    test_manifest = Manifest.load(str(cache / "test.json"))
    char_vocab = Vocab.load(str(cache / "char_vocab.json"))
    mora_vocab = Vocab.load(str(cache / "mora_vocab.json"))

    model = build_model(cfg, char_vocab, mora_vocab).to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    tcfg = cfg["train"]

    if tcfg.get("pretrain_init"):
        src = torch.load(tcfg["pretrain_init"], map_location="cpu", weights_only=False)
        src_state = src.get("ema") or src["model"]
        cur_state = model.state_dict()
        copied = 0
        for name, src_t in src_state.items():
            if name not in cur_state:
                continue
            dst_t = cur_state[name]
            if src_t.shape == dst_t.shape:
                dst_t.copy_(src_t)
                copied += 1
            elif dst_t.dim() >= 2 and dst_t.shape[0] > src_t.shape[0] and dst_t.shape[1:] == src_t.shape[1:]:
                dst_t[: src_t.shape[0]].copy_(src_t)
                copied += 1
        print(f"pretrain_init copied {copied} tensors from {tcfg['pretrain_init']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.01)
    )
    total_steps = int(len(train_manifest) // tcfg["batch_size"] * tcfg["num_epochs"])
    sched = WarmupCosine(optimizer, tcfg.get("warmup_steps", 1000), total_steps, tcfg["lr"])
    ema = EMA(model, tcfg.get("ema_decay", 0.999))
    specaug = SpecAugment() if tcfg.get("specaug", True) else None

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.shadow = ckpt["ema"]
        start_step = ckpt["step"]
        print(f"resumed from {args.resume} step={start_step}")

    train_ds = CacheDataset(train_manifest)
    test_ds = CacheDataset(test_manifest)
    lengths = [int(np.load(e["feat"]).shape[0]) for e in train_manifest.entries]

    train_loader = DataLoader(
        train_ds,
        batch_sampler=BucketedBatchSampler(lengths, tcfg["batch_size"], seed=cfg["data"]["seed"]),
        collate_fn=CollateFn(),
    )
    test_loader = DataLoader(test_ds, batch_size=tcfg["batch_size"], collate_fn=CollateFn())

    w_char = tcfg.get("ctc_weight_char", 0.3)
    w_mora = tcfg.get("ctc_weight_mora", 0.3)
    track_metric = tcfg.get("track_metric", "cer_ctc")
    best_val = float("inf")
    step = start_step
    grad_analysis = tcfg.get("grad_analysis", False)
    grad_analysis_interval = max(1, tcfg.get("grad_analysis_interval", 10))
    grad_pcgrad = tcfg.get("grad_pcgrad", False)
    named_params = list(model.named_parameters())
    trunk = [(n, p) for n, p in named_params if n.startswith("encoder.") and p.requires_grad]
    trunk_names = [n for n, _ in trunk]
    other_ps = [p for n, p in named_params if n not in trunk_names and "gate" not in n and p.requires_grad]
    gate_ps = [p for n, p in named_params if "gate" in n and p.requires_grad]
    if grad_analysis or grad_pcgrad:
        print(
            f"grad_analysis={grad_analysis} grad_pcgrad={grad_pcgrad} "
            f"trunk={len(trunk)} other={len(other_ps)} gate={len(gate_ps)}"
        )
    log_f = open(save_dir / "log.csv", "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if start_step == 0:
        writer.writerow(["step", "loss", "lr", "cer_ctc", "mer"])

    for epoch in range(1, tcfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        t0 = time.time()
        for batch in train_loader:
            feat_pad, feat_len, char_pad, char_len, mora_pad, mora_len = [
                t.to(device) for t in batch
            ]
            if specaug is not None:
                feat_pad = specaug(feat_pad)
            gate_w = None
            out = model(feat_pad, feat_len)
            char_logits, mora_logits, out_len = out[:3]
            if model.ctc_gate:
                gate_w = out[3]
                ctc_char = torch.nn.functional.ctc_loss(
                    char_logits.log_softmax(2).transpose(0, 1),
                    char_pad,
                    out_len,
                    char_len,
                    blank=char_vocab.blank_id,
                    zero_infinity=True,
                    reduction="none",
                )
                ctc_mora = torch.nn.functional.ctc_loss(
                    mora_logits.log_softmax(2).transpose(0, 1),
                    mora_pad,
                    out_len,
                    mora_len,
                    blank=mora_vocab.blank_id,
                    zero_infinity=True,
                    reduction="none",
                )
                loss = (gate_w[:, 0] * ctc_char + gate_w[:, 1] * ctc_mora).mean()
            else:
                ctc_char = torch.nn.functional.ctc_loss(
                    char_logits.log_softmax(2).transpose(0, 1),
                    char_pad,
                    out_len,
                    char_len,
                    blank=char_vocab.blank_id,
                    zero_infinity=True,
                )
                ctc_mora = torch.nn.functional.ctc_loss(
                    mora_logits.log_softmax(2).transpose(0, 1),
                    mora_pad,
                    out_len,
                    mora_len,
                    blank=mora_vocab.blank_id,
                    zero_infinity=True,
                )
                loss = w_char * ctc_char + w_mora * ctc_mora
            do_anal = grad_analysis and (step + 1) % grad_analysis_interval == 0
            optimizer.zero_grad()
            if grad_pcgrad:
                g_char, g_mora, idx = grad_pair(ctc_char.mean(), ctc_mora.mean(), named_params, trunk_names)
                tc = [g_char[idx[n]] for n in trunk_names]
                tm = [g_mora[idx[n]] for n in trunk_names]
                cos_pre = torch.dot(_flatten(tc), _flatten(tm)) / (
                    _flatten(tc).norm() * _flatten(tm).norm() + 1e-8
                )
                for p, gc, gm in zip([p for _, p in trunk], *pcgrad(tc, tm)):
                    p.grad = gc + gm
                for (n, p), gc, gm in zip(named_params, g_char, g_mora):
                    if n in trunk_names or "gate" in n:
                        continue
                    p.grad = gc if gc is not None else gm
                if gate_ps:
                    gg = torch.autograd.grad(loss, gate_ps)
                    for p, g in zip(gate_ps, gg):
                        p.grad = g
                ga_str = f" cos={cos_pre.item():.3f}(pc)"
            elif do_anal:
                g_char, g_mora, idx = grad_pair(ctc_char.mean(), ctc_mora.mean(), named_params, trunk_names)
                cos_a, rnorm_a = trunk_stats(g_char, g_mora, idx, trunk_names)
                ga_str = f" cos={cos_a:.3f} rnorm={rnorm_a:.2f}"
                loss.backward()
            else:
                ga_str = ""
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 5.0))
            optimizer.step()
            ema.update()
            step += 1
            sched.step(step)
            epoch_loss += loss.item()
            n_batch += 1
            if step % tcfg["log_interval"] == 0:
                gw = f" gw={gate_w[:, 0].mean().item():.3f}" if gate_w is not None else ""
                print(
                    f"[{step}] loss={loss.item():.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}{gw}{ga_str}"
                )
        avg = epoch_loss / max(1, n_batch)
        print(f"epoch={epoch} avg_loss={avg:.3f} time={time.time()-t0:.1f}s")

        ema.apply()
        lexicon = build_lexicon(train_manifest, test_manifest)
        metrics = evaluate(model, test_loader, lexicon)
        ema.restore()
        print(
            f"  eval ctc_cer={metrics['cer_ctc']*100:.2f}% "
            f"fst_cer={metrics['cer_fst']*100:.2f}% mer={metrics['mer']*100:.2f}%"
        )
        writer.writerow([step, f"{avg:.4f}", f"{optimizer.param_groups[0]['lr']:.2e}", f"{metrics['cer_ctc']:.4f}", f"{metrics['mer']:.4f}"])
        log_f.flush()

        if metrics[track_metric] < best_val:
            best_val = metrics[track_metric]
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": cfg,
                },
                save_dir / "best.pt",
            )
            print(f"  saved best.pt {track_metric}={best_val*100:.2f}%")

        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": cfg,
            },
            save_dir / "last.pt",
        )
    log_f.close()
    print("done")


if __name__ == "__main__":
    main()

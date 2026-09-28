from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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
    def __init__(self, manifest: Manifest, specaug: SpecAugment | None = None) -> None:
        self.manifest = manifest
        self.specaug = specaug

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, char_ids, mora_ids = load_entry(self.manifest[idx])
        if self.specaug is not None:
            feat = self.specaug(feat.unsqueeze(0)).squeeze(0)
        return feat, char_ids, mora_ids


class EMA:
    """权重滑动平均。

    `update()` 跳过 `num_batches_tracked`（它只在 `momentum=None` 时影响归一化，
    本项目用固定 momentum=0.1，因此不需要平均）。副作用是 `shadow` 缺少这些键，
    所以 `apply()` 不能用 `strict=True`：那样会在每个 epoch 求 validation 时
    直接抛 `Missing key(s) in state_dict`。这里改为 `strict=False` 并显式断言
    缺失键只能是 `num_batches_tracked`，避免把真正的结构不匹配也一起放过去。
    """

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
        inc = self.model.load_state_dict(self.shadow, strict=False)
        bad = [k for k in inc.missing_keys if "num_batches_tracked" not in k]
        if bad or inc.unexpected_keys:
            raise RuntimeError(
                f"EMA 影子权重与模型结构不匹配: missing={bad[:3]} unexpected={inc.unexpected_keys[:3]}"
            )

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


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _load_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


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
        "--fresh",
        action="store_true",
        help="忽略 save_dir 中已有的 last.pt，从零开始训练",
    )
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

    seed = int(cfg["train"].get("seed", cfg["data"].get("seed", 0)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"seed={seed}")

    cache = Path(cfg["data"]["cache_dir"])
    train_manifest = Manifest.load(str(cache / "train.json"))
    val_manifest = Manifest.load(str(cache / "val.json"))
    test_manifest = Manifest.load(str(cache / "test.json"))
    char_vocab = Vocab.load(str(cache / "char_vocab.json"))
    mora_vocab = Vocab.load(str(cache / "mora_vocab.json"))
    train_utts = {e["utt"] for e in train_manifest.entries}
    for split_name, split_manifest in (("val", val_manifest), ("test", test_manifest)):
        if len(split_manifest) == 0:
            raise RuntimeError(f"{split_name} split is empty; re-run scripts/prepare.py")
        overlap = train_utts & {e["utt"] for e in split_manifest.entries}
        if overlap:
            raise RuntimeError(
                f"{split_name} split overlaps train on {len(overlap)} utterances; refusing to train"
            )
    print(
        f"split train={len(train_manifest)} val={len(val_manifest)} test={len(test_manifest)}"
    )

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
    steps_per_epoch = (len(train_manifest) + tcfg["batch_size"] - 1) // tcfg["batch_size"]
    total_steps = steps_per_epoch * tcfg["num_epochs"]
    sched = WarmupCosine(optimizer, tcfg.get("warmup_steps", 1000), total_steps, tcfg["lr"])
    ema = EMA(model, tcfg.get("ema_decay", 0.999))
    sa = tcfg.get("specaug", {})
    if sa is True:
        specaug = SpecAugment()
    elif isinstance(sa, dict):
        specaug = SpecAugment(
            n_time_masks=sa.get("n_time_masks", 2),
            max_time_width=sa.get("max_time_width", 30),
            n_freq_masks=sa.get("n_freq_masks", 1),
            max_freq_width=sa.get("max_freq_width", 27),
        )
    else:
        specaug = None

    start_step = 0
    start_epoch = 1
    start_batch = 0
    ckpt_path = args.resume
    if ckpt_path is None and not args.fresh:
        cands = sorted(save_dir.glob("last*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            print(f"auto-resume: 检测到已有断点 {cands[0].name}")
        for cand in cands:
            try:
                ckpt = torch.load(cand, map_location="cpu", weights_only=False)
                ckpt_path = str(cand)
                break
            except Exception as e:
                print(f"  [!] checkpoint {cand.name} 损坏，尝试更早快照: {e}")
    best_val = float("inf")
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"checkpoint {ckpt_path} 与当前模型不匹配；请检查 config 或使用 --fresh 重新训练 ({e})"
            ) from e
        optimizer.load_state_dict(ckpt["optimizer"])
        missing_ema = [
            k
            for k in model.state_dict()
            if "num_batches_tracked" not in k and k not in ckpt["ema"]
        ]
        if missing_ema:
            raise RuntimeError(f"checkpoint {ckpt_path} 的 EMA 权重不完整: {missing_ema[:3]}")
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        start_step = int(ckpt["step"])
        start_epoch = int(ckpt.get("epoch", 1))
        start_batch = int(ckpt.get("batch_in_epoch", 0))
        best_val = float(ckpt.get("best", float("inf")))
        _load_rng_state(ckpt.get("rng"))
        print(
            f"resumed from {ckpt_path} step={start_step} epoch={start_epoch} "
            f"batch={start_batch} best={best_val:.4f}"
        )

    train_ds = CacheDataset(train_manifest, specaug=specaug)
    val_ds = CacheDataset(val_manifest)
    test_ds = CacheDataset(test_manifest)
    train_lengths = [int(np.load(e["feat"]).shape[0]) for e in train_manifest.entries]
    val_lengths = [int(np.load(e["feat"]).shape[0]) for e in val_manifest.entries]
    test_lengths = [int(np.load(e["feat"]).shape[0]) for e in test_manifest.entries]
    data_seed = int(cfg["data"].get("seed", 0))

    train_sampler = BucketedBatchSampler(train_lengths, tcfg["batch_size"], seed=data_seed)
    train_collate = CollateFn(speed_perturb=tcfg.get("speed_perturb", False))
    # DataLoader draws a worker base seed from the global torch RNG on every iter().
    # Route that draw to a private generator so the global RNG stream stays a pure
    # function of the training steps -- otherwise resuming mid-epoch shifts the
    # dropout stream by one draw and the trajectory stops being reproducible.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(data_seed + 1)
    val_loader = DataLoader(
        val_ds,
        batch_sampler=BucketedBatchSampler(val_lengths, tcfg["batch_size"], shuffle=False),
        collate_fn=CollateFn(),
        generator=loader_gen,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=BucketedBatchSampler(test_lengths, tcfg["batch_size"], shuffle=False),
        collate_fn=CollateFn(),
        generator=loader_gen,
    )
    lexicon = build_lexicon(train_manifest)

    w_char = tcfg.get("ctc_weight_char", 0.3)
    w_mora = tcfg.get("ctc_weight_mora", 0.3)
    track_metric = tcfg.get("track_metric", "cer_ctc")
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
    log_path = save_dir / "log.csv"
    need_header = not log_path.exists() or log_path.stat().st_size == 0
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    header = ["step", "epoch", "loss", "lr", "val_cer_ctc", "val_cer_fst", "val_mer"]
    if need_header:
        writer.writerow(header)
    ckpt_interval = max(1, tcfg.get("ckpt_interval", 500))
    lr = optimizer.param_groups[0]["lr"]

    def save_ckpt(path: Path, epoch: int, step: int, batch_in_epoch: int = 0, snap: bool = False) -> None:
        payload = {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "best": best_val,
            "rng": _rng_state(),
            "config": cfg,
        }
        tmp = save_dir / f".{path.name}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        if snap and path.name == "last.pt":
            snap_path = save_dir / f"last_step{step}.pt"
            try:
                if snap_path.exists():
                    snap_path.unlink()
                os.link(path, snap_path)
            except OSError:
                pass
            _prune_snaps(save_dir, keep=3)
        print(
            f"  saved {path.name} step={step} epoch={epoch} batch={batch_in_epoch}", flush=True
        )

    def _prune_snaps(dir_: Path, keep: int) -> None:
        snaps = sorted(
            dir_.glob("last_step*.pt"), key=lambda p: int(p.stem[len("last_step") :])
        )
        for old in snaps[:-keep]:
            old.unlink(missing_ok=True)

    for epoch in range(start_epoch, tcfg["num_epochs"] + 1):
        train_sampler.set_epoch(epoch)
        epoch_batches = list(train_sampler)
        offset = start_batch if epoch == start_epoch else 0
        if offset:
            epoch_batches = epoch_batches[offset:]
        train_loader = DataLoader(
            train_ds, batch_sampler=epoch_batches, collate_fn=train_collate, generator=loader_gen
        )
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        t0 = time.time()
        for bi, batch in enumerate(train_loader):
            feat_pad, feat_len, char_pad, char_len, mora_pad, mora_len = [
                t.to(device) for t in batch
            ]
            lr = sched.step(step)
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
            if not torch.isfinite(loss):
                print(f"[{step + 1}] [!] non-finite loss, batch skipped")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 5.0))
            optimizer.step()
            ema.update()
            step += 1
            if step % ckpt_interval == 0:
                save_ckpt(
                    save_dir / "last.pt",
                    epoch,
                    step,
                    batch_in_epoch=bi + offset + 1,
                    snap=(step % 600 == 0),
                )
            epoch_loss += loss.item()
            n_batch += 1
            if step % tcfg["log_interval"] == 0:
                gw = f" gw={gate_w[:, 0].mean().item():.3f}" if gate_w is not None else ""
                print(f"[{step}] loss={loss.item():.3f} lr={lr:.2e}{gw}{ga_str}")
        avg = epoch_loss / max(1, n_batch)
        print(
            f"epoch={epoch} avg_loss={avg:.3f} batches={n_batch}/{steps_per_epoch} "
            f"time={time.time()-t0:.1f}s"
        )

        ema.apply()
        val_metrics = evaluate(model, val_loader, lexicon)
        ema.restore()
        print(
            f"  val ctc_cer={val_metrics['cer_ctc']*100:.2f}% "
            f"fst_cer={val_metrics['cer_fst']*100:.2f}% mer={val_metrics['mer']*100:.2f}%"
        )
        writer.writerow(
            [
                step,
                epoch,
                f"{avg:.4f}",
                f"{lr:.2e}",
                f"{val_metrics['cer_ctc']:.4f}",
                f"{val_metrics['cer_fst']:.4f}",
                f"{val_metrics['mer']:.4f}",
            ]
        )
        log_f.flush()

        if val_metrics[track_metric] < best_val:
            best_val = val_metrics[track_metric]
            save_ckpt(save_dir / "best.pt", epoch, step)
            print(f"  best val {track_metric}={best_val*100:.2f}%")

        save_ckpt(save_dir / "last.pt", epoch + 1, step)
        start_batch = 0

    log_f.close()

    report = {
        "config": args.config,
        "cache_dir": str(cache),
        "train": len(train_manifest),
        "val": len(val_manifest),
        "test": len(test_manifest),
        "final_step": step,
        "epochs": tcfg["num_epochs"],
        "best_val": best_val,
        "track_metric": track_metric,
    }
    best_path = save_dir / "best.pt"
    if best_path.exists():
        try:
            best_ck = torch.load(best_path, map_location="cpu", weights_only=False)
            incompat = model.load_state_dict(best_ck["ema"], strict=False)
            bad = [k for k in incompat.missing_keys if "num_batches_tracked" not in k]
            if bad:
                raise RuntimeError(f"best.pt EMA 权重不完整: {bad[:3]}")
            test_metrics = evaluate(model, test_loader, lexicon)
            print(
                f"  test ctc_cer={test_metrics['cer_ctc']*100:.2f}% "
                f"fst_cer={test_metrics['cer_fst']*100:.2f}% mer={test_metrics['mer']*100:.2f}%"
            )
            report["best_step"] = best_ck["step"]
            report["best_epoch"] = best_ck["epoch"]
            report["test"] = test_metrics
        except Exception as e:
            print(f"  [!] final test evaluation failed: {e}")
    with open(save_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("done")


if __name__ == "__main__":
    main()

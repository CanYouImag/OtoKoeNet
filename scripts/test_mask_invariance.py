"""padding / 归一化不变式回归测试。

阶段 2 补的版本只测 `eval()` 模式，因此在 `ConvModule` 的 `BatchNorm1d` 上漏掉了
训练态的 padding 污染：普通 BatchNorm 在全 T 帧上统计，padding 的 0 帧被映射成
`-mean/std`（非零），紧邻 padding 的有效帧会读到它；`running_mean`/`running_var`
也被污染。实测同一段音频的有效帧 logits 随 batch 伙伴长度变化最大 0.88
（见 `log/stage09_bn_mask_fix.log`），而 eval 模式恒为 0。

本脚本覆盖 5 组不变式，任一失败退出 1：

1. eval  模式：同一音频单独推理 vs 与不等长音频同批 -> 有效 logits 一致
2. eval  模式：零填充到更长 vs 原始长度         -> 有效 logits 一致
3. **train 模式**：同 batch 大小、不同伙伴长度   -> 有效 logits 一致（阶段 2 漏测）
4. **train 模式**：BN 的 `running_mean`/`running_var` 与 batch 伙伴无关（阶段 2 漏测）
5. padding 位置不携带任何信号（不只 eval 模式）

    python scripts/test_mask_invariance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from otokoenet.model import DualCTC

TOL = 1e-5
IN_DIM = 80
SHORT = 137
LONG = 900
TOTAL = 1200


def build() -> DualCTC:
    torch.manual_seed(0)
    return DualCTC(
        in_dim=IN_DIM,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_dim=128,
        conv_kernel=15,
        dropout=0.0,
        n_char=40,
        n_mora=60,
    )


def make_batch(short: torch.Tensor, partner: torch.Tensor, partner_len: int) -> tuple:
    x = torch.zeros(2, TOTAL, IN_DIM)
    x[0, :SHORT] = short
    x[1, :partner_len] = partner[:partner_len]
    return x, torch.tensor([SHORT, partner_len])


def diff(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x - y).abs().max().item()


def main() -> int:
    torch.manual_seed(1)
    a = torch.randn(SHORT, IN_DIM) * 0.5
    b = torch.randn(LONG, IN_DIM) * 0.5
    failures: list[str] = []

    def check(cond: bool, name: str, detail: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} — {detail}")
        if not cond:
            failures.append(f"{name}: {detail}")

    def run(model: DualCTC, feats: torch.Tensor, lens: torch.Tensor):
        with torch.no_grad():
            char, mora, out_len = model(feats, lens)[:3]
        return char, mora, out_len

    # ---------- 1) eval 模式：单独 vs 同批 ----------
    print("== 1. eval 模式 batch 一致性 ==")
    m = build().eval()
    solo_c, solo_m, solo_l = run(m, a.unsqueeze(0), torch.tensor([SHORT]))
    valid = int(solo_l[0])
    x, lens = make_batch(a, b, LONG)
    mix_c, mix_m, mix_l = run(m, x, lens)
    d_c, d_m = diff(solo_c[0, :valid], mix_c[0, :valid]), diff(solo_m[0, :valid], mix_m[0, :valid])
    check(
        d_c <= TOL and d_m <= TOL,
        "batched-with-other",
        f"char={d_c:.3e} mora={d_m:.3e} (tol {TOL:.0e})",
    )
    z = torch.zeros(1, TOTAL, IN_DIM)
    z[0, :SHORT] = a
    z_c, z_m, _ = run(m, z, torch.tensor([SHORT]))
    d_c, d_m = diff(solo_c[0, :valid], z_c[0, :valid]), diff(solo_m[0, :valid], z_m[0, :valid])
    check(
        d_c <= TOL and d_m <= TOL,
        "zero-padded",
        f"char={d_c:.3e} mora={d_m:.3e} (tol {TOL:.0e})",
    )
    lb = int(mix_l[1])
    t_c = diff(mix_c[1, lb:], m.char_head.bias)
    t_m = diff(mix_m[1, lb:], m.mora_head.bias)
    check(t_c == 0.0 and t_m == 0.0, "padded-region", f"char={t_c:.3e} mora={t_m:.3e} (vs head bias)")

    # ---------- 2) train 模式：同 batch 大小、不同伙伴长度 ----------
    # 有效 logits 必须与伙伴无关（伙伴是另一个 batch 行，没有任何合法路径能让
    # 它的内容影响本行）；这条在修复前实测最大差 0.88。
    print("== 2. train 模式 batch 一致性（阶段 2 漏测）==")
    ref = None
    ref_partner = 0
    for partner_len in (300, 600, 900):
        m = build().train()
        x, lens = make_batch(a, b, partner_len)
        c, mo, ol = run(m, x, lens)
        v = int(ol[0])
        state = (c[0, :v].clone(), mo[0, :v].clone())
        if ref is None:
            ref, ref_partner = state, partner_len
            print(f"  ref partner_len={partner_len} out_len={v}")
            continue
        d_c, d_m = diff(ref[0], state[0]), diff(ref[1], state[1])
        check(
            d_c <= TOL and d_m <= TOL,
            f"train-batch partner={partner_len} (ref {ref_partner})",
            f"char={d_c:.3e} mora={d_m:.3e} (tol {TOL:.0e})",
        )

    # ---------- 3) train 模式：BN 统计量与 padding 长度无关 ----------
    # 注意口径：batch 统计量本来就包含同 batch 其它样本的**有效**帧，所以改变
    # 伙伴的有效长度时它理应变化（那不是 bug）。真正的不变式是：只改变时间轴
    # 尾部的 padding 长度，两句有效内容不变时，统计量必须逐元素相同。
    print("== 3. train 模式 BN 统计量不受 padding 长度影响 ==")
    stats = {}
    for total in (TOTAL, TOTAL + 800):
        m = build().train()
        bn = m.encoder.blocks[0].conv.bn
        mom = bn.momentum
        x = torch.zeros(2, total, IN_DIM)
        x[0, :SHORT] = a
        x[1, :LONG] = b
        c, mo, ol = run(m, x, torch.tensor([SHORT, LONG]))
        v = int(ol[0])
        # 恰好一次更新：running_mean = m·gmean，running_var = (1-m) + m·gvar
        stats[total] = (
            c[0, :v].clone(),
            mo[0, :v].clone(),
            bn.running_mean.clone() / mom,
            (bn.running_var.clone() - (1 - mom)) / mom,
        )
    t0, t1 = TOTAL, TOTAL + 800
    d_c, d_m = diff(stats[t0][0], stats[t1][0]), diff(stats[t0][1], stats[t1][1])
    check(
        d_c <= TOL and d_m <= TOL,
        f"train logits T={t0} vs T={t1}",
        f"char={d_c:.3e} mora={d_m:.3e} (tol {TOL:.0e})",
    )
    d_rm, d_rv = diff(stats[t0][2], stats[t1][2]), diff(stats[t0][3], stats[t1][3])
    check(
        d_rm <= TOL and d_rv <= TOL,
        f"bn batch stats T={t0} vs T={t1}",
        f"gmean={d_rm:.3e} gvar={d_rv:.3e} (tol {TOL:.0e})",
    )

    # ---------- 4) train 模式尾帧隔离 ----------
    m = build().train()
    c_short, m_short, ol_s = run(m, *make_batch(a, b, 300))
    v = int(ol_s[0])
    c_long, m_long, _ = run(m, *make_batch(a, b, 900))
    k = min(4, v)
    d_c, d_m = diff(c_short[0, v - k : v], c_long[0, v - k : v]), diff(
        m_short[0, v - k : v], m_long[0, v - k : v]
    )
    check(
        d_c <= TOL and d_m <= TOL,
        f"train tail-{k} frames",
        f"char={d_c:.3e} mora={d_m:.3e} (tol {TOL:.0e})",
    )

    # ---------- 5) 旧 checkpoint 键名兼容 ----------
    print("== 5. 旧 checkpoint 键名兼容 ==")
    import torch.nn as nn

    ref_bn_mod = nn.BatchNorm1d(128)
    mine = build()
    bn = mine.encoder.blocks[0].conv.bn
    ref_keys = set(ref_bn_mod.state_dict())
    my_keys = set(bn.state_dict())
    check(ref_keys == my_keys, "state_dict keys", f"ref-only={sorted(ref_keys - my_keys)} mine-only={sorted(my_keys - ref_keys)}")
    same_shape = all(
        tuple(ref_bn_mod.state_dict()[k].shape) == tuple(bn.state_dict()[k].shape) for k in ref_keys
    )
    check(same_shape, "state_dict shapes", "与 nn.BatchNorm1d 一致")
    bn.load_state_dict(ref_bn_mod.state_dict(), strict=True)
    check(True, "load_state_dict(strict=True)", "nn.BatchNorm1d → MaskedBatchNorm1d 互转成功")

    # 继承 nn.BatchNorm1d 的关键理由：EMA 影子权重与 ckpt["ema"] 都不含
    # num_batches_tracked，依赖 _BatchNorm._load_from_state_dict 的自动补齐。
    # 若 MaskedBatchNorm1d 写成裸 nn.Module，这里会抛 Missing key(s)，
    # train.py 的 EMA.apply() 每个 epoch 求 validation 都会炸。
    check(isinstance(bn, nn.BatchNorm1d), "继承 nn.BatchNorm1d", f"isinstance={isinstance(bn, nn.BatchNorm1d)}")
    shadow = {k: v for k, v in bn.state_dict().items() if "num_batches_tracked" not in k}
    bn2 = build().encoder.blocks[0].conv.bn
    bn2.load_state_dict(shadow, strict=True)
    check(
        "num_batches_tracked" in bn2.state_dict(),
        "缺 num_batches_tracked 的权重 strict=True 可加载",
        f"加载后 num_batches_tracked={int(bn2.num_batches_tracked)}",
    )

    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print(
        "PASS: eval/train 两态下有效 logits 均 batch 不变（%.0e）；"
        "BN running 统计与 batch 伙伴无关；padding 区无信号；键名与 nn.BatchNorm1d 兼容" % TOL
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

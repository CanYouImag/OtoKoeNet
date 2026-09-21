# OtoKoeNet 训练结果汇总

本文件汇总 OtoKoeNet（日语发音评估 / 语音识别）已完成的训练运行结果。
模型结构见 `otokoenet/model.py`（DualCTC = Conformer Encoder + 双 CTC 头，可选 SoftGate）。

## 1. 实验设置

- 语料：JSUT `basic5000` 子集，16 kHz，80 维 log-Mel fbank，留出 200 句作为 test。
- 编码器：`d_model=128`，`n_layers=6`，`n_heads=4`，`ffn_dim=512`，`conv_kernel=7`，`dropout=0.1`。
- 训练：AdamW，`lr=1e-3`，Warmup(800) + Cosine，`batch_size=8`，`num_epochs=40`，EMA(0.999)，SpecAugment，grad_clip=5.0。
- 双 CTC 头：字符（char）与莫拉（mora），指标为字符 CTC 错误率 `cer_ctc` 与莫拉错误率 `mer`。
- 运行设备：CPU。

## 2. 运行总览

| 运行 (`runs/`) | 门控配置 | 步数 / epoch | 最优 `cer_ctc` | 末次 `cer_ctc` | 末次 `mer` |
| --- | --- | --- | --- | --- | --- |
| `basic5000_base` | 无门控（固定权重 char=1.0, mora=0.5） | 24000 / 40 | **25.22%** | 25.37% | **6.59%** |
| `basic5000_gate03` | SoftGate，`gate_min_weight=0.3` | 24000 / 40 | 25.92% | 26.14% | 6.27% |
| `basic5000_ctc_baseline` | 单 CTC 基线 | 24000 / 40 | 25.69% | 25.78% | 7.02% |
| `basic5000_gate` | SoftGate，`gate_min_weight=0.05` | 24000 / 40 | 31.73% | 31.73% | 7.22% |
| `basic5000_full` | 早期配置（列名 `cer`） | 12000 / 20 | 108.05% | 114.39% | 7.08% |
| `basic5000_ar` | 早期配置（列名 `cer`） | 12000 / 20 | 89.70% | 151.21% | 7.37% |

> 说明：`basic5000_full` / `basic5000_ar` 为早期不同解码配置下的运行，日志列名为 `cer`，数值不可与其余运行直接比较。
> `basic5000_base` 与 `basic5000_gate03` 的 `fst_cer=0.00%`，表示留出集在句库最近邻解码下完全命中。

## 3. 关键结论

1. **基线（无门控）表现最好**：`basic5000_base` 训练稳定，`cer_ctc` 从 1800 步的 81.67% 收敛到 24000 步的 25.37%，最优 25.22%。
2. **SoftGate 的 `min_weight` 很关键**：下限 0.05 时门控几乎坍缩到单任务（日志 `gw=0.050` 长期不变），`cer_ctc` 退化到 31.73%；把下限提到 0.3 后恢复到 26.14%，但仍未超过无门控基线。
3. **单 CTC 基线（25.69%）与无门控双 CTC（25.22%）接近**，双头/莫拉任务未在该规模下带来字符识别增益。
4. 训练日志均以 `done` 正常结束，各运行有完整 checkpoint 与 `log.csv`。

## 4. 产出文件

- `runs/<run>/best.pt` / `last.pt`：含 `model`、`ema`、`optimizer`、`step`、`config` 字段。
- `runs/<run>/log.csv`：逐步 `loss / lr / cer_ctc / mer`。
- `log/a1_base.log`、`log/a1_gate.log`、`log/a1b_gate03.log`：完整训练 stdout。
- `runs/basic5000/best.pt`：对应 `basic5000_full` 的最优权重（同尺寸）。

## 5. 复现命令

```bash
# 无门控基线
python otokoenet/train.py --config configs/basic5000_base.yaml

# SoftGate（min_weight=0.3）
python otokoenet/train.py --config configs/basic5000_gate03.yaml

# 单 CTC 基线
python otokoenet/train.py --config configs/basic5000.yaml
```

权重与 `data/` 语料未入库，部署到新机器时需重新下载 JSUT 并生成特征缓存
（见 `README.md` 的「数据准备」），权重可另行传输。

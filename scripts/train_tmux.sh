#!/bin/bash
# OtoKoeNet 断点续训启动脚本（在 tmux 会话内运行）
# 用法: bash scripts/train_tmux.sh 或直接作为 tmux 命令执行
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/mnt/workspace/projects/otokoenet_env/bin/python
CONFIG="${1:-configs/basic5000_base.yaml}"
CACHE=data/cache/basic5000

cd "$ROOT"
mkdir -p log
LOG=log/train_base.log

# 1) 首次运行：生成特征缓存（幂等，已存在则跳过）
if [ ! -f "$CACHE/train.json" ]; then
    echo "[$(date '+%F %T')] preparing feature cache ..." | tee -a "$LOG"
    "$PY" scripts/prepare.py 2>&1 | tee -a "$LOG"
fi

# 2) 训练：自动续训（存在 runs/<run>/last.pt 时从断点继续）
echo "[$(date '+%F %T')] training start: $CONFIG" | tee -a "$LOG"
exec "$PY" -m otokoenet.train --config "$CONFIG" 2>&1 | tee -a "$LOG"
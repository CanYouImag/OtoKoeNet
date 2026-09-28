#!/bin/bash
# OtoKoeNet 断点续训守护脚本（在 tmux 会话内运行）
# 用法: bash scripts/train_tmux.sh [CONFIG]
# 特性: 特征缓存幂等生成 + 断点自动恢复 + 崩溃自动重启（最多重启 MAX_RESTARTS 次）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/mnt/workspace/projects/otokoenet_env/bin/python
CONFIG="${1:-configs/basic5000_base.yaml}"
CACHE="${CACHE:-data/cache/basic5000}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
RESTART_DELAY="${RESTART_DELAY:-15}"

cd "$ROOT"
mkdir -p log
LOG="${LOG:-log/train_$(basename "${CONFIG%.yaml}" | sed 's/basic5000_//').log}"

# 1) 首次运行：生成特征缓存（幂等，已存在则跳过）
if [ ! -f "$CACHE/train.json" ]; then
    echo "[$(date '+%F %T')] preparing feature cache at $CACHE ..." | tee -a "$LOG"
    "$PY" scripts/prepare.py --cache-dir "$CACHE" 2>&1 | tee -a "$LOG"
fi

# 2) 训练：自动续训（存在 runs/<run>/last.pt 时从断点继续）
#    崩溃 / 被回收后自动重启并续训，直到完成或达到重启上限
attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[$(date '+%F %T')] training attempt=$attempt: $CONFIG" | tee -a "$LOG"
    if "$PY" -u -m otokoenet.train --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
        echo "[$(date '+%F %T')] training done (exit 0)" | tee -a "$LOG"
        break
    fi
    if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
        echo "[$(date '+%F %T')] exceeded MAX_RESTARTS=$MAX_RESTARTS, giving up" | tee -a "$LOG"
        exit 1
    fi
    echo "[$(date '+%F %T')] training interrupted (attempt=$attempt), restarting in ${RESTART_DELAY}s ..." | tee -a "$LOG"
    sleep "$RESTART_DELAY"
done
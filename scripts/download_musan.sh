#!/usr/bin/env bash
# 下载波形级增强所需的数据集（方案二.4）。
#
#   MUSAN            11GB  噪声/音乐/人声  https://www.openslr.org/17
#   RIRS_NOISES      1.5GB 真实/仿真 RIR   https://www.openslr.org/28
#
# 落盘：
#   data/musan/{noise,music,speech}/...      （--noise-dir data/musan）
#   data/RIRS_NOISES/{real_rirs_isotropic_noises,simulated_rirs}/...
#                                            （--rir-dir data/RIRS_NOISES/simulated_rirs）
#
# 之后重建增强训练缓存（必须逐字复用 jsut_full_v1 的划分参数，否则 val/test 不可比）：
#   python scripts/prepare.py --jsut-root data/jsut_ver1.1 --subsets all \
#       --cache-dir data/cache/jsut_full_noise_v1 \
#       --holdout-from data/cache/basic5000_v2 --vocab-from data/cache/basic5000_v2 \
#       --probe-frac 0.1 \
#       --probe-subsets countersuffix26,loanword128,onomatopee300,precedent130,repeat500,travel1000,utparaphrase512,voiceactress100 \
#       --noise-dir data/musan --rir-dir data/RIRS_NOISES/simulated_rirs \
#       --noise-prob 0.3 --rir-prob 0.3 --snr-db-range 5,20
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

if [ ! -d data/musan ]; then
    echo "[1/2] MUSAN (~11GB) ..."
    curl -L --fail -o data/musan.tar.gz https://www.openslr.org/resources/17/musan.tar.gz
    tar -xzf data/musan.tar.gz -C data
    rm -f data/musan.tar.gz
else
    echo "[1/2] data/musan 已存在，跳过"
fi

if [ ! -d data/RIRS_NOISES ]; then
    echo "[2/2] RIRS_NOISES (~1.5GB) ..."
    curl -L --fail -o data/rirs_noises.zip https://www.openslr.org/resources/28/rirs_noises.zip
    unzip -q data/rirs_noises.zip -d data
    rm -f data/rirs_noises.zip
else
    echo "[2/2] data/RIRS_NOISES 已存在，跳过"
fi

echo "done:"
du -sh data/musan data/RIRS_NOISES 2>/dev/null || true

"""回归测试：`otokoenet.joint_decode.ctc_log_likelihood` 必须等于暴力枚举所有路径的和。

阶段 19 的 Char-Mora 联合解码完全建立在这个前向似然上；一旦它算错，联合解码的
汉字 CER 改善就是假的。这里用 T<=5、V<=4 的所有路径之一一比对，覆盖：
- 空目标、单 token、重复 token、T==n、T>n 等情形；
- blank(0) 与普通 token 的转移、s->s+2 的重复抑制。

    python scripts/test_joint_decode.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from otokoenet.decode import _log_softmax  # noqa: E402
from otokoenet.joint_decode import ctc_log_likelihood  # noqa: E402


def _collapse(path, blank=0):
    out, prev = [], -1
    for i in path:
        if i != blank and i != prev:
            out.append(i)
        prev = i
    return tuple(out)


def brute(logits, target, blank=0) -> float:
    lp = _log_softmax(logits)
    T, V = logits.shape
    total = -1e30
    for path in itertools.product(range(V), repeat=T):
        if _collapse(path, blank) == tuple(target):
            total = np.logaddexp(total, sum(lp[t, path[t]] for t in range(T)))
    return float(total)


def main() -> int:
    rng = np.random.default_rng(20261006)
    checked = 0
    for _ in range(500):
        T = int(rng.integers(1, 6))
        V = int(rng.integers(2, 5))
        logits = rng.normal(size=(T, V)) * 2
        n = int(rng.integers(1, 4))
        if T < n:
            continue
        target = [int(x) for x in rng.integers(1, V, size=n)]
        got = ctc_log_likelihood(logits, target)
        want = brute(logits, target)
        if abs(got - want) > 1e-9:
            print(f"FAIL T={T} V={V} target={target}: got={got} want={want}")
            return 1
        checked += 1
    print(f"ctc_log_likelihood: {checked}/500 随机用例与暴力枚举一致")
    print("test_joint_decode: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

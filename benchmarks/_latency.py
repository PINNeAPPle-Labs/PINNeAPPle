from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

try:
    import numpy as np
except Exception:
    np = None


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


def latency_summary(lat_ms: list[float]) -> dict[str, Any]:
    if len(lat_ms) == 0:
        return {
            "count": 0,
            "mean_ms": float("nan"),
            "p50_ms": float("nan"),
            "p90_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
            "max_ms": float("nan"),
        }

    if np is not None:
        arr = np.asarray(lat_ms, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }

    # fallback without numpy
    s = sorted(lat_ms)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return float(s[0])
        k = (p / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(s[int(k)])
        return float(s[f] + (s[c] - s[f]) * (k - f))

    mean = sum(s) / n
    return {
        "count": n,
        "mean_ms": float(mean),
        "p50_ms": pct(50),
        "p90_ms": pct(90),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": float(s[-1]),
    }

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TimeStats:
    count: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float


class Timer:
    def __init__(self):
        self._t0 = 0.0
        self.durations_ms: list[float] = []

    def tic(self):
        self._t0 = time.perf_counter()

    def toc(self):
        dt = (time.perf_counter() - self._t0) * 1000.0
        self.durations_ms.append(dt)
        return dt

    def stats(self) -> TimeStats:
        import numpy as np

        if not self.durations_ms:
            return TimeStats(0, 0, 0, 0, 0, 0)
        arr = np.array(self.durations_ms, dtype=np.float64)
        return TimeStats(
            count=int(arr.size),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(arr.mean()),
            max_ms=float(arr.max()),
        )

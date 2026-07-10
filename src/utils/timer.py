from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class RuntimeTracker:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self._global_start: float | None = None
        self._global_end: float | None = None

    def start_total(self) -> None:
        self._global_start = time.perf_counter()
        self._global_end = None

    def end_total(self) -> None:
        self._global_end = time.perf_counter()

    @property
    def total(self) -> float | None:
        if self._global_start is None:
            return None
        end = self._global_end if self._global_end is not None else time.perf_counter()
        return end - self._global_start

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def add(self, name: str, seconds: float | None) -> None:
        if seconds is None:
            return
        self.times[name] = self.times.get(name, 0.0) + float(seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "online_wall_clock_seconds": self.total,
            "time_breakdown": dict(self.times),
        }

"""Deterministic time helpers for ENS ordering."""

from __future__ import annotations

import threading
import time


_clock_lock = threading.Lock()
_last_seen_us = 0


def next_created_at_us() -> int:
    """Return monotonic microsecond UTC timestamp for ordering.

    Source clock: `time.time_ns() // 1000`
    Guard: `max(now_us, last_seen_us + 1)`
    """
    global _last_seen_us
    now_us = time.time_ns() // 1000
    with _clock_lock:
        candidate = now_us if now_us > _last_seen_us else (_last_seen_us + 1)
        _last_seen_us = candidate
        return candidate


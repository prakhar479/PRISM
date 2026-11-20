"""Token-budgeted batching with OOM backoff (pure; tested with injected FakeOOM)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from harness.device import empty_cache, is_oom

log = logging.getLogger("harness")


@dataclass
class Budget:
    """Tokens per forward batch (rows x sequence length). Halved on OOM down to `floor`; the smaller value is kept."""
    max_tokens: int
    max_rows: int = 64
    floor: int = 128
    backoffs: int = 0

    def rows_for(self, seq_len: int) -> int:
        return max(1, min(self.max_rows, self.max_tokens // max(1, seq_len)))

    def shrink(self) -> bool:
        if self.max_tokens <= self.floor:
            return False
        self.max_tokens = max(self.floor, self.max_tokens // 2)
        self.backoffs += 1
        log.warning("OOM: token budget -> %d", self.max_tokens)
        return True


def group_by_length(lengths: Sequence[int]) -> Dict[int, List[int]]:
    g: Dict[int, List[int]] = {}
    for i, L in enumerate(lengths):
        g.setdefault(int(L), []).append(i)
    return dict(sorted(g.items()))


def chunks(idx: Sequence[int], size: int) -> List[List[int]]:
    return [list(idx[i:i + size]) for i in range(0, len(idx), size)]


def run_with_backoff(fn: Callable[[Budget], object], budget: Budget, what: str = "", tries: int = 6):
    """Call fn(budget); on OOM free memory, halve the budget and call again. Re-raises other errors."""
    for attempt in range(tries + 1):
        try:
            return fn(budget)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or attempt == tries:
                raise
            empty_cache()
            if not budget.shrink():
                raise
            log.warning("retrying %s after OOM (%d/%d)", what, attempt + 1, tries)

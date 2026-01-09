from __future__ import annotations

from collections import deque
from typing import Optional


class HippotorchCurriculumCallback:
    """Simple curriculum driver for corridor-like envs with set_corridor_length.

    Compatible with SB3 callback API (on_step) but does not require SB3.
    Tracks a recent window of episode successes and increases length when
    threshold is met.
    """

    def __init__(
        self,
        *,
        min_length: int = 5,
        max_length: int = 30,
        step: int = 5,
        window: int = 10,
        threshold: float = 0.8,
    ) -> None:
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.step = int(step)
        self.window = int(window)
        self.threshold = float(threshold)
        self._successes: deque[float] = deque(maxlen=self.window)
        self._current_length: Optional[int] = None

    def on_episode_end(self, env, reward: float) -> None:
        self._maybe_init(env)
        self._successes.append(1.0 if reward > 0 else 0.0)
        if len(self._successes) == self.window and self._current_length is not None:
            rate = sum(self._successes) / self.window
            if rate >= self.threshold and self._current_length < self.max_length:
                new_L = min(self.max_length, self._current_length + self.step)
                try:
                    env.set_corridor_length(new_L)
                    self._current_length = new_L
                    self._successes.clear()
                except Exception:
                    pass

    # SB3-compatible hooks (no-ops unless plugged into SB3)
    def _maybe_init(self, env) -> None:
        if self._current_length is None:
            try:
                self._current_length = int(getattr(env, "length", self.min_length))
                if hasattr(env, "set_corridor_length"):
                    env.set_corridor_length(max(self._current_length, self.min_length))
            except Exception:
                self._current_length = self.min_length

    def on_step(self) -> bool:  # pragma: no cover - SB3 integration stub
        return True

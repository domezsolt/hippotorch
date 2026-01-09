from __future__ import annotations

from typing import Callable


def linear_mixture_schedule(
    start_ratio: float = 0.2,
    end_ratio: float = 0.7,
    warmup_steps: int = 10_000,
) -> Callable[[int], float]:
    """Linearly ramp semantic sampling ratio over a warmup period.

    Args:
        start_ratio: Initial semantic fraction (0.0-1.0).
        end_ratio: Final semantic fraction (0.0-1.0).
        warmup_steps: Number of steps to reach ``end_ratio``.

    Returns:
        schedule(step) -> float semantic ratio in [0.0, 1.0].
    """

    start_ratio = float(max(0.0, min(1.0, start_ratio)))
    end_ratio = float(max(0.0, min(1.0, end_ratio)))
    warmup_steps = int(max(1, warmup_steps))

    def schedule(step: int) -> float:
        if step >= warmup_steps:
            return end_ratio
        progress = max(0.0, float(step)) / float(warmup_steps)
        return start_ratio + progress * (end_ratio - start_ratio)

    return schedule


__all__ = ["linear_mixture_schedule"]

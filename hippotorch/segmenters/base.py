from __future__ import annotations

import abc
from typing import Callable, List

import torch
from torch import Tensor

from hippotorch.core.episode import Episode


class EpisodeSegmenter(abc.ABC):
    """Abstract interface for converting trajectories into episodes."""

    @abc.abstractmethod
    def segment(self, trajectory: Episode) -> List[Episode]:
        """Split a trajectory Episode into one or more Episodes."""
        raise NotImplementedError


class TerminalSegmenter(EpisodeSegmenter):
    """Splits trajectories at terminal flags."""

    def segment(self, trajectory: Episode) -> List[Episode]:
        if trajectory.dones is None:
            return [trajectory]

        done_indices = (
            torch.nonzero(trajectory.dones, as_tuple=False).flatten().tolist()
        )
        episodes: List[Episode] = []
        start = 0
        for idx in done_indices:
            end = idx + 1
            episodes.append(trajectory.slice(start, end))
            start = end
        if start < len(trajectory):
            episodes.append(trajectory.slice(start, len(trajectory)))
        return episodes


class FixedWindowSegmenter(EpisodeSegmenter):
    """Segments trajectories into fixed-length windows."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def segment(self, trajectory: Episode) -> List[Episode]:
        segments = []
        for start in range(0, len(trajectory), self.window_size):
            end = min(start + self.window_size, len(trajectory))
            segments.append(trajectory.slice(start, end))
        return segments


class GoalAchievementSegmenter(EpisodeSegmenter):
    """Segments based on a goal predicate over states."""

    def __init__(self, goal_fn: Callable[[Tensor], bool], min_length: int = 1) -> None:
        self.goal_fn = goal_fn
        self.min_length = min_length

    def segment(self, trajectory: Episode) -> List[Episode]:
        segments: List[Episode] = []
        start = 0
        for idx, state in enumerate(trajectory.states):
            goal_reached = self.goal_fn(state)
            if goal_reached and idx + 1 - start >= self.min_length:
                segments.append(trajectory.slice(start, idx + 1))
                start = idx + 1
        if start < len(trajectory):
            segments.append(trajectory.slice(start, len(trajectory)))
        return segments


class ChangePointSegmenter(EpisodeSegmenter):
    """Detects change points via state-space discontinuities."""

    def __init__(
        self, threshold: float = 2.0, min_length: int = 16, max_length: int = 256
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if min_length <= 0:
            raise ValueError("min_length must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.threshold = threshold
        self.min_length = min_length
        self.max_length = max_length

    def segment(self, trajectory: Episode) -> List[Episode]:
        if len(trajectory) < 2:
            return [trajectory]

        deltas = torch.norm(trajectory.states[1:] - trajectory.states[:-1], dim=-1)
        mean_delta = deltas.mean()
        std_delta = deltas.std().clamp(min=1e-6)

        change_points = [0]
        for i, delta in enumerate(deltas):
            since_last = i + 1 - change_points[-1]
            is_anomaly = delta > mean_delta + self.threshold * std_delta
            if (
                is_anomaly and since_last >= self.min_length
            ) or since_last >= self.max_length:
                change_points.append(i + 1)

        change_points.append(len(trajectory))

        merged_boundaries: List[tuple[int, int]] = []
        current_start = change_points[0]
        for next_point in change_points[1:]:
            if not merged_boundaries:
                proposed_start = current_start
            else:
                proposed_start = merged_boundaries[-1][0]

            segment_length = next_point - current_start
            if segment_length >= self.min_length:
                merged_boundaries.append((proposed_start, next_point))
                current_start = next_point
            else:
                if merged_boundaries:
                    merged_boundaries[-1] = (proposed_start, next_point)
                else:
                    merged_boundaries.append((0, next_point))
                current_start = next_point

        if not merged_boundaries:
            return [trajectory]

        return [trajectory.slice(start, end) for start, end in merged_boundaries]


__all__ = [
    "EpisodeSegmenter",
    "TerminalSegmenter",
    "FixedWindowSegmenter",
    "GoalAchievementSegmenter",
    "ChangePointSegmenter",
]

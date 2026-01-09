from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from hippotorch.utils.diagnostics import pca_keys


@dataclass
class ProjectionResult:
    coords: torch.Tensor  # [N, dims]
    rewards: List[float]
    contexts: List[int]


class MemoryProjector:
    def __init__(self, *, sample_size: int = 100, dims: int = 2) -> None:
        self.sample_size = int(sample_size)
        self.dims = int(dims)

    @torch.no_grad()
    def project(self, memory) -> ProjectionResult:
        n = min(self.sample_size, len(memory))
        if n == 0:
            return ProjectionResult(
                coords=torch.zeros(0, self.dims), rewards=[], contexts=[]
            )
        indices = torch.randperm(len(memory))[:n].tolist()
        keys = memory.keys[indices]
        comps, coords = pca_keys(keys, k=self.dims)
        rewards = [float(memory.episodes[i].total_reward) for i in indices]
        # contexts placeholder: 0 for lack of explicit context labels
        contexts = [0 for _ in indices]
        return ProjectionResult(coords=coords, rewards=rewards, contexts=contexts)


__all__ = ["MemoryProjector", "ProjectionResult"]

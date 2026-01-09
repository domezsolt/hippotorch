from __future__ import annotations

from dataclasses import dataclass

import torch

from hippotorch.memory.store import MemoryStore


@dataclass
class IntrinsicOutput:
    intrinsic_reward: torch.Tensor
    max_similarity: torch.Tensor


class IntrinsicConsolidator:
    """Compute intrinsic curiosity based on retrieval novelty.

    R_int = scale * max(0, target_similarity - max(similarity(query, top_k(keys))))
    """

    def __init__(
        self,
        *,
        memory: MemoryStore,
        encoder,
        target_similarity: float = 0.8,
        scale: float = 1.0,
        top_k: int = 5,
    ) -> None:
        self.memory = memory
        self.encoder = encoder
        self.target_similarity = float(target_similarity)
        self.scale = float(scale)
        self.top_k = int(top_k)

    @torch.no_grad()
    def compute(self, query_step: torch.Tensor) -> IntrinsicOutput:
        if len(self.memory) == 0:
            return IntrinsicOutput(
                intrinsic_reward=torch.tensor(self.scale * self.target_similarity),
                max_similarity=torch.tensor(0.0),
            )
        # form single-step sequence [1,1,D] expected by encoder
        x = query_step.view(1, 1, -1)
        q = self.encoder.encode_query(x)
        # Similarity via inverse distance to capture magnitude differences
        keys = self.memory.keys
        diffs = keys - q
        dists = diffs.pow(2).sum(dim=-1).sqrt()
        sims = 1.0 / (1.0 + dists)
        max_sim = sims.topk(min(self.top_k, sims.numel())).values.max()
        novelty = max(0.0, self.target_similarity - float(max_sim.item()))
        return IntrinsicOutput(
            intrinsic_reward=torch.tensor(self.scale * novelty),
            max_similarity=max_sim.detach().cpu(),
        )


__all__ = ["IntrinsicConsolidator", "IntrinsicOutput"]

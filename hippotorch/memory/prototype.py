from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from hippotorch.core.episode import Episode
from hippotorch.memory.store import MemoryStore


@dataclass
class PrototypeResult:
    num_candidates: int
    num_prototypes: int
    indices_pruned: List[int]


class PrototypeExtractor:
    def __init__(
        self, kmeans_iters: int = 10, keep_per_cluster: int = 1
    ) -> None:  # noqa: ARG002 - unused in stub
        self.kmeans_iters = kmeans_iters
        self.keep_per_cluster = keep_per_cluster

    @torch.no_grad()
    def extract_prototypes(
        self,
        memory: MemoryStore,
        encoder,
        *,
        num_prototypes: int = 2,
        reward_percentile: float = 0.5,
        prune_sources: bool = True,
    ) -> PrototypeResult:
        if len(memory) == 0:
            return PrototypeResult(0, 0, [])
        rewards = torch.tensor(
            [ep.total_reward for ep in memory.episodes], dtype=torch.float
        )
        # lower interpolation favors inclusion at the boundary
        threshold = torch.quantile(
            rewards, q=float(reward_percentile), interpolation="lower"
        )
        candidate_indices = [
            i for i, r in enumerate(rewards.tolist()) if r >= float(threshold)
        ]

        # Build a simple prototype by averaging candidate episodes' tensors
        candidates = [memory.episodes[i] for i in candidate_indices]
        tensors = [memory._episode_to_tensor(ep) for ep in candidates]
        max_len = max(t.shape[0] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - t.shape[0], t.shape[1], dtype=t.dtype, device=t.device
                )
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        avg = torch.stack(padded).mean(dim=0)
        # Split back to Episode
        state_dim = memory.episodes[0].states.shape[-1]
        action_dim = memory.episodes[0].actions.shape[-1]
        states = avg[:, :state_dim]
        actions = avg[:, state_dim : state_dim + action_dim]
        rewards_avg = avg[:, -1]
        proto = Episode(states=states, actions=actions, rewards=rewards_avg)

        indices_pruned: List[int] = []
        if prune_sources:
            # prune candidate sources and append prototype
            # remove from highest to lowest to keep indices stable
            for i in sorted(candidate_indices, reverse=True):
                # delete from keys and metadata
                if memory.keys.numel() > 0:
                    if i == 0:
                        memory.keys = memory.keys[1:]
                    elif i == memory.keys.shape[0] - 1:
                        memory.keys = memory.keys[:-1]
                    else:
                        memory.keys = torch.cat(
                            [memory.keys[:i], memory.keys[i + 1 :]], dim=0
                        )
                memory.episodes.pop(i)
                memory.key_timestamps.pop(i)
                indices_pruned.append(i)

        # append prototype at end
        x = memory._episode_to_tensor(proto).unsqueeze(0)
        key = encoder.encode_key(x)
        memory.write(key, [proto])

        return PrototypeResult(
            num_candidates=len(candidate_indices),
            num_prototypes=1,
            indices_pruned=indices_pruned,
        )


__all__ = ["PrototypeExtractor", "PrototypeResult"]

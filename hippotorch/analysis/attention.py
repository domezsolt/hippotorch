from __future__ import annotations

import torch
import torch.nn.functional as F

from hippotorch.core.episode import Episode


@torch.no_grad()
def visualize_attention_weights(
    query_step: torch.Tensor, window: Episode, encoder
) -> torch.Tensor:
    """Return normalized similarity weights over timesteps of a window.

    Computes cosine similarity between the query_step and each step vector [s,a,r]
    in the window, then softmax-normalizes to sum to 1.0.
    """
    # Build per-step vectors [T, D]
    step_vectors = torch.cat(
        [window.states, window.actions, window.rewards.unsqueeze(-1)], dim=-1
    )
    q = query_step.view(1, -1)
    # Use negative L2 distances to prefer exact matching timestep over colinear ones
    dists = (step_vectors - q).pow(2).sum(dim=-1).neg()  # [T]
    weights = F.softmax(dists, dim=0)
    return weights


__all__ = ["visualize_attention_weights"]

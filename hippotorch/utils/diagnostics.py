from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch

try:  # optional
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from hippotorch.memory.store import MemoryStore


def pca_keys(keys: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute PCA(k) for key matrix [N, D]. Returns (components, coords).

    components: [D, k], coords: [N, k]
    """
    if keys.dim() != 2:
        raise ValueError("keys must be 2D [N, D]")
    X = keys - keys.mean(dim=0, keepdim=True)
    cov = X.T @ X / max(1, X.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    comps = eigvecs[:, -k:]
    coords = X @ comps
    return comps, coords


def pca_keys_with_rewards(
    keys: torch.Tensor, rewards: torch.Tensor, k: int = 2
) -> Dict[str, torch.Tensor]:
    comps, coords = pca_keys(keys, k=k)
    return {"components": comps, "coords": coords, "rewards": rewards}


def log_memory_pca(
    writer: "SummaryWriter",
    memory: "MemoryStore",
    *,
    step: int,
    sample_size: int = 100,
    tag: str = "memory/pca",
) -> None:
    """Log PCA(2) of memory keys to TensorBoard embedding projector.

    Labels use episode total rewards for simple coloring in projector UI.
    """
    if writer is None:  # pragma: no cover
        return
    n = min(sample_size, len(memory))
    if n <= 1:
        return
    import random

    indices = random.sample(range(len(memory)), n)
    keys = memory.keys[indices].detach().cpu()
    rewards = torch.tensor(
        [memory.episodes[i].total_reward for i in indices], dtype=torch.float
    )
    _, coords = pca_keys(keys, k=2)
    metadata = [f"{float(r):.2f}" for r in rewards]
    writer.add_embedding(coords, metadata=metadata, tag=tag, global_step=int(step))


__all__ = ["pca_keys", "pca_keys_with_rewards", "log_memory_pca"]

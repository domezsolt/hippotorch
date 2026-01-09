from __future__ import annotations

import torch
import torch.nn.functional as F

_FAISS_AVAILABLE = False


class ApproximateNearestNeighborIndex:
    """Fallback ANN index that uses exact matmul as a stub.

    Rebuild stores a CPU copy of keys; search computes cosine sim and topk.
    """

    def __init__(self) -> None:
        self._keys = None

    def rebuild(self, keys: torch.Tensor) -> None:
        self._keys = keys.detach().cpu()

    def search(self, queries: torch.Tensor, top_k: int):
        if self._keys is None or self._keys.numel() == 0:
            return torch.empty(queries.size(0), 0), torch.empty(
                queries.size(0), 0, dtype=torch.long
            )
        q = F.normalize(queries.detach().cpu(), dim=-1)
        k = F.normalize(self._keys, dim=-1)
        scores = q @ k.t()
        return scores.topk(min(top_k, k.shape[0]), dim=-1)


__all__ = ["ApproximateNearestNeighborIndex", "_FAISS_AVAILABLE"]

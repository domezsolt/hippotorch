from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import Tensor, nn

from .backbone import EpisodeEncoderBackbone


class DualEncoder(nn.Module):
    """Dual encoder with momentum-updated target network."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        momentum: float = 0.995,
        refresh_interval: int = 10000,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.refresh_interval = refresh_interval
        self._steps_since_refresh = 0
        self.online = EpisodeEncoderBackbone(input_dim, embed_dim=embed_dim)
        self.target = copy.deepcopy(self.online)
        self.projector = nn.Sequential(
            nn.Linear(self.online.output_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self._sync_target()

    def _sync_target(self) -> None:
        for t_param, o_param in zip(self.target.parameters(), self.online.parameters()):
            t_param.data.copy_(o_param.data)
            t_param.requires_grad = False

    @torch.no_grad()
    def update_target(self) -> None:
        for t_param, o_param in zip(self.target.parameters(), self.online.parameters()):
            t_param.data = (
                self.momentum * t_param.data + (1 - self.momentum) * o_param.data
            )
        self._steps_since_refresh += 1

    def encode_query(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        features = self.online(x, mask=mask)
        pooled = features.mean(dim=1)
        return self.projector(pooled)

    @torch.no_grad()
    def encode_key(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        features = self.target(x, mask=mask)
        pooled = features.mean(dim=1)
        return self.projector(pooled)

    def should_refresh_keys(self) -> bool:
        """Return True when stored keys should be re-encoded."""
        return self._steps_since_refresh >= self.refresh_interval

    def mark_refreshed(self) -> None:
        """Reset refresh counter after an external key refresh."""
        self._steps_since_refresh = 0


__all__ = ["DualEncoder"]

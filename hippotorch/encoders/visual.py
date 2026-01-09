from __future__ import annotations

import torch
from torch import nn


class VisualEpisodeEncoder(nn.Module):
    """Minimal visual encoder that maps frames to per-timestep embeddings.

    Input: [B, T, C, H, W] -> Output: [B, T, output_dim]
    """

    def __init__(
        self,
        image_channels: int = 3,
        backbone: str = "nature_cnn",
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        # Simple conv trunk approximating NatureCNN shapes
        self.trunk = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[0], x.shape[1]
        x = x.view(B * T, *x.shape[2:])
        feats = self.trunk(x).view(B * T, -1)
        out = self.proj(feats).view(B, T, self.output_dim)
        return out


__all__ = ["VisualEpisodeEncoder"]

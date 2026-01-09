from __future__ import annotations

from typing import Optional

from torch import Tensor, nn


class EpisodeEncoderBackbone(nn.Module):
    """Hybrid Transformer + LSTM encoder for episodic sequences."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        lstm_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lstm = nn.LSTM(
            embed_dim, lstm_hidden, batch_first=True, bidirectional=True
        )
        self.output_dim = lstm_hidden * 2

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encode sequence [B, T, D] -> [B, T, output_dim]."""
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        outputs, _ = self.lstm(x)
        return outputs


__all__ = ["EpisodeEncoderBackbone"]

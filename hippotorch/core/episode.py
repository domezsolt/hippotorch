from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch
from torch import Tensor


@dataclass
class Episode:
    """Container for episodic transitions.

    Each episode is stored as parallel tensors for states, actions, and rewards.
    Optional fields include `dones` and `infos` for compatibility with common
    RL interfaces. All tensors share the leading time dimension.
    """

    states: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Optional[Tensor] = None
    infos: Optional[Iterable[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        time_dim = self.states.shape[0]
        for name, tensor in [("actions", self.actions), ("rewards", self.rewards)]:
            if tensor.shape[0] != time_dim:
                raise ValueError(
                    f"Episode {name} length {tensor.shape[0]} != {time_dim}"
                )
        if self.dones is not None and self.dones.shape[0] != time_dim:
            raise ValueError("Episode dones length mismatch")
        if self.infos is not None:
            if not isinstance(self.infos, list):
                self.infos = list(self.infos)
            if len(self.infos) != time_dim:
                raise ValueError("Episode infos length mismatch")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.states.shape[0]

    @property
    def device(self) -> torch.device:
        return self.states.device

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum().item())

    def slice(self, start: int, end: int) -> "Episode":
        """Return a shallow slice preserving tensor types."""
        return Episode(
            states=self.states[start:end],
            actions=self.actions[start:end],
            rewards=self.rewards[start:end],
            dones=self.dones[start:end] if self.dones is not None else None,
            infos=list(self.infos)[start:end] if self.infos is not None else None,
        )

    def to(self, device: torch.device) -> "Episode":
        """Move all tensor fields to the target device."""
        return Episode(
            states=self.states.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device) if self.dones is not None else None,
            infos=self.infos,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize episode tensors for checkpointing."""
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "infos": self.infos,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Episode":
        return cls(
            states=payload["states"],
            actions=payload["actions"],
            rewards=payload["rewards"],
            dones=payload.get("dones"),
            infos=payload.get("infos"),
        )


__all__ = ["Episode"]

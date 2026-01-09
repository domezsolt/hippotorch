from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from hippotorch.core.episode import Episode
from hippotorch.memory.replay import HippocampalReplayBuffer
from hippotorch.segmenters.base import EpisodeSegmenter, TerminalSegmenter


class SB3ReplayBufferWrapper:
    """Adapter that exposes a stable-baselines3 style API.

    The wrapper buffers transitions into trajectories using the provided
    ``EpisodeSegmenter`` and forwards finalized episodes to the underlying
    ``HippocampalReplayBuffer``. Sampling returns tensors keyed to align with
    SB3's replay buffer outputs (observations, actions, rewards, next_observations, dones).
    """

    def __init__(
        self,
        buffer: HippocampalReplayBuffer,
        segmenter: Optional[EpisodeSegmenter] = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.buffer = buffer
        self.segmenter = segmenter or TerminalSegmenter()
        self.device = torch.device(device)
        self._trajectory: Dict[str, List[Tensor]] = self._empty_trajectory()

    @property
    def size(self) -> int:
        """Number of episodes currently stored."""

        return self.buffer.size

    def add(
        self,
        obs: Tensor | float | int,
        next_obs: Tensor | float | int,
        action: Tensor | float | int,
        reward: float,
        done: bool,
        info: Optional[dict] = None,
    ) -> None:
        """Record a transition and flush completed trajectories into memory."""

        _ = info  # unused placeholder for API compatibility
        self._trajectory["states"].append(self._to_tensor(obs))
        self._trajectory["actions"].append(self._to_tensor(action))
        self._trajectory["rewards"].append(torch.tensor([reward], device=self.device))
        self._trajectory["dones"].append(torch.tensor(done, device=self.device))

        if done:
            episode = self._trajectory_to_episode(self._trajectory)
            for segment in self.segmenter.segment(episode):
                self.buffer.add_episode(segment)
            self._trajectory = self._empty_trajectory()
        else:
            # Prepare for the next step by seeding the next state
            self._trajectory["states"].append(self._to_tensor(next_obs))
            self._trajectory["actions"].append(
                torch.zeros_like(self._trajectory["actions"][-1])
            )
            self._trajectory["rewards"].append(
                torch.zeros_like(self._trajectory["rewards"][-1])
            )
            self._trajectory["dones"].append(torch.tensor(False, device=self.device))

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """Draw a batch of transitions using hybrid sampling."""

        return self.buffer.sample(batch_size=batch_size, query_state=None)

    def consolidate(self, **kwargs) -> Dict[str, float]:
        """Proxy to the underlying buffer consolidation method."""

        return self.buffer.consolidate(**kwargs)

    def _trajectory_to_episode(self, trajectory: Dict[str, List[Tensor]]) -> Episode:
        states = torch.stack(trajectory["states"]).to(self.device)
        actions = torch.stack(trajectory["actions"]).to(self.device)
        rewards = torch.stack(trajectory["rewards"]).squeeze(-1).to(self.device)
        dones = torch.stack(trajectory["dones"]).to(self.device)
        return Episode(states=states, actions=actions, rewards=rewards, dones=dones)

    def _empty_trajectory(self) -> Dict[str, List[Tensor]]:
        return {"states": [], "actions": [], "rewards": [], "dones": []}

    def _to_tensor(self, array_like) -> Tensor:
        tensor = (
            array_like
            if isinstance(array_like, torch.Tensor)
            else torch.as_tensor(array_like)
        )
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)


__all__ = ["SB3ReplayBufferWrapper"]

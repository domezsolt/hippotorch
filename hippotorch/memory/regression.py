from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from hippotorch.core.episode import Episode
from hippotorch.encoders.dual import DualEncoder
from hippotorch.memory.replay import HippocampalReplayBuffer
from hippotorch.memory.store import MemoryStore


@dataclass
class RegressionResult:
    """Lightweight container for regression harness outputs."""

    semantic_mean_reward: float
    uniform_mean_reward: float
    reward_improvement: float
    semantic_stats: Dict[str, float]
    uniform_stats: Dict[str, float]


class IdentityDualEncoder(nn.Module):
    """Deterministic encoder for synthetic regression experiments.

    The module pools inputs by mean across the sequence dimension and applies a
    linear projection to produce stable embeddings. It intentionally omits
    contrastive training dynamics; the goal is to create predictable similarity
    relationships so semantic sampling can be evaluated without optimization
    noise.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int | None = None,
        refresh_interval: int = 10_000,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim or input_dim
        self.refresh_interval = refresh_interval
        self._steps_since_refresh = 0
        self.projector = nn.Linear(input_dim, self.embed_dim, bias=False)
        nn.init.eye_(self.projector.weight)

    def encode_query(
        self, x: Tensor, mask: Tensor | None = None
    ) -> Tensor:  # noqa: ARG002 - mask unused
        pooled = x.mean(dim=1)
        # Align input dim if mismatched
        in_feats = self.projector.in_features
        if pooled.shape[-1] > in_feats:
            pooled = pooled[..., :in_feats]
        elif pooled.shape[-1] < in_feats:
            pad = torch.zeros(
                pooled.shape[0],
                in_feats - pooled.shape[-1],
                device=pooled.device,
                dtype=pooled.dtype,
            )
            pooled = torch.cat([pooled, pad], dim=-1)
        return self.projector(pooled)

    def encode_key(
        self, x: Tensor, mask: Tensor | None = None
    ) -> Tensor:  # noqa: ARG002 - mask unused
        pooled = x.mean(dim=1)
        in_feats = self.projector.in_features
        if pooled.shape[-1] > in_feats:
            pooled = pooled[..., :in_feats]
        elif pooled.shape[-1] < in_feats:
            pad = torch.zeros(
                pooled.shape[0],
                in_feats - pooled.shape[-1],
                device=pooled.device,
                dtype=pooled.dtype,
            )
            pooled = torch.cat([pooled, pad], dim=-1)
        return self.projector(pooled)

    def update_target(self) -> None:
        self._steps_since_refresh += 1

    def should_refresh_keys(self) -> bool:
        return self._steps_since_refresh >= self.refresh_interval

    def mark_refreshed(self) -> None:
        self._steps_since_refresh = 0

    @property
    def online(self) -> nn.Module:
        # Matches attribute expected by Consolidator; unused in regression harness
        return self.projector


def _generate_episode(
    length: int,
    state_dim: int,
    action_dim: int,
    state_value: float,
    reward_value: float,
) -> Episode:
    states = torch.full((length, state_dim), state_value)
    actions = torch.zeros(length, action_dim)
    rewards = torch.full((length,), reward_value)
    dones = torch.zeros(length, dtype=torch.bool)
    return Episode(states=states, actions=actions, rewards=rewards, dones=dones)


def _build_buffer(
    encoder: DualEncoder,
    episodes: List[Episode],
    *,
    mixture_ratio: float,
    top_k: int = 5,
) -> Tuple[HippocampalReplayBuffer, Tensor]:
    memory = MemoryStore(embed_dim=encoder.projector.out_features)
    buffer = HippocampalReplayBuffer(
        memory=memory, encoder=encoder, mixture_ratio=mixture_ratio
    )
    for idx, episode in enumerate(episodes):
        buffer.add_episode(episode, encoder_step=idx)
    first = episodes[0]
    query_state = torch.cat(
        [first.states[0], first.actions[0], first.rewards[0].unsqueeze(0)]
    ).unsqueeze(0)
    return buffer, query_state


def run_replay_regression(
    *,
    num_good: int = 8,
    num_distractor: int = 32,
    episode_length: int = 4,
    state_dim: int = 3,
    action_dim: int = 2,
    mixture_ratio: float = 0.6,
    batch_size: int = 16,
    top_k: int = 5,
    seed: int = 0,
) -> RegressionResult:
    """Compare semantic vs uniform replay on a synthetic reward-alignment task."""

    torch.manual_seed(seed)
    embed_dim = state_dim + action_dim + 1
    encoder = IdentityDualEncoder(input_dim=embed_dim)

    good_eps = [
        _generate_episode(
            episode_length, state_dim, action_dim, state_value=1.0, reward_value=1.0
        )
        for _ in range(num_good)
    ]
    distractor_eps = [
        _generate_episode(
            episode_length, state_dim, action_dim, state_value=-1.0, reward_value=0.1
        )
        for _ in range(num_distractor)
    ]
    episodes = good_eps + distractor_eps

    semantic_buffer, query_state = _build_buffer(
        encoder, episodes, mixture_ratio=mixture_ratio, top_k=top_k
    )
    uniform_buffer, _ = _build_buffer(encoder, episodes, mixture_ratio=0.0, top_k=top_k)

    semantic_batch = semantic_buffer.sample(
        batch_size, query_state=query_state, top_k=top_k
    )
    uniform_batch = uniform_buffer.sample(batch_size, query_state=None, top_k=top_k)

    semantic_mean_reward = float(semantic_batch["rewards"].mean().item())
    uniform_mean_reward = float(uniform_batch["rewards"].mean().item())
    improvement = (semantic_mean_reward - uniform_mean_reward) / max(
        abs(uniform_mean_reward), 1e-6
    )

    return RegressionResult(
        semantic_mean_reward=semantic_mean_reward,
        uniform_mean_reward=uniform_mean_reward,
        reward_improvement=improvement,
        semantic_stats=semantic_buffer.stats(),
        uniform_stats=uniform_buffer.stats(),
    )


__all__ = ["RegressionResult", "IdentityDualEncoder", "run_replay_regression"]

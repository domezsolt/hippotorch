import pytest

torch = pytest.importorskip("torch")

from hippotorch.core.episode import Episode
from hippotorch.memory.consolidator import Consolidator
from hippotorch.memory.regression import IdentityDualEncoder
from hippotorch.memory.store import MemoryStore


def _make_episode(
    length: int, state_dim: int, action_dim: int, reward_value: float
) -> Episode:
    states = torch.randn(length, state_dim)
    actions = torch.randn(length, action_dim)
    rewards = torch.full((length,), reward_value)
    dones = torch.zeros(length, dtype=torch.bool)
    return Episode(states=states, actions=actions, rewards=rewards, dones=dones)


def _to_tensor(episode: Episode) -> torch.Tensor:
    return torch.cat(
        [
            episode.states,
            episode.actions,
            episode.rewards.unsqueeze(-1),
        ],
        dim=-1,
    )


def test_consolidator_reports_temporal_ablation_loss():
    state_dim, action_dim = 3, 2
    embed_dim = state_dim + action_dim + 1
    encoder = IdentityDualEncoder(input_dim=embed_dim, refresh_interval=3)
    memory = MemoryStore(embed_dim=embed_dim, capacity=8)

    episodes = [
        _make_episode(5, state_dim, action_dim, reward_value=1.5),
        _make_episode(5, state_dim, action_dim, reward_value=0.2),
        _make_episode(5, state_dim, action_dim, reward_value=1.0),
    ]

    keys = []
    for idx, episode in enumerate(episodes):
        episode_tensor = _to_tensor(episode).unsqueeze(0)
        key = encoder.encode_key(episode_tensor).squeeze(0)
        keys.append(key)
    memory.write(torch.stack(keys), episodes, encoder_step=0)

    consolidator = Consolidator(
        encoder,
        temperature=0.1,
        reward_weight=0.5,
        learning_rate=1e-3,
    )

    metrics = consolidator.sleep(
        memory,
        steps=2,
        batch_size=4,
        refresh_keys=False,
        ablation_temporal_only=True,
    )

    assert "ablation_temporal_loss" in metrics
    assert metrics["ablation_temporal_loss"] >= 0
    assert metrics["reward_loss"] >= 0

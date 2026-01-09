import pytest

torch = pytest.importorskip("torch")

from hippotorch.core.episode import Episode
from hippotorch.encoders.dual import DualEncoder
from hippotorch.memory.replay import HippocampalReplayBuffer
from hippotorch.memory.store import MemoryStore


def _make_episode(length: int, state_dim: int = 2, action_dim: int = 1) -> Episode:
    states = torch.randn(length, state_dim)
    actions = torch.randn(length, action_dim)
    rewards = torch.ones(length)
    dones = torch.zeros(length, dtype=torch.bool)
    return Episode(states=states, actions=actions, rewards=rewards, dones=dones)


def test_mixture_schedule_transitions_between_uniform_and_semantic():
    input_dim = 4  # state_dim + action_dim + reward
    encoder = DualEncoder(
        input_dim=input_dim, embed_dim=8, momentum=0.9, refresh_interval=10
    )
    memory = MemoryStore(embed_dim=8, capacity=10)

    buffer = HippocampalReplayBuffer(
        memory=memory,
        encoder=encoder,
        mixture_ratio=0.0,
        mixture_schedule=lambda step: 0.0 if step < 2 else 1.0,
    )

    episode = _make_episode(3)
    buffer.add_episode(episode, encoder_step=0)
    buffer.add_episode(episode.slice(0, 2), encoder_step=1)

    query = torch.cat(
        [episode.states[0], episode.actions[0], episode.rewards[0].unsqueeze(0)]
    )

    _ = buffer.sample(batch_size=2, query_state=query, top_k=2)
    stats_uniform = buffer.stats()
    assert stats_uniform["effective_ratio"] == 0.0

    _ = buffer.sample(batch_size=2, query_state=query, top_k=2)
    stats_semantic = buffer.stats()
    assert stats_semantic["semantic"] > 0
    assert stats_semantic["effective_ratio"] > 0.0

    assert stats_semantic["memory_size"] == 2

import pytest

torch = pytest.importorskip("torch")

from hippotorch.core.episode import Episode
from hippotorch.memory.store import MemoryStore


def test_episode_slice_and_serialize():
    states = torch.randn(4, 3)
    actions = torch.randn(4, 2)
    rewards = torch.tensor([1.0, 0.5, -0.25, 2.0])
    dones = torch.tensor([False, False, True, True])

    episode = Episode(states=states, actions=actions, rewards=rewards, dones=dones)
    sliced = episode.slice(1, 3)

    assert len(sliced) == 2
    assert torch.allclose(sliced.rewards, rewards[1:3])

    payload = episode.to_dict()
    restored = Episode.from_dict(payload)
    assert torch.allclose(restored.states, states)
    assert restored.total_reward == rewards.sum().item()


def test_memory_store_retrieve_and_refresh():
    embed_dim = 6
    store = MemoryStore(embed_dim=embed_dim, capacity=5)

    base_ep = Episode(
        states=torch.randn(3, 2),
        actions=torch.randn(3, 1),
        rewards=torch.ones(3),
        dones=None,
    )

    keys = torch.randn(2, embed_dim)
    episodes = [base_ep, base_ep.slice(0, 2)]
    store.write(keys, episodes, encoder_step=1)

    queries = torch.randn(1, embed_dim)
    scores, retrieved = store.retrieve(queries, top_k=2)
    assert scores.shape == (1, min(2, len(store)))
    assert len(retrieved[0]) == min(2, len(store))

    stale = store.get_stale_keys(current_step=5, threshold=2)
    assert stale, "stale keys should be detected when threshold exceeded"

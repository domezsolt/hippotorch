import pytest

torch = pytest.importorskip("torch")

from hippotorch.core.episode import Episode
from hippotorch.memory.consolidator import Consolidator
from hippotorch.memory.regression import IdentityDualEncoder
from hippotorch.memory.replay import HippocampalReplayBuffer
from hippotorch.memory.store import MemoryStore


def _make_episode(
    T: int = 6, state_dim: int = 3, action_dim: int = 1, reward_value: float = 1.0
) -> Episode:
    states = torch.randn(T, state_dim)
    actions = torch.randn(T, action_dim)
    rewards = torch.full((T,), reward_value)
    dones = torch.zeros(T, dtype=torch.bool)
    return Episode(states=states, actions=actions, rewards=rewards, dones=dones)


def test_lazy_key_refresh_on_retrieve_updates_key_and_timestamp():
    embed_dim = (
        6  # state_dim + action_dim + 1 using defaults in IdentityDualEncoder helpers
    )
    encoder = IdentityDualEncoder(input_dim=embed_dim, refresh_interval=5)
    store = MemoryStore(embed_dim=embed_dim, capacity=4)

    # build a single episode and write its key at step 0
    ep = _make_episode(T=4, state_dim=3, action_dim=2, reward_value=2.0)
    x = torch.cat([ep.states, ep.actions, ep.rewards.unsqueeze(-1)], dim=-1).unsqueeze(
        0
    )
    key0 = encoder.encode_key(x).squeeze(0)
    store.write(key0.unsqueeze(0), [ep], encoder_step=0)

    # form a query equal to the stored key to ensure it is retrieved
    query = key0.unsqueeze(0)

    # mutate encoder so refreshed key differs (scale projector weights)
    with torch.no_grad():
        encoder.projector.weight.mul_(2.0)

    # mark key as stale relative to current step and retrieve with lazy_refresh
    current_step = encoder.refresh_interval
    store.key_timestamps[0] = 0
    scores, eps = store.retrieve(
        query, top_k=1, encoder=encoder, current_step=current_step, lazy_refresh=True
    )
    assert scores.shape[-1] == 1
    assert eps[0], "should retrieve at least one episode"

    # key should have been updated to new encoder output and timestamp bumped
    new_key = encoder.encode_key(x).squeeze(0)
    assert torch.allclose(store.keys[0], new_key), "lazy refresh should update the key"
    assert store.key_timestamps[0] == current_step


def test_sample_windows_shapes_and_lengths():
    # Use IdentityDualEncoder for deterministic shapes
    state_dim, action_dim = 3, 1
    embed_dim = state_dim + action_dim + 1
    encoder = IdentityDualEncoder(input_dim=embed_dim)
    memory = MemoryStore(embed_dim=embed_dim, capacity=10)
    buffer = HippocampalReplayBuffer(memory=memory, encoder=encoder, mixture_ratio=0.0)

    # add a few episodes of sufficient length
    for r in [1.0, 0.5, 1.5]:
        ep = _make_episode(
            T=8, state_dim=state_dim, action_dim=action_dim, reward_value=r
        )
        # prepare key and store via buffer API
        x = torch.cat(
            [ep.states, ep.actions, ep.rewards.unsqueeze(-1)], dim=-1
        ).unsqueeze(0)
        key = encoder.encode_key(x)
        memory.write(key, [ep], encoder_step=0)

    batch = buffer.sample_windows(batch_size=3, window_size=5, query_state=None)
    assert set(batch.keys()) == {"states", "actions", "rewards", "dones", "lengths"}
    assert batch["states"].shape[:2] == (3, 5)
    assert batch["actions"].shape[:2] == (3, 5)
    assert batch["rewards"].shape[:2] == (3, 5)
    assert batch["dones"].shape[:2] == (3, 5)
    assert batch["lengths"].shape[0] == 3
    # windows are fixed-size slices; lengths should equal window_size
    assert torch.all(batch["lengths"] == 5)


def test_rank_weighting_invariant_to_scale():
    # construct anchors/positives and weights, compare loss with scaled weights
    B, D = 6, 8
    # simple identity encoder for consolidator (unused in this test)
    encoder = IdentityDualEncoder(input_dim=D)
    cons = Consolidator(encoder, temperature=0.1, reward_weight=0.5, learning_rate=1e-3)

    anchors = torch.randn(B, D)
    positives = anchors + 0.01 * torch.randn(B, D)  # keep positives near anchors
    weights = torch.tensor([0.1, 0.5, 0.2, 0.9, 0.3, 0.7], dtype=torch.float)
    loss1 = cons._reward_weighted_info_nce(anchors, positives, weights)
    loss2 = cons._reward_weighted_info_nce(anchors, positives, 10 * weights)
    assert torch.isfinite(loss1) and torch.isfinite(loss2)
    assert abs(loss1.item() - loss2.item()) < 1e-6

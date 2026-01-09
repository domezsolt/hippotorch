import pytest

torch = pytest.importorskip("torch")
from torch import nn

from hippotorch.analysis.attention import visualize_attention_weights
from hippotorch.analysis.projector import MemoryProjector
from hippotorch.core.episode import Episode
from hippotorch.memory.intrinsic import IntrinsicConsolidator
from hippotorch.memory.store import MemoryStore


def _episode_to_tensor(episode: Episode) -> torch.Tensor:
    return torch.cat(
        [episode.states, episode.actions, episode.rewards.unsqueeze(-1)], dim=-1
    )


class _DummyEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.refresh_interval = 1
        self.online = nn.Identity()
        self.target = nn.Identity()
        self.projector = nn.Identity()
        self._device_param = nn.Parameter(torch.zeros(1))
        self.input_dim = input_dim

    def encode_query(
        self, x: torch.Tensor, mask=None
    ):  # noqa: D401, ARG002 - mask unused
        return x.mean(dim=1)

    def encode_key(
        self, x: torch.Tensor, mask=None
    ):  # noqa: D401, ARG002 - mask unused
        return x.mean(dim=1)

    def update_target(self) -> None:
        pass

    def should_refresh_keys(self) -> bool:
        return False

    def mark_refreshed(self) -> None:
        pass

    def parameters(
        self, recurse: bool = True
    ):  # noqa: D401 - delegate to internal param for device checks
        return [self._device_param]


def _build_episode(length: int, value: float) -> Episode:
    states = torch.full((length, 2), value)
    actions = torch.ones(length, 1) * value
    rewards = torch.ones(length) * value
    dones = torch.zeros(length, dtype=torch.bool)
    return Episode(states=states, actions=actions, rewards=rewards, dones=dones)


def test_memory_projector_returns_coords() -> None:
    encoder = _DummyEncoder(input_dim=4)
    memory = MemoryStore(embed_dim=encoder.input_dim)

    for idx, val in enumerate([0.1, 0.5, 0.9, 1.3]):
        episode = _build_episode(3, val)
        key = encoder.encode_key(_episode_to_tensor(episode).unsqueeze(0))
        memory.write(key, [episode], encoder_step=idx)

    projector = MemoryProjector(sample_size=3, dims=2)
    result = projector.project(memory)

    assert result.coords.shape[1] == 2
    assert len(result.rewards) == len(result.contexts) == min(3, len(memory))
    assert result.coords.shape[0] == len(result.rewards)


def test_attention_weights_peak_at_matching_step() -> None:
    encoder = _DummyEncoder(input_dim=4)
    window = _build_episode(length=3, value=0.0)
    window.states[1] = torch.tensor([1.0, 1.0])
    window.states[2] = torch.tensor([2.0, 2.0])
    query_step = _episode_to_tensor(window)[2]

    weights = visualize_attention_weights(query_step, window, encoder)

    assert weights.shape[0] == len(window)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-4)
    assert int(weights.argmax().item()) == 2


def test_intrinsic_consolidator_rewards_novel_queries() -> None:
    encoder = _DummyEncoder(input_dim=4)
    memory = MemoryStore(embed_dim=encoder.input_dim)
    base_episode = _build_episode(4, 0.0)
    novel_episode = _build_episode(4, 2.0)

    memory.write(
        encoder.encode_key(_episode_to_tensor(base_episode).unsqueeze(0)),
        [base_episode],
        encoder_step=0,
    )
    memory.write(
        encoder.encode_key(_episode_to_tensor(novel_episode).unsqueeze(0)),
        [novel_episode],
        encoder_step=1,
    )

    intrinsic = IntrinsicConsolidator(
        memory=memory, encoder=encoder, target_similarity=0.8, scale=1.0, top_k=2
    )

    baseline_query = _episode_to_tensor(base_episode)[0]
    novel_query = torch.ones_like(baseline_query) * 3.0

    baseline_output = intrinsic.compute(baseline_query)
    curious_output = intrinsic.compute(novel_query)

    assert curious_output.intrinsic_reward.item() > 0
    assert baseline_output.intrinsic_reward.item() == 0

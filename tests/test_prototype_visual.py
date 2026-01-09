import torch

from hippotorch.core.episode import Episode
from hippotorch.encoders.visual import VisualEpisodeEncoder
from hippotorch.memory.prototype import PrototypeExtractor
from hippotorch.memory.regression import IdentityDualEncoder
from hippotorch.memory.store import MemoryStore


def _make_episode(
    state_value: float, reward_value: float, *, length: int = 3
) -> Episode:
    states = torch.full((length, 2), state_value)
    actions = torch.zeros(length, 1)
    rewards = torch.full((length,), reward_value)
    return Episode(states=states, actions=actions, rewards=rewards)


def test_prototype_extraction_compresses_memory() -> None:
    torch.manual_seed(0)
    episodes = [
        _make_episode(1.0, 1.0),
        _make_episode(1.1, 1.2),
        _make_episode(0.9, 0.9),
        _make_episode(-1.0, 0.1),
    ]

    encoder = IdentityDualEncoder(input_dim=3)
    memory = MemoryStore(embed_dim=encoder.projector.out_features)
    tensors = memory.episodes_to_tensor(episodes)
    memory.write(encoder.encode_key(tensors), episodes)

    extractor = PrototypeExtractor(kmeans_iters=10, keep_per_cluster=1)
    result = extractor.extract_prototypes(
        memory, encoder, num_prototypes=2, reward_percentile=0.5, prune_sources=True
    )

    assert result.num_candidates == 3
    assert result.num_prototypes >= 1
    assert len(memory) <= len(episodes)

    prototype = memory.episodes[-1]
    assert prototype.rewards.mean() > 0.5
    assert torch.allclose(prototype.states.mean(), torch.tensor(1.0), atol=0.2)


def test_visual_encoder_nature_cnn_shape() -> None:
    encoder = VisualEpisodeEncoder(
        image_channels=3, backbone="nature_cnn", output_dim=32
    )
    frames = torch.randn(2, 4, 3, 84, 84)
    outputs = encoder(frames)

    assert outputs.shape == (2, 4, encoder.output_dim)
    assert torch.isfinite(outputs).all()

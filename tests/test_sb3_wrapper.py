import pytest

torch = pytest.importorskip("torch")

from hippotorch.encoders.dual import DualEncoder
from hippotorch.memory.replay import HippocampalReplayBuffer
from hippotorch.memory.sb3 import SB3ReplayBufferWrapper
from hippotorch.memory.store import MemoryStore


def test_sb3_wrapper_adds_episodes_and_samples():
    obs_dim = 2
    action_dim = 1
    input_dim = obs_dim + action_dim + 1

    encoder = DualEncoder(
        input_dim=input_dim, embed_dim=8, momentum=0.9, refresh_interval=10
    )
    memory = MemoryStore(embed_dim=8, capacity=10)
    buffer = HippocampalReplayBuffer(memory=memory, encoder=encoder, mixture_ratio=0.0)
    wrapper = SB3ReplayBufferWrapper(buffer)

    obs = torch.zeros(obs_dim)
    next_obs = torch.ones(obs_dim)
    action = torch.tensor([0.5])

    # Add a short trajectory of two steps
    wrapper.add(obs, next_obs, action, reward=1.0, done=False)
    wrapper.add(next_obs, next_obs + 1, action, reward=0.5, done=True)

    assert wrapper.size == 1

    batch = wrapper.sample(batch_size=1)
    assert set(batch.keys()) == {"states", "actions", "rewards", "next_states", "dones"}
    assert batch["states"].shape[0] == 1
    assert batch["actions"].shape[-1] == action_dim

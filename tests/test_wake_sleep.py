import pytest

torch = pytest.importorskip("torch")

from hippotorch.encoders.dual import DualEncoder
from hippotorch.memory.consolidator import Consolidator
from hippotorch.memory.replay import HippocampalReplayBuffer
from hippotorch.memory.store import MemoryStore
from hippotorch.memory.wake_sleep import WakeSleepTrainer
from hippotorch.segmenters import TerminalSegmenter


class DummyAgent:
    def __init__(self):
        self.update_calls = 0

    def act(self, state, explore=True):
        return torch.zeros(1)

    def update(self, batch):  # pragma: no cover - behavior asserted indirectly
        self.update_calls += 1


class DummyEnv:
    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self.t = 0

    def reset(self):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        reward = 1.0
        done = self.t >= self.horizon
        return self._obs(), reward, done, False, {}

    def _obs(self):
        return torch.tensor([float(self.t)], dtype=torch.float)


def test_wake_sleep_trainer_runs_with_consolidation():
    input_dim = 3  # state + action + reward
    embed_dim = 16

    encoder = DualEncoder(
        input_dim=input_dim, embed_dim=embed_dim, momentum=0.9, refresh_interval=5
    )
    memory = MemoryStore(embed_dim=embed_dim, capacity=64)
    consolidator = Consolidator(
        encoder,
        temperature=0.1,
        reward_weight=0.7,
        learning_rate=5e-4,
    )
    buffer = HippocampalReplayBuffer(
        memory,
        encoder,
        mixture_ratio=0.5,
        consolidator=consolidator,
    )

    agent = DummyAgent()

    trainer = WakeSleepTrainer(
        agent=agent,
        buffer=buffer,
        segmenter=TerminalSegmenter(),
        consolidation_schedule="periodic",
        consolidation_interval=4,
        consolidation_steps=2,
        min_buffer_episodes=1,
        batch_size=4,
    )

    metrics = trainer.train(DummyEnv(horizon=6), total_steps=12)

    assert buffer.size > 0
    assert trainer.sleep_steps > 0
    assert agent.update_calls > 0
    assert metrics["consolidation_loss"], "Consolidation should produce metrics"
    assert trainer.buffer.consolidator is not None
    assert trainer.buffer.consolidator.total_steps >= trainer.sleep_steps
    assert (
        trainer.buffer.consolidator.get_representation_quality(memory)[
            "retrieval_sharpness"
        ]
        >= 0
    )

    # Ensure wake updates occurred
    assert trainer.buffer.memory.keys.shape[1] == embed_dim
    assert isinstance(trainer.buffer.mixture_ratio, float)

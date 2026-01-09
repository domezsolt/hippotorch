# Quickstart

This quickstart shows how to store episodes, run consolidation with reward-aware contrastive learning, and sample hybrid batches.

Prereqs
- PyTorch tensors for states/actions/rewards
- Input dimension = state_dim + action_dim + 1 (reward)

Example
```python
import torch
from hippotorch import (
    Episode,
    DualEncoder,
    MemoryStore,
    Consolidator,
    HippocampalReplayBuffer,
    linear_mixture_schedule,
)

# Fake data
state_dim, action_dim = 4, 2
input_dim = state_dim + action_dim + 1

# Build encoder/memory
encoder = DualEncoder(input_dim=input_dim, embed_dim=32, momentum=0.995, refresh_interval=10_000)
memory = MemoryStore(embed_dim=32, capacity=10_000)

# Replay buffer with linear ramp from 20%→70% semantic sampling
buffer = HippocampalReplayBuffer(
    memory=memory,
    encoder=encoder,
    mixture_ratio=0.2,
    mixture_schedule=linear_mixture_schedule(0.2, 0.7, warmup_steps=50_000),
)

# Consolidator for reward-aware contrastive learning
consolidator = Consolidator(encoder, temperature=0.07, reward_weight=0.5, learning_rate=1e-4)
buffer.attach_consolidator(consolidator)

# Create and add an episode
T = 16
states = torch.randn(T, state_dim)
actions = torch.randn(T, action_dim)
rewards = torch.randn(T)
dones = torch.zeros(T, dtype=torch.bool)
episode = Episode(states=states, actions=actions, rewards=rewards, dones=dones)

buffer.add_episode(episode, encoder_step=0)

# Consolidate (sleep)
metrics = buffer.consolidate(steps=50, batch_size=64, refresh_keys=True, report_quality=True)
print("consolidation:", metrics)

# Sample a hybrid batch for RL training
query_state = torch.cat([states[0], actions[0], rewards[0].unsqueeze(0)])
batch = buffer.sample(batch_size=128, query_state=query_state, top_k=10)
print({k: v.shape for k, v in batch.items()})
```

Tips
- Increase `mixture_ratio` later in training or use `linear_mixture_schedule` for stability.
- Set `refresh_interval` to re-encode stale keys periodically.
- Use `report_quality=True` during consolidation to log representation diagnostics.


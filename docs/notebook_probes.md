# Notebook Probes

Copy-paste these cells into a Jupyter/VSCode notebook to sanity‑check core functionality without any Gym deps.

## 0) Imports
```python
import torch
from hippotorch import (
    Episode,
    DualEncoder,
    MemoryStore,
    Consolidator,
    HippocampalReplayBuffer,
    linear_mixture_schedule,
    run_replay_regression,
)
```

## 1) Regression harness (needle‑in‑haystack)
```python
torch.manual_seed(0)
result = run_replay_regression(
    num_good=6, num_distractor=18, episode_length=3, batch_size=12
)
print(result)
print("improvement:", result.reward_improvement)
```
Expected: `semantic_mean_reward > uniform_mean_reward` and positive `reward_improvement`.

## 2) Build memory, add synthetic episodes, sample
```python
# Synthetic episode generator
state_dim, action_dim = 3, 2
embed_dim = state_dim + action_dim + 1

def gen_episode(T=4, s_val=1.0, r_val=1.0):
    states = torch.full((T, state_dim), s_val)
    actions = torch.zeros(T, action_dim)
    rewards = torch.full((T,), r_val)
    dones = torch.zeros(T, dtype=torch.bool)
    return Episode(states=states, actions=actions, rewards=rewards, dones=dones)

# Encoder + memory
encoder = DualEncoder(input_dim=embed_dim, embed_dim=embed_dim)
memory = MemoryStore(embed_dim=embed_dim, capacity=128)

# Add a few episodes
episodes = [gen_episode(5, 1.0, 1.0) for _ in range(4)] + [gen_episode(5, -1.0, 0.1) for _ in range(8)]
keys = []
for ep in episodes:
    x = torch.cat([ep.states, ep.actions, ep.rewards.unsqueeze(-1)], dim=-1).unsqueeze(0)
    keys.append(encoder.encode_key(x).squeeze(0))
memory.write(torch.stack(keys), episodes, encoder_step=0)

# Uniform samples (as episodes)
uniform_eps = memory.sample_uniform(3)
len(uniform_eps), type(uniform_eps[0])
```

## 3) Consolidation before/after diagnostics
```python
consolidator = Consolidator(encoder, temperature=0.07, reward_weight=0.5, learning_rate=1e-4)
print("before:", consolidator.get_representation_quality(memory))
metrics = consolidator.sleep(memory, steps=50, batch_size=16, refresh_keys=True, report_quality=True)
print("sleep:", metrics)
print("after:", consolidator.get_representation_quality(memory))
```
Expected: `reward_alignment` should increase on this synthetic data.

## 4) Hybrid sampling probe (no Gym)
```python
buffer = HippocampalReplayBuffer(memory=memory, encoder=encoder, mixture_ratio=0.5)
# Query vector = [state, action_scalar, reward]; we can use first state, zeros for action/reward
first = episodes[0]
query = torch.cat([first.states[0], torch.zeros(1), first.rewards[0].unsqueeze(0)])

batch = buffer.sample(batch_size=8, query_state=query, top_k=5)
print({k: v.shape for k, v in batch.items()})
print("stats:", buffer.stats())
```

## 5) Mixture schedule quick check
```python
sched = linear_mixture_schedule(0.2, 0.7, warmup_steps=1000)
print([sched(s) for s in [0, 250, 500, 750, 1000, 2000]])
```

If all cells run and outputs look sensible, your installation is ready to try the full examples.


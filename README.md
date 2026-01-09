# hippotorch 🧠🔥

> **Differentiable episodic memory for RL. Retrieves what matters. Forgets what doesn't.**

**hippotorch** is a PyTorch library that replaces standard replay buffers with a learnable memory system. It uses **reward-aware contrastive learning** to organize experiences and **hybrid sampling** to retrieve them—solving the temporal credit assignment problem in sparse-reward, long-horizon tasks.

---

### Key Hyperparameters

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `mixture_ratio` | 0.5 | Start low (0.2), ramp to 0.5 after warmup |
| `momentum` | 0.995 | Higher = more stable keys, slower adaptation |
| `temperature` | 0.07 | Lower = sharper retrieval (try 0.05 for sparse rewards) |
| `reward_weight` | 0.5 | Higher = clusters by reward; lower = clusters by time |

---

## When to Use (and When Not To)

Hippotorch adds overhead. Use it where episodic structure matters.

| Scenario | Benefit | Recommendation |
|----------|---------|----------------|
| **Sparse rewards** (Montezuma, long corridors) | ✅ High | Use hippotorch—retrieval surfaces rare successes |
| **Partial observability** (POMDPs, visual RL) | ✅ High | Use hippotorch—pattern completion reconstructs context |
| **Long-horizon tasks** (100+ steps to goal) | ✅ High | Use hippotorch—bridges temporal credit gap |
| **Curriculum / transfer learning** | ✅ High | Use hippotorch—retains skills across task stages |
| **Dense rewards, full observability** | ⚠️ Low | Use standard replay—uniform sampling is sufficient |
| **Short episodes** (<20 steps) | ⚠️ Low | Use standard replay—no retrieval advantage |

### Rule of Thumb

> If your agent "forgets" how to solve early tasks, or struggles to connect actions to delayed rewards, hippotorch can help. If training already converges well with a standard buffer, you don't need it.

---

## Installation
```bash
pip install hippotorch

# Optional: Fast retrieval for large memories (1M+ episodes)
pip install faiss-gpu
```

**Requirements:** Python ≥3.9, PyTorch ≥2.0

---

## Quickstart

Create a dual encoder + memory store, attach a consolidator, and use the hybrid replay buffer.

```python
import torch
from hippotorch import Episode, DualEncoder, MemoryStore, Consolidator, HippocampalReplayBuffer

state_dim, action_dim = 4, 1
input_dim = state_dim + action_dim + 1  # +1 for reward

# 1) Encoder and memory
encoder = DualEncoder(input_dim=input_dim, embed_dim=128, momentum=0.995)
memory = MemoryStore(embed_dim=128, capacity=50_000)

# 2) Reward-aware consolidator (sleep phase optimizer)
consolidator = Consolidator(encoder, temperature=0.07, reward_weight=0.5)

# 3) Hybrid replay buffer (semantic + uniform)
buffer = HippocampalReplayBuffer(memory=memory, encoder=encoder, mixture_ratio=0.3,
                                 consolidator=consolidator)

# 4) Record an episode (toy tensors)
T = 32
states = torch.randn(T, state_dim)
actions = torch.randn(T, action_dim)
rewards = torch.randn(T)
dones = torch.zeros(T, dtype=torch.bool)
episode = Episode(states=states, actions=actions, rewards=rewards, dones=dones)
buffer.add_episode(episode)

# 5) Sample with semantic + uniform mixing
query_state = torch.cat([states[0], torch.zeros(action_dim), rewards[0].unsqueeze(0)])
batch = buffer.sample(batch_size=64, query_state=query_state, top_k=5)
agent.update(batch)

# 6) Periodic consolidation ("sleep")
metrics = buffer.consolidate(steps=50, batch_size=64, report_quality=True)
print(metrics)
```

SB3 users can keep their rollout API unchanged with the adapter:
```python
from hippotorch import SB3ReplayBufferWrapper, TerminalSegmenter
sb3_buffer = SB3ReplayBufferWrapper(buffer, segmenter=TerminalSegmenter())
# sb3_buffer.add(obs, next_obs, action, reward, done)
```

## Portable Brains (Hub)

Export a trained memory so another agent can load it instantly:
```python
from hippotorch import (
    DualEncoder,
    HippocampalReplayBuffer,
    MemoryStore,
    push_memory_to_hub,
    load_memory_from_hub,
)

obs_dim = 42
encoder = DualEncoder(input_dim=obs_dim, embed_dim=128)
memory = MemoryStore(embed_dim=128, capacity=2048)

# Push to hub (requires real hub backend, e.g., huggingface_hub)
push_memory_to_hub(memory, repo_id="user/fetch-reach-expert", private=False)

# Later, load
restored = load_memory_from_hub("user/fetch-reach-expert")

# Or operate via the buffer convenience wrappers
buffer = HippocampalReplayBuffer(memory=memory, encoder=encoder)
buffer.save_to_hub("user/fetch-reach-expert")
restored_memory = buffer.load_memory_from_hub("user/fetch-reach-expert")
```

Note
- The hub utilities in this repo are minimal stubs for testing. To use the Hub in production,
  integrate a real backend (e.g., `huggingface_hub`) by adapting `hippotorch.utils.hub`.

## Quick Experiment Scripts
Convenience scripts in `scripts/` run short, repeatable experiments:

- Rank‑weighted consolidation ablation:
  - `bash scripts/run_rank_ablation.sh`
- Consolidation micro‑bench (synthetic):
  - `bash scripts/run_consolidation_micro.sh`
- CartPole parity (short run with logging):
  - `bash scripts/quick_cartpole.sh`
- Zero‑noise corridor (Amnesiac):
  - `bash scripts/corridor_multiseed_zn.sh`
  - Faster: `SEEDS=2 EPISODES=150 CONS_EVERY=10 CONS_STEPS=50 bash scripts/corridor_multiseed_zn.sh`
- Curriculum corridor (progressively increases length):
  - `bash scripts/corridor_curriculum.sh`
- TensorBoard embedding snapshot (PCA):
  - `bash scripts/log_tb_embedding.sh` then `tensorboard --logdir runs/hippo_tb`

See `docs/benchmarks.md` and `docs/curriculum.md` for details.

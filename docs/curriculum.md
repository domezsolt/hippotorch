# Curriculum Learning

Hippotorch includes a small callback to grow environment difficulty as the agent succeeds, useful for long-horizon tasks that otherwise suffer vanishing gradients.

Callback
- `hippotorch.curriculum.callbacks.HippotorchCurriculumCallback`
  - Tracks a rolling success window and calls `env.set_corridor_length(L+step)` when `win_rate >= threshold`.
  - SB3-compatible hook shape (on_step), but works standalone via `on_episode_end(env, reward)`.

Example (Corridor)
- Multi-seed runner (short, finishes in ~15 min):
```bash
bash scripts/corridor_curriculum.sh
# overrides
SEEDS=2 EPISODES=200 CONS_EVERY=10 CONS_STEPS=75 bash scripts/corridor_curriculum.sh
```

Manual use
```python
from hippotorch.curriculum.callbacks import HippotorchCurriculumCallback
cb = HippotorchCurriculumCallback(min_length=5, max_length=30, step=5, window=10, threshold=0.8)
# after each episode
cb.on_episode_end(env, reward=episode_return)
```

Notes
- Corridor env in `examples/amnesiac_corridor.py` supports `set_corridor_length(L)` and the `--curriculum` flag.
- Start with shorter episodes; hybrid replay keeps early solutions accessible while consolidation organizes memory.


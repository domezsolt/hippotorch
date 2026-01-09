# Hybrid Sampling

hippotorch mixes semantic retrieval with uniform sampling to maintain stability while exploiting relevant experiences.

Core idea
- Semantic: retrieve episodes similar to the current context (query embedding).
- Uniform: draw transitions uniformly to preserve I.I.D.-like properties.

Controls
- `mixture_ratio` in `HippocampalReplayBuffer` sets the semantic fraction.
- Provide a schedule via `linear_mixture_schedule(start, end, warmup_steps)` to ramp up semantic sampling.

API snapshot
```python
from hippotorch import HippocampalReplayBuffer, linear_mixture_schedule

buffer = HippocampalReplayBuffer(
    memory=memory,
    encoder=encoder,
    mixture_ratio=0.2,
    mixture_schedule=linear_mixture_schedule(0.2, 0.7, warmup_steps=50_000),
)

batch = buffer.sample(batch_size=256, query_state=state, top_k=10)
stats = buffer.stats()  # includes effective_ratio, semantic vs uniform counts
```

Sequences for POMDPs (LSTM/Transformer)
- Use temporal windows rather than atomic transitions so agents can burn in hidden state.
```python
# Sample padded windows (states/actions/rewards/dones/lengths)
seq_batch = buffer.sample_windows(batch_size=64, window_size=10, query_state=state, top_k=10)
```

Diagnostics
- `buffer.stats()` exposes the recent effective ratio and memory staleness.
- Use `Consolidator.get_representation_quality()` for reward alignment and retrieval sharpness.

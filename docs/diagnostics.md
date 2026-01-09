# Diagnostics

Consolidation and retrieval quality can be tracked to ensure embeddings remain useful.

Representation quality
```python
metrics = consolidator.get_representation_quality(memory)
# keys: retrieval_sharpness, reward_alignment, key_dispersion, avg_staleness
```

Sampling diagnostics
```python
stats = buffer.stats()
# includes: memory_size, semantic_fraction (target), effective_ratio (actual), staleness_*, encoder_step
```

Common patterns
- Falling `reward_alignment` → run a sleep phase or increase consolidation steps.
- High `avg_staleness` → lower `refresh_interval` to re-encode stale keys more often.
- Diverging `effective_ratio` vs `mixture_ratio` → increase `top_k` or memory size to provide more semantic candidates.


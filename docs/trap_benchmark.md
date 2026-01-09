# Key-Door-Trap Benchmark (Proposal)

A long-horizon POMDP designed to stress episodic retrieval:
- Phase 1 (t=0): observe colored key (Red/Blue).
- Phase 2 (t=1..100): long corridor (no distinguishing features).
- Phase 3 (t=101): choose door matching key color.

Hypothesis
- Uniform replay: LSTM forgets the color; gradients vanish over the corridor.
- hippotorch: Consolidation clusters key→reward structure; at the door, semantic retrieval surfaces key-context windows, enabling correct choice.

What to measure
- Success rate vs steps; retrieval_sharpness and reward_alignment during training.
- Ablation: temporal-only InfoNCE vs reward-weighted.

Implementation notes
- Use `sample_windows(window_size=10)` for agent burn-in.
- Lazy key refresh avoids stalls as memory grows.
- Smart eviction protects rare successes.


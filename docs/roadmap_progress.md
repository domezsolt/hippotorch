# Roadmap Progress

This document maps roadmap items to current code/tests.

Phase 1: Foundation
- Episode dataclass, segmenters, encoder backbone, dual encoder, memory store: implemented in `hippotorch/`.
- SB3 wrapper and wake/sleep orchestration: `hippotorch/memory/sb3.py`, `hippotorch/memory/wake_sleep.py`.
- Needle-in-haystack milestone: covered by regression harness test `tests/test_regression_harness.py` (semantic > uniform reward).

Phase 2: Consolidation
- Reward-aware contrastive consolidator: `hippotorch/memory/consolidator.py`.
- Diagnostics: retrieval sharpness, reward alignment, staleness in consolidator + buffer stats.
- Clustering benchmark: example added (`examples/consolidation_clustering.py`), docs in `docs/consolidation_benchmark.md`.

Phase 3: RL Integration
- Hybrid replay + scheduling: `hippotorch/memory/replay.py` with `linear_mixture_schedule` in `hippotorch/memory/schedules.py`.
- Stability checks and wake/sleep tests: `tests/test_sampling_schedule.py`, `tests/test_wake_sleep.py`.
- CartPole wake/sleep example: `examples/cartpole_wakesleep.py` (see `docs/cartpole.md`).
- CartPole milestone achieved: hybrid matches/exceeds uniform (see logs under `results/`).

Phase 4: Validation
- Documentation pages added (usage, hybrid sampling, diagnostics, wake/sleep, CartPole, consolidation benchmark, sensitivity, benchmarks). Benchmarks progressing:
  - CartPole example and multi-seed eval added
  - CartPole sensitivity CLI added (`examples/cartpole_sensitivity.py`)
    - Sweeps: mixture ratio, temperature, momentum, consolidation interval/steps
  - MiniGrid example scaffold added
  - FetchReach random-agent scaffold added (agent integration pending)
  - Sensitivity (synthetic) added
  - Zero-noise corridor (Amnesiac) multi-seed completed: hybrid outperforms uniform (e.g., SEEDS=2, EPISODES=150 → delta ≈ +0.846). See `results/corridor_zn.csv` and `results/corridor_zn.png`.

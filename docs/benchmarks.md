# Benchmark Suite

Targets
- CartPole-v1 (discrete): DQN + wake/sleep with hybrid replay.
- MiniGrid (discrete): DQN + wake/sleep (observation flattening scaffold).
- FetchReach (continuous): random agent scaffold; integrate SAC/DDPG for learning.

Scripts
- `examples/cartpole_wakesleep.py` (single run) and `examples/cartpole_multiseed_eval.py` (multi-seed)
- `examples/minigrid_wakesleep.py`
- `examples/fetchreach_random_wakesleep.py`
- `examples/cartpole_sensitivity.py` (mixture ratio and temperature across seeds)
  - Supports momentum and consolidation budget sweeps as well
- `examples/amnesiac_corridor.py` (Amnesiac's Corridor POMDP demonstrating credit assignment via retrieval)

Zero‑Noise Corridor (Amnesiac)
- Purpose: remove distractor rewards and test retrieval’s ability to bridge long horizons.
- Quick run
  - `bash scripts/corridor_multiseed_zn.sh`
  - Override defaults for faster runs:
    - `SEEDS=2 EPISODES=150 CONS_EVERY=10 CONS_STEPS=50 bash scripts/corridor_multiseed_zn.sh`
- Output: CSV at `results/corridor_zn.csv` and summary stats on stdout. Optional plot:
  - `python scripts/plot_corridor_regret.py --csv results/corridor_zn.csv --out results/corridor_zn.png`

Curriculum Corridor (Amnesiac)
- Purpose: demonstrate progressive length increase driven by recent success.
- Quick run
  - `bash scripts/corridor_curriculum.sh`
  - Override: `SEEDS=2 EPISODES=200 CONS_EVERY=10 CONS_STEPS=75 bash scripts/corridor_curriculum.sh`

Validation goal
- Phase 3: CartPole — hybrid matches/exceeds uniform replay. Use `--compare-baseline` or multi-seed script.
- Phase 4: Extend to MiniGrid and FetchReach with appropriate agents.

# MiniGrid Wake/Sleep (Scaffold)

This example flattens MiniGrid observations and trains a DQN agent with hybrid replay and consolidation.

Run
```bash
pip install gymnasium gymnasium-minigrid
python -m examples.minigrid_wakesleep --steps 100000 --batch-size 256 --schedule interleaved \
  --cons-interval 2000 --cons-steps 100 --warmup-steps 100000
```

Notes
- Observations are flattened (`image` tensor by default). For custom environments, adapt `obs_to_vec` in the example.
- This script focuses on pipeline integration. For stronger policies, consider curriculum tasks and tuning schedule/agent hyperparameters.

Key file
- `examples/minigrid_wakesleep.py`


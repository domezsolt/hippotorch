# CartPole Wake/Sleep Example

A minimal end-to-end example that trains a DQN agent on `CartPole-v1` while running consolidation with reward-aware contrastive learning and hybrid sampling.

Run
```bash
python -m examples.cartpole_wakesleep --steps 50000 --batch-size 256 --schedule periodic \
  --cons-interval 1000 --cons-steps 100 --warmup-steps 50000
```

Notes
- The encoder input uses `[obs, action_scalar, reward]`. For semantic retrieval, we set a `query_state_fn` that maps observation to `[obs, 0, 0]` as a proxy.
- Use `linear_mixture_schedule(0.2, 0.7, warmup_steps)` to ramp semantic sampling over training.
- To compare against uniform replay, set `mixture_ratio=0.0` when creating the replay buffer.

Key files
- `examples/cartpole_wakesleep.py`
- `hippotorch/agents/dqn.py`

Compare against uniform replay (with flags)
```bash
python -m examples.cartpole_wakesleep \
  --steps 50000 --batch-size 256 \
  --schedule periodic --cons-interval 1000 --cons-steps 100 --warmup-steps 50000 \
  --mixture-ratio 0.2 --schedule-start 0.2 --schedule-end 0.7 \
  --compare-baseline --eval-episodes 10
```
This runs both hybrid and uniform-replay training and prints average evaluation rewards for comparison.

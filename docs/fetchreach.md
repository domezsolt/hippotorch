# FetchReach Wake/Sleep (Random Agent Scaffold)

This example demonstrates memory integration and consolidation for a continuous-control task with a random agent. For learning, integrate with an actor-critic (e.g., SAC/DDPG) and use the SB3 wrapper or a custom agent.

Run
```bash
pip install gymnasium gymnasium-robotics
python -m examples.fetchreach_random_wakesleep --steps 50000 --batch-size 256
```

Notes
- Observations are concatenated from `observation`, `achieved_goal`, and `desired_goal`.
- The random agent is a placeholder. To train, swap in a continuous-control agent and keep `query_state_fn` that maps observations to encoder input.

Key file
- `examples/fetchreach_random_wakesleep.py`


# Hyperparameter Sensitivity (Synthetic)

Use the synthetic consolidation sensitivity script to sweep consolidation parameters and observe how they affect representation quality.

Run
```bash
python -m examples.sensitivity_consolidation \
  --temperatures 0.05 0.07 0.1 0.2 \
  --reward-weights 0.0 0.25 0.5 0.75 1.0 \
  --steps 200 --batch-size 64
```

Output columns
- temperature, reward_weight, retrieval_sharpness, reward_alignment, loss

Notes
- This assesses consolidation behavior independent of an RL loop. For task-level sensitivity (e.g., CartPole), run multiple training seeds while varying `mixture_ratio`, schedule warmup, `temperature`, and `momentum`.

---

# CartPole Sensitivity (Mixture Ratio, Temperature)

Run across seeds and export CSV (including momentum and consolidation budget sweeps):
```bash
python -m examples.cartpole_sensitivity \
  --seeds 5 --steps 50000 --batch-size 256 \
  --mixture-ratios 0.0 0.2 0.4 0.6 0.8 \
  --temperatures 0.05 0.07 0.1 \
  --momentums 0.9 0.99 0.995 \
  --cons-intervals 500 1000 \
  --cons-steps-list 50 100 \
  --aggregate --out cartpole_sensitivity.csv
```

CSV columns
- `seed, mixture_ratio, temperature, momentum, cons_interval, cons_steps, avg_eval_reward`
Aggregated output prints mean reward per full configuration when `--aggregate` is set.

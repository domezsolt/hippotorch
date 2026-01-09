#!/usr/bin/env bash
set -euo pipefail

# Quick CartPole head-to-head with logging

bash scripts/compare_cartpole.sh \
  --seeds "${SEEDS:-3}" \
  --steps "${STEPS:-10000}" \
  --batch "${BATCH:-256}" \
  --schedule "${SCHEDULE:-periodic}" \
  --cons-interval "${CONS_INTERVAL:-1000}" \
  --cons-steps "${CONS_STEPS:-50}}" \
  --warmup "${WARMUP:-8000}" \
  --mixture-ratio "${MIXTURE_RATIO:-0.2}" \
  --schedule-start "${SCHED_START:-0.0}" \
  --schedule-end "${SCHED_END:-0.5}" \
  --log-interval "${LOG_INTERVAL:-1000}"


#!/usr/bin/env bash
set -euo pipefail

# Zero-noise corridor multi-seed runner with CSV output

SEEDS=${SEEDS:-3}
EPISODES=${EPISODES:-300}
LENGTH=${LENGTH:-30}
OUT=${OUT:-results/corridor_zn.csv}
CONS_EVERY=${CONS_EVERY:-5}
CONS_STEPS=${CONS_STEPS:-100}

python -u -m examples.corridor_multiseed \
  --seeds ${SEEDS} --episodes ${EPISODES} --length ${LENGTH} \
  --coin-reward 0.0 --uniform 0.0 --mixture 0.5 \
  --schedule-start 0.0 --schedule-end 0.5 --warmup-episodes 250 \
  --reward-weight 0.9 --temperature 0.05 \
  --cons-every ${CONS_EVERY} --cons-steps ${CONS_STEPS} \
  --out ${OUT}

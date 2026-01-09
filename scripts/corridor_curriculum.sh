#!/usr/bin/env bash
set -euo pipefail

# Curriculum Corridor (multi-seed) with short defaults for quick runs.

SEEDS=${SEEDS:-2}
EPISODES=${EPISODES:-200}
LENGTH=${LENGTH:-30}
OUT=${OUT:-results/corridor_curriculum.csv}
CONS_EVERY=${CONS_EVERY:-10}
CONS_STEPS=${CONS_STEPS:-75}
MIN_LENGTH=${MIN_LENGTH:-5}
LENGTH_STEP=${LENGTH_STEP:-5}
SUCCESS_WINDOW=${SUCCESS_WINDOW:-30}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}

python -u -m examples.corridor_multiseed \
  --seeds ${SEEDS} --episodes ${EPISODES} --length ${LENGTH} \
  --coin-reward 0.03 --uniform 0.0 --mixture 0.4 \
  --schedule-start 0.0 --schedule-end 0.6 --warmup-episodes 150 \
  --reward-weight 0.9 --temperature 0.05 \
  --cons-every ${CONS_EVERY} --cons-steps ${CONS_STEPS} \
  --curriculum --min-length ${MIN_LENGTH} --length-step ${LENGTH_STEP} \
  --success-window ${SUCCESS_WINDOW} --success-threshold ${SUCCESS_THRESHOLD} \
  --out ${OUT}


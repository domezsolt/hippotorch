#!/usr/bin/env bash
set -euo pipefail

# Zero-noise corridor runner: sets coin_reward=0.0 and compares uniform vs hybrid.

EPISODES=${EPISODES:-600}
LENGTH=${LENGTH:-30}
REWARD_WEIGHT=${REWARD_WEIGHT:-0.9}
TEMP=${TEMP:-0.05}
CONS_EVERY=${CONS_EVERY:-5}
CONS_STEPS=${CONS_STEPS:-200}
MIXTURE=${MIXTURE:-0.5}
SCHED_START=${SCHED_START:-0.0}
SCHED_END=${SCHED_END:-0.5}
WARMUP_EPS=${WARMUP_EPS:-600}

python -u -m examples.amnesiac_corridor \
  --episodes ${EPISODES} --length ${LENGTH} \
  --coin-reward 0.0 --mixture ${MIXTURE} \
  --schedule-start ${SCHED_START} --schedule-end ${SCHED_END} --warmup-episodes ${WARMUP_EPS} \
  --reward-weight ${REWARD_WEIGHT} --temperature ${TEMP} \
  --cons-every ${CONS_EVERY} --cons-steps ${CONS_STEPS}


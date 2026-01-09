#!/usr/bin/env bash
set -euo pipefail

# Simple experiment harness to compare hippotorch (hybrid replay) vs baseline (uniform replay)
# on CartPole using the provided example scripts.

# Config (override via env or CLI args)
SEEDS="${SEEDS:-3}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-256}"
WARMUP="${WARMUP:-50000}"
SCHEDULE="${SCHEDULE:-periodic}"
CONS_INTERVAL="${CONS_INTERVAL:-1000}"
CONS_STEPS="${CONS_STEPS:-100}"
MIXTURE_RATIO="${MIXTURE_RATIO:-0.2}"
SCHED_START="${SCHED_START:-0.2}"
SCHED_END="${SCHED_END:-0.7}"
EVAL_EPS="${EVAL_EPS:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --batch|--batch-size) BATCH="$2"; shift 2;;
    --warmup|--warmup-steps) WARMUP="$2"; shift 2;;
    --schedule) SCHEDULE="$2"; shift 2;;
    --cons-interval) CONS_INTERVAL="$2"; shift 2;;
    --cons-steps) CONS_STEPS="$2"; shift 2;;
    --mixture-ratio) MIXTURE_RATIO="$2"; shift 2;;
    --schedule-start) SCHED_START="$2"; shift 2;;
    --schedule-end) SCHED_END="$2"; shift 2;;
    --eval-episodes) EVAL_EPS="$2"; shift 2;;
    --log-interval) LOG_INTERVAL="$2"; shift 2;;
    --help) echo "Usage: scripts/compare_cartpole.sh [--seeds N] [--steps N] [--batch N] [--warmup N] [--schedule periodic|interleaved|adaptive] [--cons-interval N] [--cons-steps N] [--mixture-ratio F] [--schedule-start F] [--schedule-end F] [--eval-episodes N]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: SEEDS=3 STEPS=20000 BATCH=256 WARMUP=50000 scripts/compare_cartpole.sh"
  exit 0
fi

mkdir -p results

printf "[1/2] Single-run head-to-head - hybrid vs uniform baseline\n"
python -u -m examples.cartpole_wakesleep \
  --steps "${STEPS}" \
  --batch-size "${BATCH}" \
  --schedule "${SCHEDULE}" \
  --cons-interval "${CONS_INTERVAL}" \
  --cons-steps "${CONS_STEPS}" \
  --warmup-steps "${WARMUP}" \
  --mixture-ratio "${MIXTURE_RATIO}" \
  --schedule-start "${SCHED_START}" \
  --schedule-end "${SCHED_END}" \
  --compare-baseline \
  --eval-episodes "${EVAL_EPS}" \
  --log-interval "${LOG_INTERVAL}" \
  2>&1 | tee "results/cartpole_single_${STEPS}_${BATCH}.log"

printf "\n[2/2] Multi-seed evaluation - aggregated means\n"
python -u -m examples.cartpole_multiseed_eval \
  --seeds "${SEEDS}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH}" \
  --warmup-steps "${WARMUP}" \
  --hybrid-mixture "${MIXTURE_RATIO}" \
  --schedule-start "${SCHED_START}" \
  --schedule-end "${SCHED_END}" \
  2>&1 | tee "results/cartpole_multiseed_${SEEDS}_${STEPS}_${BATCH}.log"

printf "\nDone. Logs written under results/\n"

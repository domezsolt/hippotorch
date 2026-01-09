#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from statistics import mean
from typing import Tuple

from examples.amnesiac_corridor import run_variant

# Multi-seed Amnesiac Corridor runner with CSV reporter.
# Runs uniform and hybrid variants across seeds and writes a CSV.
# Usage:
#   python -m examples.corridor_multiseed --seeds 5 --episodes 800 --length 30 \\
#     --coin-reward 0.0 --mixture 0.5 --schedule-start 0.0 --schedule-end 0.5 \\
#     --warmup-episodes 600 --reward-weight 0.9 --temperature 0.05 \\
#     --cons-every 5 --cons-steps 200 --out results/corridor_multiseed.csv


def run_pair(
    seed: int, args: argparse.Namespace
) -> Tuple[float, float, float, float, float]:
    # Uniform
    u_mean, _ = run_variant(
        args.length,
        args.episodes,
        mixture_ratio=args.uniform,
        coin_reward=args.coin_reward,
        loiter_penalty=args.loiter_penalty,
        success_reward=args.success_reward,
        failure_penalty=args.failure_penalty,
        reward_weight=args.reward_weight,
        temperature=args.temperature,
        cons_every=args.cons_every,
        cons_steps=args.cons_steps,
        curriculum=args.curriculum,
        min_length=args.min_length,
        length_step=args.length_step,
        success_window=args.success_window,
        success_threshold=args.success_threshold,
        oracle=args.oracle,
        seed=seed,
    )
    # Hybrid
    h_mean, _ = run_variant(
        args.length,
        args.episodes,
        mixture_ratio=args.mixture,
        coin_reward=args.coin_reward,
        loiter_penalty=args.loiter_penalty,
        success_reward=args.success_reward,
        failure_penalty=args.failure_penalty,
        schedule_start=args.schedule_start,
        schedule_end=args.schedule_end,
        warmup_episodes=args.warmup_episodes,
        reward_weight=args.reward_weight,
        temperature=args.temperature,
        cons_every=args.cons_every,
        cons_steps=args.cons_steps,
        curriculum=args.curriculum,
        min_length=args.min_length,
        length_step=args.length_step,
        success_window=args.success_window,
        success_threshold=args.success_threshold,
        oracle=args.oracle,
        seed=seed,
    )
    # Oracle-retrieval upper-bound for hybrid (if requested)
    h_oracle = h_mean
    if args.oracle_retrieval:
        h_oracle, _ = run_variant(
            args.length,
            args.episodes,
            mixture_ratio=args.mixture,
            coin_reward=args.coin_reward,
            loiter_penalty=args.loiter_penalty,
            success_reward=args.success_reward,
            failure_penalty=args.failure_penalty,
            schedule_start=args.schedule_start,
            schedule_end=args.schedule_end,
            warmup_episodes=args.warmup_episodes,
            reward_weight=args.reward_weight,
            temperature=args.temperature,
            cons_every=args.cons_every,
            cons_steps=args.cons_steps,
            curriculum=args.curriculum,
            min_length=args.min_length,
            length_step=args.length_step,
            success_window=args.success_window,
            success_threshold=args.success_threshold,
            oracle_retrieval=True,
            seed=seed,
        )
    return (
        float(u_mean),
        float(h_mean),
        float(h_mean - u_mean),
        float(u_mean),
        float(h_oracle),
    )


def main(args: argparse.Namespace) -> None:
    rows = []
    for seed in range(args.seeds):
        u_mean, h_mean, delta, _, h_oracle = run_pair(seed, args)
        rows.append(
            [
                seed,
                args.uniform,
                args.mixture,
                args.coin_reward,
                args.schedule_start,
                args.schedule_end,
                args.warmup_episodes,
                args.reward_weight,
                args.temperature,
                args.cons_every,
                args.cons_steps,
                int(args.curriculum),
                args.min_length,
                args.length_step,
                args.success_window,
                args.success_threshold,
                int(args.oracle),
                u_mean,
                h_mean,
                delta,
                h_oracle,
                (h_oracle - h_mean),
            ]
        )

    header = [
        "seed",
        "uniform_mixture",
        "hybrid_mixture",
        "coin_reward",
        "schedule_start",
        "schedule_end",
        "warmup_episodes",
        "reward_weight",
        "temperature",
        "cons_every",
        "cons_steps",
        "curriculum",
        "min_length",
        "length_step",
        "success_window",
        "success_threshold",
        "oracle",
        "uniform_mean_last50",
        "hybrid_mean_last50",
        "delta",
        "hybrid_oracle_retrieval_mean_last50",
        "regret_oracle_minus_hybrid",
    ]

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    else:
        print(",".join(header))
        for r in rows:
            print(",".join(str(x) for x in r))

    print(
        {
            "uniform_mean": mean(r[-5] for r in rows),
            "hybrid_mean": mean(r[-4] for r in rows),
            "delta": mean(r[-3] for r in rows),
            "hybrid_oracle_retrieval_mean": mean(r[-2] for r in rows),
            "regret": mean(r[-1] for r in rows),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--length", type=int, default=30)
    p.add_argument("--uniform", type=float, default=0.0)
    p.add_argument("--mixture", type=float, default=0.5)
    p.add_argument("--coin-reward", dest="coin_reward", type=float, default=0.0)
    p.add_argument("--loiter-penalty", dest="loiter_penalty", type=float, default=-0.01)
    p.add_argument("--success-reward", dest="success_reward", type=float, default=12.0)
    p.add_argument(
        "--failure-penalty", dest="failure_penalty", type=float, default=-5.0
    )
    p.add_argument("--schedule-start", dest="schedule_start", type=float, default=0.0)
    p.add_argument("--schedule-end", dest="schedule_end", type=float, default=0.5)
    p.add_argument("--warmup-episodes", dest="warmup_episodes", type=int, default=600)
    p.add_argument("--reward-weight", dest="reward_weight", type=float, default=0.9)
    p.add_argument("--temperature", dest="temperature", type=float, default=0.05)
    p.add_argument("--cons-every", dest="cons_every", type=int, default=5)
    p.add_argument("--cons-steps", dest="cons_steps", type=int, default=200)
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--min-length", dest="min_length", type=int, default=5)
    p.add_argument("--length-step", dest="length_step", type=int, default=5)
    p.add_argument("--success-window", dest="success_window", type=int, default=50)
    p.add_argument(
        "--success-threshold", dest="success_threshold", type=float, default=0.8
    )
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--oracle-retrieval", dest="oracle_retrieval", action="store_true")
    p.add_argument("--out", type=str, default="")
    main(p.parse_args())

#!/usr/bin/env python3
"""CartPole sensitivity sweep over mixture ratio, temperature, momentum,
and consolidation budget across seeds.

Outputs CSV rows with columns:
seed,mixture_ratio,temperature,momentum,cons_interval,cons_steps,avg_eval_reward
and (optionally) aggregated means per configuration.

Usage:
  python -m examples.cartpole_sensitivity \
    --seeds 5 --steps 50000 --batch-size 256 \
    --mixture-ratios 0.0 0.2 0.4 0.6 0.8 \
    --temperatures 0.05 0.07 0.1 \
    --aggregate --out results.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, Tuple

import torch

try:
    import gymnasium as gym
except Exception as e:  # pragma: no cover - optional dependency
    raise SystemExit("Please install gymnasium: pip install gymnasium") from e

from hippotorch import (Consolidator, DualEncoder, HippocampalReplayBuffer,
                        MemoryStore, TerminalSegmenter, WakeSleepTrainer)
from hippotorch.agents import DQNAgent, DQNConfig


def make_env():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return env, obs_dim, action_dim


@torch.no_grad()
def evaluate_agent(env, agent, episodes: int = 10) -> float:
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total += float(reward)
    return total / max(1, episodes)


def run_one(
    seed: int,
    mixture_ratio: float,
    temperature: float,
    momentum: float,
    *,
    steps: int,
    batch_size: int,
    cons_interval: int,
    cons_steps: int,
    eval_episodes: int,
) -> float:
    torch.manual_seed(seed)
    env, obs_dim, action_dim = make_env()

    input_dim = obs_dim + 1 + 1
    embed_dim = 64
    encoder = DualEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
        momentum=float(momentum),
        refresh_interval=10_000,
    )
    memory = MemoryStore(embed_dim=embed_dim, capacity=100_000)
    buffer = HippocampalReplayBuffer(
        memory=memory,
        encoder=encoder,
        mixture_ratio=float(mixture_ratio),
    )
    consolidator = Consolidator(
        encoder, temperature=float(temperature), reward_weight=0.5, learning_rate=1e-4
    )
    buffer.attach_consolidator(consolidator)

    agent = DQNAgent(DQNConfig(state_dim=obs_dim, action_dim=action_dim, device="cpu"))
    trainer = WakeSleepTrainer(
        agent=agent,
        buffer=buffer,
        segmenter=TerminalSegmenter(),
        consolidation_schedule="periodic",
        consolidation_interval=cons_interval,
        consolidation_steps=cons_steps,
        min_buffer_episodes=5,
        batch_size=batch_size,
    )
    # Build query vector [obs, 0, 0] for semantic sampling
    trainer.query_state_fn = lambda obs: torch.cat(
        [torch.as_tensor(obs, dtype=torch.float32), torch.zeros(2)]
    )

    _ = trainer.train(env, total_steps=steps)
    score = evaluate_agent(env, agent, episodes=eval_episodes)
    return float(score)


def write_csv(
    rows: Iterable[Tuple[int, float, float, float, int, int, float]], path: str | None
) -> None:
    header = [
        "seed",
        "mixture_ratio",
        "temperature",
        "momentum",
        "cons_interval",
        "cons_steps",
        "avg_eval_reward",
    ]
    if path:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in rows:
                writer.writerow(r)
    else:
        print(",".join(header))
        for r in rows:
            print(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}")


def main(args: argparse.Namespace) -> None:
    rows = []
    for seed in range(args.seeds):
        for mix in args.mixture_ratios:
            for temp in args.temperatures:
                for mom in args.momentums:
                    for ci in args.cons_intervals:
                        for cs in args.cons_steps_list:
                            score = run_one(
                                seed,
                                mix,
                                temp,
                                mom,
                                steps=args.steps,
                                batch_size=args.batch_size,
                                cons_interval=int(ci),
                                cons_steps=int(cs),
                                eval_episodes=args.eval_episodes,
                            )
                            rows.append(
                                (
                                    seed,
                                    float(mix),
                                    float(temp),
                                    float(mom),
                                    int(ci),
                                    int(cs),
                                    float(score),
                                )
                            )

    write_csv(rows, args.out)

    if args.aggregate:
        groups: Dict[Tuple[float, float, float, int, int], list[float]] = defaultdict(
            list
        )
        for _, mix, temp, mom, ci, cs, score in rows:
            groups[(mix, temp, mom, ci, cs)].append(score)
        print(
            "\n# Aggregated means (mixture_ratio, temperature, momentum, cons_interval,"
        )
        print("# cons_steps, mean_score)")
        print("mixture_ratio,temperature,momentum,cons_interval,cons_steps,mean_score")
        for (mix, temp, mom, ci, cs), scores in sorted(groups.items()):
            print(f"{mix},{temp},{mom},{ci},{cs},{mean(scores):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument(
        "--mixture-ratios", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8]
    )
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.05, 0.07, 0.1])
    p.add_argument("--momentums", nargs="+", type=float, default=[0.9, 0.99, 0.995])
    p.add_argument("--cons-intervals", nargs="+", type=int, default=[1_000])
    p.add_argument("--cons-steps-list", nargs="+", type=int, default=[100])
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--out", type=str, default=None)
    main(p.parse_args())

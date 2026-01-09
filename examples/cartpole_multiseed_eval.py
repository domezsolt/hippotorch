#!/usr/bin/env python3
"""Multi-seed CartPole evaluation comparing hybrid vs uniform replay.

Outputs average returns across seeds for both settings.
"""
from __future__ import annotations

import argparse
from statistics import mean
from typing import Tuple

import torch

try:
    import gymnasium as gym
except Exception as e:  # pragma: no cover - optional dependency
    raise SystemExit("Please install gymnasium: pip install gymnasium") from e

from hippotorch import (Consolidator, DualEncoder, HippocampalReplayBuffer,
                        MemoryStore, TerminalSegmenter, WakeSleepTrainer,
                        linear_mixture_schedule)
from hippotorch.agents import DQNAgent, DQNConfig


def make_env() -> Tuple[object, int, int]:
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


def run(
    seed: int,
    steps: int,
    batch_size: int,
    warmup_steps: int,
    *,
    hybrid_mixture: float,
    use_schedule: bool,
    schedule_start: float,
    schedule_end: float,
) -> Tuple[float, float]:
    torch.manual_seed(seed)
    env, obs_dim, action_dim = make_env()

    def train_setting(mixture_ratio: float, schedule=True) -> Tuple[float, float]:
        input_dim = obs_dim + 1 + 1
        embed_dim = 64
        encoder = DualEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            momentum=0.995,
            refresh_interval=10_000,
        )
        memory = MemoryStore(embed_dim=embed_dim, capacity=100_000)
        buffer = HippocampalReplayBuffer(
            memory=memory,
            encoder=encoder,
            mixture_ratio=mixture_ratio,
            mixture_schedule=(
                linear_mixture_schedule(
                    schedule_start, schedule_end, warmup_steps=warmup_steps
                )
                if schedule
                else None
            ),
        )
        consolidator = Consolidator(
            encoder, temperature=0.07, reward_weight=0.5, learning_rate=1e-4
        )
        buffer.attach_consolidator(consolidator)
        agent = DQNAgent(
            DQNConfig(state_dim=obs_dim, action_dim=action_dim, device="cpu")
        )
        trainer = WakeSleepTrainer(
            agent=agent,
            buffer=buffer,
            segmenter=TerminalSegmenter(),
            consolidation_schedule="periodic",
            consolidation_interval=1000,
            consolidation_steps=100,
            min_buffer_episodes=5,
            batch_size=batch_size,
        )
        trainer.query_state_fn = lambda obs: torch.cat(
            [torch.as_tensor(obs, dtype=torch.float32), torch.zeros(2)]
        )
        _ = trainer.train(env, total_steps=steps)
        score = evaluate_agent(env, agent, episodes=10)
        return score, score

    hybrid, _ = train_setting(hybrid_mixture, schedule=use_schedule)
    uniform, _ = train_setting(0.0, schedule=False)
    return hybrid, uniform


def main(args: argparse.Namespace) -> None:
    hybrids, uniforms = [], []
    for seed in range(args.seeds):
        h, u = run(
            seed,
            steps=args.steps,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            hybrid_mixture=args.hybrid_mixture,
            use_schedule=not args.no_schedule,
            schedule_start=args.schedule_start,
            schedule_end=args.schedule_end,
        )
        hybrids.append(h)
        uniforms.append(u)
    print(
        {
            "hybrid_mean": mean(hybrids),
            "uniform_mean": mean(uniforms),
            "delta": mean(hybrids) - mean(uniforms),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=50_000)
    p.add_argument("--hybrid-mixture", dest="hybrid_mixture", type=float, default=0.2)
    p.add_argument("--schedule-start", dest="schedule_start", type=float, default=0.2)
    p.add_argument("--schedule-end", dest="schedule_end", type=float, default=0.7)
    p.add_argument("--no-schedule", action="store_true")
    main(p.parse_args())

#!/usr/bin/env python3
"""CartPole wake/sleep training with hybrid sampling.

Requires: gymnasium

Usage:
  python -m examples.cartpole_wakesleep --steps 50000
"""
from __future__ import annotations

import argparse
from typing import Tuple

import torch

try:
    import gymnasium as gym
except Exception as e:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Please install gymnasium to run this example: pip install gymnasium"
    ) from e

from hippotorch import (Consolidator, DualEncoder, HippocampalReplayBuffer,
                        MemoryStore, TerminalSegmenter, WakeSleepTrainer,
                        linear_mixture_schedule)
from hippotorch.agents import DQNAgent, DQNConfig


def make_env() -> Tuple[object, int, int]:
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return env, obs_dim, action_dim


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    env, obs_dim, action_dim = make_env()

    # Encoder input = state_dim + action_dim(1 for discrete) + 1 reward
    input_dim = obs_dim + 1 + 1
    embed_dim = 64

    encoder = DualEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
        momentum=0.995,
        refresh_interval=10_000,
    )
    memory = MemoryStore(embed_dim=embed_dim, capacity=100_000)
    schedule = (
        None
        if args.no_schedule
        else linear_mixture_schedule(
            args.schedule_start, args.schedule_end, warmup_steps=args.warmup_steps
        )
    )
    buffer = HippocampalReplayBuffer(
        memory=memory,
        encoder=encoder,
        mixture_ratio=args.mixture_ratio,
        mixture_schedule=schedule,
    )
    consolidator = Consolidator(
        encoder, temperature=0.07, reward_weight=0.5, learning_rate=1e-4
    )
    buffer.attach_consolidator(consolidator)

    agent = DQNAgent(DQNConfig(state_dim=obs_dim, action_dim=action_dim, device="cpu"))
    trainer = WakeSleepTrainer(
        agent=agent,
        buffer=buffer,
        segmenter=TerminalSegmenter(),
        consolidation_schedule=args.schedule,
        consolidation_interval=args.cons_interval,
        consolidation_steps=args.cons_steps,
        min_buffer_episodes=5,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )

    # Build a query vector as [obs, last_action, last_reward], using zeros for missing fields
    def query_builder(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        return torch.cat([obs_t, torch.zeros(2)])  # action scalar + reward

    trainer.query_state_fn = query_builder

    metrics = trainer.train(env, total_steps=args.steps)
    print("training complete; episodes:", len(metrics["rewards"]))
    if metrics["consolidation_loss"]:
        print(
            "avg consolidation loss:",
            sum(metrics["consolidation_loss"]) / len(metrics["consolidation_loss"]),
        )
    print("buffer stats:", buffer.stats())

    if args.compare_baseline:
        # Evaluate current (hybrid) agent
        hybrid_score = evaluate_agent(env, agent, episodes=args.eval_episodes)
        print(f"hybrid average reward: {hybrid_score:.2f}")

        # Train a uniform-replay baseline agent
        env_b, _, _ = make_env()
        encoder_b = DualEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            momentum=0.995,
            refresh_interval=10_000,
        )
        memory_b = MemoryStore(embed_dim=embed_dim, capacity=100_000)
        buffer_b = HippocampalReplayBuffer(
            memory=memory_b, encoder=encoder_b, mixture_ratio=0.0
        )
        consolidator_b = Consolidator(
            encoder_b, temperature=0.07, reward_weight=0.5, learning_rate=1e-4
        )
        buffer_b.attach_consolidator(consolidator_b)
        agent_b = DQNAgent(
            DQNConfig(state_dim=obs_dim, action_dim=action_dim, device="cpu")
        )
        trainer_b = WakeSleepTrainer(
            agent=agent_b,
            buffer=buffer_b,
            segmenter=TerminalSegmenter(),
            consolidation_schedule=args.schedule,
            consolidation_interval=args.cons_interval,
            consolidation_steps=args.cons_steps,
            min_buffer_episodes=5,
            batch_size=args.batch_size,
        )
        trainer_b.query_state_fn = query_builder
        _ = trainer_b.train(env_b, total_steps=args.steps)

        baseline_score = evaluate_agent(env_b, agent_b, episodes=args.eval_episodes)
        print(f"uniform baseline average reward: {baseline_score:.2f}")
        if hybrid_score + 1e-6 >= baseline_score:
            print("Result: hybrid matches or exceeds standard replay on CartPole.")
        else:
            print(
                "Result: hybrid underperforms baseline; consider tuning "
                "mixture/schedule."
            )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument(
        "--schedule",
        type=str,
        default="periodic",
        choices=["periodic", "interleaved", "adaptive"],
    )
    parser.add_argument(
        "--cons-interval", dest="cons_interval", type=int, default=1_000
    )
    parser.add_argument("--cons-steps", dest="cons_steps", type=int, default=100)
    parser.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=50_000)
    parser.add_argument(
        "--mixture-ratio", dest="mixture_ratio", type=float, default=0.2
    )
    parser.add_argument(
        "--schedule-start", dest="schedule_start", type=float, default=0.2
    )
    parser.add_argument("--schedule-end", dest="schedule_end", type=float, default=0.7)
    parser.add_argument("--no-schedule", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare-baseline", dest="compare_baseline", action="store_true"
    )
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=10)
    parser.add_argument(
        "--log-interval",
        dest="log_interval",
        type=int,
        default=0,
        help="log every N steps (0=off)",
    )
    main(parser.parse_args())

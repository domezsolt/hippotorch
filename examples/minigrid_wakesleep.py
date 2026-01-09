#!/usr/bin/env python3
"""MiniGrid wake/sleep training (scaffold).

Requires: gymnasium, gymnasium-minigrid

This example flattens MiniGrid observations and runs DQN with hybrid replay.
"""
from __future__ import annotations

import argparse
from typing import Tuple

import torch

try:
    import gymnasium as gym
    import minigrid  # noqa: F401 - ensure plugin registered
except Exception as e:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Please install dependencies: pip install gymnasium gymnasium-minigrid"
    ) from e

from hippotorch import (Consolidator, DualEncoder, HippocampalReplayBuffer,
                        MemoryStore, TerminalSegmenter, WakeSleepTrainer,
                        linear_mixture_schedule)
from hippotorch.agents import DQNAgent, DQNConfig


def obs_to_vec(obs) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.flatten().float()
    if isinstance(obs, dict):
        if "image" in obs:
            return torch.as_tensor(obs["image"]).flatten().float()
        # Fallback: flatten all values
        parts = []
        for v in obs.values():
            t = torch.as_tensor(v)
            parts.append(t.flatten().float())
        return torch.cat(parts)
    return torch.as_tensor(obs, dtype=torch.float32).flatten()


def make_env() -> Tuple[object, int, int]:
    env = gym.make("MiniGrid-Empty-8x8-v0")
    obs, _ = env.reset()
    obs_dim = int(obs_to_vec(obs).numel())
    action_dim = env.action_space.n
    return env, obs_dim, action_dim


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    env, obs_dim, action_dim = make_env()

    # Encoder input = obs_vec + action_scalar + reward
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
        mixture_ratio=0.3,
        mixture_schedule=linear_mixture_schedule(
            0.2, 0.7, warmup_steps=args.warmup_steps
        ),
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
        min_buffer_episodes=10,
        batch_size=args.batch_size,
    )

    def query_builder(o):
        v = obs_to_vec(o)
        return torch.cat([v, torch.zeros(2)])  # action scalar + reward

    trainer.query_state_fn = query_builder

    # Monkey-patch trainer._to_tensor to use obs_to_vec for states
    orig_to_tensor = trainer._to_tensor

    def _to_tensor_obs(x):
        try:
            return obs_to_vec(x)
        except Exception:
            return orig_to_tensor(x)

    trainer._to_tensor = _to_tensor_obs  # type: ignore[assignment]

    metrics = trainer.train(env, total_steps=args.steps)
    print("training complete; episodes:", len(metrics["rewards"]))
    if metrics["consolidation_loss"]:
        print(
            "avg consolidation loss:",
            sum(metrics["consolidation_loss"]) / len(metrics["consolidation_loss"]),
        )
    print("buffer stats:", buffer.stats())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument(
        "--schedule",
        type=str,
        default="interleaved",
        choices=["periodic", "interleaved", "adaptive"],
    )
    p.add_argument("--cons-interval", dest="cons_interval", type=int, default=2_000)
    p.add_argument("--cons-steps", dest="cons_steps", type=int, default=100)
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())

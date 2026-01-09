#!/usr/bin/env python3
"""Synthetic sensitivity analysis for consolidation hyperparameters.

Sweeps temperature and reward_weight on a two-cluster synthetic dataset and
reports reward_alignment and retrieval_sharpness.
"""
from __future__ import annotations

import argparse
import itertools
from typing import Tuple

import torch

from hippotorch import Consolidator, DualEncoder, Episode, MemoryStore


def gen_episode(
    T: int, state_dim: int, action_dim: int, s_val: float, r_val: float
) -> Episode:
    states = torch.full((T, state_dim), s_val)
    actions = torch.zeros(T, action_dim)
    rewards = torch.full((T,), r_val)
    dones = torch.zeros(T, dtype=torch.bool)
    return Episode(states=states, actions=actions, rewards=rewards, dones=dones)


def build_memory(
    num_good: int, num_bad: int, T: int, state_dim: int, action_dim: int, embed_dim: int
) -> Tuple[DualEncoder, MemoryStore]:
    input_dim = state_dim + action_dim + 1
    encoder = DualEncoder(
        input_dim=input_dim, embed_dim=embed_dim, refresh_interval=10_000
    )
    memory = MemoryStore(embed_dim=embed_dim, capacity=10_000)
    episodes = [
        gen_episode(T, state_dim, action_dim, 1.0, 1.0) for _ in range(num_good)
    ] + [gen_episode(T, state_dim, action_dim, -1.0, 0.1) for _ in range(num_bad)]
    keys = []
    for ep in episodes:
        x = torch.cat([ep.states, ep.actions, ep.rewards.unsqueeze(-1)], dim=-1)
        keys.append(encoder.encode_key(x.unsqueeze(0)).squeeze(0))
    memory.write(torch.stack(keys), episodes, encoder_step=0)
    return encoder, memory


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    encoder, memory = build_memory(
        num_good=args.num_good,
        num_bad=args.num_bad,
        T=args.length,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        embed_dim=args.embed_dim,
    )

    temps = args.temperatures
    weights = args.reward_weights
    print("temperature,reward_weight,retrieval_sharpness,reward_alignment,loss")
    for temp, w in itertools.product(temps, weights):
        consolidator = Consolidator(
            encoder, temperature=float(temp), reward_weight=float(w), learning_rate=1e-4
        )
        metrics = consolidator.sleep(
            memory,
            steps=args.steps,
            batch_size=args.batch_size,
            refresh_keys=True,
            report_quality=True,
        )
        after = consolidator.get_representation_quality(memory)
        print(
            f"{temp},{w},{after['retrieval_sharpness']:.4f},{after['reward_alignment']:.4f},{metrics['loss']:.4f}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--state-dim", type=int, default=3)
    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--num-good", type=int, default=16)
    p.add_argument("--num-bad", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.05, 0.07, 0.1, 0.2]
    )
    p.add_argument(
        "--reward-weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())

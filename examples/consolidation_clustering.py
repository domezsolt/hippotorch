#!/usr/bin/env python3
"""Consolidation clustering benchmark on synthetic episodes.

Measures reward_alignment and retrieval_sharpness before/after sleep.
"""
from __future__ import annotations

import argparse

import torch

from hippotorch import Consolidator, DualEncoder, Episode, MemoryStore


def gen_episode(
    T: int, state_dim: int, action_dim: int, state_value: float, reward_value: float
) -> Episode:
    states = torch.full((T, state_dim), state_value)
    actions = torch.zeros(T, action_dim)
    rewards = torch.full((T,), reward_value)
    dones = torch.zeros(T, dtype=torch.bool)
    return Episode(states=states, actions=actions, rewards=rewards, dones=dones)


def to_input(ep: Episode) -> torch.Tensor:
    return torch.cat([ep.states, ep.actions, ep.rewards.unsqueeze(-1)], dim=-1)


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    input_dim = args.state_dim + args.action_dim + 1
    encoder = DualEncoder(
        input_dim=input_dim, embed_dim=args.embed_dim, refresh_interval=10_000
    )
    memory = MemoryStore(embed_dim=args.embed_dim, capacity=10_000)

    # Build synthetic dataset with two reward groups
    good_eps = [
        gen_episode(args.length, args.state_dim, args.action_dim, 1.0, 1.0)
        for _ in range(args.num_good)
    ]
    bad_eps = [
        gen_episode(args.length, args.state_dim, args.action_dim, -1.0, 0.1)
        for _ in range(args.num_bad)
    ]
    episodes = good_eps + bad_eps

    keys = []
    for ep in episodes:
        key = encoder.encode_key(to_input(ep).unsqueeze(0)).squeeze(0)
        keys.append(key)
    memory.write(torch.stack(keys), episodes, encoder_step=0)

    consolidator = Consolidator(
        encoder, temperature=0.07, reward_weight=0.5, learning_rate=1e-4
    )

    before = consolidator.get_representation_quality(memory)
    print("before:", before)

    metrics = consolidator.sleep(
        memory,
        steps=args.steps,
        batch_size=args.batch_size,
        refresh_keys=True,
        report_quality=True,
    )
    print("consolidation:", metrics)

    after = consolidator.get_representation_quality(memory)
    print("after:", after)


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
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())

#!/usr/bin/env python3
"""The Amnesiac's Corridor: validates reward-aware retrieval on a long-horizon task.

Dependencies: gymnasium, numpy, torch

This script implements:
- A simple POMDP environment with a distractor corridor and a signal at t=0
- A HippoAgent that conditions its policy on a retrieved memory embedding
- A training loop that compares hybrid replay vs uniform baseline

Run:
  python -m examples.amnesiac_corridor --episodes 400 --mixture 0.5 --uniform 0.0
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover - optional dependency
    raise SystemExit("Please install gymnasium: pip install gymnasium") from e

from hippotorch import (Consolidator, DualEncoder, HippocampalReplayBuffer,
                        MemoryStore, SB3ReplayBufferWrapper,
                        linear_mixture_schedule)


class AmnesiacCorridor(gym.Env):
    def __init__(
        self,
        length: int = 30,
        *,
        coin_reward: float = 0.1,
        loiter_penalty: float = -0.01,
        success_reward: float = 10.0,
        failure_penalty: float = -5.0,
    ):
        super().__init__()
        self.length = length
        self.coin_reward = float(coin_reward)
        self.loiter_penalty = float(loiter_penalty)
        self.success_reward = float(success_reward)
        self.failure_penalty = float(failure_penalty)
        # [progress, signal_red, signal_blue, is_junction]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        # 0: left, 1: forward, 2: right
        self.action_space = spaces.Discrete(3)
        self.t = 0
        self.signal = 0

    def reset(self, *, seed: int | None = None, options=None):  # noqa: D401 - Gym API
        super().reset(seed=seed)
        self.t = 0
        self.signal = int(np.random.choice([0, 1]))
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        obs = np.zeros((4,), dtype=np.float32)
        obs[0] = float(self.t) / float(self.length)
        if self.t == 0:
            obs[1] = 1.0 if self.signal == 0 else 0.0
            obs[2] = 1.0 if self.signal == 1 else 0.0
        elif self.t >= self.length:
            obs[3] = 1.0
        return obs

    def step(self, action: int):  # noqa: D401 - Gym API
        reward = 0.0
        done = False
        if self.t < self.length:
            if action == 1:  # forward
                self.t += 1
                reward = self.coin_reward
            else:
                reward = self.loiter_penalty
        else:
            correct = 0 if self.signal == 0 else 2
            reward = self.success_reward if action == correct else self.failure_penalty
            done = True
        return self._obs(), reward, done, False, {}

    def set_length(self, new_length: int) -> None:
        self.length = int(max(1, new_length))


@dataclass
class HippoConfig:
    obs_dim: int = 4
    action_dim: int = 3
    embed_dim: int = 32
    gamma: float = 0.99
    lr: float = 1e-3


class HippoAgent(nn.Module):
    """Policy/Q-network conditioned on retrieved memory embedding."""

    def __init__(
        self, cfg: HippoConfig, buffer: HippocampalReplayBuffer, *, window_size: int = 8
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.buffer = buffer
        self.window_size = int(max(1, window_size))
        self.history: deque[tuple[np.ndarray, float, float]] = deque(
            maxlen=self.window_size
        )
        self.oracle_retrieval: bool = False
        self.episode_signal: int | None = None
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim + cfg.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, cfg.action_dim),
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def reset_history(self) -> None:
        self.history.clear()
        self.episode_signal = None

    def record_transition(
        self, obs: np.ndarray, action: int | float, reward: float
    ) -> None:
        self.history.append(
            (np.array(obs, dtype=np.float32), float(action), float(reward))
        )

    def set_signal(self, signal: int | None) -> None:
        self.episode_signal = int(signal) if signal is not None else None

    @torch.no_grad()
    def _context(self, obs: np.ndarray) -> torch.Tensor:
        # Build a short query sequence from recent history plus current obs
        seq_elems = list(self.history)
        # Append current obs with zero action/reward placeholder
        seq_elems.append((np.array(obs, dtype=np.float32), 0.0, 0.0))
        xs = []
        for o, a, r in seq_elems[-self.window_size :]:
            o_t = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            xs.append(
                torch.cat(
                    [
                        o_t,
                        torch.tensor([[a]], dtype=torch.float32),
                        torch.tensor([[r]], dtype=torch.float32),
                    ],
                    dim=-1,
                )
            )
        query_input = torch.stack(xs, dim=1)  # [1, T, D]
        query = self.buffer.encoder.encode_query(query_input)
        scores, eps_lists = self.buffer.memory.retrieve(
            query,
            top_k=1,
            encoder=self.buffer.encoder,
            current_step=0,
            lazy_refresh=True,
        )
        if scores.shape[-1] == 0 or not eps_lists or not eps_lists[0]:
            return torch.zeros(1, self.cfg.embed_dim)
        ep = eps_lists[0][0]
        # encode first step of retrieved episode
        x = torch.cat(
            [ep.states[0:1], ep.actions[0:1], ep.rewards[0:1].unsqueeze(-1)], dim=-1
        ).unsqueeze(0)
        return self.buffer.encoder.encode_key(x)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.1) -> int:
        # At the junction, attempt memory-guided decision first
        if obs[3] > 0.5:
            # Oracle retrieval: feed start-frame context to policy
            if self.oracle_retrieval and self.episode_signal is not None:
                ctx = self._oracle_context()
                x = torch.cat(
                    [torch.tensor(obs, dtype=torch.float32).unsqueeze(0), ctx], dim=-1
                )
                q = self.net(x)
                return int(q.argmax(dim=-1).item())
            # Heuristic override: pick door by retrieved signal
            sig = self._retrieved_signal(obs)
            if sig is not None and np.random.rand() >= epsilon:
                return 0 if sig == 0 else 2
        # Epsilon-greedy fallback policy
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.cfg.action_dim))
        ctx = self._context(obs)
        x = torch.cat(
            [torch.tensor(obs, dtype=torch.float32).unsqueeze(0), ctx], dim=-1
        )
        q = self.net(x)
        return int(q.argmax(dim=-1).item())

    @torch.no_grad()
    def _retrieved_signal(self, obs: np.ndarray) -> int | None:
        """Infer start signal (0=red, 1=blue) via a sequence query.

        Uses the top retrieved episode and inspects its start frame.
        """
        # Use the same sequence query as in _context
        seq_elems = list(self.history)
        seq_elems.append((np.array(obs, dtype=np.float32), 0.0, 0.0))
        xs = []
        for o, a, r in seq_elems[-self.window_size :]:
            o_t = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            xs.append(
                torch.cat(
                    [
                        o_t,
                        torch.tensor([[a]], dtype=torch.float32),
                        torch.tensor([[r]], dtype=torch.float32),
                    ],
                    dim=-1,
                )
            )
        query_input = torch.stack(xs, dim=1)
        query = self.buffer.encoder.encode_query(query_input)
        scores, eps_lists = self.buffer.memory.retrieve(
            query,
            top_k=5,
            encoder=self.buffer.encoder,
            current_step=0,
            lazy_refresh=True,
        )
        if scores.shape[-1] == 0 or not eps_lists or not eps_lists[0]:
            return None
        # Prefer the retrieved episode with highest total reward (bias toward successes)
        candidates = eps_lists[0]
        ep = max(candidates, key=lambda e: float(getattr(e, "total_reward", 0.0)))
        start_obs = ep.states[0]
        # signal bits are indices 1 and 2 per env encoding
        if start_obs[1] > start_obs[2]:
            return 0
        if start_obs[2] > start_obs[1]:
            return 1
        return None

    @torch.no_grad()
    def _oracle_context(self) -> torch.Tensor:
        """Encode an oracle context built from the true start signal frame."""
        if self.episode_signal is None:
            return torch.zeros(1, self.cfg.embed_dim)
        # Build start observation: progress=0, one-hot signal bits, not at junction
        start_obs = np.zeros((4,), dtype=np.float32)
        start_obs[0] = 0.0
        if self.episode_signal == 0:
            start_obs[1] = 1.0
        else:
            start_obs[2] = 1.0
        x = torch.tensor(start_obs, dtype=torch.float32).view(1, 1, -1)
        x = torch.cat([x, torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)], dim=-1)
        return self.buffer.encoder.encode_key(x)

    def update(self, batch: dict, *, use_context: bool = True) -> float:
        states = batch["states"].float()
        actions = batch["actions"].long().view(-1)
        rewards = batch["rewards"].float()
        next_states = batch["next_states"].float()
        dones = batch["dones"].float()

        if use_context:
            ctx = self._batch_context(states)
            x = torch.cat([states, ctx], dim=-1)
            ctx_next = self._batch_context(next_states)
            x_next = torch.cat([next_states, ctx_next], dim=-1)
        else:
            x, x_next = states, next_states

        q_pred = self.net(x).gather(1, actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next = self.net(x_next).max(dim=-1).values
            target = rewards + self.cfg.gamma * (1.0 - dones) * q_next
        loss = nn.functional.mse_loss(q_pred, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item())

    @torch.no_grad()
    def _batch_context(self, states: torch.Tensor) -> torch.Tensor:
        # states: [B, obs_dim]
        ctxs = []
        for s in states:
            ctxs.append(self._context(s.cpu().numpy()))
        return torch.cat(ctxs, dim=0)


def run_variant(
    env_len: int,
    episodes: int,
    mixture_ratio: float,
    *,
    seed: int = 0,
    coin_reward: float = 0.1,
    loiter_penalty: float = -0.01,
    success_reward: float = 10.0,
    failure_penalty: float = -5.0,
    schedule_start: float | None = None,
    schedule_end: float | None = None,
    warmup_episodes: int = 200,
    reward_weight: float = 0.5,
    temperature: float = 0.07,
    cons_every: int = 10,
    cons_steps: int = 100,
    curriculum: bool = False,
    min_length: int = 5,
    length_step: int = 5,
    success_window: int = 50,
    success_threshold: float = 0.8,
    oracle: bool = False,
    oracle_retrieval: bool = False,
) -> Tuple[float, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = AmnesiacCorridor(
        length=env_len,
        coin_reward=coin_reward,
        loiter_penalty=loiter_penalty,
        success_reward=success_reward,
        failure_penalty=failure_penalty,
    )

    obs_dim, action_dim = 4, 3
    input_dim = obs_dim + 1 + 1
    embed_dim = 32
    encoder = DualEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
        momentum=0.995,
        refresh_interval=10_000,
    )
    memory = MemoryStore(embed_dim=embed_dim, capacity=10_000)
    warmup_steps_est = warmup_episodes * (env_len + 2)
    schedule = (
        None
        if (schedule_start is None or schedule_end is None)
        else linear_mixture_schedule(
            float(schedule_start),
            float(schedule_end),
            warmup_steps=int(warmup_steps_est),
        )
    )
    buffer = HippocampalReplayBuffer(
        memory=memory,
        encoder=encoder,
        mixture_ratio=mixture_ratio,
        mixture_schedule=schedule,
    )
    wrapper = SB3ReplayBufferWrapper(buffer)  # accumulate transitions into episodes
    consolidator = Consolidator(
        encoder,
        temperature=float(temperature),
        reward_weight=float(reward_weight),
        learning_rate=1e-4,
    )
    buffer.attach_consolidator(consolidator)

    agent = HippoAgent(
        HippoConfig(obs_dim=obs_dim, action_dim=action_dim, embed_dim=embed_dim),
        buffer,
        window_size=8,
    )

    total_rewards: list[float] = []
    successes = deque(maxlen=success_window)

    if curriculum:
        env.set_length(min_length)
    for ep in range(episodes):
        obs, _ = env.reset()
        agent.reset_history()
        agent.oracle_retrieval = bool(oracle_retrieval)
        agent.set_signal(env.signal)
        done = False
        ep_reward = 0.0
        while not done:
            if oracle and env.t >= env.length:
                # Oracle decision at junction
                action = 0 if env.signal == 0 else 2
            else:
                action = agent.act(obs, epsilon=max(0.05, 1.0 - ep / 200.0))
            next_obs, reward, done, _, _ = env.step(action)
            wrapper.add(
                obs,
                next_obs,
                np.array([action], dtype=np.float32),
                float(reward),
                bool(done),
            )
            # record transition for sequence queries on the next step
            agent.record_transition(obs, action, reward)

            if buffer.size > 10:  # enough episodes collected
                obs_t = torch.tensor(obs, dtype=torch.float32)
                query_vec = torch.cat(
                    [obs_t, torch.zeros(2)]
                )  # [obs, action_scalar, reward]
                batch = buffer.sample(batch_size=32, query_state=query_vec)
                agent.update(batch, use_context=True)

            obs = next_obs
            ep_reward += float(reward)

        total_rewards.append(ep_reward)
        # track success at junction
        if done:
            successes.append(1.0 if reward > 0 else 0.0)
        # curriculum progression
        if curriculum and len(successes) == successes.maxlen and env.length < env_len:
            if (sum(successes) / len(successes)) >= success_threshold:
                env.set_length(min(env_len, env.length + length_step))
                successes.clear()
        # periodic consolidation
        if (ep + 1) % max(1, cons_every) == 0 and buffer.size > 10:
            _ = buffer.consolidate(steps=int(cons_steps), report_quality=False)

    return float(np.mean(total_rewards[-50:])), total_rewards


def main(args: argparse.Namespace) -> None:
    print("Running Amnesiac's Corridor...")
    mean_uniform, rewards_uniform = run_variant(
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
    )
    mean_hybrid, rewards_hybrid = run_variant(
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
    )
    print(
        {
            "uniform_mean_last50": round(mean_uniform, 2),
            "hybrid_mean_last50": round(mean_hybrid, 2),
            "delta": round(mean_hybrid - mean_uniform, 2),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=30)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--mixture", type=float, default=0.5, help="hybrid mixture ratio")
    p.add_argument("--uniform", type=float, default=0.0, help="uniform baseline ratio")
    p.add_argument("--coin-reward", dest="coin_reward", type=float, default=0.1)
    p.add_argument("--loiter-penalty", dest="loiter_penalty", type=float, default=-0.01)
    p.add_argument("--success-reward", dest="success_reward", type=float, default=10.0)
    p.add_argument(
        "--failure-penalty", dest="failure_penalty", type=float, default=-5.0
    )
    p.add_argument("--schedule-start", dest="schedule_start", type=float, default=0.0)
    p.add_argument("--schedule-end", dest="schedule_end", type=float, default=0.5)
    p.add_argument("--warmup-episodes", dest="warmup_episodes", type=int, default=200)
    p.add_argument("--reward-weight", dest="reward_weight", type=float, default=0.7)
    p.add_argument("--temperature", dest="temperature", type=float, default=0.07)
    p.add_argument("--cons-every", dest="cons_every", type=int, default=10)
    p.add_argument("--cons-steps", dest="cons_steps", type=int, default=100)
    p.add_argument(
        "--curriculum", action="store_true", help="enable corridor length curriculum"
    )
    p.add_argument("--min-length", dest="min_length", type=int, default=5)
    p.add_argument("--length-step", dest="length_step", type=int, default=5)
    p.add_argument("--success-window", dest="success_window", type=int, default=50)
    p.add_argument(
        "--success-threshold", dest="success_threshold", type=float, default=0.8
    )
    p.add_argument(
        "--oracle",
        action="store_true",
        help="use oracle action at junction for regret analysis",
    )
    main(p.parse_args())

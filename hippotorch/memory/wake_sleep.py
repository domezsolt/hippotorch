from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from hippotorch.core.episode import Episode
from hippotorch.memory.replay import HippocampalReplayBuffer
from hippotorch.segmenters.base import EpisodeSegmenter


class WakeSleepTrainer:
    """Training loop that alternates wake (RL) and sleep (consolidation)."""

    def __init__(
        self,
        agent: object,
        buffer: HippocampalReplayBuffer,
        segmenter: EpisodeSegmenter,
        *,
        consolidation_schedule: str = "periodic",
        consolidation_interval: int = 1000,
        consolidation_steps: int = 100,
        min_buffer_episodes: int = 10,
        batch_size: int = 256,
        mixture_schedule: Optional[Callable[[int], float]] = None,
        report_quality: bool = False,
        log_interval: int = 0,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.agent = agent
        self.buffer = buffer
        self.segmenter = segmenter
        self.schedule = consolidation_schedule
        self.interval = consolidation_interval
        self.consolidation_steps = consolidation_steps
        self.min_buffer_episodes = min_buffer_episodes
        self.batch_size = batch_size
        self.report_quality = report_quality
        if mixture_schedule is not None:
            self.buffer.mixture_schedule = mixture_schedule

        self.total_steps = 0
        self.wake_steps = 0
        self.sleep_steps = 0
        self._steps_since_consolidation = 0
        self._representation_quality_history: List[float] = []
        # Optional callable to build query feature vector for semantic sampling
        # Signature: fn(state) -> Tensor of shape [input_dim]
        self.query_state_fn: Optional[Callable] = None
        self.log_interval = int(max(0, log_interval))
        if log_fn is None:

            def _default_log(msg: str) -> None:
                # ensure immediate visibility even when stdout is piped
                print(msg, flush=True)

            self._log = _default_log
        else:
            self._log = log_fn

    def train(
        self,
        env,
        total_steps: int,
        *,
        eval_interval: int = 0,
        eval_episodes: int = 0,
    ) -> Dict[str, List[float]]:
        """Run wake/sleep training for a fixed number of environment steps."""

        metrics = {
            "rewards": [],
            "consolidation": [],
            "consolidation_loss": [],
            "reward_alignment": [],
            "retrieval_sharpness": [],
            "eval_rewards": [],
        }

        state, _ = env.reset()
        trajectory = self._empty_trajectory_like(state)
        episode_reward = 0.0

        for step in range(total_steps):
            action = self.agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            trajectory["states"].append(self._to_tensor(state))
            trajectory["actions"].append(self._to_tensor(action))
            trajectory["rewards"].append(torch.tensor(reward).view(1))
            trajectory["dones"].append(torch.tensor(done))

            episode_reward += float(reward)

            if self.buffer.size >= self.min_buffer_episodes and hasattr(
                self.agent, "update"
            ):
                query_vec = self._build_query(state)
                batch = self.buffer.sample(
                    batch_size=self.batch_size,
                    query_state=query_vec,
                )
                self.agent.update(batch)

            if done:
                metrics["rewards"].append(episode_reward)
                trajectory_episode = self._trajectory_to_episode(trajectory)
                for segment in self.segmenter.segment(trajectory_episode):
                    self.buffer.add_episode(segment)
                state, _ = env.reset()
                trajectory = self._empty_trajectory_like(state)
                episode_reward = 0.0
            else:
                state = next_state

            self.wake_steps += 1
            self.total_steps += 1
            self._steps_since_consolidation += 1

            if self._should_consolidate():
                consolidation_metrics = self._consolidate()
                metrics["consolidation"].append(consolidation_metrics)
                metrics["consolidation_loss"].append(
                    consolidation_metrics.get("loss", 0.0)
                )
                if "reward_alignment" in consolidation_metrics:
                    metrics["reward_alignment"].append(
                        consolidation_metrics["reward_alignment"]
                    )
                if "retrieval_sharpness" in consolidation_metrics:
                    metrics["retrieval_sharpness"].append(
                        consolidation_metrics["retrieval_sharpness"]
                    )

            if (
                eval_interval
                and step > 0
                and step % eval_interval == 0
                and eval_episodes > 0
            ):
                metrics["eval_rewards"].append(self._evaluate(env, eval_episodes))

            # Progress logging
            if self.log_interval and (step + 1) % self.log_interval == 0:
                stats = self.buffer.stats()
                msg = (
                    f"[wake] step={self.total_steps} wake_steps={self.wake_steps} "
                    f"sleep_steps={self.sleep_steps} mem_eps={int(stats.get('memory_size', 0))} "
                    f"eff_ratio={stats.get('effective_ratio', 0.0):.2f}"
                )
                try:
                    self._log(msg)
                except Exception:
                    pass

        return metrics

    def _should_consolidate(self) -> bool:
        if self.buffer.size < self.min_buffer_episodes:
            return False

        if self.schedule == "periodic":
            return self._steps_since_consolidation >= self.interval
        if self.schedule == "interleaved":
            return self._steps_since_consolidation >= max(1, self.interval // 10)
        if self.schedule == "adaptive":
            if self._steps_since_consolidation < max(1, self.interval // 2):
                return False
            if self.buffer.consolidator is None:
                return False
            quality = self.buffer.consolidator.get_representation_quality(
                self.buffer.memory
            )
            reward_alignment = quality.get("reward_alignment", 0.0)
            self._representation_quality_history.append(reward_alignment)
            if len(self._representation_quality_history) < 5:
                return False
            recent = self._representation_quality_history[-5:]
            trend = sum(recent[:-1]) / len(recent[:-1])
            return (
                reward_alignment < trend - 0.05
                or self._steps_since_consolidation >= self.interval
            )
        return False

    def _consolidate(self) -> Dict[str, float]:
        steps = self.consolidation_steps
        if self.schedule == "interleaved":
            steps = max(1, self.consolidation_steps // 10)

        metrics = self.buffer.consolidate(
            steps=steps, report_quality=self.report_quality
        )
        self.sleep_steps += steps
        self._steps_since_consolidation = 0
        return {k: float(v) for k, v in metrics.items()}

    def _evaluate(self, env, n_episodes: int) -> float:
        total_reward = 0.0
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = (
                    self.agent.act(state, explore=False)
                    if hasattr(self.agent, "act")
                    else self.agent(state)
                )
                state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                total_reward += float(reward)
        return total_reward / max(1, n_episodes)

    def _empty_trajectory_like(self, state) -> Dict[str, List[torch.Tensor]]:
        return {"states": [], "actions": [], "rewards": [], "dones": []}

    def _trajectory_to_episode(
        self, trajectory: Dict[str, List[torch.Tensor]]
    ) -> Episode:
        states = torch.stack(trajectory["states"])
        actions = torch.stack(trajectory["actions"])
        rewards = torch.stack(trajectory["rewards"]).squeeze(-1)
        dones = torch.stack(trajectory["dones"])
        return Episode(states=states, actions=actions, rewards=rewards, dones=dones)

    def _to_tensor(self, array_like) -> torch.Tensor:
        tensor = (
            array_like
            if isinstance(array_like, torch.Tensor)
            else torch.as_tensor(array_like)
        )
        # Ensure scalars become 1D vectors (e.g., actions/rewards)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _build_query(self, state) -> Optional[torch.Tensor]:
        if self.query_state_fn is None:
            return self._to_tensor(state)
        q = self.query_state_fn(state)
        return q if isinstance(q, torch.Tensor) else torch.as_tensor(q)


__all__ = ["WakeSleepTrainer"]

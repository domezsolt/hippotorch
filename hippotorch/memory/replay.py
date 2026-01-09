from __future__ import annotations

from os import PathLike
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from hippotorch.core.episode import Episode
from hippotorch.encoders.dual import DualEncoder
from hippotorch.memory.consolidator import Consolidator
from hippotorch.memory.store import MemoryStore


class HippocampalReplayBuffer:
    """Hybrid replay buffer mixing semantic and uniform sampling."""

    def __init__(
        self,
        memory: MemoryStore,
        encoder: DualEncoder,
        mixture_ratio: float = 0.5,
        mixture_schedule: Optional[Callable[[int], float]] = None,
        consolidator: Optional[Consolidator] = None,
    ) -> None:
        self.memory = memory
        self.encoder = encoder
        self.mixture_ratio = mixture_ratio
        self.mixture_schedule = mixture_schedule
        self.consolidator = consolidator
        self._total_steps = 0
        self._encoder_step = 0
        self._last_ratio = mixture_ratio
        self._sampling_stats: Dict[str, float] = {
            "semantic": 0.0,
            "uniform": 0.0,
            "retrieved_candidates": 0.0,
            "effective_ratio": mixture_ratio,
        }

    def add_episode(self, episode: Episode, encoder_step: Optional[int] = None) -> None:
        """Encode and store an episode, tracking encoder freshness."""
        episode_tensor = self._episode_to_tensor(episode).unsqueeze(0)
        # encode_key returns shape [1, embed_dim] for a single episode batch
        key = self.encoder.encode_key(episode_tensor)

        if encoder_step is None:
            self._encoder_step += 1
            step_value = self._encoder_step
        else:
            step_value = int(encoder_step)
            self._encoder_step = max(self._encoder_step, step_value)

        self.memory.write(key, [episode], encoder_step=step_value)

    @property
    def size(self) -> int:
        """Number of stored episodes."""

        return len(self.memory)

    def attach_consolidator(self, consolidator: Consolidator) -> None:
        """Attach a consolidator post-construction."""

        self.consolidator = consolidator

    def sample(
        self, batch_size: int, query_state: Optional[Tensor] = None, top_k: int = 5
    ) -> Dict[str, Tensor]:
        if len(self.memory) == 0:
            raise ValueError("Memory is empty; add episodes before sampling.")

        ratio = self._current_ratio()
        self._last_ratio = ratio
        n_semantic = min(int(batch_size * ratio), batch_size)
        n_uniform = batch_size - n_semantic

        samples: List[Tuple[Episode, int]] = []
        retrieved_candidates = 0

        if n_semantic > 0 and query_state is not None:
            query = self._prepare_query(query_state)
            _, retrieved = self.memory.retrieve(
                query,
                top_k=top_k,
                encoder=self.encoder,
                current_step=self._encoder_step,
                lazy_refresh=True,
            )
            retrieved_eps = retrieved[0] if retrieved else []
            retrieved_candidates = len(retrieved_eps)
            if retrieved_eps:
                samples.extend(self._scatter_sample(retrieved_eps, n_semantic))
            else:
                n_uniform += n_semantic

        if n_uniform > 0:
            uniform_eps = self.memory.sample_uniform(n_uniform)
            for episode in uniform_eps:
                if len(episode) <= 1:
                    continue
                idx = torch.randint(0, len(episode) - 1, (1,)).item()
                samples.append((episode, idx))

        if not samples:
            raise ValueError("Unable to draw samples from memory.")

        self._total_steps += len(samples)
        total_drawn = len(samples)
        effective_ratio = (
            (total_drawn - n_uniform) / total_drawn if total_drawn else 0.0
        )
        self._sampling_stats = {
            "semantic": float(total_drawn - n_uniform),
            "uniform": float(n_uniform),
            "retrieved_candidates": float(retrieved_candidates),
            "effective_ratio": float(effective_ratio),
        }
        return self._samples_to_batch(samples[:batch_size])

    def sample_with_intrinsic(
        self,
        batch_size: int,
        *,
        query_state: Optional[Tensor] = None,
        top_k: int = 5,
        intrinsic_top_k: int = 5,
        intrinsic_scale: float = 1.0,
    ) -> Dict[str, Tensor]:
        """Sample a batch and attach an intrinsic curiosity reward.

        Computes inverse-distance novelty for each sampled state against memory keys.
        """
        batch = self.sample(batch_size=batch_size, query_state=query_state, top_k=top_k)
        try:
            from .intrinsic import IntrinsicConsolidator

            ic = IntrinsicConsolidator(
                memory=self.memory,
                encoder=self.encoder,
                target_similarity=1.0,
                scale=float(intrinsic_scale),
                top_k=int(intrinsic_top_k),
            )
            intrinsic = []
            for s in batch["states"]:
                out = ic.compute(torch.cat([s, torch.zeros(2, device=s.device)]))
                intrinsic.append(out.intrinsic_reward)
            batch["intrinsic"] = torch.stack(intrinsic)
        except Exception:
            pass
        return batch

    def sample_windows(
        self,
        batch_size: int,
        *,
        window_size: int = 10,
        query_state: Optional[Tensor] = None,
        top_k: int = 5,
    ) -> Dict[str, Tensor]:
        """Draw a batch of temporal windows for POMDP agents.

        Returns a dictionary with padded tensors: states, actions, rewards, dones, lengths.
        """
        if len(self.memory) == 0:
            raise ValueError("Memory is empty; add episodes before sampling.")

        ratio = self._current_ratio()
        n_semantic = min(int(batch_size * ratio), batch_size)
        n_uniform = batch_size - n_semantic

        windows: List[Episode] = []

        if n_semantic > 0 and query_state is not None:
            query = self._prepare_query(query_state)
            _, retrieved = self.memory.retrieve(
                query,
                top_k=top_k,
                encoder=self.encoder,
                current_step=self._encoder_step,
                lazy_refresh=True,
            )
            retrieved_eps = retrieved[0] if retrieved else []
            if retrieved_eps:
                windows.extend(
                    self._scatter_sample_windows(
                        retrieved_eps, n_semantic, window_size=window_size
                    )
                )
            else:
                n_uniform += n_semantic

        if n_uniform > 0:
            uniform_eps = self.memory.sample_uniform(n_uniform)
            windows.extend(
                self._scatter_sample_windows(
                    uniform_eps, n_uniform, window_size=window_size
                )
            )

        if not windows:
            raise ValueError("Unable to draw sequence windows from memory.")

        return self._windows_to_batch(windows[:batch_size])

    def consolidate(
        self,
        *,
        steps: int = 1,
        batch_size: int = 32,
        refresh_keys: bool = True,
        report_quality: bool = False,
        tb_writer=None,
        tb_log_every: int = 0,
    ) -> Dict[str, float]:
        """Run consolidation through the attached Consolidator."""

        if self.consolidator is None:
            raise RuntimeError("No consolidator attached to replay buffer")

        metrics = self.consolidator.sleep(
            self.memory,
            steps=steps,
            batch_size=batch_size,
            refresh_keys=refresh_keys,
            report_quality=report_quality,
            tb_writer=tb_writer,
            tb_log_every=tb_log_every,
        )
        return metrics

    def save_to_hub(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        revision: str = "main",
        private: bool = False,
        exist_ok: bool = True,
        filename: str = "memory.pt",
        workdir: Optional[str] = None,
        readme: Optional[str] = None,
    ) -> PathLike[str] | str:
        """Upload the attached memory to the HuggingFace Hub.

        This is a thin convenience wrapper around
        :func:`hippotorch.utils.hub.push_memory_to_hub` that keeps the buffer API
        ergonomic for transfer learning workflows.
        """

        from hippotorch.utils.hub import push_memory_to_hub

        return push_memory_to_hub(
            self.memory,
            repo_id,
            token=token,
            revision=revision,
            private=private,
            exist_ok=exist_ok,
            filename=filename,
            workdir=workdir,
            readme=readme,
        )

    def load_memory_from_hub(
        self,
        repo_id: str,
        *,
        filename: str = "memory.pt",
        revision: str = "main",
        token: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        cache_dir: Optional[str] = None,
    ) -> MemoryStore:
        """Replace the underlying memory with a checkpoint from the Hub."""

        from hippotorch.utils.hub import load_memory_from_hub

        target_device = device or getattr(self.memory, "device", torch.device("cpu"))
        loaded = load_memory_from_hub(
            repo_id,
            filename=filename,
            revision=revision,
            token=token,
            device=target_device,
            cache_dir=cache_dir,
        )
        self._validate_memory_compatibility(loaded)
        self.memory = loaded
        return loaded

    def _validate_memory_compatibility(self, memory: MemoryStore) -> None:
        """Ensure loaded memory matches encoder and buffer expectations."""

        encoder_embed = getattr(self.encoder.projector[0], "out_features", None)
        if encoder_embed is not None and memory.embed_dim != int(encoder_embed):
            raise ValueError(
                (
                    "Loaded memory embed_dim "
                    f"{memory.embed_dim} does not match encoder embed_dim {encoder_embed}."
                )
            )

        if (
            hasattr(self.memory, "embed_dim")
            and memory.embed_dim != self.memory.embed_dim
        ):
            raise ValueError(
                (
                    "Loaded memory embed_dim "
                    f"{memory.embed_dim} does not match buffer embed_dim {self.memory.embed_dim}."
                )
            )

    def _current_ratio(self) -> float:
        if self.mixture_schedule is None:
            return self.mixture_ratio
        return float(self.mixture_schedule(self._total_steps))

    def _prepare_query(self, state: Tensor) -> Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2:
            state = state.unsqueeze(0)
        # Align feature dimension with encoder input expectations (obs + action + reward)
        try:
            in_feats = self.encoder.online.input_proj.in_features  # type: ignore[attr-defined]
        except Exception:
            in_feats = state.shape[-1]
        d = state.shape[-1]
        if d < in_feats:
            pad = torch.zeros(
                *state.shape[:-1], in_feats - d, device=state.device, dtype=state.dtype
            )
            state = torch.cat([state, pad], dim=-1)
        elif d > in_feats:
            state = state[..., :in_feats]
        return self.encoder.encode_query(state)

    def _scatter_sample(
        self, episodes: List[Episode], n_samples: int
    ) -> List[Tuple[Episode, int]]:
        samples: List[Tuple[Episode, int]] = []
        per_episode = max(1, n_samples // len(episodes))
        for episode in episodes:
            if len(episode) <= 1:
                continue
            indices = torch.randint(0, len(episode) - 1, (per_episode,))
            for idx in indices.tolist():
                samples.append((episode, idx))
                if len(samples) >= n_samples:
                    return samples
        return samples

    def _episode_to_tensor(self, episode: Episode) -> Tensor:
        return torch.cat(
            [episode.states, episode.actions, episode.rewards.unsqueeze(-1)], dim=-1
        )

    def _samples_to_batch(
        self, samples: List[Tuple[Episode, int]]
    ) -> Dict[str, Tensor]:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for episode, idx in samples:
            states.append(episode.states[idx])
            actions.append(episode.actions[idx])
            rewards.append(episode.rewards[idx])
            done_value = (
                episode.dones[idx]
                if episode.dones is not None
                else torch.zeros((), dtype=torch.bool, device=episode.states.device)
            )
            dones.append(done_value)
            if idx + 1 < len(episode):
                next_states.append(episode.states[idx + 1])
            else:
                next_states.append(episode.states[idx])

        return {
            "states": torch.stack(states),
            "actions": torch.stack(actions),
            "rewards": torch.stack(rewards),
            "next_states": torch.stack(next_states),
            "dones": torch.stack(dones),
        }

    # --- Hub portability helpers ---
    # (hub helper methods defined above; duplicates removed)

    def _scatter_sample_windows(
        self, episodes: List[Episode], n_samples: int, *, window_size: int
    ) -> List[Episode]:
        """Sample random windows of length `window_size` from episodes."""
        if not episodes:
            return []
        out: List[Episode] = []
        attempts = 0
        while len(out) < n_samples and attempts < n_samples * 5:
            idx = torch.randint(0, len(episodes), (1,)).item()
            ep = episodes[idx]
            if len(ep) <= window_size:
                attempts += 1
                continue
            start = torch.randint(0, len(ep) - window_size, (1,)).item()
            out.append(ep.slice(start, start + window_size))
            attempts += 1
        return out

    def _windows_to_batch(self, windows: List[Episode]) -> Dict[str, Tensor]:
        """Pad variable-length windows and stack into batch tensors.

        Returns a dict with keys: states [B, T, Ds], actions [B, T, Da],
        rewards [B, T], dones [B, T], lengths [B].
        """
        if not windows:
            raise ValueError("No windows provided")
        lengths = torch.tensor([len(w) for w in windows], dtype=torch.long)
        max_len = int(lengths.max().item())
        states_b, actions_b, rewards_b, dones_b = [], [], [], []
        for w in windows:
            s, a, r = w.states, w.actions, w.rewards
            d = (
                w.dones
                if w.dones is not None
                else torch.zeros(len(w), dtype=torch.bool, device=s.device)
            )
            if len(w) < max_len:
                s = torch.cat(
                    [
                        s,
                        torch.zeros(
                            max_len - len(w),
                            *s.shape[1:],
                            device=s.device,
                            dtype=s.dtype,
                        ),
                    ],
                    dim=0,
                )
                a = torch.cat(
                    [
                        a,
                        torch.zeros(
                            max_len - len(w),
                            *a.shape[1:],
                            device=a.device,
                            dtype=a.dtype,
                        ),
                    ],
                    dim=0,
                )
                r = torch.cat(
                    [r, torch.zeros(max_len - len(w), device=r.device, dtype=r.dtype)],
                    dim=0,
                )
                d = torch.cat(
                    [d, torch.zeros(max_len - len(w), device=d.device, dtype=d.dtype)],
                    dim=0,
                )
            states_b.append(s)
            actions_b.append(a)
            rewards_b.append(r)
            dones_b.append(d)
        return {
            "states": torch.stack(states_b),
            "actions": torch.stack(actions_b),
            "rewards": torch.stack(rewards_b),
            "dones": torch.stack(dones_b),
            "lengths": lengths,
        }

    def stats(self) -> Dict[str, float]:
        """Return diagnostic statistics about sampling and storage."""

        stats = {
            "memory_size": float(len(self.memory)),
            "semantic_fraction": float(self._last_ratio),
            "encoder_step": float(self._encoder_step),
            "sampled_steps": float(self._total_steps),
        }
        staleness = self.memory.staleness_stats(current_step=self._encoder_step)
        stats.update({f"staleness_{k}": v for k, v in staleness.items()})
        stats.update(self._sampling_stats)
        return stats

    def sampling_stats(self) -> Dict[str, float]:
        """Expose the most recent sampling diagnostics."""

        return dict(self._sampling_stats)


__all__ = ["HippocampalReplayBuffer"]

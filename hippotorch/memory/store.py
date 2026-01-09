from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from hippotorch.core.episode import Episode
from hippotorch.utils.ann import ApproximateNearestNeighborIndex

if TYPE_CHECKING:  # pragma: no cover - import guarded for type checking only
    from hippotorch.encoders.dual import DualEncoder


class MemoryStore:
    """Lightweight key-value episodic memory."""

    def __init__(
        self,
        embed_dim: int,
        capacity: int = 1024,
        device: torch.device | str = "cpu",
        *,
        indexer: ApproximateNearestNeighborIndex | None = None,
    ) -> None:
        self.embed_dim = embed_dim
        self.capacity = capacity
        self.device = torch.device(device)
        self.keys = torch.empty(0, embed_dim, device=self.device)
        self.episodes: List[Episode | None] = []
        self.key_timestamps: List[int] = []
        # Advanced bookkeeping
        self.access_counts: List[int] = []
        self.insertion_steps: List[int] = []
        self._write_count: int = 0
        # Eviction weights: (reward_rank, access_count, recency)
        self.eviction_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
        self.indexer = indexer

    def to_checkpoint(self) -> Dict[str, object]:
        """Serialize the store for saving to disk or the HuggingFace Hub."""

        episodes = [
            self._get_episode(i).to(torch.device("cpu"))
            for i in range(len(self.episodes))
        ]
        config: Dict[str, object] = {
            "embed_dim": self.embed_dim,
            "capacity": self.capacity,
            "device": str(self.device),
        }
        if isinstance(self, TieredMemoryStore):
            config.update(
                {"cold_dir": str(self.cold_dir), "hot_fraction": self.hot_fraction}
            )

        return {
            "type": self.__class__.__name__,
            "config": config,
            "keys": self.keys.detach().cpu(),
            "episodes": [ep.to_dict() for ep in episodes],
            "key_timestamps": list(self.key_timestamps),
            "access_counts": list(self.access_counts),
            "insertion_steps": list(self.insertion_steps),
            "write_count": self._write_count,
            "eviction_weights": tuple(self.eviction_weights),
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Dict[str, object],
        *,
        device: torch.device | str = "cpu",
    ) -> "MemoryStore":
        """Restore a store from a checkpoint dictionary."""

        target_device = torch.device(device)
        config = payload.get("config", {})
        embed_dim = int(config.get("embed_dim", 0))
        if embed_dim <= 0:
            embed_dim = int(payload["keys"].shape[1])
        capacity = int(config.get("capacity", 0)) or len(payload.get("episodes", []))
        store_type = payload.get("type", cls.__name__)
        store_cls: type[MemoryStore]
        if store_type == "TieredMemoryStore":
            cold_dir = config.get("cold_dir", ".hippotorch_cache")
            hot_fraction = float(config.get("hot_fraction", 0.05))
            store_cls = TieredMemoryStore
            store = store_cls(
                embed_dim,
                capacity=capacity,
                device=target_device,
                cold_dir=cold_dir,
                hot_fraction=hot_fraction,
                indexer=None,
            )
        else:
            store_cls = cls
            store = store_cls(embed_dim, capacity=capacity, device=target_device)
        store.keys = payload["keys"].to(target_device)
        ep_payloads = payload.get("episodes", [])
        store.episodes = [Episode.from_dict(ep).to(target_device) for ep in ep_payloads]
        store.key_timestamps = list(
            payload.get("key_timestamps", [0] * len(store.episodes))
        )
        store.access_counts = list(
            payload.get("access_counts", [0] * len(store.episodes))
        )
        store.insertion_steps = list(
            payload.get("insertion_steps", list(range(1, len(store.episodes) + 1)))
        )
        store._write_count = int(payload.get("write_count", len(store.episodes)))
        store.eviction_weights = tuple(
            payload.get("eviction_weights", store.eviction_weights)
        )
        if isinstance(store, TieredMemoryStore):
            store._episode_paths = [
                store._persist_episode(i, ep) for i, ep in enumerate(store.episodes)
            ]
            store._rebalance_cache()
        store._sync_index()
        return store

    def save(self, path: str | Path) -> None:
        """Persist the store checkpoint to ``path``."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_checkpoint()
        torch.save(payload, target)

    @classmethod
    def load(
        cls, path: str | Path, *, device: torch.device | str = "cpu"
    ) -> "MemoryStore":
        """Load a store checkpoint saved with :meth:`save`."""

        payload = torch.load(Path(path), map_location="cpu")
        return cls.from_checkpoint(payload, device=device)

    def __len__(self) -> int:
        return len(self.episodes)

    def write(
        self, keys: Tensor, episodes: Sequence[Episode], encoder_step: int | None = None
    ) -> None:
        if keys.shape[0] != len(episodes):
            raise ValueError("keys and episodes must align")
        if keys.shape[1] != self.embed_dim:
            raise ValueError("invalid key dimension")

        keys = keys.detach().to(self.device)
        self.keys = torch.cat([self.keys, keys], dim=0)
        self.episodes.extend(episodes)

        step_value = 0 if encoder_step is None else int(encoder_step)
        self.key_timestamps.extend([step_value] * len(episodes))
        # initialize access/recency tracking
        for _ in range(len(episodes)):
            self.access_counts.append(0)
            self._write_count += 1
            self.insertion_steps.append(self._write_count)

        self._enforce_capacity()
        self._sync_index()

    def prune_indices(self, indices: Sequence[int]) -> None:
        """Remove specific episodes and associated keys.

        Args:
            indices: Collection of indices to drop. Out-of-range entries are
                ignored to keep the method robust to stale callers.
        """

        if not indices or len(self.episodes) == 0:
            return

        mask = torch.ones(len(self.episodes), dtype=torch.bool)
        for idx in indices:
            if 0 <= idx < len(self.episodes):
                mask[idx] = False

        if mask.all():
            return

        self._prune_with_mask(mask)

    def _enforce_capacity(self) -> None:
        while len(self.episodes) > self.capacity:
            idx = self._select_evict_index()
            # remove row idx from keys and associated metadata
            if self.keys.shape[0] > 0:
                if idx == 0:
                    self.keys = self.keys[1:]
                elif idx == self.keys.shape[0] - 1:
                    self.keys = self.keys[:-1]
                else:
                    self.keys = torch.cat(
                        [self.keys[:idx], self.keys[idx + 1 :]], dim=0
                    )
            self._drop_episode(idx)
            self.key_timestamps.pop(idx)
            if self.access_counts:
                self.access_counts.pop(idx)
            if self.insertion_steps:
                self.insertion_steps.pop(idx)
        self._sync_index()

    def _select_evict_index(self) -> int:
        """Select index to evict using weighted importance (lowest score evicted)."""
        n = len(self.episodes)
        if n <= 1:
            return 0
        w1, w2, w3 = self.eviction_weights
        rewards = torch.tensor(
            [self._get_episode(ep_idx).total_reward for ep_idx in range(n)],
            dtype=torch.float,
        )
        order = torch.argsort(rewards)  # low to high
        ranks = torch.empty(n, dtype=torch.float)
        ranks[order] = torch.arange(n, dtype=torch.float)
        reward_rank = (ranks / max(1, n - 1)).tolist()
        max_acc = max(1, max(self.access_counts) if self.access_counts else 1)
        acc_norm = (
            [c / max_acc for c in self.access_counts]
            if self.access_counts
            else [0.0] * n
        )
        now = max(1, self._write_count)
        ages = [max(0, now - step) for step in (self.insertion_steps or [0] * n)]
        max_age = max(1, max(ages))
        recency = [1.0 - (age / max_age) for age in ages]
        scores = [
            w1 * r + w2 * a + w3 * rc
            for r, a, rc in zip(reward_rank, acc_norm, recency)
        ]
        return int(torch.tensor(scores).argmin().item())

    @torch.no_grad()
    def retrieve(
        self,
        queries: Tensor,
        top_k: int = 5,
        *,
        encoder: "DualEncoder" | None = None,
        current_step: int | None = None,
        lazy_refresh: bool = False,
    ) -> Tuple[Tensor, List[List[Episode]]]:
        if len(self.episodes) == 0:
            return torch.empty(queries.size(0), 0, device=queries.device), [
                [] for _ in range(queries.size(0))
            ]

        queries = F.normalize(queries, dim=-1)
        if self.indexer is not None:
            top_scores, indices = self.indexer.search(queries, top_k)
        else:
            keys = F.normalize(self.keys, dim=-1)
            scores = queries @ keys.t()
            top_scores, indices = scores.topk(min(top_k, len(self.episodes)), dim=-1)
        batch_episodes: List[List[Episode]] = []
        refresh_needed = False
        for row in indices:
            idxs = row.tolist()
            # access count update
            for i in idxs:
                if 0 <= i < len(self.access_counts):
                    self.access_counts[i] += 1
            # lazy key refresh for retrieved indices only
            if lazy_refresh and encoder is not None:
                ref_step = 0 if current_step is None else int(current_step)
                enc_dev = next(encoder.parameters()).device
                for i in idxs:
                    if 0 <= i < len(self.key_timestamps):
                        age = ref_step - int(self.key_timestamps[i])
                        if age >= max(1, int(encoder.refresh_interval)):
                            ep = self._get_episode(i)
                            ep_tensor = (
                                self._episode_to_tensor(ep).unsqueeze(0).to(enc_dev)
                            )
                            new_key = (
                                encoder.encode_key(ep_tensor).detach().to(self.device)
                            )
                            self.keys[i : i + 1] = new_key
                            self.key_timestamps[i] = ref_step
                            refresh_needed = True
            batch_episodes.append([self._get_episode(i) for i in idxs])
        if refresh_needed:
            self._sync_index()
        return top_scores, batch_episodes

    def get_stale_keys(self, current_step: int, threshold: int) -> List[int]:
        """Return indices of keys that exceed the staleness threshold."""

        if threshold <= 0:
            return []

        stale_indices = []
        for idx, timestamp in enumerate(self.key_timestamps):
            if current_step - timestamp >= threshold:
                stale_indices.append(idx)
        return stale_indices

    @torch.no_grad()
    def refresh_keys(
        self,
        indices: Sequence[int],
        encoder: "DualEncoder",
        encoder_step: int | None = None,
    ) -> None:
        """Re-encode stored episodes to refresh stale keys."""

        if not indices:
            return

        encoder_device = next(encoder.parameters()).device
        for idx in indices:
            if idx < 0 or idx >= len(self.episodes):
                continue
            episode = self._get_episode(idx)
            episode_tensor = (
                self._episode_to_tensor(episode).unsqueeze(0).to(encoder_device)
            )
            new_key = encoder.encode_key(episode_tensor).detach().to(self.device)
            self.keys[idx : idx + 1] = new_key
            if encoder_step is not None:
                self.key_timestamps[idx] = encoder_step
        self._sync_index()

    def sample_uniform(self, batch_size: int) -> List[Episode]:
        if len(self.episodes) == 0:
            return []
        indices = torch.randint(0, len(self.episodes), (batch_size,))
        return [self._get_episode(i) for i in indices.tolist()]

    def sample_contrastive_batch(
        self,
        batch_size: int,
        window_size: int = 2,
        reward_weighted: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        valid: List[Tuple[int, Episode]] = [
            (idx, self._get_episode(idx))
            for idx in range(len(self.episodes))
            if len(self._get_episode(idx)) >= 2 * window_size
        ]
        if not valid:
            raise ValueError("No episodes available for contrastive sampling")

        if reward_weighted:
            rewards = torch.tensor(
                [ep.total_reward for _, ep in valid], dtype=torch.float
            )
            denom = rewards.std(unbiased=False).clamp(min=0.1)
            probs = F.softmax(rewards / denom, dim=0)
            indices = torch.multinomial(probs, batch_size, replacement=True)
        else:
            indices = torch.randperm(len(valid))[:batch_size]

        anchors, positives, weights = [], [], []
        for idx in indices:
            _, episode = valid[idx.item()]
            max_start = len(episode) - 2 * window_size
            anchor_start = torch.randint(0, max_start + 1, (1,)).item()
            positive_start = anchor_start + window_size
            anchor_window = episode.slice(anchor_start, anchor_start + window_size)
            positive_window = episode.slice(
                positive_start, positive_start + window_size
            )
            anchors.append(self._episode_to_tensor(anchor_window))
            positives.append(self._episode_to_tensor(positive_window))
            weights.append(episode.total_reward)

        anchors_t = torch.stack(anchors)
        positives_t = torch.stack(positives)
        weight_t = torch.tensor(weights)
        weight_t = (weight_t - weight_t.mean()) / (weight_t.std() + 1e-8)
        weight_t = torch.sigmoid(weight_t)
        return anchors_t, positives_t, weight_t

    def staleness_stats(self, current_step: int | None = None) -> Dict[str, float]:
        """Return summary statistics about key staleness.

        Args:
            current_step: Optional reference step (e.g., encoder step). If not
                provided, the most recent timestamp in the store is used.
        """

        if not self.key_timestamps:
            return {"avg_staleness": 0.0, "max_staleness": 0.0, "min_staleness": 0.0}

        reference = (
            self.key_timestamps[-1] if current_step is None else int(current_step)
        )
        ages = [max(0, reference - ts) for ts in self.key_timestamps]

        return {
            "avg_staleness": float(sum(ages) / len(ages)),
            "max_staleness": float(max(ages)),
            "min_staleness": float(min(ages)),
        }

    def episodes_to_tensor(self, episodes: Sequence[Episode]) -> Tensor:
        """Batch episodes into a tensor for encoding.

        The tensor layout concatenates states, actions, and rewards to match the
        expected encoder input shape. Episodes are zero-padded to the longest
        length in the batch to allow stacking.
        """

        base_tensors = [self._episode_to_tensor(ep) for ep in episodes]
        if not base_tensors:
            raise ValueError("No episodes provided for batching")

        max_len = max(t.shape[0] for t in base_tensors)
        feat_dim = base_tensors[0].shape[-1]
        padded: list[Tensor] = []
        for tensor in base_tensors:
            if tensor.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - tensor.shape[0],
                    feat_dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                tensor = torch.cat([tensor, pad], dim=0)
            padded.append(tensor)

        return torch.stack(padded)

    def _episode_to_tensor(self, episode: Episode) -> Tensor:
        return torch.cat(
            [
                episode.states,
                episode.actions,
                episode.rewards.unsqueeze(-1),
            ],
            dim=-1,
        )

    def _get_episode(self, index: int) -> Episode:
        return self.episodes[index]

    def _drop_episode(self, index: int) -> None:
        self.episodes.pop(index)

    def _sync_index(self) -> None:
        if self.indexer is not None:
            self.indexer.rebuild(self.keys.detach().cpu())

    def _prune_with_mask(self, mask: torch.Tensor) -> None:
        """Apply a boolean mask to prune all store data structures."""

        if mask.numel() != len(self.episodes):
            raise ValueError("Prune mask must align with stored episodes")

        keep_list = mask.tolist()
        if self.keys.numel():
            self.keys = self.keys[mask]
        self.episodes = [ep for ep, keep in zip(self.episodes, keep_list) if keep]
        self.key_timestamps = [
            ts for ts, keep in zip(self.key_timestamps, keep_list) if keep
        ]
        self.access_counts = [
            ac for ac, keep in zip(self.access_counts, keep_list) if keep
        ]
        self.insertion_steps = [
            step for step, keep in zip(self.insertion_steps, keep_list) if keep
        ]

        # TieredMemoryStore already defined above; stub removed to avoid redefinition
        self._sync_index()

    def clone(self, device: torch.device | str | None = None) -> "MemoryStore":
        """Deep copy the store for background processing."""

        new_device = self.device if device is None else torch.device(device)
        clone_store = MemoryStore(
            self.embed_dim,
            capacity=self.capacity,
            device=new_device,
            indexer=None,
        )
        clone_store.keys = self.keys.detach().to(new_device).clone()
        clone_store.episodes = copy.deepcopy(self.episodes)
        clone_store.key_timestamps = list(self.key_timestamps)
        clone_store.access_counts = list(self.access_counts)
        clone_store.insertion_steps = list(self.insertion_steps)
        clone_store._write_count = self._write_count
        clone_store.eviction_weights = self.eviction_weights
        return clone_store


class TieredMemoryStore(MemoryStore):
    """Memory store with hot/cold tiers using disk offload for episodes.

    Keys remain in device memory for fast similarity search, while older
    episodes are serialized to disk and loaded on demand. The most recent
    ``hot_fraction`` of episodes stay resident to avoid slowing the training
    loop.
    """

    def __init__(
        self,
        embed_dim: int,
        capacity: int = 1024,
        *,
        device: torch.device | str = "cpu",
        cold_dir: str | Path = ".hippotorch_cache",
        hot_fraction: float = 0.05,
        indexer: ApproximateNearestNeighborIndex | None = None,
    ) -> None:
        super().__init__(embed_dim, capacity, device=device, indexer=indexer)
        self.cold_dir = Path(cold_dir)
        self.cold_dir.mkdir(parents=True, exist_ok=True)
        self.hot_fraction = max(0.0, min(1.0, hot_fraction))
        self._episode_paths: List[Path] = []
        self._file_counter = 0

    def write(
        self, keys: Tensor, episodes: Sequence[Episode], encoder_step: int | None = None
    ) -> None:
        super().write(keys, episodes, encoder_step=encoder_step)
        start_idx = max(0, len(self.episodes) - len(episodes))
        for offset, episode in enumerate(episodes):
            idx = start_idx + offset
            path = self._persist_episode(idx, episode)
            if len(self._episode_paths) <= idx:
                self._episode_paths.append(path)
            else:
                self._episode_paths[idx] = path
        self._rebalance_cache()

    def _drop_episode(self, index: int) -> None:
        if 0 <= index < len(self._episode_paths):
            path = self._episode_paths.pop(index)
            try:
                path.unlink()
            except FileNotFoundError:  # pragma: no cover - best effort cleanup
                pass
        super()._drop_episode(index)

    def _enforce_capacity(self) -> None:
        while len(self.episodes) > self.capacity:
            idx = self._select_evict_index()
            if self.keys.shape[0] > 0:
                if idx == 0:
                    self.keys = self.keys[1:]
                elif idx == self.keys.shape[0] - 1:
                    self.keys = self.keys[:-1]
                else:
                    self.keys = torch.cat(
                        [self.keys[:idx], self.keys[idx + 1 :]], dim=0
                    )
            self._drop_episode(idx)
            self.key_timestamps.pop(idx)
            if self.access_counts:
                self.access_counts.pop(idx)
            if self.insertion_steps:
                self.insertion_steps.pop(idx)
        self._sync_index()

    def _get_episode(self, index: int) -> Episode:
        episode = self.episodes[index]
        if episode is None:
            episode = self._load_episode(index)
            if self._is_hot(index):
                self.episodes[index] = episode
        return episode

    def _persist_episode(self, index: int, episode: Episode) -> Path:
        self._file_counter += 1
        path = self.cold_dir / f"episode_{self._file_counter:08d}.pt"
        torch.save(episode, path)
        return path

    def _load_episode(self, index: int) -> Episode:
        path = self._episode_paths[index]
        return torch.load(path)

    def _rebalance_cache(self) -> None:
        hot_count = max(1, int(self.capacity * self.hot_fraction))
        hot_start = max(0, len(self.episodes) - hot_count)
        for idx in range(len(self.episodes)):
            if idx < hot_start and self.episodes[idx] is not None:
                self.episodes[idx] = None
            elif idx >= hot_start and self.episodes[idx] is None:
                self.episodes[idx] = self._load_episode(idx)

    def _is_hot(self, index: int) -> bool:
        hot_count = max(1, int(self.capacity * self.hot_fraction))
        hot_start = max(0, len(self.episodes) - hot_count)
        return index >= hot_start

    def prune_indices(self, indices: Sequence[int]) -> None:
        if not indices or len(self.episodes) == 0:
            return

        mask = torch.ones(len(self.episodes), dtype=torch.bool)
        for idx in indices:
            if 0 <= idx < len(self.episodes):
                mask[idx] = False
                if idx < len(self._episode_paths):
                    try:
                        self._episode_paths[idx].unlink()
                    except FileNotFoundError:  # pragma: no cover - best effort cleanup
                        pass

        if mask.all():
            return

        keep_list = mask.tolist()
        self._episode_paths = [
            p for p, keep in zip(self._episode_paths, keep_list) if keep
        ]
        self._prune_with_mask(mask)
        self._rebalance_cache()

    def clone(self, device: torch.device | str | None = None) -> "TieredMemoryStore":
        new_device = self.device if device is None else torch.device(device)
        clone_store = TieredMemoryStore(
            self.embed_dim,
            capacity=self.capacity,
            device=new_device,
            cold_dir=self.cold_dir,
            hot_fraction=self.hot_fraction,
            indexer=None,
        )
        clone_store.keys = self.keys.detach().to(new_device).clone()
        clone_store.episodes = copy.deepcopy(self.episodes)
        clone_store._episode_paths = list(self._episode_paths)
        clone_store.key_timestamps = list(self.key_timestamps)
        clone_store.access_counts = list(self.access_counts)
        clone_store.insertion_steps = list(self.insertion_steps)
        clone_store._write_count = self._write_count
        clone_store.eviction_weights = self.eviction_weights
        return clone_store


__all__ = ["MemoryStore", "TieredMemoryStore"]

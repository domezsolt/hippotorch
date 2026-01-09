from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from hippotorch.encoders.dual import DualEncoder
from hippotorch.memory.store import MemoryStore


class Consolidator(torch.nn.Module):
    """Reward-aware contrastive learner for episodic memory."""

    def __init__(
        self,
        encoder: DualEncoder,
        temperature: float = 0.05,
        reward_weight: float = 0.8,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.reward_weight = reward_weight
        self.optimizer = torch.optim.Adam(
            self.encoder.online.parameters(), lr=learning_rate
        )
        self.loss_history: list[float] = []
        self.total_steps = 0

    def sleep(
        self,
        memory: MemoryStore,
        steps: int = 1,
        batch_size: int = 32,
        refresh_keys: bool = True,
        report_quality: bool = False,
        reward_weighted: bool = True,
        ablation_temporal_only: bool = False,
        *,
        tb_writer=None,
        tb_log_every: int = 0,
    ) -> Dict[str, float]:
        """Run contrastive consolidation against stored episodes."""
        if len(memory) == 0:
            return {
                "loss": 0.0,
                "temporal_loss": 0.0,
                "reward_loss": 0.0,
                "keys_refreshed": 0,
            }

        total_loss = 0.0
        total_temporal = 0.0
        total_reward = 0.0
        ablation_losses: list[float] = []

        stale_indices: list[int] = []

        for _ in range(steps):
            anchors, positives, weights = memory.sample_contrastive_batch(
                batch_size=batch_size, reward_weighted=reward_weighted
            )

            anchor_embeds = self.encoder.encode_query(anchors)
            positive_embeds = self.encoder.encode_query(positives)

            temporal_loss = self._info_nce(anchor_embeds, positive_embeds)
            reward_loss = (
                self._reward_weighted_info_nce(anchor_embeds, positive_embeds, weights)
                if reward_weighted
                else torch.zeros_like(temporal_loss)
            )

            effective_reward_weight = self.reward_weight if reward_weighted else 0.0
            loss = (
                1 - effective_reward_weight
            ) * temporal_loss + effective_reward_weight * reward_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.encoder.update_target()

            total_loss += loss.item()
            total_temporal += temporal_loss.item()
            total_reward += reward_loss.item()
            self.total_steps += 1

            if ablation_temporal_only:
                ablation_losses.append(
                    float(
                        self._temporal_only_baseline(
                            memory, batch_size=batch_size
                        ).item()
                    )
                )

            # optional TensorBoard logging
            if (
                tb_writer is not None
                and tb_log_every
                and (self.total_steps % max(1, tb_log_every) == 0)
            ):
                try:
                    from hippotorch.utils.diagnostics import log_memory_pca

                    log_memory_pca(
                        tb_writer, memory, step=self.total_steps, sample_size=100
                    )
                except Exception:  # pragma: no cover - best effort
                    pass

        if refresh_keys and self.encoder.should_refresh_keys():
            stale_indices = memory.get_stale_keys(
                current_step=self.total_steps, threshold=self.encoder.refresh_interval
            )
            if stale_indices:
                memory.refresh_keys(
                    stale_indices, self.encoder, encoder_step=self.total_steps
                )
            self.encoder.mark_refreshed()

        avg_loss = total_loss / steps
        self.loss_history.append(avg_loss)

        metrics: Dict[str, float] = {
            "loss": avg_loss,
            "temporal_loss": total_temporal / steps,
            "reward_loss": total_reward / steps,
            "steps": self.total_steps,
            "keys_refreshed": len(stale_indices),
        }

        if ablation_losses:
            metrics["ablation_temporal_loss"] = sum(ablation_losses) / len(
                ablation_losses
            )

        if report_quality:
            metrics.update(self.get_representation_quality(memory))

        return metrics

    @torch.no_grad()
    def get_representation_quality(
        self,
        memory: MemoryStore,
        *,
        sample_size: int = 64,
        top_k: int = 5,
    ) -> Dict[str, float]:
        """Diagnostic metrics describing retrieval quality and staleness."""

        if len(memory) == 0:
            return {
                "retrieval_sharpness": 0.0,
                "reward_alignment": 0.0,
                "key_dispersion": 0.0,
                "avg_staleness": 0.0,
            }

        device = next(self.encoder.parameters()).device
        num_samples = min(sample_size, len(memory))
        indices = torch.randperm(len(memory))[:num_samples]
        episodes = [memory.episodes[i] for i in indices.tolist()]
        episode_tensors = memory.episodes_to_tensor(episodes).to(device)

        query_keys = F.normalize(self.encoder.encode_key(episode_tensors), dim=-1)
        stored_keys = F.normalize(memory.keys.to(device), dim=-1)

        similarities = query_keys @ stored_keys.t()
        topk_vals = similarities.topk(min(top_k, similarities.size(-1)), dim=-1).values
        retrieval_sharpness = topk_vals.mean().item()

        rewards = torch.tensor([ep.total_reward for ep in episodes], device=device)
        top1 = topk_vals[:, 0]
        reward_alignment = self._pearson(top1, rewards)

        key_norms = memory.keys.norm(dim=-1)
        key_dispersion = key_norms.std().item() if key_norms.numel() else 0.0

        timestamps = torch.tensor(
            memory.key_timestamps, device=device, dtype=torch.float
        )
        avg_staleness = (
            (float(self.total_steps) - timestamps).clamp(min=0).mean().item()
            if timestamps.numel()
            else 0.0
        )

        return {
            "retrieval_sharpness": retrieval_sharpness,
            "reward_alignment": reward_alignment,
            "key_dispersion": key_dispersion,
            "avg_staleness": avg_staleness,
        }

    def _info_nce(self, anchors: Tensor, positives: Tensor) -> Tensor:
        logits = anchors @ positives.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def _reward_weighted_info_nce(
        self, anchors: Tensor, positives: Tensor, weights: Tensor
    ) -> Tensor:
        """Rank-based weighting for robustness to scale and sign of rewards."""
        logits = anchors @ positives.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        per_sample = F.cross_entropy(logits, labels, reduction="none")
        # ranks in [0, B-1]
        if weights.numel() <= 1:
            norm_w = torch.ones_like(per_sample)
        else:
            order = torch.argsort(weights)
            ranks = torch.empty_like(weights, dtype=torch.float)
            ranks[order] = torch.arange(
                weights.numel(), dtype=torch.float, device=weights.device
            )
            norm_w = 0.1 + 0.9 * (ranks / max(1.0, weights.numel() - 1.0))
        norm_w = norm_w.to(per_sample.device)
        return (per_sample * norm_w).mean()

    @torch.no_grad()
    def _temporal_only_baseline(
        self, memory: MemoryStore, *, batch_size: int
    ) -> Tensor:
        """Compute a temporal-only InfoNCE loss for ablation diagnostics."""

        anchors, positives, _ = memory.sample_contrastive_batch(
            batch_size=batch_size, reward_weighted=False
        )
        anchor_embeds = self.encoder.encode_key(anchors)
        positive_embeds = self.encoder.encode_key(positives)
        return self._info_nce(anchor_embeds, positive_embeds)

    def _pearson(self, x: Tensor, y: Tensor) -> float:
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = x_centered.norm() * y_centered.norm() + 1e-8
        return ((x_centered * y_centered).sum() / denom).item()


__all__ = ["Consolidator"]

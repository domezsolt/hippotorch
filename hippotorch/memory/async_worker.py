from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Dict, Optional

import torch

from hippotorch.memory.consolidator import Consolidator
from hippotorch.memory.store import MemoryStore


class AsyncConsolidationWorker:
    """Run consolidation in a background process to avoid blocking training."""

    def __init__(
        self,
        consolidator: Consolidator,
        memory: MemoryStore,
        *,
        start_method: str = "spawn",
        device: torch.device | str | None = None,
    ) -> None:
        self.consolidator = consolidator
        self.memory = memory
        self.ctx = mp.get_context(start_method)
        self._requests: mp.Queue = self.ctx.Queue()
        self._results: mp.Queue = self.ctx.Queue()
        self._process: mp.Process | None = None
        self._device = (
            torch.device(device)
            if device is not None
            else next(consolidator.parameters()).device
        )

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        worker_consolidator = self.consolidator.cpu()
        self._process = self.ctx.Process(
            target=_worker_loop,
            args=(worker_consolidator, self._requests, self._results),
            daemon=True,
        )
        self._process.start()
        self.consolidator.to(self._device)

    def submit(
        self,
        *,
        steps: int = 1,
        batch_size: int = 32,
        refresh_keys: bool = True,
        report_quality: bool = False,
        reward_weighted: bool = True,
    ) -> None:
        """Queue a consolidation job using a CPU snapshot of the memory store."""

        self.start()
        snapshot = self.memory.clone(device="cpu")
        self._requests.put(
            {
                "memory": snapshot,
                "steps": steps,
                "batch_size": batch_size,
                "refresh_keys": refresh_keys,
                "report_quality": report_quality,
                "reward_weighted": reward_weighted,
            }
        )

    def poll(
        self, *, apply: bool = True, timeout: float = 0.0
    ) -> Optional[Dict[str, float]]:
        """Return metrics if a background job completed."""

        try:
            result = self._results.get(timeout=timeout)
        except queue.Empty:
            return None

        if apply:
            self._apply_result(result)
        return result.get("metrics")

    def shutdown(self) -> None:
        if self._process is None:
            return
        self._requests.put({"stop": True})
        self._process.join(timeout=1.0)
        if self._process.is_alive():  # pragma: no cover - safety net
            self._process.terminate()
        self._process = None

    def _apply_result(self, payload: Dict) -> None:
        encoder_state = payload.get("encoder_state")
        if encoder_state:
            self.consolidator.encoder.load_state_dict(encoder_state)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            self.consolidator.optimizer.load_state_dict(optimizer_state)
        self.consolidator.total_steps = int(
            payload.get("steps", self.consolidator.total_steps)
        )

        key_tensor: torch.Tensor | None = payload.get("keys")
        if key_tensor is not None:
            self.memory.keys = key_tensor.to(self.memory.device)
            self.memory.key_timestamps = payload.get(
                "key_timestamps", self.memory.key_timestamps
            )
            if hasattr(self.memory, "_sync_index"):
                self.memory._sync_index()  # type: ignore[attr-defined]

        self.consolidator.to(self._device)


def _worker_loop(
    consolidator: Consolidator, request_q: mp.Queue, result_q: mp.Queue
) -> None:
    consolidator = consolidator.cpu()
    while True:
        task = request_q.get()
        if isinstance(task, dict) and task.get("stop"):
            break
        memory: MemoryStore = task["memory"]
        metrics = consolidator.sleep(
            memory,
            steps=task.get("steps", 1),
            batch_size=task.get("batch_size", 32),
            refresh_keys=task.get("refresh_keys", True),
            report_quality=task.get("report_quality", False),
            reward_weighted=task.get("reward_weighted", True),
        )
        payload = {
            "metrics": metrics,
            "encoder_state": consolidator.encoder.state_dict(),
            "optimizer_state": consolidator.optimizer.state_dict(),
            "steps": consolidator.total_steps,
            "keys": memory.keys.detach(),
            "key_timestamps": list(memory.key_timestamps),
        }
        result_q.put(payload)


__all__ = ["AsyncConsolidationWorker"]

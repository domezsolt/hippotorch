from __future__ import annotations

"""Lightweight hub portability stubs.

The test suite monkeypatches these functions, so default implementations can
be minimal. Real integrations can target Hugging Face Hub.
"""

from typing import Any


def push_memory_to_hub(
    memory, repo_id: str, **kwargs: Any
):  # pragma: no cover - delegated in tests
    return {"repo_id": repo_id, "size": len(memory)}


def load_memory_from_hub(
    repo_id: str, *, device: str | None = None, **kwargs: Any
):  # pragma: no cover
    # Default: return a simple placeholder dict; tests monkeypatch this
    return {"repo_id": repo_id, "device": device}


__all__ = ["push_memory_to_hub", "load_memory_from_hub"]

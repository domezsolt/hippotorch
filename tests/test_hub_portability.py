import pytest

from hippotorch.encoders.dual import DualEncoder
from hippotorch.memory.replay import HippocampalReplayBuffer
from hippotorch.memory.store import MemoryStore


def _make_buffer(embed_dim: int = 8) -> HippocampalReplayBuffer:
    memory = MemoryStore(embed_dim=embed_dim, capacity=8)
    encoder = DualEncoder(input_dim=3, embed_dim=embed_dim)
    return HippocampalReplayBuffer(memory=memory, encoder=encoder)


def test_save_to_hub_delegates(monkeypatch, tmp_path):
    buffer = _make_buffer()
    called = {}

    def fake_push(memory, repo_id, **kwargs):  # pragma: no cover - delegated call
        called["memory"] = memory
        called["repo_id"] = repo_id
        called["kwargs"] = kwargs
        return tmp_path

    monkeypatch.setattr("hippotorch.utils.hub.push_memory_to_hub", fake_push)

    result = buffer.save_to_hub("user/test-repo", filename="memory.pt", private=True)

    assert result == tmp_path
    assert called["memory"] is buffer.memory
    assert called["repo_id"] == "user/test-repo"
    assert called["kwargs"]["filename"] == "memory.pt"
    assert called["kwargs"]["private"] is True


def test_load_memory_from_hub_replaces_store(monkeypatch):
    buffer = _make_buffer()
    loaded_memory = MemoryStore(
        embed_dim=buffer.memory.embed_dim, capacity=2, device=buffer.memory.device
    )

    monkeypatch.setattr(
        "hippotorch.utils.hub.load_memory_from_hub", lambda *_, **__: loaded_memory
    )

    result = buffer.load_memory_from_hub("user/test-repo", device=buffer.memory.device)

    assert result is loaded_memory
    assert buffer.memory is loaded_memory


def test_load_memory_from_hub_validates_embed_dim(monkeypatch):
    buffer = _make_buffer(embed_dim=4)
    incompatible_memory = MemoryStore(embed_dim=6)

    monkeypatch.setattr(
        "hippotorch.utils.hub.load_memory_from_hub",
        lambda *_, **__: incompatible_memory,
    )

    with pytest.raises(ValueError):
        buffer.load_memory_from_hub("user/test-repo")

"""Tests for ModelManager LoRA registry logic (stubbed pipeline, no GPU)."""

from __future__ import annotations

from sdxl_runner.params import GenerationParams, LoraSpec
from sdxl_runner.server.pipeline import ModelManager


class StubPipe:
    """Records LoRA calls instead of touching diffusers."""

    def __init__(self) -> None:
        self.loaded: list[tuple] = []
        self.adapters: list[tuple] = []
        self.unloaded = 0

    def load_lora_weights(self, source, weight_name=None, adapter_name=None):
        self.loaded.append((source, weight_name, adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.adapters.append((list(names), list(adapter_weights or [])))

    def unload_lora_weights(self):
        self.unloaded += 1


def _manager_with_stub() -> tuple[ModelManager, StubPipe]:
    manager = ModelManager(mock=False)
    stub = StubPipe()
    manager._txt2img = stub  # type: ignore[assignment]
    return manager, stub


def test_load_lora_returns_alias():
    manager, _ = _manager_with_stub()
    alias = manager.load_lora("org/repo", "style.safetensors", 0.8)
    assert alias == "org/repo::style.safetensors"
    assert manager.list_loras()["registered"] == [alias]


def test_sync_loras_loads_and_activates():
    manager, stub = _manager_with_stub()
    specs = [
        LoraSpec("org/a", "a.safetensors", 0.7),
        LoraSpec("org/b", None, 1.0),
    ]
    manager._sync_loras(specs)

    assert [c[0] for c in stub.loaded] == ["org/a", "org/b"]
    assert stub.adapters[-1][0] == ["org/a::a.safetensors", "org/b"]
    assert stub.adapters[-1][1] == [0.7, 1.0]
    assert manager.list_loras()["active"] == ["org/a::a.safetensors", "org/b"]


def test_sync_loras_noop_when_unchanged():
    manager, stub = _manager_with_stub()
    specs = [LoraSpec("org/a", "a.safetensors", 0.7)]
    manager._sync_loras(specs)
    loads_before = len(stub.loaded)
    manager._sync_loras(specs)  # identical request
    assert len(stub.loaded) == loads_before  # not reloaded


def test_sync_loras_clears_when_empty():
    manager, stub = _manager_with_stub()
    manager._sync_loras([LoraSpec("org/a", "a.safetensors", 1.0)])
    manager._sync_loras([])
    assert stub.unloaded >= 1
    assert manager.list_loras()["active"] == []


def test_mock_generate_returns_placeholders():
    manager = ModelManager(mock=True)
    images = manager.generate(GenerationParams(prompt="x", num_images=3))
    assert len(images) == 3
    assert manager.last_seed is not None

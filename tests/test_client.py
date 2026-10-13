"""Tests for SdxlClient argument building, with gradio_client monkeypatched."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from sdxl_runner.client import SdxlClient, _parse_loras, _split_result
from sdxl_runner.params import GenerationParams


class FakeGradioClient:
    """Records predict calls and returns canned (gallery, status) tuples."""

    instances: ClassVar[list[FakeGradioClient]] = []

    def __init__(self, src, **kwargs):
        self.src = src
        self.kwargs = kwargs
        self.calls: list[tuple] = []
        self._gallery: list[str] = []
        FakeGradioClient.instances.append(self)

    def predict(self, *args, api_name=None, **kwargs):
        self.calls.append((args, api_name))
        if api_name == "/loras":
            return json.dumps({"registered": ["a"], "active": ["a"]})
        # Simulate gradio_client downloading two files to download_files dir.
        out_dir = Path(self.kwargs["download_files"])
        files = []
        for i in range(2):
            path = out_dir / f"tmp{i}.png"
            path.write_bytes(b"png")
            files.append({"path": str(path), "name": f"tmp{i}.png"})
        status = json.dumps(
            {"seed": 123, "width": 832, "height": 1216, "num_images": 2}
        )
        return (files, status)


@pytest.fixture(autouse=True)
def patch_gradio(monkeypatch):
    FakeGradioClient.instances.clear()
    fake_module = types.ModuleType("gradio_client")
    fake_module.Client = FakeGradioClient
    fake_module.handle_file = lambda path: {"path": str(path), "meta": "file"}
    monkeypatch.setitem(sys.modules, "gradio_client", fake_module)


def test_txt2img_builds_params_json(tmp_path):
    client = SdxlClient("http://example.test", download_dir=tmp_path)
    params = GenerationParams(prompt="1girl", width=832, height=1216, seed=7)
    result = client.txt2img(params)

    (args, api_name) = client._client.calls[0]
    assert api_name == "/txt2img"
    assert json.loads(args[0])["prompt"] == "1girl"
    assert json.loads(args[0])["seed"] == 7

    assert result.seed == 123
    assert len(result.images) == 2
    assert all(p.exists() for p in result.images)
    assert all("832x1216" in p.name and "seed123" in p.name for p in result.images)


def test_img2img_requires_init_image(tmp_path):
    client = SdxlClient("http://example.test", download_dir=tmp_path)
    with pytest.raises(ValueError):
        client.img2img(GenerationParams(prompt="x"))


def test_img2img_sends_init_image(tmp_path):
    client = SdxlClient("http://example.test", download_dir=tmp_path)
    init = tmp_path / "init.png"
    init.write_bytes(b"png")
    params = GenerationParams(prompt="x", init_image=str(init))
    result = client.img2img(params)

    args, api_name = client._client.calls[0]
    assert api_name == "/img2img"
    assert len(args) == 2  # params_json + handle_file wrapper
    assert args[1]["path"] == str(init)
    assert len(result.images) == 2


def test_iter_paths_handles_gallery_dicts():
    from sdxl_runner.client import _iter_paths

    gallery = [{"image": {"path": "/a.png"}, "caption": None}, {"path": "/b.png"}]
    assert _iter_paths(gallery) == ["/a.png", "/b.png"]


def test_generate_quality_and_auto_size(tmp_path):
    client = SdxlClient("http://example.test", download_dir=tmp_path)
    client.generate(
        prompt="1girl", width=800, height=1200, quality_tags=True, auto_size=True
    )
    payload = json.loads(client._client.calls[0][0][0])
    assert (payload["width"], payload["height"]) in [
        (832, 1216),
        (768, 1344),
        (896, 1152),
    ]
    assert "masterpiece" in payload["prompt"]


def test_loras_endpoint(tmp_path):
    client = SdxlClient("http://example.test", download_dir=tmp_path)
    assert client.loras() == ["a"]


def test_auth_forwarded(tmp_path):
    client = SdxlClient("http://example.test", download_dir=tmp_path, auth=("u", "p"))
    assert client._client.kwargs["auth"] == ("u", "p")


def test_split_result_tolerates_bare_gallery():
    gallery, status = _split_result(["a.png", "b.png"])
    assert gallery == ["a.png", "b.png"]
    assert status == {}


def test_parse_loras_dict():
    assert _parse_loras({"unet": ["x", "y"], "text_encoder": ["x"]}) == ["x", "y"]

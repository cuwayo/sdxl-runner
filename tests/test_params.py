"""Tests for presets and GenerationParams (no GPU, no network)."""

from __future__ import annotations

import pytest

from sdxl_runner.params import GenerationParams, LoraSpec
from sdxl_runner.presets import (
    DEFAULT_NEGATIVE_PROMPT,
    QUALITY_TAGS,
    RESOLUTIONS,
    snap_resolution,
    with_quality_tags,
)


def test_snap_exact_bucket():
    for bucket in RESOLUTIONS:
        assert snap_resolution(*bucket) == bucket


def test_snap_preserves_orientation():
    # Portrait-ish input stays portrait, landscape stays landscape.
    w, h = snap_resolution(800, 1200)
    assert h > w
    w, h = snap_resolution(1200, 800)
    assert w > h


def test_snap_near_square():
    assert snap_resolution(1000, 1000) == (1024, 1024)


def test_snap_rejects_nonpositive():
    with pytest.raises(ValueError):
        snap_resolution(0, 100)


def test_with_quality_tags_appends():
    result = with_quality_tags("1girl, smile")
    assert result.startswith("1girl, smile")
    for tag in QUALITY_TAGS:
        assert tag in result


def test_with_quality_tags_idempotent():
    once = with_quality_tags("1girl")
    twice = with_quality_tags(once)
    assert once == twice


def test_params_roundtrip():
    original = GenerationParams(
        prompt="1girl, cat",
        width=832,
        height=1216,
        seed=42,
        num_images=2,
        loras=[LoraSpec(source="org/lora", weight_name="a.safetensors", scale=0.8)],
    )
    restored = GenerationParams.from_json(original.to_json())
    assert restored.to_dict() == original.to_dict()
    assert restored.loras[0].scale == 0.8


def test_params_default_negative_prompt():
    params = GenerationParams(prompt="x")
    assert params.negative_prompt == DEFAULT_NEGATIVE_PROMPT


def test_validate_rejects_bad_size():
    with pytest.raises(ValueError):
        GenerationParams(prompt="x", width=833).validate()


def test_validate_rejects_empty_prompt():
    with pytest.raises(ValueError):
        GenerationParams(prompt="   ").validate()


def test_validate_rejects_bad_strength():
    with pytest.raises(ValueError):
        GenerationParams(prompt="x", strength=1.5).validate()


def test_with_quality_tags_mutates_and_returns_self():
    params = GenerationParams(prompt="1girl")
    returned = params.with_quality_tags()
    assert returned is params
    assert QUALITY_TAGS[0] in params.prompt


def test_snap_size_mutates():
    params = GenerationParams(prompt="x", width=800, height=1200)
    params.snap_size()
    assert (params.width, params.height) in RESOLUTIONS

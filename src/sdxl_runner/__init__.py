"""sdxl-runner: run Animagine XL 4.0 Zero on a cloud GPU from your machine."""

from __future__ import annotations

from .client import GenerationResult, SdxlClient
from .params import GenerationParams, LoraSpec
from .presets import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    MODEL_ID,
    QUALITY_TAGS,
    RESOLUTIONS,
    snap_resolution,
    with_quality_tags,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_HEIGHT",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_STEPS",
    "DEFAULT_WIDTH",
    "MODEL_ID",
    "QUALITY_TAGS",
    "RESOLUTIONS",
    "GenerationParams",
    "GenerationResult",
    "LoraSpec",
    "SdxlClient",
    "main",
    "snap_resolution",
    "with_quality_tags",
]


def main() -> None:
    """Back-compat entry point; forwards to the CLI."""
    from .cli import main as cli_main

    raise SystemExit(cli_main())

"""Generation parameters shared between the client and the server.

Everything that crosses the wire is a plain JSON string produced by
:meth:`GenerationParams.to_json` and parsed by
:meth:`GenerationParams.from_json`, so the transport (Gradio) only ever carries
a single text field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .presets import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    snap_resolution,
    with_quality_tags,
)

# SDXL requires dimensions that are multiples of 8.
_SIZE_MULTIPLE = 8


@dataclass
class LoraSpec:
    """A LoRA adapter to load and apply during generation."""

    source: str
    weight_name: str | None = None
    scale: float = 1.0

    @property
    def alias(self) -> str:
        """Stable identifier used as the diffusers ``adapter_name``."""
        if self.weight_name:
            return f"{self.source}::{self.weight_name}"
        return self.source

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "weight_name": self.weight_name,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraSpec:
        return cls(
            source=str(data["source"]),
            weight_name=data.get("weight_name"),
            scale=float(data.get("scale", 1.0)),
        )


@dataclass
class GenerationParams:
    """A single generation request (text-to-image or image-to-image)."""

    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    num_inference_steps: int = DEFAULT_STEPS
    seed: int = -1  # -1 means "pick a random seed"
    num_images: int = 1
    init_image: str | None = None  # path/reference for img2img
    strength: float = 0.6  # img2img denoising strength
    loras: list[LoraSpec] = field(default_factory=list)

    def validate(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width % _SIZE_MULTIPLE or self.height % _SIZE_MULTIPLE:
            raise ValueError(
                f"width and height must be multiples of {_SIZE_MULTIPLE} "
                f"(got {self.width}x{self.height})"
            )
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be >= 0")
        if self.num_images < 1:
            raise ValueError("num_images must be >= 1")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be between 0.0 and 1.0")
        for lora in self.loras:
            if not lora.source:
                raise ValueError("each LoRA needs a source")

    # -- helpers ---------------------------------------------------------

    def with_quality_tags(self, tags: list[str] | None = None) -> GenerationParams:
        """Return a copy with the model's quality tags appended."""
        self.prompt = with_quality_tags(self.prompt, tags)
        return self

    def snap_size(self) -> GenerationParams:
        """Return a copy snapped to the nearest recommended resolution bucket."""
        self.width, self.height = snap_resolution(self.width, self.height)
        return self

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "num_images": self.num_images,
            "init_image": self.init_image,
            "strength": self.strength,
            "loras": [lora.to_dict() for lora in self.loras],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationParams:
        return cls(
            prompt=str(data["prompt"]),
            negative_prompt=data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            width=int(data.get("width", DEFAULT_WIDTH)),
            height=int(data.get("height", DEFAULT_HEIGHT)),
            guidance_scale=float(data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
            num_inference_steps=int(data.get("num_inference_steps", DEFAULT_STEPS)),
            seed=int(data.get("seed", -1)),
            num_images=int(data.get("num_images", 1)),
            init_image=data.get("init_image"),
            strength=float(data.get("strength", 0.6)),
            loras=[LoraSpec.from_dict(item) for item in data.get("loras", [])],
        )

    @classmethod
    def from_json(cls, raw: str) -> GenerationParams:
        return cls.from_dict(json.loads(raw))

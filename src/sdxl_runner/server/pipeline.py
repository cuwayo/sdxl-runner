"""Server-side model manager for Animagine XL 4.0 Zero.

Loads the SDXL pipeline lazily on first use, keeps a shared img2img pipeline,
manages LoRA adapters, and serialises generation behind a lock because diffusers
pipelines are not thread-safe.

A ``mock`` mode skips all heavy imports and returns placeholder images, which
lets the full client <-> server <-> tunnel path be tested without a GPU.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..params import GenerationParams, LoraSpec
from ..presets import MODEL_ID

# diffusers callback signature: (pipe, step, timestep, kwargs) -> kwargs
ProgressFn = Callable[[int, int], None]


@dataclass
class _LoadedLora:
    spec: LoraSpec
    loaded: bool = False


class ModelManager:
    """Owns the diffusers pipelines and the LoRA registry."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        mock: bool = False,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.mock = mock
        self._device = device
        self._lock = threading.Lock()
        self._txt2img: Any = None
        self._img2img: Any = None
        self._loras: dict[str, _LoadedLora] = {}
        self._active: list[str] = []

    # -- lifecycle -------------------------------------------------------

    @property
    def device(self) -> str:
        if self._device:
            return self._device
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _ensure_loaded(self) -> None:
        if self.mock or self._txt2img is not None:
            return
        import torch
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLPipeline,
        )

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            add_watermarker=False,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        self._apply_memory_optimizations(pipe)

        img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
        # from_pipe shares weights; reapply scheduler + offload on the derived
        # pipeline (documented diffusers gotcha).
        img2img.scheduler = EulerAncestralDiscreteScheduler.from_config(
            img2img.scheduler.config
        )
        self._apply_memory_optimizations(img2img)

        self._txt2img = pipe
        self._img2img = img2img

    def _apply_memory_optimizations(self, pipe: Any) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover - optional
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        if self.device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:  # noqa: BLE001 - fall back if accelerate is absent
                pipe.to("cuda")
        else:
            pipe.to("cpu")

    # -- generation ------------------------------------------------------

    def generate(
        self,
        params: GenerationParams,
        progress: ProgressFn | None = None,
    ) -> list[Any]:
        """Run txt2img or img2img depending on ``params.init_image``."""
        params.validate()
        if self.mock:
            return self._mock_images(params)

        with self._lock:
            self._ensure_loaded()
            self._sync_loras(params.loras)
            pipe = self._img2img if params.init_image else self._txt2img
            callback = self._make_callback(params.num_inference_steps, progress)
            common = {
                "prompt": params.prompt,
                "negative_prompt": params.negative_prompt or None,
                "guidance_scale": params.guidance_scale,
                "num_inference_steps": params.num_inference_steps,
                "num_images_per_prompt": params.num_images,
                "callback_on_step_end": callback,
            }
            seeds = self._seeds(params)
            if params.init_image:
                image = self._load_image(params.init_image)
                kwargs = {
                    "image": image,
                    "strength": params.strength,
                    "width": params.width,
                    "height": params.height,
                    "generator": seeds,
                }
            else:
                kwargs = {
                    "width": params.width,
                    "height": params.height,
                    "generator": seeds,
                }
            result = pipe(**common, **kwargs)
        return list(result.images)

    # -- LoRA ------------------------------------------------------------

    def load_lora(
        self, source: str, weight_name: str | None = None, scale: float = 1.0
    ) -> str:
        """Register a LoRA (loaded lazily on next use). Returns its alias."""
        spec = LoraSpec(source=source, weight_name=weight_name, scale=scale)
        self._loras[spec.alias] = _LoadedLora(spec=spec)
        return spec.alias

    def unload_loras(self) -> None:
        """Forget all adapters and restore base weights."""
        with self._lock:
            self._loras.clear()
            self._active = []
            if not self.mock and self._txt2img is not None:
                with contextlib.suppress(Exception):  # pragma: no cover
                    self._txt2img.unload_lora_weights()
                    self._img2img.unload_lora_weights()

    def list_loras(self) -> dict[str, Any]:
        """Return registered + currently active adapters."""
        return {
            "registered": list(self._loras.keys()),
            "active": list(self._active),
        }

    def _sync_loras(self, requested: list[LoraSpec]) -> None:
        """Load requested adapters and set the active set (no-op if unchanged)."""
        desired: list[str] = []
        scales: list[float] = []
        for spec in requested:
            alias = self.load_lora(spec.source, spec.weight_name, spec.scale)
            desired.append(alias)
            scales.append(spec.scale)

        if desired == self._active:
            return

        for alias in desired:
            entry = self._loras[alias]
            if not entry.loaded:
                self._txt2img.load_lora_weights(
                    entry.spec.source,
                    weight_name=entry.spec.weight_name,
                    adapter_name=alias,
                )
                entry.loaded = True

        if desired:
            self._txt2img.set_adapters(desired, adapter_weights=scales)
        else:
            self._txt2img.unload_lora_weights()
            for entry in self._loras.values():
                entry.loaded = False
        self._active = desired

    # -- helpers ---------------------------------------------------------

    def _seeds(self, params: GenerationParams) -> Any:
        import torch

        generator = torch.Generator(device=self.device)
        seed = params.seed
        if seed is None or seed < 0:
            seed = int(generator.seed())
        self._last_seed = seed
        return generator.manual_seed(seed)

    def _load_image(self, ref: str) -> Any:
        from PIL import Image

        return Image.open(ref).convert("RGB")

    def _make_callback(self, total: int, progress: ProgressFn | None) -> Any:
        if progress is None:
            return None

        def callback(pipe: Any, step: int, timestep: Any, kwargs: dict) -> dict:
            progress(step + 1, total)
            return kwargs

        return callback

    def _mock_images(self, params: GenerationParams) -> list[Any]:
        from PIL import Image

        seed = params.seed if params.seed >= 0 else 1234
        base = (
            seed * 2654435761 % 256,
            (seed * 40503) % 256,
            (seed * 12289 + 40) % 256,
        )
        images = []
        for index in range(params.num_images):
            color = tuple((channel + index * 20) % 256 for channel in base)
            images.append(Image.new("RGB", (params.width, params.height), color))
        self._last_seed = seed
        return images

    @property
    def last_seed(self) -> int | None:
        return getattr(self, "_last_seed", None)

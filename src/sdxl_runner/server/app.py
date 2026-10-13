"""Gradio Blocks app exposing named endpoints for the remote client.

Named endpoints (used by ``gradio_client``):
    /txt2img : (params_json)                -> (gallery, status_json)
    /img2img : (params_json, init_image)    -> (gallery, status_json)
    /loras   : ()                           -> status_json

``params_json`` is a :class:`~sdxl_runner.params.GenerationParams` serialized
with ``to_json()``; the status JSON carries metadata such as the resolved seed.

The JSON-payload endpoints are bound to hidden components so the client contract
stays stable and can carry LoRA specs. A small interactive UI is also provided
for humans and is hidden from the API.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from ..params import GenerationParams
from .pipeline import ModelManager

# Where the API endpoints stage generated PNGs for the client to download.
_EXPORT_DIR = Path(tempfile.gettempdir()) / "sdxl-runner-export"

try:  # Gradio is only needed on the server.
    import gradio as gr
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The server requires Gradio. Install with: pip install 'sdxl-runner[server]'"
    ) from exc

# Module-level singleton so it is not re-created on every call (B008).
_PROGRESS = gr.Progress()


def build_app(manager: ModelManager) -> gr.Blocks:
    """Construct the Gradio app bound to ``manager``."""

    def _run(
        params: GenerationParams, progress: Any | None = None
    ) -> tuple[list[Any], str]:
        report = None
        if progress is not None:

            def report(step: int, total: int) -> None:  # type: ignore[misc]
                progress(step / max(total, 1), desc=f"step {step}/{total}")

        images = manager.generate(params, progress=report)
        status = {
            "seed": manager.last_seed,
            "width": params.width,
            "height": params.height,
            "num_images": len(images),
        }
        return images, json.dumps(status)

    def _export(images: list[Any], seed: int | None) -> list[str]:
        """Save images as PNG files and return their paths for download."""
        _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        seed_part = "random" if seed is None else str(seed)
        stamp = int(time.time() * 1000)
        paths: list[str] = []
        for index, image in enumerate(images):
            path = _EXPORT_DIR / f"animagine_{seed_part}_{stamp}_{index}.png"
            image.save(path, format="PNG")
            paths.append(str(path))
        return paths

    # --- named API endpoints (called by gradio_client) -----------------

    def txt2img(params_json: str, progress: gr.Progress = _PROGRESS):
        params = GenerationParams.from_json(params_json)
        images, status = _run(params, progress)
        return _export(images, manager.last_seed), status

    def img2img(
        params_json: str,
        init_image: str,
        progress: gr.Progress = _PROGRESS,
    ):
        params = GenerationParams.from_json(params_json)
        params.init_image = init_image  # authoritative server-side path
        images, status = _run(params, progress)
        return _export(images, manager.last_seed), status

    def loras() -> str:
        return json.dumps(manager.list_loras())

    # --- interactive UI handlers (share the same generation path) ------

    def ui_txt2img(prompt, negative, width, height, steps, cfg, seed, batch, quality):
        params = GenerationParams(
            prompt=prompt,
            negative_prompt=negative,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            seed=int(seed),
            num_images=int(batch),
        )
        if quality:
            params.with_quality_tags()
        return _run(params, gr.Progress())

    def ui_img2img(init, prompt, negative, strength, steps, cfg, seed):
        params = GenerationParams(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            seed=int(seed),
            strength=float(strength),
            init_image=init,
        )
        return _run(params, gr.Progress())

    with gr.Blocks(title="Animagine XL 4.0 Zero") as demo:
        gr.Markdown(
            "# Animagine XL 4.0 Zero\n"
            "Remote SDXL generation. Drive it with the `sdxl-runner` client, or "
            "use the tabs below directly."
        )

        # Hidden components that expose the JSON API endpoints. Files (not
        # Gallery) so full-quality PNGs are delivered unchanged to the client.
        with gr.Group(visible=False):
            api_txt_params = gr.Textbox()
            api_txt_files = gr.Files()
            api_txt_status = gr.Textbox()
            api_txt_btn = gr.Button()
            api_txt_btn.click(
                txt2img,
                inputs=[api_txt_params],
                outputs=[api_txt_files, api_txt_status],
                api_name="txt2img",
            )

            api_i2i_params = gr.Textbox()
            api_i2i_init = gr.Image(type="filepath")
            api_i2i_files = gr.Files()
            api_i2i_status = gr.Textbox()
            api_i2i_btn = gr.Button()
            api_i2i_btn.click(
                img2img,
                inputs=[api_i2i_params, api_i2i_init],
                outputs=[api_i2i_files, api_i2i_status],
                api_name="img2img",
            )

            api_loras_out = gr.Textbox()
            api_loras_btn = gr.Button()
            api_loras_btn.click(
                loras, inputs=None, outputs=[api_loras_out], api_name="loras"
            )

        with gr.Tab("txt2img"):
            txt_prompt = gr.Textbox(label="Prompt", lines=3)
            txt_neg = gr.Textbox(label="Negative prompt", lines=2)
            with gr.Row():
                txt_w = gr.Slider(256, 1536, value=832, step=8, label="Width")
                txt_h = gr.Slider(256, 1536, value=1216, step=8, label="Height")
            with gr.Row():
                txt_steps = gr.Slider(1, 60, value=28, step=1, label="Steps")
                txt_cfg = gr.Slider(0, 15, value=5.0, step=0.5, label="CFG")
            with gr.Row():
                txt_seed = gr.Number(value=-1, label="Seed (-1 = random)")
                txt_batch = gr.Slider(1, 8, value=1, step=1, label="Batch")
            txt_quality = gr.Checkbox(label="Append quality tags", value=True)
            txt_btn = gr.Button("Generate", variant="primary")
            txt_gallery = gr.Gallery(label="Result", columns=2)
            txt_status = gr.Textbox(label="Status (JSON)", visible=False)
            txt_btn.click(
                ui_txt2img,
                inputs=[
                    txt_prompt,
                    txt_neg,
                    txt_w,
                    txt_h,
                    txt_steps,
                    txt_cfg,
                    txt_seed,
                    txt_batch,
                    txt_quality,
                ],
                outputs=[txt_gallery, txt_status],
            )

        with gr.Tab("img2img"):
            i2i_init = gr.Image(type="filepath", label="Initial image")
            i2i_prompt = gr.Textbox(label="Prompt", lines=3)
            i2i_neg = gr.Textbox(label="Negative prompt", lines=2)
            i2i_strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Strength")
            with gr.Row():
                i2i_steps = gr.Slider(1, 60, value=28, step=1, label="Steps")
                i2i_cfg = gr.Slider(0, 15, value=5.0, step=0.5, label="CFG")
            i2i_seed = gr.Number(value=-1, label="Seed (-1 = random)")
            i2i_btn = gr.Button("Generate", variant="primary")
            i2i_gallery = gr.Gallery(label="Result", columns=2)
            i2i_status = gr.Textbox(label="Status (JSON)", visible=False)
            i2i_btn.click(
                ui_img2img,
                inputs=[
                    i2i_init,
                    i2i_prompt,
                    i2i_neg,
                    i2i_strength,
                    i2i_steps,
                    i2i_cfg,
                    i2i_seed,
                ],
                outputs=[i2i_gallery, i2i_status],
            )

        with gr.Tab("LoRAs"):
            gr.Markdown("Adapters currently loaded on the server.")
            lora_btn = gr.Button("Refresh")
            lora_out = gr.Textbox(label="LoRAs (JSON)")
            lora_btn.click(loras, inputs=None, outputs=lora_out)

    return demo

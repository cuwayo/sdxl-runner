"""Command-line client: ``sdxl-runner``.

Examples
--------
    sdxl-runner generate --url https://xxxx.trycloudflare.com \
        --quality --auto-size -p "1girl, arima kana, oshi no ko" -o ./out

    sdxl-runner img2img --url http://127.0.0.1:8000 \
        -p "1girl, sunset" -i ./photo.png --strength 0.6

    sdxl-runner loras --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys

from .client import SdxlClient
from .params import GenerationParams, LoraSpec
from .presets import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", required=True, help="Server URL (tunnel or host).")
    parser.add_argument("-p", "--prompt", required=True, help="Positive prompt.")
    parser.add_argument(
        "-n", "--negative", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt."
    )
    parser.add_argument(
        "-W", "--width", type=int, default=DEFAULT_WIDTH, help="Image width."
    )
    parser.add_argument(
        "-H", "--height", type=int, default=DEFAULT_HEIGHT, help="Image height."
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, help="Inference steps."
    )
    parser.add_argument(
        "--cfg", type=float, default=DEFAULT_GUIDANCE_SCALE, help="Guidance scale."
    )
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 for random).")
    parser.add_argument("-b", "--batch", type=int, default=1, help="Number of images.")
    parser.add_argument(
        "--quality", action="store_true", help="Append the model's quality tags."
    )
    parser.add_argument(
        "--auto-size",
        action="store_true",
        help="Snap width/height to the nearest recommended resolution.",
    )
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="SOURCE[:WEIGHT_NAME][@SCALE]",
        help="LoRA to apply; repeatable. Example: org/lora:file.safetensors@0.8",
    )
    parser.add_argument("-o", "--out", default=".", help="Output directory for images.")
    parser.add_argument(
        "--auth", default=None, metavar="USER:PASS", help="Basic auth credentials."
    )
    parser.add_argument(
        "--timeout", type=float, default=None, help="Request timeout in seconds."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print a JSON summary of results."
    )


def _parse_lora(spec: str) -> LoraSpec:
    scale = 1.0
    if "@" in spec:
        spec, _, scale_str = spec.rpartition("@")
        scale = float(scale_str)
    weight_name = None
    if ":" in spec:
        source, _, weight_name = spec.partition(":")
    else:
        source = spec
    return LoraSpec(source=source, weight_name=weight_name or None, scale=scale)


def _build_params(args: argparse.Namespace) -> GenerationParams:
    params = GenerationParams(
        prompt=args.prompt,
        negative_prompt=args.negative,
        width=args.width,
        height=args.height,
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        seed=args.seed,
        num_images=args.batch,
        strength=getattr(args, "strength", 0.6),
        init_image=getattr(args, "init", None),
        loras=[_parse_lora(spec) for spec in args.lora],
    )
    if args.quality:
        params.with_quality_tags()
    if args.auto_size:
        params.snap_size()
    return params


def _make_client(args: argparse.Namespace) -> SdxlClient:
    auth = None
    if args.auth:
        user, _, password = args.auth.partition(":")
        auth = (user, password)
    return SdxlClient(args.url, download_dir=args.out, auth=auth, timeout=args.timeout)


def _report(result, args: argparse.Namespace) -> int:
    if args.json:
        print(
            json.dumps(
                {
                    "seed": result.seed,
                    "images": [str(path) for path in result.images],
                }
            )
        )
    else:
        if result.seed is not None:
            print(f"seed: {result.seed}")
        for path in result.images:
            print(path)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    client = _make_client(args)
    params = _build_params(args)
    return _report(client.txt2img(params), args)


def _cmd_img2img(args: argparse.Namespace) -> int:
    client = _make_client(args)
    params = _build_params(args)
    return _report(client.img2img(params), args)


def _cmd_loras(args: argparse.Namespace) -> int:
    client = _make_client(args)
    names = client.loras()
    if args.json:
        print(json.dumps({"loras": names}))
    else:
        for name in names:
            print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdxl-runner",
        description="Run Animagine XL 4.0 on a remote GPU via sdxl-server.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Text-to-image generation.")
    _add_common_args(gen)
    gen.set_defaults(func=_cmd_generate)

    i2i = sub.add_parser("img2img", help="Image-to-image generation.")
    _add_common_args(i2i)
    i2i.add_argument("-i", "--init", required=True, help="Initial image path.")
    i2i.add_argument(
        "--strength", type=float, default=0.6, help="Denoising strength (0-1)."
    )
    i2i.set_defaults(func=_cmd_img2img)

    loras = sub.add_parser("loras", help="List LoRAs loaded on the server.")
    loras.add_argument("--url", required=True, help="Server URL.")
    loras.add_argument(
        "-o", "--out", default=".", help="Scratch directory for the client."
    )
    loras.add_argument("--auth", default=None, metavar="USER:PASS")
    loras.add_argument("--timeout", type=float, default=None)
    loras.add_argument("--json", action="store_true")
    loras.set_defaults(func=_cmd_loras)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

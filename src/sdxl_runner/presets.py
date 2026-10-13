"""Model presets and recommended defaults for Animagine XL 4.0 Zero.

Values here mirror the model card:
https://huggingface.co/cagliostrolab/animagine-xl-4.0-zero
"""

from __future__ import annotations

import math

MODEL_ID = "cagliostrolab/animagine-xl-4.0-zero"

# The model card's recommended negative prompt.
DEFAULT_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
    "fewer digits, cropped, worst quality, low quality, low score, bad score, "
    "average score, signature, watermark, username, blurry"
)

# Quality-enhancement tags appended to the end of a prompt.
QUALITY_TAGS = ["masterpiece", "high score", "great score", "absurdres"]

# Recommended resolution buckets (portrait / square / landscape).
RESOLUTIONS: list[tuple[int, int]] = [
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
    (896, 1152),
    (832, 1216),
    (768, 1344),
    (640, 1536),
]

# Optimal settings from the model card.
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 1216
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_STEPS = 28
DEFAULT_SAMPLER = "euler_a"


def with_quality_tags(prompt: str, tags: list[str] | None = None) -> str:
    """Append the model's quality tags to ``prompt`` (idempotent)."""
    tags = QUALITY_TAGS if tags is None else tags
    existing = {part.strip().lower() for part in prompt.split(",")}
    extra = [tag for tag in tags if tag.lower() not in existing]
    base = prompt.strip().rstrip(",")
    if not extra:
        return base
    return ", ".join([base, *extra])


def snap_resolution(width: int, height: int) -> tuple[int, int]:
    """Return the recommended bucket closest to ``(width, height)``.

    Distance weighs aspect-ratio mismatch more heavily than area mismatch so the
    orientation and framing of the request are preserved.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    target_ar = math.log(width / height)
    target_area = math.log(width * height)

    def score(bucket: tuple[int, int]) -> float:
        bw, bh = bucket
        return 2.0 * abs(math.log(bw / bh) - target_ar) + abs(
            math.log(bw * bh) - target_area
        )

    return min(RESOLUTIONS, key=score)

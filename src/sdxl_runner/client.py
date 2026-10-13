"""Local client that talks to a remote ``sdxl-server`` over Gradio.

The server exposes three named endpoints (``/txt2img``, ``/img2img``,
``/loras``). Each generation endpoint returns a ``(gallery, status_json)`` pair;
this client downloads the gallery images to a local directory and parses the
status JSON for metadata such as the resolved seed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .params import GenerationParams, LoraSpec

_TXT2IMG_API = "/txt2img"
_IMG2IMG_API = "/img2img"
_LORAS_API = "/loras"


@dataclass
class GenerationResult:
    """Images returned by a generation call, plus metadata."""

    images: list[Path] = field(default_factory=list)
    seed: int | None = None
    params: GenerationParams | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)


class SdxlClient:
    """Thin wrapper around :class:`gradio_client.Client`.

    Parameters
    ----------
    src:
        Public URL of the server (``https://xxxx.trycloudflare.com``) or a
        Gradio Space id.
    download_dir:
        Directory that received files are written to. Created if missing.
    auth:
        Optional ``(username, password)`` for a password-protected server.
    timeout:
        Per-request timeout in seconds, passed through to ``gradio_client``.
    """

    def __init__(
        self,
        src: str,
        download_dir: str | Path = ".",
        auth: tuple[str, str] | None = None,
        timeout: float | None = None,
        verbose: bool = False,
    ) -> None:
        # Imported lazily so the server-only install does not need gradio_client.
        from gradio_client import Client, handle_file

        self._handle_file = handle_file
        self.src = src.rstrip("/")
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "download_files": str(self.download_dir),
            "verbose": verbose,
        }
        if auth is not None:
            kwargs["auth"] = auth
        if timeout is not None:
            # gradio_client forwards httpx kwargs to the underlying transport.
            kwargs["httpx_kwargs"] = {"timeout": timeout}

        self._client = Client(self.src, **kwargs)

    # -- endpoints -------------------------------------------------------

    def txt2img(self, params: GenerationParams) -> GenerationResult:
        """Generate images from a text prompt."""
        params.validate()
        gallery, status = self._call(_TXT2IMG_API, params)
        return self._collect(gallery, status, params)

    def img2img(self, params: GenerationParams) -> GenerationResult:
        """Generate images from an initial image plus a text prompt."""
        if not params.init_image:
            raise ValueError("img2img requires params.init_image")
        params.validate()
        # The init image is uploaded as a separate Gradio file input.
        gallery, status = self._call(
            _IMG2IMG_API, params, extra=(self._handle_file(params.init_image),)
        )
        return self._collect(gallery, status, params)

    def loras(self) -> list[str]:
        """Return the adapter names currently loaded on the server."""
        raw = self._client.predict(api_name=_LORAS_API)
        return _parse_loras(raw)

    # -- convenience -----------------------------------------------------

    def generate(self, **kwargs: Any) -> GenerationResult:
        """Build a :class:`GenerationParams` from ``kwargs`` and run txt2img.

        Accepts the same field names as :class:`GenerationParams`, e.g.
        ``client.generate(prompt="1girl", width=832, quality_tags=True)``.
        """
        quality = kwargs.pop("quality_tags", False)
        auto_size = kwargs.pop("auto_size", False)
        loras = kwargs.pop("loras", None)
        if loras is not None:
            kwargs["loras"] = [
                LoraSpec.from_dict(item) if isinstance(item, dict) else item
                for item in loras
            ]
        params = GenerationParams(**kwargs)
        if quality:
            params.with_quality_tags()
        if auto_size:
            params.snap_size()
        return self.txt2img(params)

    # -- internals -------------------------------------------------------

    def _call(
        self,
        api_name: str,
        params: GenerationParams,
        extra: tuple[Any, ...] = (),
    ) -> tuple[Any, dict[str, Any]]:
        args: list[Any] = [params.to_json(), *extra]
        result = self._client.predict(*args, api_name=api_name)
        gallery, status = _split_result(result)
        return gallery, status

    def _collect(
        self,
        gallery: Any,
        status: dict[str, Any],
        params: GenerationParams,
    ) -> GenerationResult:
        paths = [Path(p) for p in _iter_paths(gallery)]
        seed = status.get("seed")
        width = status.get("width", params.width)
        height = status.get("height", params.height)

        named: list[Path] = []
        for index, path in enumerate(paths):
            target = self.download_dir / _filename(
                width, height, seed, index, path.suffix or ".png"
            )
            try:
                if path.resolve() != target.resolve():
                    path.replace(target)
            except OSError:
                # Cross-device rename or already-moved file: copy instead.
                target.write_bytes(path.read_bytes())
            named.append(target)

        _prune_empty_dirs(self.download_dir)
        return GenerationResult(images=named, seed=seed, params=params, raw=status)


def _split_result(result: Any) -> tuple[Any, dict[str, Any]]:
    """Split a Gradio ``(gallery, status_json)`` tuple into its parts.

    A two-output event returns a 2-tuple, but a bare two-image gallery is also a
    2-element sequence, so the second element is only treated as the status when
    it actually looks like one (a dict or a JSON object string).
    """
    if isinstance(result, (tuple, list)) and len(result) == 2:
        gallery, status = result
        parsed = _maybe_status(status)
        if parsed is not None:
            return gallery, parsed
    return result, {}


def _maybe_status(status: Any) -> dict[str, Any] | None:
    """Return ``status`` as a dict if it is a status payload, else ``None``."""
    if isinstance(status, dict):
        return status
    if isinstance(status, str) and status.strip():
        try:
            parsed = json.loads(status)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _iter_paths(gallery: Any) -> Sequence[str]:
    """Yield local file paths from a Gradio Files/Image/Gallery output."""
    if gallery is None:
        return []
    if isinstance(gallery, str):
        return [gallery]
    if isinstance(gallery, dict):
        value = gallery.get("path") or gallery.get("image")
        if isinstance(value, dict):
            value = value.get("path")
        return [str(value)] if value else []
    if isinstance(gallery, (list, tuple)):
        paths: list[str] = []
        for item in gallery:
            paths.extend(_iter_paths(item))
        return paths
    return []


def _parse_loras(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [raw] if raw else []
    if isinstance(raw, dict):
        # get_list_adapters() returns {component: [names]}; flatten and dedupe.
        names: list[str] = []
        for value in raw.values():
            for name in value or []:
                if name not in names:
                    names.append(name)
        return names
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return []


def _prune_empty_dirs(root: Path) -> None:
    """Remove empty subdirectories left behind by gradio_client downloads."""
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir():
            try:
                child.rmdir()  # only succeeds when empty
            except OSError:
                pass


def _filename(
    width: int, height: int, seed: int | None, index: int, suffix: str
) -> str:
    seed_part = "random" if seed is None else str(seed)
    return f"animagine_{width}x{height}_seed{seed_part}_{index}{suffix}"

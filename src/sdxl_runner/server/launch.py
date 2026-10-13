"""Server entry point: ``sdxl-server``.

Launches the Gradio app, optionally fronts it with a cloudflared quick tunnel,
checks which transport protocol Gradio negotiated (quick tunnels do not support
SSE), and prints the URL to paste into the local ``sdxl-runner`` client.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

from ..presets import MODEL_ID
from .app import build_app
from .pipeline import ModelManager
from .tunnel import Tunnel, start_tunnel

_BANNER = "=" * 72


def _detect_protocol(local_url: str) -> str | None:
    """Read the transport protocol Gradio negotiated from ``/config``."""
    try:
        with urllib.request.urlopen(
            local_url.rstrip("/") + "/config", timeout=10
        ) as response:
            config = json.load(response)
        return config.get("protocol")
    except Exception:  # noqa: BLE001 - protocol detection is best-effort
        return None


def _warn_if_sse(protocol: str | None) -> None:
    if protocol and protocol.startswith("sse"):
        print(
            "[sdxl-runner] note: Gradio negotiated the SSE protocol "
            f"({protocol!r}). Cloudflare quick tunnels buffer SSE, so generation "
            "still completes but live progress will not stream. If requests "
            "instead hang, restart with --share (gradio.live) or --no-tunnel on a "
            "host with an open port.",
            file=sys.stderr,
        )


def launch_server(
    port: int = 8000,
    tunnel: bool = True,
    share: bool = False,
    mock: bool = False,
    model_id: str = MODEL_ID,
    host: str = "127.0.0.1",
) -> None:
    """Start the server and block until interrupted."""
    manager = ModelManager(model_id=model_id, mock=mock)
    demo = build_app(manager)

    _, local_url, share_url = demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        prevent_thread_lock=True,
        show_api=True,
        quiet=True,
    )
    local_url = local_url or f"http://{host}:{port}"

    protocol = _detect_protocol(local_url)
    _warn_if_sse(protocol)

    active_tunnel: Tunnel | None = None
    public_url: str | None = share_url
    if tunnel and not share:
        active_tunnel = start_tunnel(port)
        public_url = active_tunnel.url

    _print_banner(local_url, public_url, protocol, mock)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[sdxl-runner] shutting down", file=sys.stderr)
    finally:
        if active_tunnel is not None:
            active_tunnel.stop()


def _print_banner(
    local_url: str, public_url: str | None, protocol: str | None, mock: bool
) -> None:
    print(_BANNER)
    print(
        " Animagine XL 4.0 Zero server is running" + ("  [MOCK MODE]" if mock else "")
    )
    print(_BANNER)
    print(f" Local URL : {local_url}")
    if public_url:
        print(f" Public URL: {public_url}")
    if protocol:
        print(f" Transport : {protocol}")
    print(_BANNER)
    if public_url:
        print(" From your machine, run:\n")
        print(
            "   sdxl-runner generate \\\n"
            f"     --url {public_url} \\\n"
            "     --quality \\\n"
            '     -p "1girl, masterpiece, looking at viewer, smile, outdoors" \\\n'
            "     -o ./out"
        )
        print("\n" + _BANNER)
    print(" Press Ctrl+C to stop.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sdxl-server",
        description="Run Animagine XL 4.0 Zero behind a Gradio API.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Local port.")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)."
    )
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Do not start a cloudflared tunnel (use on RunPod/Vast).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Use Gradio's own gradio.live share link instead of cloudflared.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return placeholder images without loading the model.",
    )
    parser.add_argument(
        "--model", default=MODEL_ID, help="Hugging Face model id to load."
    )
    args = parser.parse_args(argv)

    launch_server(
        port=args.port,
        tunnel=not args.no_tunnel,
        share=args.share,
        mock=args.mock,
        model_id=args.model,
        host=args.host,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

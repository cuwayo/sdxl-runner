"""Start ``sdxl-server`` on a Colab VM in the background and print its public URL.

Runs *on the Colab VM* (via ``colab exec -f``). It launches the Gradio server
plus a cloudflared quick tunnel as a detached subprocess, waits for the tunnel
URL to appear in the server log, writes it to ``/content/sdxl-url.txt``, and
prints ``PUBLIC_URL=...`` so the local wrapper can capture it.

The background process keeps running after this cell returns, so you can drive
the server with ``sdxl-runner`` from your own machine until you ``colab stop``
the session.

Usage (from the local machine):
    colab exec -s sdxl --timeout 300 -f scripts/colab_serve.py

Set ``SDXL_MOCK=1`` to start in mock mode (no model download) for a quick
wiring test.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

PORT = int(os.environ.get("SDXL_PORT", "8000"))
RUN_DIR = Path(os.environ.get("SDXL_RUN_DIR", "/content"))
LOG_PATH = RUN_DIR / "sdxl-server.log"
URL_PATH = RUN_DIR / "sdxl-url.txt"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
TIMEOUT_S = 300.0


def _already_running() -> str | None:
    """Return the URL if a server is already serving, else ``None``."""
    if URL_PATH.exists():
        url = URL_PATH.read_text().strip()
        if url:
            return url
    return None


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    existing = _already_running()
    if existing:
        print(f"Server already running: {existing}")
        print(f"PUBLIC_URL={existing}")
        return 0

    # Detach so the server survives this cell returning.
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "sdxl_runner.server.launch",
        "--port",
        str(PORT),
        "--host",
        "127.0.0.1",
    ]
    if os.environ.get("SDXL_MOCK", "").lower() in {"1", "true", "yes"}:
        cmd.append("--mock")  # smoke-test the wiring without loading the model

    with LOG_PATH.open("w") as log:
        subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        if LOG_PATH.exists():
            text = LOG_PATH.read_text(errors="ignore")
            match = URL_RE.search(text)
            if match:
                url = match.group(0)
                URL_PATH.write_text(url + "\n")
                print(f"PUBLIC_URL={url}")
                return 0
        time.sleep(2)

    print("Timed out waiting for the tunnel URL. Server log tail:")
    if LOG_PATH.exists():
        print(LOG_PATH.read_text(errors="ignore")[-4000:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

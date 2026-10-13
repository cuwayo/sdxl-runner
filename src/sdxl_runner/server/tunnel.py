"""cloudflared quick-tunnel helper.

Downloads the ``cloudflared`` binary on demand, starts a quick tunnel pointing
at the local Gradio port, and parses the public ``*.trycloudflare.com`` URL from
the process output.

Note: Cloudflare quick tunnels do **not** support Server-Sent Events (SSE); they
do carry WebSockets and plain HTTP. The launcher checks which protocol Gradio
negotiated and warns if it is SSE (see :mod:`sdxl_runner.server.launch`).
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_CACHE_DIR = Path.home() / ".cache" / "sdxl-runner"
_BIN_PATH = _CACHE_DIR / "cloudflared"
_DOWNLOAD_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)


def find_cloudflared() -> Path | None:
    """Return a usable cloudflared binary path, if one exists."""
    on_path = shutil.which("cloudflared")
    if on_path:
        return Path(on_path)
    if _BIN_PATH.exists():
        return _BIN_PATH
    return None


def ensure_cloudflared(download: bool = True) -> Path:
    """Return the cloudflared binary, downloading it if necessary."""
    found = find_cloudflared()
    if found is not None:
        return found
    if not download:
        raise FileNotFoundError("cloudflared not found; install it or allow download")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _BIN_PATH.with_suffix(".download")
    print(f"[sdxl-runner] downloading cloudflared -> {_BIN_PATH}", file=sys.stderr)
    urllib.request.urlretrieve(_DOWNLOAD_URL, tmp)
    tmp.chmod(tmp.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    tmp.replace(_BIN_PATH)
    return _BIN_PATH


@dataclass
class Tunnel:
    """A running cloudflared quick tunnel."""

    process: subprocess.Popen
    url: str

    def stop(self) -> None:
        stop_tunnel(self.process)


def start_tunnel(
    port: int,
    binary: str | Path | None = None,
    timeout: float = 30.0,
) -> Tunnel:
    """Start a quick tunnel to ``http://localhost:PORT`` and return it."""
    exe = Path(binary) if binary is not None else ensure_cloudflared()
    process = subprocess.Popen(
        [str(exe), "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    deadline = time.time() + timeout
    url: str | None = None
    assert process.stdout is not None
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        match = _URL_RE.search(line)
        if match:
            url = match.group(0)
            break

    if url is None:
        stop_tunnel(process)
        raise RuntimeError(
            "cloudflared did not report a tunnel URL within "
            f"{timeout:.0f}s; check its output for errors"
        )
    return Tunnel(process=process, url=url)


def stop_tunnel(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        process.kill()

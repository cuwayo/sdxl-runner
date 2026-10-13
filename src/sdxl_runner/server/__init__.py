"""Server-side components (require the ``server`` extra)."""

from __future__ import annotations

__all__ = ["ModelManager", "build_app", "launch_server"]


def __getattr__(name: str):
    # Import lazily so importing the client package does not require torch/gradio.
    if name == "ModelManager":
        from .pipeline import ModelManager

        return ModelManager
    if name == "build_app":
        from .app import build_app

        return build_app
    if name == "launch_server":
        from .launch import launch_server

        return launch_server
    raise AttributeError(name)

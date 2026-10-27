"""Runnable reference examples for the quality pipeline.

``pipe_wear_twin`` is the end-to-end benchmark (stages A-G over a 3D
pipeline-wear twin). It is imported lazily by ``__main__`` so that simply
importing this package does not build a scene.
"""
from __future__ import annotations

__all__ = ["pipe_wear_twin"]

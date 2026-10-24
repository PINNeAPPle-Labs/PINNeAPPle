"""What a run used: versions of the NVIDIA and PINNeAPPle packages, the hardware, and step timings.

    info = RunInfo()
    with info.step("train"):
        ...
    info.mark("Isaac Sim", ran=False, why="needs an RTX GPU")
    info.save("results/run_info.json")

Every example in ``examples/nvidia`` writes one, so each figure can be traced to the machine and the versions that
produced it, and the steps that were *not* executed (GPU, API key) are listed as such.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import time
from contextlib import contextmanager
from typing import Any

__all__ = ["RunInfo", "environment"]

PACKAGES = ("pinneapple", "torch", "numpy", "scipy", "matplotlib", "nvidia-physicsnemo", "nvidia-physicsnemo.sym",
            "usd-core", "isaacsim", "isaaclab", "torch_geometric", "torch_scatter", "openai", "requests")


def _cpu() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment() -> dict[str, Any]:
    """Python, OS, CPU, GPU (CUDA through torch, if any) and the installed versions of the relevant packages."""
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": _cpu(),
            "cpu_count": os.cpu_count(), "gpu": gpu, "packages": vers}


class RunInfo:
    def __init__(self, name: str = ""):
        self.name = name
        self.t0 = time.time()
        self.steps: dict[str, float] = {}
        self.components: list[dict[str, Any]] = []
        self.env = environment()

    @contextmanager
    def step(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.steps[name] = round(time.perf_counter() - t, 3)

    def mark(self, component: str, *, ran: bool, why: str = "", how: str = "") -> None:
        """Record whether an NVIDIA component was executed here (``ran``) and why not / how."""
        self.components.append({"component": component, "ran": bool(ran), "why": why, "how": how})

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.t0)),
                "wall_s": round(time.time() - self.t0, 3), "steps_s": self.steps, "components": self.components,
                "environment": self.env}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

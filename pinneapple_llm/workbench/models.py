"""Which local model to run, from the machine's memory.

No local model matches the large hosted models on long, multi-file agent work. The recommendations below are the
strongest open-weight models with tool calling that fit each memory size, served by Ollama (Metal on Apple Silicon,
CUDA on NVIDIA). ``python -m pinneapple_llm.workbench eval`` measures the model you pick on the workbench's own tasks,
so you can compare models on your machine rather than trust a table.
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

__all__ = ["Machine", "Recommendation", "detect_machine", "recommend", "RECOMMENDATIONS"]


@dataclass(frozen=True)
class Machine:
    system: str              # "Darwin", "Linux", "Windows"
    arch: str                # "arm64", "x86_64"
    memory_gb: float         # unified memory on Apple Silicon, RAM elsewhere
    apple_silicon: bool


@dataclass(frozen=True)
class Recommendation:
    min_memory_gb: float     # total memory the machine needs (model + context + the OS and your apps)
    model: str               # Ollama tag
    approx_size_gb: float    # download / resident size of the default quantisation
    context: int             # context window the workbench asks for (num_ctx)
    note: str


# Ordered from the largest machine down; the first entry that fits is chosen.
RECOMMENDATIONS = (
    Recommendation(96, "gpt-oss:120b", 65, 65536,
                   "MoE, 5B active parameters: fast for its size; strongest choice here for planning and review"),
    Recommendation(48, "qwen3-coder:30b", 19, 65536,
                   "MoE coding model with native tool calls; leaves room for a long context"),
    Recommendation(32, "qwen3-coder:30b", 19, 32768,
                   "MoE coding model with native tool calls (3B active: fast on Apple Silicon)"),
    Recommendation(24, "gpt-oss:20b", 14, 32768, "MoE, 3.6B active; good tool use for its size"),
    Recommendation(16, "qwen3:8b", 5.2, 16384, "fits 16 GB; expect shorter, simpler tasks to work best"),
    Recommendation(0, "qwen3:4b", 2.6, 8192, "small machines only; single-file edits and questions"),
)


def _memory_bytes() -> float:
    if platform.system() == "Darwin":
        try:
            return float(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                        timeout=5).stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0.0
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return 0.0


def detect_machine() -> Machine:
    system, arch = platform.system(), platform.machine()
    return Machine(system=system, arch=arch, memory_gb=round(_memory_bytes() / 2**30, 1),
                   apple_silicon=system == "Darwin" and arch == "arm64")


def recommend(memory_gb: float | None = None) -> Recommendation:
    """The default model for a machine with ``memory_gb`` of memory (detected when omitted)."""
    mem = detect_machine().memory_gb if memory_gb is None else memory_gb
    for rec in RECOMMENDATIONS:
        if mem >= rec.min_memory_gb:
            return rec
    return RECOMMENDATIONS[-1]

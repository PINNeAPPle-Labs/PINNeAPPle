"""Peak memory of one operation, measured rather than guessed.

The rest of ``pinneapple_analysis.cost`` reports ``mem_source`` next to every peak; this module is where those
numbers come from, and where the limits of each method were established by measurement (issue #54). Nothing here
estimates: when a method cannot see an allocation it says so and the peak is reported as a lower bound or as
``unknown``, never as a plausible number.

What each method can and cannot see, measured on this library's own environment (CPython 3.15, torch 2.15,
NumPy 2.5, macOS arm64; the same experiment is ``tests/pinneapple_analysis/test_cost_memory.py``):

============================  ==========================  ==========================  ==================
method                        reference 200 MiB NumPy     reference 100 MiB torch     sees
============================  ==========================  ==========================  ==================
``tracemalloc``               200.0003 MiB (exact)        0.0007 MiB (blind)          Python + NumPy
PyTorch profiler (memory)     blind                       100.0000 MiB (exact)        CPU tensors
process RSS watermark         exact (single use)          exact (single use)          everything
``process`` (forked child)    200.6 MiB (exact)           exact, **fails on autograd** everything
``psutil`` RSS sampling       lower bound (2 ms poll)     lower bound (2 ms poll)     everything
============================  ==========================  ==========================  ==================

Two findings shaped the design:

* **NumPy is visible to ``tracemalloc``.** It allocates through ``PyDataMem``, which registers with tracemalloc, so
  a Python/NumPy peak is exact rather than the lower bound the old docstring claimed. PyTorch's CPU allocator is
  *not* registered, so tracemalloc is blind to tensors — the two are complements, not alternatives.
* **The in-process RSS watermark (``getrusage().ru_maxrss``) cannot be used per operation.** It is a lifetime
  high-water mark: it measures the first operation correctly and every smaller one after it as zero. Making it
  reusable means a fresh process per operation (``process`` method), and that process cannot run a
  ``backward()`` — PyTorch refuses fork with an active autograd thread pool (*"Unable to handle autograd's
  threading in combination with fork-based multiprocessing"*). So the fork method is offered as an independent
  cross-check for forward-only work, never as the default for training.

Hence ``auto``: CUDA when a GPU is present (its allocator reports the peak exactly), otherwise the PyTorch
profiler plus ``tracemalloc``, summed. The sum is exact when both allocators are live at the same time (the
common case: the data arrays are resident while the network runs) and an upper bound when they take turns.

A third finding matters for anyone tempted to check these numbers against ``/usr/bin/time -l`` or a process
monitor: **resident memory is not a reference for a repeated operation.** Running the same PINN step a second
time grows RSS by 45 % of the peak at one size and 12 % at another (the allocator reuses what the first run
freed), while the live-allocation peak does not move at all — which is the correct answer, since the operation
allocates the same bytes. RSS is the footprint to provision a machine with; live allocation is what the code
asks for. This module measures the second and says so.

RSS or peak, and the unit of ``ru_maxrss``
------------------------------------------
Two different questions hide behind "memory of an operation", and this module answers the second:

* **RSS (sampled, ``rss``)** is the resident footprint now, sampled every 2 ms. Its peak is what a machine is
  provisioned with, and it is a *lower bound* on what the code asked for: an allocation shorter than the poll
  interval is missed and a reused arena never appears twice.
* **Peak of live allocations (``torch``, ``tracemalloc``, ``cuda``, ``process``)** is the high-water mark of what
  the operation held at once, which is the number a budget, a scaling fit or a memory limit is about. This is the
  default and what ``exact`` marks.

The kernel reports the second of those, per process, as ``getrusage().ru_maxrss`` — but **in different units
per platform**: kibibytes on Linux (and on the other Linux-like systems CPython follows) and plain bytes on
macOS and the BSDs. Reading it with the wrong factor is a silent factor-1024 error, the kind of mistake that
turns 200 MiB into 200 KiB and makes a budget pass. :func:`rusage_unit_bytes` is the single place that decides,
:func:`to_bytes` applies it, and both are tested against the Linux and the macOS convention
(``test_ru_maxrss_unit_is_kib_on_linux_and_bytes_on_darwin``). ``process`` is additionally cross-checked against
the kernel's number for a *separate* process (``test_process_peak_agrees_with_the_kernel_number_of_a_separate_process``),
so the value is not merely self-consistent.
"""
from __future__ import annotations

import os
import resource
import sys
import threading
import time
import tracemalloc
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

MB = 1024 * 1024


@dataclass
class PeakMemory:
    """One measured peak, with what it covers and whether it can be trusted as an exact value."""
    bytes: int | None
    source: str                                   # cuda | torch+trace | torch | tracemalloc | process | rss | unknown
    exact: bool
    components: dict[str, int] = field(default_factory=dict)   # allocator -> bytes it contributed
    notes: str = ""

    @property
    def mib(self) -> float | None:
        return None if self.bytes is None else self.bytes / MB

    def to_dict(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "source": self.source, "exact": self.exact,
                "components": dict(self.components), "notes": self.notes}


def unavailable(reason: str) -> PeakMemory:
    """Peak that could not be measured. ``bytes`` stays ``None`` so no caller can mistake it for a number."""
    return PeakMemory(bytes=None, source="unknown", exact=False, notes=reason)


# --------------------------------------------------------------------------- CUDA
def cuda_peak(fn: Callable[[], Any]) -> int | None:
    """Peak allocated by the CUDA caching allocator during ``fn``; ``None`` when there is no GPU."""
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        return int(max(0, torch.cuda.max_memory_allocated() - base))
    except Exception:
        return None


# --------------------------------------------------------------------------- PyTorch CPU
def torch_peak(fn: Callable[[], Any]) -> int | None:
    """Peak CPU tensor memory during ``fn``, from the PyTorch profiler's allocation records.

    ``profile_memory=True`` records a signed ``self_cpu_memory_usage`` per event; walking the events in
    chronological order and keeping the running maximum of the sum gives the high-water mark of the CPU
    allocator. ``self_`` (not the inclusive ``cpu_memory_usage``) matters: the inclusive value is also reported
    on the parent ``aten::`` op, so summing it double-counts and reports exactly twice the true peak — measured,
    not assumed. Returns ``None`` when PyTorch is missing or the profiler fails, ``0`` when the op allocates no
    tensor memory.
    """
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        return None
    try:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
            fn()
        running = 0
        peak = 0
        for event in prof.events():
            running += int(getattr(event, "self_cpu_memory_usage", 0) or 0)
            peak = max(peak, running)
        return int(peak)
    except Exception:
        return None


# --------------------------------------------------------------------------- Python / NumPy
def traced_peak(fn: Callable[[], Any]) -> int | None:
    """Peak traced allocation during ``fn`` from ``tracemalloc``.

    Exact for objects the interpreter allocates and for NumPy arrays (``PyDataMem`` reports to tracemalloc, so a
    200 MiB array measures 200.0003 MiB). Blind to PyTorch CPU tensors, whose allocator bypasses it.
    """
    started = tracemalloc.is_tracing()
    if not started:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    try:
        fn()
        return int(max(0, tracemalloc.get_traced_memory()[1] - base))
    except Exception:
        return None
    finally:
        if not started:
            tracemalloc.stop()


# --------------------------------------------------------------------------- whole process
def process_peak(fn: Callable[[], Any]) -> int | None:
    """Whole-process peak of ``fn``, measured in a forked child by the kernel's RSS watermark.

    Exact, and it sees native allocations the Python-level methods miss, because the child is fresh: its
    ``ru_maxrss`` starts at its (copy-on-write) footprint instead of inheriting the parent's lifetime peak. That
    freshness is the whole point — in-process the watermark only fires once.

    Two limits, both measured: it needs ``fork`` (POSIX), and it cannot run autograd — PyTorch raises *"Unable to
    handle autograd's threading in combination with fork-based multiprocessing"*. Returns ``None`` in either case
    instead of a number; callers here treat it as a cross-check, not as the default.
    """
    if not hasattr(os, "fork") or sys.platform.startswith("win"):
        return None
    read_fd, write_fd = os.pipe()
    try:
        with warnings.catch_warnings():
            # "this process is multi-threaded, use of fork() may lead to deadlocks in the child" — the child here
            # allocates and exits, and its failure is reported through the pipe, so the warning is noise.
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
    except OSError:
        os.close(read_fd)
        os.close(write_fd)
        return None
    if pid == 0:                                              # child
        os.close(read_fd)
        try:
            base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            fn()
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base
            payload = f"ok {int(max(0, peak))}"
        except BaseException as exc:                          # report, never a silent zero
            payload = f"err {type(exc).__name__}: {exc}"
        try:
            os.write(write_fd, payload.encode())
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(read_fd)
        os.waitpid(pid, 0)
    text = b"".join(chunks).decode(errors="replace").strip()
    status, _, detail = text.partition(" ")
    if status == "ok" and detail.isdigit():
        return to_bytes(int(detail))
    return None


def rusage_unit_bytes(platform_name: str | None = None) -> int:
    """Bytes in one unit of ``getrusage().ru_maxrss`` on ``platform_name`` (``sys.platform`` by default).

    ``ru_maxrss`` is a *number of kilobytes* on Linux and a *number of bytes* on macOS and the BSDs; the two
    conventions differ by 1024, so the unit is decided in one function instead of being assumed at each call site.
    Anything else is treated as bytes (the POSIX-bytes convention), which is what CPython's own test suite does.
    """
    name = (sys.platform if platform_name is None else platform_name).lower()
    return 1024 if name.startswith("linux") else 1


def to_bytes(ru_maxrss: int, platform_name: str | None = None) -> int:
    """``ru_maxrss`` in bytes, with the platform's unit applied (``rusage_unit_bytes``)."""
    return int(ru_maxrss) * rusage_unit_bytes(platform_name)


# --------------------------------------------------------------------------- sampled RSS
class _PeakRSS:
    """Samples process RSS on a thread; ``peak`` is the increase over the starting RSS.

    A lower bound by construction: an allocation shorter than the sampling interval can be missed, and a reused
    arena never shows up at all. Kept for the case where neither tracemalloc nor the PyTorch profiler can be used.
    """

    def __init__(self, interval: float = 0.002):
        import psutil
        self._p = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self.peak = 0

    def __enter__(self):
        self._base = self._p.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._p.memory_info().rss - self._base)
            time.sleep(self._interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._p.memory_info().rss - self._base)


def rss_peak(fn: Callable[[], Any]) -> int | None:
    """Sampled RSS peak; ``None`` when ``psutil`` is not installed (it is not a dependency of this package)."""
    try:
        with _PeakRSS() as probe:
            fn()
        return int(probe.peak)
    except ImportError:
        return None


# --------------------------------------------------------------------------- entry point
def measure_peak(fn: Callable[[], Any], *, method: str = "auto") -> PeakMemory:
    """Peak memory of ``fn()``.

    ``method``:

    - ``"auto"`` (default): CUDA when available, otherwise the PyTorch profiler and ``tracemalloc`` summed.
      Neither is a guess and each one is exact for the allocations it can see.
    - ``"torch"``: PyTorch profiler only. Blind to NumPy and Python allocations, so a lower bound for mixed work.
    - ``"tracemalloc"``: interpreter and NumPy only. Blind to tensors, so a lower bound for mixed work.
    - ``"process"``: forked child, kernel watermark. Exact total; fails on autograd, so treat it as a cross-check.
    - ``"rss"``: sampled RSS. Lower bound; needs ``psutil``.

    The two CPU methods run in two passes, so neither perturbs the other; the sum is exact when the two
    allocators are live at the same time (the usual case for a training step, whose data arrays are resident
    while the network runs) and an upper bound when they take turns.
    """
    method = method.strip().lower()
    if method == "cuda":
        cuda = cuda_peak(fn)
        return unavailable("no CUDA device") if cuda is None else PeakMemory(cuda, "cuda", True, {"cuda": cuda})

    if method == "process":
        peak = process_peak(fn)
        return (unavailable("fork unavailable, or the operation cannot run in a forked child (autograd)")
                if peak is None else PeakMemory(peak, "process", True, {"process": peak}))

    if method == "rss":
        peak = rss_peak(fn)
        return (unavailable("psutil is not installed, so RSS cannot be sampled")
                if peak is None else PeakMemory(peak, "rss", False, {"rss": peak},
                                                notes="sampled RSS: a lower bound, short-lived allocations can be missed"))

    if method == "torch":
        peak = torch_peak(fn)
        return (unavailable("PyTorch is not installed") if peak is None
                else PeakMemory(peak, "torch", True, {"torch": peak}))

    if method == "tracemalloc":
        peak = traced_peak(fn)
        return (unavailable("tracemalloc could not trace the operation") if peak is None
                else PeakMemory(peak, "tracemalloc", True, {"python": peak}))

    if method != "auto":
        raise ValueError(f"unknown method {method!r}; use auto, cuda, process, rss, torch or tracemalloc")

    cuda = cuda_peak(fn)
    if cuda is not None:
        return PeakMemory(cuda, "cuda", True, {"cuda": cuda})
    torch_mem = torch_peak(fn)
    if torch_mem is None:                                     # no PyTorch: NumPy and Python are all there is
        traced = traced_peak(fn)
        rss = rss_peak(fn) if traced is None else None
        if traced is not None:
            return PeakMemory(traced, "tracemalloc", True, {"python": traced},
                              notes="no PyTorch: tracemalloc covers Python and NumPy, not native libraries")
        if rss is not None:
            return PeakMemory(rss, "rss", False, {"rss": rss}, notes="sampled RSS: a lower bound")
        return unavailable("no method could measure this operation")
    traced = traced_peak(fn) or 0
    total = int(torch_mem) + int(traced)
    if traced <= 0:
        return PeakMemory(total, "torch", True, {"torch": int(torch_mem)},
                          notes="PyTorch profiler: exact for CPU tensor allocations")
    components = {"torch": int(torch_mem), "python": int(traced)}
    return PeakMemory(total, "torch+trace", True, components,
                      notes=("PyTorch profiler plus tracemalloc, summed: exact when both allocators are live at "
                             "the same time, an upper bound when they take turns"))

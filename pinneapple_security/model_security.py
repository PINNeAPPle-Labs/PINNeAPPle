"""Model security: scan checkpoints for code execution before loading, load safely, and measure adversarial
sensitivity of surrogates.

PyTorch ``.pt``/``.pth`` files (and plain pickles) can run arbitrary code when loaded: the pickle stream names Python
callables that are called during unpickling. :func:`scan_checkpoint` reads the pickle opcodes *without executing them*
and lists every global the file would import; anything outside the allow-list (tensor rebuilding, OrderedDict, NumPy
array reconstruction) is reported, and known-dangerous ones (``os``, ``subprocess``, ``builtins.eval``...) make the file
unsafe. :func:`safe_load` refuses unsafe files and loads the rest with ``torch.load(weights_only=True)``.

:func:`adversarial_sensitivity` runs a projected-gradient (PGD, Madry et al. 2018, arXiv:1706.06083) search for the
input perturbation of size ``eps`` (relative to each input's range) that moves a surrogate's output the most: a large
change from a small, physically plausible perturbation means a noisy or manipulated sensor can steer the model.
"""
from __future__ import annotations

import io
import pickletools
import zipfile
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ScanReport", "scan_pickle_bytes", "scan_checkpoint", "safe_load", "UnsafeModelError", "adversarial_sensitivity"]

SAFE_PREFIXES = ("torch.", "collections.OrderedDict", "numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct",
                 "numpy.ndarray", "numpy.dtype", "numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar",
                 "_codecs.encode", "builtins.set", "builtins.frozenset", "builtins.slice", "builtins.complex",
                 "builtins.bytearray", "builtins.range")
DANGEROUS = ("os.", "posix.", "nt.", "subprocess.", "sys.", "socket.", "shutil.", "runpy.", "importlib.", "pty.",
             "webbrowser.", "requests.", "urllib.", "http.", "ctypes.", "pickle.", "marshal.", "code.", "commands.",
             "builtins.eval", "builtins.exec", "builtins.compile", "builtins.open", "builtins.__import__",
             "builtins.getattr", "builtins.setattr", "builtins.globals", "builtins.input", "__builtin__.")


class UnsafeModelError(Exception):
    pass


@dataclass
class ScanReport:
    path: str
    globals: set[str] = field(default_factory=set)
    dangerous: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return not self.dangerous and not self.errors

    @property
    def weights_only_loadable(self) -> bool:
        return self.safe and not self.unknown

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "safe": self.safe, "weights_only_loadable": self.weights_only_loadable,
                "dangerous": self.dangerous, "unknown": self.unknown, "errors": self.errors, "globals": sorted(self.globals)}


def _globals_in(data: bytes) -> set[str]:
    out: set[str] = set()
    strings: list[str] = []                    # recent string pushes, for STACK_GLOBAL (protocol >= 4)
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ("GLOBAL", "INST"):
            mod, name = str(arg).split(" ", 1) if " " in str(arg) else (str(arg), "")
            out.add(f"{mod}.{name}")
        elif op.name == "STACK_GLOBAL":
            if len(strings) >= 2:
                out.add(f"{strings[-2]}.{strings[-1]}")
            else:
                out.add("<unresolved STACK_GLOBAL>")
        elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strings.append(str(arg))
        elif op.name in ("MEMOIZE", "BINGET", "LONG_BINGET", "GET", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
    return out


def _classify(rep: ScanReport) -> ScanReport:
    for g in sorted(rep.globals):
        if any(g.startswith(d) or g == d.rstrip(".") for d in DANGEROUS) or g == "<unresolved STACK_GLOBAL>":
            rep.dangerous.append(g)
        elif not any(g.startswith(s) for s in SAFE_PREFIXES):
            rep.unknown.append(g)
    return rep


def scan_pickle_bytes(data: bytes, name: str = "<bytes>") -> ScanReport:
    rep = ScanReport(name)
    try:
        rep.globals |= _globals_in(data)
    except Exception as e:
        rep.errors.append(f"unreadable pickle stream: {e}")
    return _classify(rep)


def scan_checkpoint(path: str) -> ScanReport:
    """Scan a ``torch.save`` zip archive (every ``*.pkl`` member) or a legacy/plain pickle file."""
    rep = ScanReport(path)
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                pkls = [n for n in z.namelist() if n.endswith(".pkl")]
                if not pkls:
                    rep.errors.append("zip archive without a pickle (not a torch checkpoint)")
                for n in pkls:
                    rep.globals |= _globals_in(z.read(n))
        else:
            with open(path, "rb") as f:
                rep.globals |= _globals_in(f.read())
    except Exception as e:
        rep.errors.append(f"unreadable: {e}")
    return _classify(rep)


def safe_load(path: str, map_location: str = "cpu", expected_sha256: str | None = None):
    """Verify the digest (if given), scan, then ``torch.load(weights_only=True)``. Raises :class:`UnsafeModelError`."""
    import torch

    if expected_sha256 is not None:
        from .integrity import sha256_file
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise UnsafeModelError(f"{path}: sha256 {actual[:12]}… differs from the expected {expected_sha256[:12]}…")
    rep = scan_checkpoint(path)
    if not rep.safe:
        raise UnsafeModelError(f"{path} is unsafe to load: {', '.join(rep.dangerous + rep.errors)}")
    return torch.load(path, map_location=map_location, weights_only=True)


def adversarial_sensitivity(model, X, eps: float = 0.01, steps: int = 20, step_frac: float = 0.25,
                            input_range: tuple[Any, Any] | None = None) -> dict[str, Any]:
    """Worst-case relative output change under an L∞ input perturbation of ``eps`` × the input range (per feature).

    Returns ``max_rel_change`` (largest ‖f(x+δ)-f(x)‖ / ‖f(x)‖ over the samples), its mean, the random-noise baseline of
    the same size for comparison (``random_rel_change``), and the amplification ratio of the two."""
    import torch

    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    lo, hi = (X.min(0).values, X.max(0).values) if input_range is None else map(lambda v: torch.as_tensor(v, dtype=torch.float32), input_range)
    radius = eps * (hi - lo).clamp_min(1e-12)
    with torch.no_grad():
        y0 = model(X)
        y0 = y0.y if hasattr(y0, "y") else y0
    norm0 = y0.reshape(len(X), -1).norm(dim=1).clamp_min(1e-12)

    def rel(d):
        y = model(X + d)
        y = y.y if hasattr(y, "y") else y
        return (y - y0).reshape(len(X), -1).norm(dim=1) / norm0

    delta = (torch.rand_like(X) * 2 - 1) * radius
    for _ in range(steps):
        delta.requires_grad_(True)
        r = rel(delta).sum()
        (g,) = torch.autograd.grad(r, delta)
        with torch.no_grad():
            delta = torch.max(torch.min(delta + step_frac * radius * g.sign(), radius), -radius)
    with torch.no_grad():
        worst = rel(delta)
        rand = rel((torch.rand_like(X) * 2 - 1) * radius)
    return {"eps": eps, "max_rel_change": float(worst.max()), "mean_rel_change": float(worst.mean()),
            "random_rel_change": float(rand.mean()),
            "amplification": float(worst.mean() / rand.mean().clamp_min(1e-12)), "worst_index": int(worst.argmax())}

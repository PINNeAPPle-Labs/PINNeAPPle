"""``pinneapple_lab.experiment(...)``: wrap a script or an example so it leaves a manifest, a Lab item and a grade.

    from pinneapple_lab import experiment

    with experiment("plate_hole", title="Plate with a hole, Kt", domain="structural mechanics",
                    methods=["FEM (CalculiX)"], params={"mesh": 0.5}) as ex:
        kt = solve(...)
        ex.log_metric("Kt", kt, unit="", how="max von Mises / nominal stress at the hole edge")
        ex.check("kt_vs_peterson", value=kt, reference=3.03, rtol=0.03, kind="reference", label="Kt vs Peterson")
        fig.savefig("results/stress.png")          # recorded automatically (and stamped files too)
        ex.add_file("results/wake.mp4")             # movies and viewers

    @experiment("oscillator_demo", title="Damped oscillator", domain="system dynamics")
    def main(ex): ...

On exit it records the git commit, the environment and the wall time, writes ``experiment.json`` (default: the
script's folder), imports it into the Lab as an item (the ``manifest`` experiment, so its checks are graded A-F
with next steps), and re-curates. ``register=False`` writes the manifest only; ``curate=False`` skips the
re-curation. A failure inside the block is recorded (the Lab run fails, so the item is graded F) and re-raised.
"""
from __future__ import annotations

import functools
import inspect
import os
import platform
import sys
import time
from typing import Any

from .manifest import FIGURE_EXT, VIDEO_EXT, Manifest, _git, _iso, save, validate

__all__ = ["Capture", "experiment", "register_manifest"]


class Capture:
    def __init__(self, name: str, *, title: str = "", description: str = "", domain: str = "",
                 methods=(), modules=(), params: dict | None = None, out_dir: str | None = None,
                 links: dict | None = None, references=(), lab: Any = None, register: bool = True,
                 curate: bool = True):
        self.name, self.out_dir, self.lab, self.do_register, self.do_curate = name, out_dir, lab, register, curate
        self.manifest = Manifest(id=name, title=title or name.replace("_", " ").capitalize(), description=description,
                                 domain=domain, methods=list(methods), modules=list(modules),
                                 params=dict(params or {}), links=dict(links or {}), references=list(references))
        self.path: str | None = None
        self.item: str | None = None
        self.error: str | None = None
        self._orig = None

    # -- recording ---------------------------------------------------------------------------------------------
    def param(self, **kw) -> None:
        self.manifest.params.update(kw)

    def log_metric(self, name: str, value: float, unit: str = "", how: str = "") -> float:
        """Record a measured metric (a finite number, its unit, and how it was measured)."""
        self.manifest.metric(name, float(value), unit, how or "measured by the script")
        return value

    def check(self, name: str, passed: Any = None, *, value=None, reference=None, rtol=None, atol=None, max=None,
              min=None, detail: str = "", kind: str | None = None, label: str = "") -> bool:
        """A validation check, recorded as ``RunContext.check`` records it (same arguments, same pass rule)."""
        if passed is None:
            v, ok = float(value), True
            if reference is not None:
                ok &= abs(v - float(reference)) <= (atol or 0.0) + (rtol or 0.0) * abs(float(reference))
            if max is not None:
                ok &= v <= max
            if min is not None:
                ok &= v >= min
            passed = ok
        from .curation import infer_check_kind
        rec = {"name": name, "passed": bool(passed), "value": value, "reference": reference, "rtol": rtol,
               "atol": atol, "max": max, "min": min, "detail": detail}
        if label:
            rec["label"] = label
        rec["kind"] = kind or infer_check_kind(rec)
        self.manifest.checks.append(rec)
        return bool(passed)

    def add_file(self, path: str, kind: str | None = None) -> str:
        """Record a figure, movie, viewer or data file the experiment produced."""
        ext = os.path.splitext(path)[1].lower()
        kind = kind or ("videos" if ext in VIDEO_EXT else "figures" if ext in FIGURE_EXT else
                        "viewers" if ext == ".html" else "data")
        rel = os.path.relpath(os.path.abspath(path), self._base())
        if rel not in self.manifest.files[kind]:
            self.manifest.files[kind].append(rel)
        return path

    # -- the block ---------------------------------------------------------------------------------------------
    def _base(self) -> str:
        if self.out_dir is None:
            main = sys.modules.get("__main__")
            script = getattr(main, "__file__", None)
            self.out_dir = os.path.dirname(os.path.abspath(script)) if script else os.getcwd()
            if script:
                self.manifest.scripts.append(os.path.basename(script))
        return self.out_dir

    def __enter__(self) -> Capture:
        from matplotlib.figure import Figure
        self._base()
        self._t0 = time.time()
        self._orig = Figure.savefig
        cap = self

        def savefig(fig, fname, *a, **kw):
            out = cap._orig(fig, fname, *a, **kw)
            if isinstance(fname, (str, os.PathLike)):
                cap.add_file(os.fspath(fname))
            return out

        Figure.savefig = savefig
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        from matplotlib.figure import Figure
        if self._orig is not None:
            Figure.savefig = self._orig
        if exc is not None:
            self.error = f"{exc_type.__name__}: {exc}"
        self.manifest.provenance = {"source": "experiment()", "git": _git(), "created": _iso(self._t0),
                                    "seconds": round(time.time() - self._t0, 3), "error": self.error,
                                    "environment": {"python": platform.python_version(),
                                                    "platform": platform.platform()}}
        self.manifest.files = {k: [p for p in v if os.path.exists(os.path.join(self.out_dir, p))]
                               for k, v in self.manifest.files.items()}
        self.path = save(self.manifest, self.out_dir)
        if self.do_register:
            self.item = register_manifest(self.path, self.lab, curate=self.do_curate)
        return False                                       # never swallow the exception

    def __call__(self, fn):
        """Decorator form: the function gets the capture as its first argument (or none if it takes none)."""
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with self as ex:
                return fn(ex, *a, **kw) if inspect.signature(fn).parameters else fn(*a, **kw)
        return wrapper


def experiment(name: str, **kw) -> Capture:
    """A :class:`Capture`: use it as ``with experiment(...) as ex:`` or as ``@experiment(...)``."""
    return Capture(name, **kw)


def register_manifest(path: str, lab: Any = None, *, curate: bool = True) -> str:
    """Import a manifest into the Lab as an item (``manifest/<id>``, its checks graded A-F) and re-curate; returns
    the item key. Refuses an invalid manifest."""
    from .manifest import load
    from .runner import run
    from .store import LabStore
    m = load(path)
    base = os.path.dirname(os.path.abspath(path)) if path.endswith(".json") else os.path.abspath(path)
    errors = validate(m, base)
    if errors:
        raise ValueError(f"{path}: invalid manifest: " + "; ".join(errors))
    store = lab if isinstance(lab, LabStore) else LabStore(lab)
    p = os.path.abspath(path if path.endswith(".json") else os.path.join(path, "experiment.json"))
    run("manifest", {"id": m.id, "path": p}, root=store.root, force=True)
    if curate:
        from .curation import curate as _curate
        _curate(store)
    return f"manifest/{m.id}"

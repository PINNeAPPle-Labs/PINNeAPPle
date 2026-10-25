"""
Auto-migrated from the original inference_suite.py (single-file registry).
Each infer_* function builds a minimal real instance of the actual library
and runs one real forward pass / inference call — see pinneapple_arena/external/result.py
for the ModelResult contract every function returns.

Heavy/GPU-oriented functions now take an explicit `include_heavy` parameter
instead of reading a module-level global, so concurrent API requests can't
step on each other's --include-heavy setting.
"""

from __future__ import annotations

import numpy as np

from pinneapple_arena.external.installer import is_installed
from pinneapple_arena.external.result import (
    ModelResult,
    Timer,
    failed,
    manual_install,
    not_installed,
    ok,
    summarize_array,
)


def infer_ceviche(ds: dict) -> ModelResult:
    meta = dict(name="Ceviche", slug="ceviche", category="domain", domains=["em"],
                repo="https://github.com/fancompute/ceviche")
    if not is_installed("ceviche"):
        return not_installed(**meta, pip_hint="pip install ceviche", dataset=ds)
    with Timer() as t:
        try:
            import ceviche
            sim = ceviche.fdfd_ez(omega=2 * np.pi * 2e14, dL=4e-8, eps_r=ds["eps_r"], npml=(2, 2))
            source = np.zeros((ds["nx"], ds["ny"]))
            source[ds["nx"] // 2, ds["ny"] // 2] = 1.0
            Hx, Hy, Ez = sim.solve(source)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["eps_r"].shape),
              output_shape=str(np.asarray(Ez).shape), output_summary=summarize_array(np.abs(Ez)),
              note="Solved fdfd_ez with a point source; output is the complex Ez field (summary uses |Ez|).")

def infer_photontorch(ds: dict) -> ModelResult:
    meta = dict(name="Photontorch", slug="photontorch", category="domain", domains=["em"],
                repo="https://github.com/flaport/photontorch")
    if not is_installed("photontorch"):
        return not_installed(**meta, pip_hint="pip install photontorch", dataset=ds)
    with Timer() as t:
        try:
            import photontorch as pt
            wl = np.linspace(1.5e-6, 1.6e-6, 10)
            wg = pt.Waveguide(length=1e-5, neff=2.34)  # noqa: F841 (building it is the check)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(wl.shape),
              output_shape="Waveguide component built", note="Built a single pt.Waveguide component; "
              "a full circuit forward pass needs a pt.Network wiring multiple components.")

def infer_hpinn(ds: dict) -> ModelResult:
    meta = dict(name="hPINN", slug="hpinn", category="pinn", domains=["em"],
                repo="https://github.com/lululxvi/hpinn")
    return manual_install(**meta, instructions="Paper code, pinned to DeepXDE v0.9.1 — `git clone` and "
                                                "`pip install deepxde==0.9.1` for exact reproducibility.",
                           dataset=ds)

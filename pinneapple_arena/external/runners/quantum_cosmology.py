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


def infer_netket(ds: dict) -> ModelResult:
    meta = dict(name="NetKet", slug="netket", category="domain", domains=["quantum"],
                repo="https://github.com/netket/netket")
    if not is_installed("netket"):
        return not_installed(**meta, pip_hint="pip install netket", dataset=ds)
    with Timer() as t:
        try:
            import netket as nk
            hi = nk.hilbert.Spin(s=0.5, N=ds["n_sites"])
            ma = nk.models.RBM(alpha=1)
            sampler = nk.sampler.MetropolisLocal(hi)
            vs = nk.vqs.MCState(sampler, ma, n_samples=64)
            samples = vs.samples
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=f"{ds['n_sites']} spin-1/2 sites",
              output_shape=str(np.asarray(samples).shape), output_summary=summarize_array(np.asarray(samples)),
              note="Built an RBM neural quantum state over a Metropolis sampler and drew MC samples.")

def infer_ferminet(ds: dict) -> ModelResult:
    meta = dict(name="FermiNet", slug="ferminet", category="domain", domains=["quantum"],
                repo="https://github.com/google-deepmind/ferminet")
    return manual_install(**meta, instructions="Not on PyPI — `git clone` and `pip install -e .` per "
                                                "repo README (pulls in JAX, KFAC-JAX, PySCF, and other "
                                                "heavy dependencies regardless of --include-heavy).",
                           dataset=ds)

def infer_flowpm_jaxpm(ds: dict) -> ModelResult:
    meta = dict(name="FlowPM / JaxPM", slug="flowpm-jaxpm", category="diffsim", domains=["cosmology"],
                repo="https://github.com/modichirag/flowpm")
    return manual_install(**meta, instructions="Research code, not pip-packaged — `git clone` and install "
                                                "requirements (mesh-tensorflow or jax) per repo README.",
                           dataset=ds)

def infer_camels(ds: dict) -> ModelResult:
    meta = dict(name="CAMELS", slug="camels", category="domain", domains=["cosmology"],
                repo="https://github.com/franciscovillaescusa/CAMELS")
    return manual_install(**meta, instructions="Simulation suite + analysis scripts, not a pip package — "
                                                "`git clone` and download data per repo instructions.",
                           dataset=ds)

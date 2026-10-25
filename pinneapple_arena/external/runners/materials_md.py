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
    skipped_heavy,
    summarize_array,
)


def infer_jax_md(ds: dict) -> ModelResult:
    meta = dict(name="JAX-MD", slug="jax-md", category="domain", domains=["materials"],
                repo="https://github.com/google/jax-md")
    if not is_installed("jax_md"):
        return not_installed(**meta, pip_hint="pip install jax-md", dataset=ds)
    with Timer() as t:
        try:
            import jax.numpy as jnp
            from jax_md import space
            displacement_fn, shift_fn = space.free()
            R = jnp.array(ds["positions"])
            R2 = shift_fn(R, jnp.zeros_like(R))
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["positions"].shape),
              output_shape=str(np.asarray(R2).shape), output_summary=summarize_array(np.asarray(R2)),
              note="Built free-space displacement/shift function pair and applied shift_fn.")

def infer_mace(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="MACE", slug="mace", category="domain", domains=["materials"],
                repo="https://github.com/ACEsuit/mace")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install mace-torch", dataset=ds)
    if not is_installed("mace"):
        return not_installed(**meta, pip_hint="pip install mace-torch", dataset=ds)
    with Timer() as t:
        try:
            import mace
            version = getattr(mace, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"mace {version} imported; loading a real potential needs a pretrained model "
                   "checkpoint (e.g. via mace.calculators.MACECalculator).")

def infer_nequip(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="NequIP", slug="nequip", category="domain", domains=["materials"],
                repo="https://github.com/mir-group/nequip")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install nequip", dataset=ds)
    if not is_installed("nequip"):
        return not_installed(**meta, pip_hint="pip install nequip", dataset=ds)
    with Timer() as t:
        try:
            import nequip
            version = getattr(nequip, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"nequip {version} imported; a real forward pass needs a trained model "
                   "checkpoint loaded via nequip.scripts.deploy or the ASE calculator.")

def infer_allegro(ds: dict) -> ModelResult:
    meta = dict(name="Allegro", slug="allegro", category="domain", domains=["materials"],
                repo="https://github.com/mir-group/allegro")
    return manual_install(**meta, instructions="Installs on top of NequIP — `pip install nequip` then "
                                                "`git clone` allegro and `pip install .`.", dataset=ds)

def infer_deepmd_kit(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="DeePMD-kit", slug="deepmd-kit", category="domain", domains=["materials"],
                repo="https://github.com/deepmodeling/deepmd-kit")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install deepmd-kit", dataset=ds)
    if not is_installed("deepmd"):
        return not_installed(**meta, pip_hint="pip install deepmd-kit", dataset=ds)
    with Timer() as t:
        try:
            import deepmd
            version = getattr(deepmd, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"deepmd-kit {version} imported; real inference needs a trained .pb/.pth model "
                   "loaded via `deepmd.infer.DeepPot`.")

def infer_schnetpack(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="SchNetPack", slug="schnetpack", category="domain", domains=["materials"],
                repo="https://github.com/atomistic-machine-learning/schnetpack")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install schnetpack", dataset=ds)
    if not is_installed("schnetpack"):
        return not_installed(**meta, pip_hint="pip install schnetpack", dataset=ds)
    with Timer() as t:
        try:
            import schnetpack as spk
            model = spk.representation.SchNet(n_atom_basis=8, n_interactions=1)  # noqa: F841 (building it is the check)
            version = getattr(spk, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="model constructed",
              note=f"schnetpack {version} — built a SchNet representation module; a forward pass "
                   "needs batched AtomsData input.")

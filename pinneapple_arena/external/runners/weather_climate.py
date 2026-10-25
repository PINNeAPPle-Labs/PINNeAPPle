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

from pinneapple_arena.external.installer import is_installed
from pinneapple_arena.external.result import (
    ModelResult,
    Timer,
    failed,
    manual_install,
    not_installed,
    ok,
    skipped_heavy,
)


def infer_graphcast(ds: dict) -> ModelResult:
    meta = dict(name="GraphCast", slug="graphcast", category="graph", domains=["weather"],
                repo="https://github.com/google-deepmind/graphcast")
    return manual_install(**meta, instructions="Research code, not pip-packaged — `git clone` and install "
                                                "requirements (dm-haiku, jraph, jax). Pretrained weights are "
                                                "CC-BY-NC-SA.", dataset=ds)

def infer_pangu_weather(ds: dict) -> ModelResult:
    meta = dict(name="Pangu-Weather", slug="pangu-weather", category="domain", domains=["weather"],
                repo="https://github.com/198808xc/Pangu-Weather")
    return manual_install(**meta, instructions="Inference-only ONNX weights, non-commercial license — see "
                                                "github.com/ecmwf-lab/ai-models-panguweather for a "
                                                "pip-installable runner.", dataset=ds)

def infer_fourcastnet(ds: dict) -> ModelResult:
    meta = dict(name="FourCastNet", slug="fourcastnet", category="domain", domains=["weather"],
                repo="https://github.com/NVlabs/FourCastNet")
    return manual_install(**meta, instructions="Research code, not pip-packaged — `git clone` and follow "
                                                "repo README.", dataset=ds)

def infer_neuralgcm(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="NeuralGCM", slug="neuralgcm", category="domain", domains=["weather"],
                repo="https://github.com/google-research/neuralgcm")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install neuralgcm", dataset=ds)
    if not is_installed("neuralgcm"):
        return not_installed(**meta, pip_hint="pip install neuralgcm", dataset=ds)
    with Timer() as t:
        try:
            import neuralgcm
            version = getattr(neuralgcm, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"neuralgcm {version} imported; a real forecast needs a pretrained checkpoint "
                   "(gs://neuralgcm bucket) loaded per repo README.")

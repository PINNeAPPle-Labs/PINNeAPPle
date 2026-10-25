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
    not_installed,
    ok,
    skipped_heavy,
    summarize_array,
)


def infer_brax(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="Brax", slug="brax", category="rigidbody", domains=["rigidbody"],
                repo="https://github.com/google/brax")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install brax", dataset=ds)
    if not is_installed("brax"):
        return not_installed(**meta, pip_hint="pip install brax", dataset=ds)
    with Timer() as t:
        try:
            version = __import__("brax").__version__
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"brax {version} imported; a real step() needs a loaded MJCF/Brax system config.")

def infer_mujoco_mjx(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="MuJoCo MJX", slug="mujoco-mjx", category="rigidbody", domains=["rigidbody"],
                repo="https://github.com/google-deepmind/mujoco")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install mujoco", dataset=ds)
    if not is_installed("mujoco"):
        return not_installed(**meta, pip_hint="pip install mujoco", dataset=ds)
    with Timer() as t:
        try:
            import mujoco
            from mujoco import mjx
            xml = "<mujoco><worldbody><body><freejoint/><geom type='sphere' size='0.1'/></body></worldbody></mujoco>"
            model = mujoco.MjModel.from_xml_string(xml)
            mjx_model = mjx.put_model(model)
            data = mjx.make_data(mjx_model)
            data = mjx.forward(mjx_model, data)
            qpos = np.asarray(data.qpos)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape=str(qpos.shape),
              output_summary=summarize_array(qpos),
              note="Built a 1-body free-joint MJX model and ran one mjx.forward() step.")

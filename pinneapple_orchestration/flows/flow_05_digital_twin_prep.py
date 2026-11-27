"""flow_05_digital_twin_prep — build a live DigitalTwin from a
registry-stored, stage-gated model."""
from __future__ import annotations

from typing import Any

from pinneapple_registry import ArtifactRegistry
from pinneapple_systems.digital_twin import DigitalTwin

from .._prefect_compat import flow
from ..common.provenance import flow_run_manifest
from ..tasks import prepare_digital_twin_task


@flow(name="flow_05_digital_twin_prep")
def flow_digital_twin_prep(
    registry: ArtifactRegistry,
    problem_id: str,
    field_names: list[str],
    coord_names: list[str] | None = None,
    *,
    version: str | None = None,
    require_stage: str | None = "production",
    manifest_dir: str = "data/manifests",
    **init_kwargs: Any,
) -> dict[str, Any]:
    with flow_run_manifest(
        "flow_05_digital_twin_prep", {"problem_id": problem_id, "require_stage": require_stage}, manifest_dir=manifest_dir
    ) as manifest:
        twin: DigitalTwin = prepare_digital_twin_task(
            registry, problem_id, field_names, coord_names,
            version=version, require_stage=require_stage, **init_kwargs,
        )
        manifest.log_task("prepare_digital_twin", "ok")
        return {"twin": twin, "problem_id": problem_id, "run_id": manifest.run_id}

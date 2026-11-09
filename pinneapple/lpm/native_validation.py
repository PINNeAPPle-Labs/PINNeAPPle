"""Native VeriPhysics ValidationCase for LPM: evidence without invented approval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pinneapple_veriphysics.validation_case import (
    CheckVerdict, ModelVersion, ProblemRef, ValidationCase, RunProvenance,
)
from pinneapple_veriphysics.usecase import ValidationProfile


def evaluate(root, records, metrics, campaign):
    """Create an auditable native validation case.

    A holdout against the same numerical solver is *not* an independent
    physical reference. Consequently, this integration never automatically
    promotes a model to 'validated'. A separate reference and an authorized
    expert review are required by the native ValidationCase policy.
    """
    root=Path(root)
    checkpoint=root/"training"/"model.pt"
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    threshold=float(campaign.gates.get("max_relative_l2",0.5))
    profile=ValidationProfile(
        name=campaign.name,
        objective="industrial",
        primary_metric="relative_l2",
        target_error_pct=max(0.001,min(99.0,100*threshold)),
        reference_evidence=("experimental data",),
        required_checks=("holdout_relative_l2",),
        decision="surrogate-assisted engineering screening",
    )
    case=ValidationCase.open(
        profile=profile,
        model=ModelVersion(campaign.training.get("model","mlp"),version=digest[:12],checkpoint_sha256=digest),
        problem=ProblemRef(description=f"{campaign.family} / {campaign.solver}"),
        requested_use="engineering screening, subject to independent verification",
        actor="pinneapple.lpm",
    )
    value=float(metrics["test"]["relative_l2"])
    case=case.with_checks([CheckVerdict(
        check_id="holdout_relative_l2",category="verification",
        passed=value<=threshold,value=value,threshold=threshold,
        critical=True,evidence=str(root/"training"/"metrics.json"),
    )],actor="pinneapple.lpm")
    case=case.with_limitations([
        "Holdout comparisons are against the training solver, not independent physical measurements",
        "Mesh/time convergence, conservation and uncertainty calibration have not been demonstrated",
        "An authorized engineer must review the intended scope of use",
    ],actor="pinneapple.lpm")
    status=case.assess().status
    result=dict(passed=status=="validated",status=status,case_id=case.case_id,
                reason=case.assess().reason,
                revisions=[r.to_dict() for r in case.history])
    (root/"veriphysics_case.json").write_text(json.dumps(case.to_dict(),indent=2,default=str))\n    (root/"veriphysics_assessment.json").write_text(json.dumps(result,indent=2,default=str))
    return result

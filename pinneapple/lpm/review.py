"""Attach independent evidence and human review to a native VeriPhysics case."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from pinneapple_veriphysics.validation_case import (
    ValidationCase,ReferenceComparison,ExpertReview,
)


def _case(root):
    return ValidationCase.from_dict(json.loads((root/"veriphysics_case.json").read_text()))


def _save(root,case):
    (root/"veriphysics_case.json").write_text(json.dumps(case.to_dict(),indent=2,default=str))
    status=case.assess()
    (root/"veriphysics_assessment.json").write_text(json.dumps(
        dict(passed=status.status=="validated",status=status.status,
             reason=status.reason,case_id=case.case_id),indent=2))
    return status


def add_reference(root,artifact,kind,metric,value_pct,tolerance_pct,description):
    root=Path(root);artifact=Path(artifact)
    if not artifact.is_file():raise FileNotFoundError(artifact)
    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    case=_case(root)
    evidence=ReferenceComparison(
        evidence_id="reference-"+digest[:16],kind=kind,metric=metric,
        value_pct=float(value_pct),tolerance_pct=float(tolerance_pct),
        independent=True,description=description,
        source=str(artifact.resolve()),artefact_sha256=digest)
    case=case.with_reference(evidence,actor="lpm-evidence-import")
    return _save(root,case)


def review(root,reviewer,decision,justification,role,scope):
    root=Path(root);case=_case(root)
    decision_record=ExpertReview(
        reviewer=reviewer,decision=decision,justification=justification,
        role=role,scope=(("use",scope),))
    case=case.with_review(decision_record,actor=reviewer)
    status=_save(root,case)
    deployment=root/"deployment"
    cardpath=deployment/"model_card.json"
    if not cardpath.exists():raise RuntimeError("Train campaign before review")
    card=json.loads(cardpath.read_text())
    checkpoint=root/"training"/"model.pt"
    if not checkpoint.exists():raise FileNotFoundError(checkpoint)
    original=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if case.model.checkpoint_sha256!=original:
        raise RuntimeError("Checkpoint changed since validation; cannot approve")
    data=card.get("validation",{}).get("data",{})
    model=card.get("validation",{}).get("model",{})
    # The CLI does not authenticate reviewer identity or independently calculate\n    # reference errors. Never authorize deployment from self-attested evidence.\n    eligible=False
    card["status"]="blocked"\n    card["production_blocker"]="Unverified reviewer identity and independently uncomputed reference metric"
    card["native_veriphysics_status"]=status.status
    card["checkpoint_sha256"]=original
    deployment.mkdir(parents=True,exist_ok=True)
    cardpath.write_text(json.dumps(card,indent=2,default=str))
    if eligible:
        shutil.copy2(checkpoint,deployment/"approved_model.pt")
    else:
        (deployment/"approved_model.pt").unlink(missing_ok=True)
    return status


def main():
    parser=argparse.ArgumentParser(description="Native LPM evidence and expert review")
    parser.add_argument("root")
    sub=parser.add_subparsers(dest="action",required=True)
    ref=sub.add_parser("reference")
    ref.add_argument("--artifact",required=True)
    ref.add_argument("--kind",required=True)
    ref.add_argument("--metric",required=True)
    ref.add_argument("--value-pct",required=True,type=float)
    ref.add_argument("--tolerance-pct",required=True,type=float)
    ref.add_argument("--description",required=True)
    rv=sub.add_parser("review")
    rv.add_argument("--reviewer",required=True)
    rv.add_argument("--decision",required=True,choices=["approved","approved with conditions","rejected","more evidence requested","suspended"])
    rv.add_argument("--justification",required=True)
    rv.add_argument("--role",required=True)
    rv.add_argument("--scope",required=True)
    args=parser.parse_args()
    if args.action=="reference":
        status=add_reference(args.root,args.artifact,args.kind,args.metric,args.value_pct,args.tolerance_pct,args.description)
    else:
        status=review(args.root,args.reviewer,args.decision,args.justification,args.role,args.scope)
    print(status.status+": "+status.reason)


if __name__=="__main__":main()

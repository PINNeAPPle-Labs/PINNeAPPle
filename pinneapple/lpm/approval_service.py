"""Trusted LPM approval boundary.

Reviewers authenticate via a deployment-configured OIDC issuer. Independent
reference metrics are calculated from arrays, never accepted as user input.
A validated VeriPhysics case is necessary but not sufficient for promotion.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_reference(path: Path) -> tuple[np.ndarray,np.ndarray]:
    # NPZ arrays must be generated independently and aligned to the model
    # evaluation grid. Never allow pickle/object arrays.
    if not path.is_file() or path.suffix.lower()!=".npz":
        raise ValueError("Independent reference must be an existing .npz file")
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!={"prediction","reference"}:
            raise ValueError("Expected exactly prediction and reference arrays")
        prediction=data["prediction"].copy()
        reference=data["reference"].copy()
    if prediction.shape!=reference.shape or prediction.ndim<1 or not prediction.size:
        raise ValueError("Reference and prediction shapes must agree and be nonempty")
    if not all(np.isfinite(v).all() for v in (prediction,reference)):
        raise ValueError("Nonfinite independent reference data")
    return prediction.astype("float64"),reference.astype("float64")


def compute_reference(path: Path) -> dict:
    prediction,reference=_load_reference(path)
    diff=prediction-reference
    rel=float(np.linalg.norm(diff.ravel())/(np.linalg.norm(reference.ravel())+1e-12))
    return {"relative_l2":rel,"relative_l2_pct":100*rel,
            "rmse":float(np.sqrt(np.mean(diff*diff))),
            "mae":float(np.mean(abs(diff))),
            "sha256":_sha256(path),"samples":int(reference.size)}


def verify_identity(authorization: str | None) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise PermissionError("Missing bearer token")
    issuer=os.environ.get("PINNEAPPLE_LPM_OIDC_ISSUER")
    audience=os.environ.get("PINNEAPPLE_LPM_OIDC_AUDIENCE")
    jwks_url=os.environ.get("PINNEAPPLE_LPM_OIDC_JWKS_URL")
    reviewers=set(filter(None,os.environ.get("PINNEAPPLE_LPM_REVIEWER_SUBJECTS","").split(",")))
    if not issuer or not audience or not jwks_url or not reviewers:
        raise RuntimeError("OIDC issuer, audience, JWKS and reviewer allowlist must be configured")
    if not jwks_url.startswith("https://"):
        raise RuntimeError("OIDC JWKS URL must use HTTPS")
    import jwt
    from jwt import PyJWKClient
    token=authorization.removeprefix("Bearer ").strip()
    header=jwt.get_unverified_header(token)
    if header.get("alg")!="RS256":
        raise PermissionError("Only RS256 tokens are accepted")
    key=PyJWKClient(jwks_url,cache_jwk_set=True,lifespan=300).get_signing_key_from_jwt(token).key
    claims=jwt.decode(token,key,algorithms=["RS256"],issuer=issuer,audience=audience,
                      options={"require":["exp","iat","iss","aud","sub"]},leeway=30)
    if claims.get("sub") not in reviewers:
        raise PermissionError("Reviewer is not authorized")
    return claims


def _atomic_copy(src:Path,dst:Path):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent,delete=False) as temp:
        name=Path(temp.name)
        with src.open("rb") as source:shutil.copyfileobj(source,temp)
    os.replace(name,dst)


def approve(root: str | Path, reference_path: str | Path, authorization: str,
            justification: str, scope: str, expected_reference_sha256: str) -> dict:
    from pinneapple_veriphysics.validation_case import (
        ValidationCase,ReferenceComparison,ExpertReview,
    )
    claims=verify_identity(authorization)
    if not justification.strip() or not scope.strip():
        raise ValueError("Justification and scope are mandatory")
    root=Path(root).resolve()
    evidence=Path(reference_path).resolve()
    # Evidence must be independently supplied and immutable at approval time.
    checkpoint=root/"training"/"model.pt"\n    values=compute_reference(evidence,checkpoint)
    if not expected_reference_sha256 or values["sha256"]!=expected_reference_sha256:
        raise ValueError("Independent reference digest mismatch")
    card_path=root/"deployment"/"model_card.json"
    checkpoint=root/"training"/"model.pt"
    case_path=root/"veriphysics_case.json"
    if not all(p.is_file() for p in (card_path,checkpoint,case_path)):
        raise RuntimeError("Missing campaign model, model card or validation case")
    card=json.loads(card_path.read_text())
    case=ValidationCase.from_dict(json.loads(case_path.read_text()))
    if card.get("validation",{}).get("data",{}).get("passed") is not True:
        raise RuntimeError("Data quality gate failed")
    if card.get("validation",{}).get("model",{}).get("passed") is not True:
        raise RuntimeError("Held-out model gate failed")
    digest=_sha256(checkpoint)
    if case.model.checkpoint_sha256!=digest:
        raise RuntimeError("Checkpoint changed since VeriPhysics validation")
    threshold=float(card["validation"]["model"]["threshold"])*100
    comparison=ReferenceComparison(
        evidence_id="independent-"+values["sha256"][:16],
        kind="experimental data",metric="relative_l2",
        value_pct=values["relative_l2_pct"],tolerance_pct=threshold,
        independent=True,description="Independent reference array comparison",
        source=str(evidence),artefact_sha256=values["sha256"],
    )
    case=case.with_reference(comparison,actor=claims["sub"])
    review=ExpertReview(
        reviewer=claims["sub"],decision="approved",
        justification=justification,role="authenticated engineering reviewer",
        scope=(("use",scope),),
    )
    case=case.with_review(review,actor=claims["sub"])
    assessment=case.assess()
    if assessment.status!="validated":
        raise RuntimeError(f"VeriPhysics approval gate not met: {assessment.status}: {assessment.reason}")
    if values["relative_l2_pct"]>threshold:
        raise RuntimeError("Independent reference error exceeds threshold")
    # Write model last, only after all gates succeed.
    deployment=root/"deployment"
    _atomic_copy(checkpoint,deployment/"approved_model.pt")
    card.update(status="approved",native_veriphysics_status="validated",
                checkpoint_sha256=digest,reviewer_subject=claims["sub"],
                independent_reference=values,approval_scope=scope)
    card_path.write_text(json.dumps(card,indent=2))
    case_path.write_text(json.dumps(case.to_dict(),indent=2,default=str))
    (root/"veriphysics_assessment.json").write_text(json.dumps(
        {"status":assessment.status,"reason":assessment.reason,"reviewer":claims["sub"]},indent=2))
    return {"approved":True,"checkpoint_sha256":digest,"reference":values,
            "reviewer":claims["sub"],"case_id":case.case_id}


def create_app(campaign_root: str | Path):
    from fastapi import FastAPI,Header,HTTPException
    from pydantic import BaseModel,Field
    root=Path(campaign_root).resolve()
    app=FastAPI(title="PINNeAPPLe LPM Approval Service",docs_url=None,redoc_url=None)
    class ApprovalRequest(BaseModel):
        reference_path:str
        reference_sha256:str=Field(min_length=64,max_length=64)
        justification:str=Field(min_length=20,max_length=5000)
        scope:str=Field(min_length=3,max_length=1000)
    @app.post("/approve")
    def approve_model(body:ApprovalRequest,authorization:str|None=Header(default=None)):
        try:
            return approve(root,body.reference_path,authorization or "",body.justification,
                           body.scope,body.reference_sha256)
        except PermissionError as exc:raise HTTPException(status_code=403,detail=str(exc))
        except (ValueError,RuntimeError,FileNotFoundError) as exc:
            raise HTTPException(status_code=422,detail=str(exc))
    return app

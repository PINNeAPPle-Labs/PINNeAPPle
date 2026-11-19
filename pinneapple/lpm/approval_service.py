"""Authenticated, campaign-scoped LPM evidence ingestion and approval.

S3 Object Lock COMPLIANCE retains evidence; SQLite BEGIN IMMEDIATE serializes
approvals, audit events and unique campaign promotion. Approval is fail-closed.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from .evidence_store import EvidenceStore


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_identity(authorization):
    if not authorization or not authorization.startswith("Bearer "):
        raise PermissionError("Missing bearer token")
    issuer=os.environ.get("PINNEAPPLE_LPM_OIDC_ISSUER")
    audience=os.environ.get("PINNEAPPLE_LPM_OIDC_AUDIENCE")
    jwks_url=os.environ.get("PINNEAPPLE_LPM_OIDC_JWKS_URL")
    if not all((issuer,audience,jwks_url)) or not jwks_url.startswith("https://"):
        raise RuntimeError("OIDC configuration is incomplete")
    import jwt
    from jwt import PyJWKClient
    token=authorization[7:].strip()
    header=jwt.get_unverified_header(token)
    if header.get("alg")!="RS256":
        raise PermissionError("Unsupported JWT algorithm")
    key=PyJWKClient(jwks_url,cache_jwk_set=True,lifespan=300).get_signing_key_from_jwt(token).key
    return jwt.decode(token,key,algorithms=["RS256"],issuer=issuer,audience=audience,
        options={"require":["exp","iat","iss","aud","sub"]},leeway=30)


def authorize_campaign(claims,campaign,action):
    """Campaign-scoped RBAC from server-managed subject-to-campaign policy.

    Format: {"subject": {"campaign-name": ["ingest", "approve"]}}.
    No token-controlled role or wildcard campaign authorization is accepted.
    """
    policy=json.loads(os.environ.get("PINNEAPPLE_LPM_CAMPAIGN_ACL","{}"))
    subject=claims.get("sub")
    if not subject or action not in policy.get(subject,{}).get(campaign,[]):
        raise PermissionError(f"Not authorized to {action} campaign {campaign}")
    return subject


def _campaign(root):
    manifest=json.loads((root/"manifest.json").read_text())
    name=manifest["name"]
    if not name or "/" in name or "\\" in name or name in {".",".."}:
        raise ValueError("Invalid campaign name")
    return name


def _reference_metrics(blob,checkpoint):
    from .training import predict
    with np.load(io.BytesIO(blob),allow_pickle=False) as data:
        if set(data.files)!={"features","reference"}:
            raise ValueError("Evidence NPZ requires exactly features and reference")
        features=data["features"].copy()
        reference=data["reference"].copy()
    if features.ndim!=2 or reference.ndim!=2 or not features.size or not reference.size:
        raise ValueError("Evidence arrays must be nonempty two-dimensional arrays")
    if features.shape[0]>100000 or features.shape[1]>128:
        raise ValueError("Evidence array exceeds configured size limits")
    if not all(np.isfinite(x).all() for x in (features,reference)):
        raise ValueError("Evidence contains nonfinite values")
    prediction=predict(checkpoint,features)[0]
    if prediction.shape!=reference.shape:raise ValueError("Reference shape does not match prediction")
    diff=prediction.astype("float64")-reference.astype("float64")
    rel=float(np.linalg.norm(diff.ravel())/(np.linalg.norm(reference.ravel())+1e-12))
    return dict(relative_l2=rel,relative_l2_pct=100*rel,
                rmse=float(np.sqrt(np.mean(diff**2))),
                mae=float(np.mean(abs(diff))),samples=int(reference.size))


def ingest(root,authorization,blob,retention_days=365):
    root=Path(root).resolve()
    claims=verify_identity(authorization)
    campaign=_campaign(root)
    actor=authorize_campaign(claims,campaign,"ingest")
    # Check NPZ is well-formed before sending to the immutable store.
    with np.load(io.BytesIO(blob),allow_pickle=False) as data:
        if set(data.files)!={"features","reference"}:
            raise ValueError("Expected features and reference")
    return EvidenceStore().ingest(campaign,actor,blob,retention_days)


def approve(root,evidence_id,authorization,justification,scope):
    from pinneapple_veriphysics.validation_case import (
        ValidationCase,ReferenceComparison,ExpertReview,
    )
    root=Path(root).resolve()
    claims=verify_identity(authorization)
    campaign=_campaign(root)
    actor=authorize_campaign(claims,campaign,"approve")
    if len(justification.strip())<20 or not scope.strip():
        raise ValueError("Meaningful justification and scope required")
    store=EvidenceStore()
    card_path=root/"deployment"/"model_card.json"
    checkpoint=root/"training"/"model.pt"
    case_path=root/"veriphysics_case.json"
    if not all(p.is_file() for p in (card_path,checkpoint,case_path)):
        raise RuntimeError("Incomplete training or validation artifacts")
    # BEGIN IMMEDIATE: only one writer can promote a campaign at a time.
    with store.transaction() as db:
        if db.execute("SELECT 1 FROM approvals WHERE campaign=?",(campaign,)).fetchone():
            raise RuntimeError("Campaign already approved; new model requires a new campaign revision")
        card=json.loads(card_path.read_text())
        case=ValidationCase.from_dict(json.loads(case_path.read_text()))
        if card.get("status")=="approved":
            raise RuntimeError("Already approved")
        if card.get("validation",{}).get("data",{}).get("passed") is not True:
            raise RuntimeError("Data gate failed")
        if card.get("validation",{}).get("model",{}).get("passed") is not True:
            raise RuntimeError("Holdout gate failed")
        digest=_sha256(checkpoint)
        if case.model.checkpoint_sha256!=digest:
            raise RuntimeError("Checkpoint does not match VeriPhysics case")
        blob,evidence_hash=store.load(db,campaign,evidence_id)
        evidence_owner=db.execute("SELECT uploaded_by FROM evidence WHERE id=?",(evidence_id,)).fetchone()[0]
        if evidence_owner==actor:
            raise PermissionError("Approver must differ from evidence uploader")
        metrics=_reference_metrics(blob,checkpoint)
        threshold=float(card["validation"]["model"]["threshold"])*100
        if metrics["relative_l2_pct"]>threshold:
            raise RuntimeError("Independent comparison exceeds accepted error threshold")
        comparison=ReferenceComparison(
            evidence_id="independent-"+evidence_hash[:16],
            kind="experimental data",metric="relative_l2",
            value_pct=metrics["relative_l2_pct"],tolerance_pct=threshold,
            independent=True,description="Approved external reference comparison",
            source=f"s3-object-lock:{evidence_id}",artefact_sha256=evidence_hash,
        )
        case=case.with_reference(comparison,actor=actor)
        case=case.with_review(ExpertReview(
            reviewer=actor,decision="approved",justification=justification,
            role="OIDC authenticated campaign reviewer",scope=(("use",scope),),
        ),actor=actor)
        assessment=case.assess()
        if assessment.status!="validated":
            raise RuntimeError(f"Native VeriPhysics not validated: {assessment.status}: {assessment.reason}")
        # Serialize a durable approval decision before publication. The
        # inference service also checks the checkpoint digest and model card.
        db.execute("INSERT INTO approvals VALUES(?,?,?,?,?)",
            (campaign,digest,evidence_id,actor,__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()))
        store.audit(db,campaign,actor,"approval.granted",
            dict(checkpoint_sha256=digest,evidence_id=evidence_id,metrics=metrics,scope=scope))
        deployment=root/"deployment"
        deployment.mkdir(parents=True,exist_ok=True)
        def atomic_write(path,data):
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as fh:
                temp=Path(fh.name);fh.write(data);fh.flush();os.fsync(fh.fileno())
            os.replace(temp,path)
        # Model card is written last: inference cannot see an approved card
        # before the approved checkpoint exists.
        atomic_write(deployment/"approved_model.pt",checkpoint.read_bytes())
        card.update(status="approved",native_veriphysics_status="validated",
            checkpoint_sha256=digest,reviewer_subject=actor,
            independent_reference=metrics,approval_scope=scope,
            evidence_id=evidence_id)
        atomic_write(case_path,json.dumps(case.to_dict(),indent=2,default=str).encode())
        atomic_write(card_path,json.dumps(card,indent=2).encode())
    return dict(approved=True,campaign=campaign,reviewer=actor,
                checkpoint_sha256=digest,evidence_id=evidence_id,metrics=metrics)


def create_app(campaign_root):
    from fastapi import FastAPI,Header,HTTPException,UploadFile,File
    from pydantic import BaseModel,Field
    root=Path(campaign_root).resolve()
    app=FastAPI(title="PINNeAPPLe LPM Approval Service",docs_url=None,redoc_url=None,openapi_url=None)
    class ApprovalRequest(BaseModel):
        evidence_id:str=Field(min_length=32,max_length=32)
        justification:str=Field(min_length=20,max_length=5000)
        scope:str=Field(min_length=3,max_length=1000)
    @app.post("/evidence")
    async def ingest_evidence(file:UploadFile=File(...),authorization:str|None=Header(default=None)):
        try:
            # Bound upload before allocating arbitrary-sized request bodies.
            blob=await file.read(100*1024*1024+1)
            if len(blob)>100*1024*1024:raise ValueError("Evidence exceeds 100 MiB")
            return ingest(root,authorization or "",blob)
        except PermissionError as exc:raise HTTPException(status_code=403,detail=str(exc))
        except (ValueError,RuntimeError) as exc:raise HTTPException(status_code=422,detail=str(exc))
    @app.post("/approve")
    def approve_model(body:ApprovalRequest,authorization:str|None=Header(default=None)):
        try:return approve(root,body.evidence_id,authorization or "",body.justification,body.scope)
        except PermissionError as exc:raise HTTPException(status_code=403,detail=str(exc))
        except (ValueError,RuntimeError) as exc:raise HTTPException(status_code=422,detail=str(exc))
    return app

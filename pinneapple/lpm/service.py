"""Fail-closed inference service; authentication and provenance required."""
import hashlib
import hmac
import json
import os
from pathlib import Path

def create_app(deployment_dir):
    from fastapi import FastAPI,HTTPException,Header
    from pydantic import BaseModel,Field
    import numpy as np
    from .training import predict
    root=Path(deployment_dir)
    card_path=root/"model_card.json"
    checkpoint=root/"approved_model.pt"
    if not card_path.is_file() or not checkpoint.is_file():
        raise RuntimeError("No approved deployment artifacts")
    card=json.loads(card_path.read_text())
    if card.get("status")!="approved" or card.get("native_veriphysics_status")!="validated":
        raise RuntimeError("Missing native VeriPhysics production approval")
    expected=card.get("checkpoint_sha256")
    if not expected or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected:
        raise RuntimeError("Checkpoint integrity verification failed")
    api_key=os.environ.get("PINNEAPPLE_LPM_API_KEY")
    if not api_key or len(api_key)<32:
        raise RuntimeError("PINNEAPPLE_LPM_API_KEY must be configured with at least 32 characters")
    max_nodes=int(os.environ.get("PINNEAPPLE_LPM_MAX_NODES","50000"))
    if not 0<max_nodes<=1000000:raise RuntimeError("Invalid PINNEAPPLE_LPM_MAX_NODES")
    app=FastAPI(title="PINNeAPPLe LPM",docs_url=None,redoc_url=None,openapi_url=None)
    class Request(BaseModel):
        features:list[list[float]]=Field(min_length=1,max_length=max_nodes)
    def authorize(key):
        if not key or not hmac.compare_digest(key,api_key):
            raise HTTPException(status_code=401,detail="Unauthorized")
    @app.get("/health")
    def health():
        return {"status":"ok"}
    @app.post("/predict")
    def infer(body:Request,x_api_key:str|None=Header(default=None)):
        authorize(x_api_key)
        try:
            x=np.asarray(body.features,dtype="float32")
            if x.ndim!=2 or not np.isfinite(x).all():
                raise ValueError("Expected finite [nodes, features]")
            return {"values":predict(checkpoint,x)[0].tolist(),"campaign":card["campaign"]}
        except (ValueError,TypeError,RuntimeError) as exc:
            raise HTTPException(status_code=422,detail=str(exc))
    return app

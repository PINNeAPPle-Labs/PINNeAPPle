"""Optional inference API, available only for approved deployments."""
import json
from pathlib import Path
import numpy as np
from .training import predict

def create_app(deployment_dir):
    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel
    root=Path(deployment_dir)
    if not (root/"approved_model.pt").is_file():raise RuntimeError("No approved model checkpoint")
    card=json.loads((root/"model_card.json").read_text())
    if card.get("status")!="approved":raise RuntimeError("Model card not approved")
    app=FastAPI(title="PINNeAPPLe LPM")
    class Request(BaseModel):
        features:list[list[float]]
    @app.get("/health")
    def health():return {"status":"ok","campaign":card["campaign"],"model":card["model"]}
    @app.post("/predict")
    def infer(body:Request):
        try:
            x=np.asarray(body.features,dtype="float32")
            if x.ndim!=2 or not np.isfinite(x).all():raise ValueError("Expected finite [nodes, features]")
            return {"values":predict(root/"approved_model.pt",x)[0].tolist(),"campaign":card["campaign"]}
        except Exception as exc:raise HTTPException(status_code=422,detail=str(exc))
    return app

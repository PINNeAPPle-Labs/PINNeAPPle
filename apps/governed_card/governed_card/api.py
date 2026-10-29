"""GovernedCard: model cards that cannot be issued without a physics check of the model behind them.

``POST /v1/generate_verified_card`` (job) runs ``pinneapple_hub.verified_card.certify_model``: train a model of a
registered preset (or load uploaded ``state_dict`` weights), run ``PhysicsGuardrail``, and build a ``ModelCard``
whose ``validation_metrics`` are only that run's numbers; a card that fails ``ModelCard.validate()`` is refused.
The finished job is the audit trail (``GET /v1/compliance_report/{id}``: every check, value, threshold, time).
``POST /v1/audit_card`` re-runs the schema check on any card JSON; it cannot re-execute a card's physics claims,
because the weights and preset kwargs do not travel with the card.

How it relates to the other apps: ``apps/model_gateway`` / ``apps/hub`` publish and serve models with their cards;
``apps/validation_lab`` validates results and models in depth with an evidence ledger. This app is the narrow
"train or load, guardrail, card or refusal" service.

Run:  uvicorn governed_card.api:app --port 8148   (from apps/governed_card)

Environment (all optional; the rest in apps/_shared/README.md):
  GCARD_USER / GCARD_PASSWORD(_HASH)   login on everything except /health
  GCARD_FREE_CARDS, GCARD_FREE_AUDITS  free cards (20) and audits (1000) per caller and month
  GCARD_MAX_EPOCHS                     largest training budget per job (default 3000)
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
MAX_EPOCHS = int(os.environ.get("GCARD_MAX_EPOCHS") or 3000)
LIMITATION = ("Schema check only (ModelCard.validate()): it confirms validation_metrics are present and "
              "reference_source is cited. It cannot re-execute the physics claims in the card without the "
              "original weights and preset kwargs, which a card JSON does not carry.")

app = FastAPI(title="GovernedCard", version=VERSION,
              description="Model cards backed by a guardrail run against the model, refused otherwise.")
install(app, prefix="GCARD")
rt = install_runtime(app, prefix="GCARD", name="governed_card", units={"cards": 20, "audits": 1000}, max_jobs=1)
WEIGHTS = rt.root / "weights"


def card_job(p: dict, ctx) -> dict:
    from pinneapple_core.jsonable import to_jsonable
    from pinneapple_hub.verified_card import CardRefused, certify_model

    weights = p.pop("weights_path", None)
    ctx.progress(0.05, "loading weights" if weights else "training")
    try:
        state = open(weights, "rb").read() if weights else None
        out = certify_model(state_dict=state, **p)
    except CardRefused as e:
        return {"issued": False, "refused": e.problems}
    finally:
        if weights:
            try:
                os.unlink(weights)
            except OSError:
                pass
    card = out.pop("card")
    return to_jsonable({"issued": True, "model_card": card.to_dict(), "model_card_markdown": card.to_markdown(),
                        **out, "created_at": datetime.now(timezone.utc).isoformat()})


rt.jobs.register("card", card_job)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.post("/v1/generate_verified_card")
async def generate(request: Request, preset: str = Form(...), name: str = Form(..., min_length=1, max_length=200),
                   preset_kwargs_json: str = Form("{}"), description: str = Form(""),
                   architecture: str = Form("modified_mlp"), architecture_config_json: str = Form("{}"),
                   train_epochs: int = Form(300, ge=1), n_collocation: int = Form(1024, ge=64, le=8192),
                   residual_threshold: float = Form(1e-2, gt=0), reference_benchmark: str | None = Form(None),
                   license: str = Form("apache-2.0"), citation: str = Form(""), tags_csv: str = Form(""),
                   seed: int | None = Form(None), state_dict_file: UploadFile | None = File(None)):  # noqa: B008
    import pinneapple_physics as pp

    try:
        kwargs, config = json.loads(preset_kwargs_json), json.loads(architecture_config_json)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"preset_kwargs_json / architecture_config_json must be JSON: {e}") from None
    if not isinstance(kwargs, dict) or not isinstance(config, dict):
        raise HTTPException(422, "preset_kwargs_json and architecture_config_json must be JSON objects")
    try:
        pp.get_preset(preset, **kwargs)
    except (KeyError, TypeError) as e:
        raise HTTPException(422, str(e)) from None
    params: dict[str, Any] = {"preset": preset, "name": name, "preset_kwargs": kwargs, "architecture": architecture,
                              "architecture_config": config, "train_epochs": min(train_epochs, MAX_EPOCHS),
                              "n_collocation": n_collocation, "residual_threshold": residual_threshold,
                              "reference_benchmark": reference_benchmark or None, "seed": seed,
                              "description": description, "citation": citation, "license": license,
                              "tags": [t.strip() for t in tags_csv.split(",") if t.strip()]}
    if state_dict_file is not None:
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        path = WEIGHTS / f"{uuid.uuid4().hex}.pt"
        path.write_bytes(await state_dict_file.read())
        params["weights_path"] = str(path)
    return rt.submit(request, "card", params, units={"cards": 1})


@app.get("/v1/compliance_report/{job_id}")
def compliance_report(request: Request, job_id: str):
    return rt.job_for(request, job_id)


@app.post("/v1/audit_card")
def audit_card(request: Request, card: dict[str, Any]):
    from pinneapple_hub.model_card import ModelCard

    rt.consume(request, "audits")
    try:
        problems = ModelCard(**card).validate()
    except TypeError as e:
        return {"schema_valid": False, "schema_problems": [f"not a ModelCard: {e}"], "limitation": LIMITATION,
                "card_name": card.get("name")}
    return {"schema_valid": not problems, "schema_problems": problems, "limitation": LIMITATION,
            "card_name": card.get("name")}

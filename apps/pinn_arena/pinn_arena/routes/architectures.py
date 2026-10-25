"""Architectures: the catalog (``pinneapple_arena.pointwise.ARCHITECTURES_CATALOG``) with filters, a schematic
layer diagram per family (illustrative, not read from the model), and thumbs-up/down feedback.

Each entry carries ``registered``: whether the PINNeAPPle registry has the model. A registered model can still fail
to build for pointwise data; an experiment then reports it as failed with the reason (nothing else is trained in
its place).
"""
# ruff: noqa: B008  (FastAPI Depends/Query/File defaults)
from __future__ import annotations

import functools
import logging

from fastapi import APIRouter, HTTPException, Query

from pinn_arena.db import SessionLocal
from pinn_arena.orm import ArchitectureFeedback
from pinn_arena.schemas import ArchitectureFamily, ArchitectureFeedbackRequest
from pinneapple_arena.pointwise.architectures import ARCHITECTURES_CATALOG, catalog_status
from pinneapple_arena.pointwise.problems import PROBLEMS_CATALOG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/architectures", tags=["architectures"])


@functools.lru_cache(maxsize=1)
def _registered() -> dict[str, bool]:
    return catalog_status()


def _dump(arch) -> dict:
    return {**arch.model_dump(), "registered": _registered().get(arch.id, False)}


@router.get("", summary="List all neural architectures")
async def list_architectures(
    family: ArchitectureFamily | None = Query(None, description="Filter by architecture family"),
    supports_physics_loss: bool | None = Query(None, description="Filter by physics loss support"),
    supports_3d: bool | None = Query(None, description="Filter by 3D support"),
    inverse_only: bool | None = Query(None, description="If true, only inverse-only architectures; if false, only non-inverse-only ones"),
    problem_id: str | None = Query(None, description="Filter architectures recommended for a problem"),
):
    """
    List all available neural architectures with optional filtering.

    Filters:
    - family: PINN, NEURAL_OPERATOR, TRANSFORMER, GRAPH, etc.
    - supports_physics_loss: only architectures that can use PDE residual loss
    - supports_3d: only architectures that handle 3D inputs natively
    - inverse_only: filter by inverse_only flag
    - problem_id: only architectures listed in best_for for that problem
    """
    results = list(ARCHITECTURES_CATALOG.values())

    if family is not None:
        results = [a for a in results if a.family == family]

    if supports_physics_loss is not None:
        results = [a for a in results if a.supports_physics_loss == supports_physics_loss]

    if supports_3d is not None:
        results = [a for a in results if a.supports_3d == supports_3d]

    if inverse_only is not None:
        results = [a for a in results if a.inverse_only == inverse_only]

    if problem_id is not None:
        if problem_id not in PROBLEMS_CATALOG:
            raise HTTPException(
                status_code=404,
                detail=f"Problem '{problem_id}' not found in catalog.",
            )
        results = [a for a in results if problem_id in (a.best_for or [])]

    return {
        "total": len(results),
        "architectures": [_dump(a) for a in results],
    }


@router.get("/families", summary="List available architecture families")
async def list_families():
    """
    Return the list of distinct architecture families present in the catalog,
    with a count of architectures in each family.
    """
    counts: dict[str, int] = {}
    for arch in ARCHITECTURES_CATALOG.values():
        key = arch.family.value
        counts[key] = counts.get(key, 0) + 1

    return {
        "families": [
            {"family": k, "count": v}
            for k, v in sorted(counts.items())
        ]
    }


@router.get("/{architecture_id}", summary="Get a single architecture")
async def get_architecture(architecture_id: str):
    """
    Get the full metadata for a specific neural architecture by ID.

    Returns family, description, physics loss support, 3D support,
    recommended problems, default hyperparameters, and paper reference.
    """
    arch = ARCHITECTURES_CATALOG.get(architecture_id)
    if arch is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Architecture '{architecture_id}' not found. "
                f"Available: {list(ARCHITECTURES_CATALOG.keys())}"
            ),
        )
    return _dump(arch)


@router.get("/{architecture_id}/recommended-problems", summary="Get problems recommended for an architecture")
async def get_recommended_problems(architecture_id: str):
    """
    Return the physics problems that the architecture performs well on,
    cross-referenced with the problems catalog.
    """
    arch = ARCHITECTURES_CATALOG.get(architecture_id)
    if arch is None:
        raise HTTPException(status_code=404, detail=f"Architecture '{architecture_id}' not found.")

    recommended = []
    for pid in (arch.best_for or []):
        problem = PROBLEMS_CATALOG.get(pid)
        if problem is not None:
            recommended.append({
                "id": problem.id,
                "name": problem.name,
                "domain": problem.domain.value,
                "difficulty": problem.difficulty.value,
            })

    return {
        "architecture_id": architecture_id,
        "recommended_problems": recommended,
    }


@router.get("/{architecture_id}/layers", summary="Get schematic layer structure for diagram rendering")
async def get_architecture_layers(architecture_id: str):
    """
    Return a schematic node/edge list describing the architecture's layer structure.
    Clients use this to render an SVG block diagram without loading the full PyTorch model.
    """
    arch = ARCHITECTURES_CATALOG.get(architecture_id)
    if arch is None:
        raise HTTPException(status_code=404, detail=f"Architecture '{architecture_id}' not found.")

    params = arch.parameters
    family = arch.family.value
    nodes = []
    edges = []

    def _node(nid: str, label: str, sublabel: str = "", ntype: str = "block"):
        nodes.append({"id": nid, "label": label, "sublabel": sublabel, "type": ntype})

    def _edge(src: str, dst: str):
        edges.append({"from": src, "to": dst})

    _node("input", "Input", f"in_dim={params.get('in_dim', 'D')}", "input")

    if family == "PINN":
        hidden = params.get("hidden_dim", 128)
        n_layers = params.get("n_layers", 4)
        act = str(params.get("activation", "tanh"))
        prev = "input"
        for i in range(n_layers):
            lid = f"fc{i+1}"
            _node(lid, f"Linear({hidden}→{hidden})", act, "linear")
            _edge(prev, lid)
            prev = lid
        _node("output", "Output", f"out_dim={params.get('out_dim', 1)}", "output")
        _edge(prev, "output")

    elif family == "NEURAL_OPERATOR":
        modes = params.get("modes", params.get("modes1", 16))
        hidden = params.get("hidden_channels", params.get("hidden_dim", 64))
        n_layers = params.get("n_layers", 4)
        prev = "input"
        _node("lift", f"Lifting({hidden})", "pointwise linear", "linear")
        _edge(prev, "lift")
        prev = "lift"
        for i in range(n_layers):
            lid = f"fourier{i+1}"
            _node(lid, "Fourier Layer", f"modes={modes}", "block")
            _edge(prev, lid)
            prev = lid
        _node("proj", f"Projection({hidden}→out)", "pointwise linear", "linear")
        _edge(prev, "proj")
        _node("output", "Output", "", "output")
        _edge("proj", "output")

    elif family == "TRANSFORMER":
        d_model = params.get("d_model", 128)
        n_heads = params.get("n_heads", 8)
        n_layers = params.get("n_layers", 6)
        prev = "input"
        _node("embed", f"Embedding({d_model})", "", "linear")
        _edge(prev, "embed")
        prev = "embed"
        for i in range(n_layers):
            bid = f"block{i+1}"
            _node(bid, f"Transformer Block {i+1}", f"{n_heads} heads · d={d_model}", "attention")
            _edge(prev, bid)
            prev = bid
        _node("head", "MLP Head", "", "linear")
        _edge(prev, "head")
        _node("output", "Output", "", "output")
        _edge("head", "output")

    elif family == "RECURRENT":
        hidden = params.get("hidden_dim", 128)
        n_layers = params.get("n_layers", 2)
        prev = "input"
        for i in range(n_layers):
            rid = f"rnn{i+1}"
            cell = "LSTM" if "lstm" in architecture_id else "GRU"
            _node(rid, f"{cell} Layer {i+1}", f"hidden={hidden}", "recurrent")
            _edge(prev, rid)
            prev = rid
        _node("output", "Output", "", "output")
        _edge(prev, "output")

    elif family == "CONVOLUTIONAL":
        n_filters = params.get("n_filters", 64)
        n_layers = params.get("n_layers", 4)
        prev = "input"
        for i in range(n_layers):
            cid = f"conv{i+1}"
            dim = "3D" if "3d" in architecture_id else ("2D" if "2d" in architecture_id else "1D")
            _node(cid, f"Conv{dim}({n_filters})", "BatchNorm + ReLU", "conv")
            _edge(prev, cid)
            prev = cid
        _node("pool", "Global Avg Pool", "", "block")
        _edge(prev, "pool")
        _node("output", "Output", "", "output")
        _edge("pool", "output")

    elif family == "GRAPH":
        hidden = params.get("hidden_dim", params.get("node_dim", 128))
        n_mp = params.get("n_message_passing", params.get("n_layers", 6))
        prev = "input"
        _node("enc", "Node Encoder", f"hidden={hidden}", "linear")
        _edge(prev, "enc")
        for i in range(min(n_mp, 6)):
            mid = f"mp{i+1}"
            _node(mid, f"Message Passing {i+1}", "agg=sum", "block")
            _edge("enc" if i == 0 else f"mp{i}", mid)
        _node("dec", "Node Decoder", "", "linear")
        _edge(f"mp{min(n_mp, 6)}", "dec")
        _node("output", "Output", "", "output")
        _edge("dec", "output")

    elif family == "AUTOENCODER":
        hidden = params.get("hidden_dim", params.get("latent_dim", 64))
        prev = "input"
        _node("enc1", "Encoder Block 1", f"dim→{hidden*2}", "linear")
        _edge(prev, "enc1")
        _node("enc2", "Encoder Block 2", f"{hidden*2}→{hidden}", "linear")
        _edge("enc1", "enc2")
        _node("latent", "Latent Space", f"dim={hidden}", "block")
        _edge("enc2", "latent")
        _node("dec1", "Decoder Block 1", f"{hidden}→{hidden*2}", "linear")
        _edge("latent", "dec1")
        _node("dec2", "Decoder Block 2", f"{hidden*2}→out", "linear")
        _edge("dec1", "dec2")
        _node("output", "Output", "", "output")
        _edge("dec2", "output")

    elif family == "CONTINUOUS":
        hidden = params.get("hidden_dim", params.get("latent_dim", 64))
        _node("enc", "Encoder", f"dim={hidden}", "linear")
        _edge("input", "enc")
        _node("ode", "ODE Solver", "f_θ(z,t)", "block")
        _edge("enc", "ode")
        _node("dec", "Decoder", "", "linear")
        _edge("ode", "dec")
        _node("output", "Output", "", "output")
        _edge("dec", "output")

    elif family == "PHYSICS_AWARE":
        d_model = params.get("d_model", 256)
        n_layers = params.get("n_layers", 8)
        prev = "input"
        _node("embed", f"Physics Embedding({d_model})", "symmetry group projection", "linear")
        _edge(prev, "embed")
        prev = "embed"
        for i in range(n_layers):
            bid = f"pblock{i+1}"
            _node(bid, f"Conservation Block {i+1}", f"d={d_model}", "attention")
            _edge(prev, bid)
            prev = bid
        _node("output", "Output", "conservation-preserving", "output")
        _edge(prev, "output")

    else:
        _node("block", "Network Block", family, "block")
        _edge("input", "block")
        _node("output", "Output", "", "output")
        _edge("block", "output")

    return {
        "architecture_id": architecture_id,
        "family": family,
        "nodes": nodes,
        "edges": edges,
    }


@router.post("/{architecture_id}/feedback", summary="Submit feedback for an architecture")
async def submit_architecture_feedback(
    architecture_id: str,
    body: ArchitectureFeedbackRequest,
):
    """
    Submit a thumbs-up or thumbs-down rating for an architecture, with an optional comment.
    Feedback is stored per architecture and aggregated for display.
    """
    if architecture_id not in ARCHITECTURES_CATALOG:
        raise HTTPException(status_code=404, detail=f"Architecture '{architecture_id}' not found.")

    db = SessionLocal()
    try:
        fb = ArchitectureFeedback(
            arch_id=architecture_id,
            is_positive=1 if body.is_positive else 0,
            comment=body.comment,
        )
        db.add(fb)
        db.commit()
        return {"status": "ok", "arch_id": architecture_id, "is_positive": body.is_positive}
    finally:
        db.close()


@router.get("/{architecture_id}/feedback/summary", summary="Get aggregate feedback summary for an architecture")
async def get_architecture_feedback_summary(architecture_id: str):
    """
    Return aggregate thumbs-up / thumbs-down counts and the 10 most recent comments
    for the specified architecture.
    """
    if architecture_id not in ARCHITECTURES_CATALOG:
        raise HTTPException(status_code=404, detail=f"Architecture '{architecture_id}' not found.")

    db = SessionLocal()
    try:
        rows = db.query(ArchitectureFeedback).filter(
            ArchitectureFeedback.arch_id == architecture_id
        ).order_by(ArchitectureFeedback.created_at.desc()).all()

        thumbs_up = sum(1 for r in rows if r.is_positive == 1)
        thumbs_down = sum(1 for r in rows if r.is_positive == 0)
        recent_comments = [
            {"is_positive": bool(r.is_positive), "comment": r.comment, "created_at": str(r.created_at)}
            for r in rows if r.comment
        ][:10]

        return {
            "arch_id": architecture_id,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "total": len(rows),
            "recent_comments": recent_comments,
        }
    finally:
        db.close()

"""
HTTP routes of the external model catalog, mounted under ``/api/external-catalog``.

The catalog and its runners are ``pinneapple_arena.external`` (references to 95 third-party physics-ML libraries,
each run through its own package at run time). Every result carries a ``status`` (``ok`` / ``not_installed`` /
``skipped_heavy`` / ``manual_install`` / ``failed`` / ``finetune_not_supported``); a model that can not run here
says so and gives the exact local command to get a real result.
"""

# ruff: noqa: B008  (FastAPI Depends/Query/File defaults)
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pinn_arena.external.config import settings
from pinn_arena.external.jobs import await_job, submit
from pinn_arena.external.jobs import get as get_job
from pinn_arena.external.jobs import to_dict as job_to_dict
from pinneapple_arena.external.arch_meta import ARCH_META
from pinneapple_arena.external.benchmark import run_benchmark, run_compare, select_slugs
from pinneapple_arena.external.dataset_probe import probe_dataset
from pinneapple_arena.external.datasets_data import DATASETS, DATASETS_BY_SLUG
from pinneapple_arena.external.domain_meta import DOMAIN_META
from pinneapple_arena.external.finetune import run_finetune
from pinneapple_arena.external.installer import is_installed, pip_install
from pinneapple_arena.external.local_command import local_command_for, pip_install_command
from pinneapple_arena.external.models_data import MODELS, MODELS_BY_SLUG
from pinneapple_arena.external.runners.registry import call_infer, get_dataset_preview
from pinneapple_arena.external.status import install_state

router = APIRouter(prefix="/external-catalog", tags=["external-catalog"])


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #

def _spec_dict(spec) -> dict:
    d = spec.to_dict()
    d["install_state"] = install_state(spec)
    d["local_command_infer"] = local_command_for(spec, action="infer")
    d["local_command_finetune"] = local_command_for(spec, action="finetune")
    d["pip_install_command"] = pip_install_command(spec)
    return d


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

@router.get("/health", summary="External-catalog health check")
def health():
    return {"status": "ok", "models_indexed": len(MODELS), "datasets_indexed": len(DATASETS)}


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #

@router.get("/catalog/domains", summary="List the 41 physical domains in the external catalog")
def list_domains():
    return {k: {"key": v.key, "label": v.label, "glyph": v.glyph, "desc": v.desc}
            for k, v in DOMAIN_META.items()}


@router.get("/catalog/categories", summary="List the architecture-family groupings")
def list_categories():
    return {k: {"key": v.key, "label": v.label} for k, v in ARCH_META.items()}


@router.get("", summary="List all external catalog models")
def list_models(
    q: str | None = Query(None, description="free-text search over name/org/desc/tags"),
    domain: str | None = Query(None),
    category: str | None = Query(None),
    status: str | None = Query(None, description="installed | not_installed | manual_only | heavy"),
):
    items = MODELS
    if domain:
        items = [m for m in items if domain in m.domains]
    if category:
        items = [m for m in items if m.arch == category]
    if q:
        needle = q.lower()
        def hay(m):
            return f"{m.name} {m.org} {m.desc} {' '.join(m.tags)}".lower()
        items = [m for m in items if needle in hay(m)]

    results = [_spec_dict(m) for m in items]
    if status:
        results = [r for r in results if r["install_state"] == status]

    return {"count": len(results), "total": len(MODELS), "models": results}


@router.get("/models/{slug}", summary="Get a single external catalog model")
def get_model(slug: str):
    spec = MODELS_BY_SLUG.get(slug)
    if not spec:
        raise HTTPException(404, f"Unknown model slug: {slug}")
    return _spec_dict(spec)


@router.get("/models/{slug}/sample-input", summary="Preview the synthetic demo dataset for a model")
def sample_input(slug: str):
    if slug not in MODELS_BY_SLUG:
        raise HTTPException(404, f"Unknown model slug: {slug}")
    return get_dataset_preview(slug)


@router.get("/models/{slug}/api-snippet", summary="Ready-to-use curl/python snippet for this model")
def api_snippet(slug: str, request: Request):
    spec = MODELS_BY_SLUG.get(slug)
    if not spec:
        raise HTTPException(404, f"Unknown model slug: {slug}")
    base = (settings.public_base_url or str(request.base_url)).rstrip("/")
    curl = (
        f"curl -X POST {base}/api/external-catalog/models/{slug}/infer \\\n"
        f'  -H "Content-Type: application/json" -d \'{{"include_heavy": {str(spec.heavy).lower()}}}\''
    )
    python = (
        "import requests\n\n"
        f'resp = requests.post(\n    "{base}/api/external-catalog/models/{slug}/infer",\n'
        f'    json={{"include_heavy": {spec.heavy}}},\n)\n'
        "result = resp.json()\nprint(result[\"status\"], result.get(\"output_summary\"))"
    )
    return {"slug": slug, "base_url": base, "curl": curl, "python": python,
            "openapi_docs": f"{base}/docs"}


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

class InferRequest(BaseModel):
    include_heavy: bool = False


@router.post("/models/{slug}/infer", summary="Run real inference for a model")
def infer(slug: str, body: InferRequest = InferRequest()):
    if slug not in MODELS_BY_SLUG:
        raise HTTPException(404, f"Unknown model slug: {slug}")
    job = submit("infer", lambda: call_infer(slug, include_heavy=body.include_heavy).to_dict())
    job = await_job(job, timeout=settings.sync_wait_seconds)
    return job_to_dict(job)


# --------------------------------------------------------------------------- #
# Fine-tuning / reconfiguration
# --------------------------------------------------------------------------- #

class FinetuneRequest(BaseModel):
    params: dict = Field(default_factory=dict)


@router.post("/models/{slug}/finetune", summary="Fine-tune or reconfigure a model")
def finetune(slug: str, body: FinetuneRequest = FinetuneRequest()):
    if slug not in MODELS_BY_SLUG:
        raise HTTPException(404, f"Unknown model slug: {slug}")
    job = submit("finetune", lambda: run_finetune(slug, body.params).to_dict())
    job = await_job(job, timeout=settings.sync_wait_seconds)
    return job_to_dict(job)


# --------------------------------------------------------------------------- #
# Jobs (poll a long-running infer/finetune/benchmark that didn't finish
# within the synchronous wait window)
# --------------------------------------------------------------------------- #

@router.get("/jobs/{job_id}", summary="Poll a long-running job")
def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Unknown job id: {job_id}")
    return job_to_dict(job)


# --------------------------------------------------------------------------- #
# Benchmark / compare
# --------------------------------------------------------------------------- #

class BenchmarkRequest(BaseModel):
    slugs: list[str] | None = None
    category: str | None = None
    domain: str | None = None
    include_heavy: bool = False


@router.post("/benchmark", summary="Run the whole catalog (or a slice) as a smoke-test suite")
def benchmark(body: BenchmarkRequest):
    try:
        slugs = select_slugs(body.slugs, body.category, body.domain)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    if len(slugs) > settings.max_benchmark_models:
        raise HTTPException(
            400,
            f"{len(slugs)} models requested, max is {settings.max_benchmark_models}. "
            "Narrow with slugs/category/domain.",
        )
    job = submit("benchmark", lambda: run_benchmark(slugs, include_heavy=body.include_heavy))
    job = await_job(job, timeout=settings.sync_wait_seconds)
    return job_to_dict(job)


class CompareRequest(BaseModel):
    slugs: list[str]
    include_heavy: bool = False


@router.post("/compare", summary="Run 2-6 models side by side")
def compare(body: CompareRequest):
    if not (2 <= len(body.slugs) <= 6):
        raise HTTPException(400, "Compare between 2 and 6 models at a time.")
    unknown = [s for s in body.slugs if s not in MODELS_BY_SLUG]
    if unknown:
        raise HTTPException(400, f"Unknown model slug(s): {', '.join(unknown)}")
    job = submit("compare", lambda: run_compare(body.slugs, include_heavy=body.include_heavy))
    job = await_job(job, timeout=settings.sync_wait_seconds)
    return job_to_dict(job)


# --------------------------------------------------------------------------- #
# Optional server-side install (off by default; admin only; see config.py)
# --------------------------------------------------------------------------- #

@router.post("/models/{slug}/install", summary="Best-effort server-side pip install (disabled by default)")
def install(slug: str, request: Request):
    rt = getattr(request.app.state, "runtime", None)
    if not settings.allow_server_side_install or rt is None or not rt.is_admin(request):
        raise HTTPException(
            403,
            "Server-side installs are disabled on this deployment (or you are not its admin). Run the pip_install_command "
            "from GET /api/external-catalog/models/{slug} on your own machine instead.",
        )
    spec = MODELS_BY_SLUG.get(slug)
    if not spec:
        raise HTTPException(404, f"Unknown model slug: {slug}")
    if not spec.pip_package:
        raise HTTPException(400, "This model has no pip package to install (manual-install only).")
    ok, err = pip_install(spec.pip_package)
    return {"slug": slug, "installed": ok, "error": None if ok else err}


# --------------------------------------------------------------------------- #
# Datasets — a parallel catalog (see DatasetSpec in registry/schema.py). The
# only action is a "probe": load one real sample of one subset.
# --------------------------------------------------------------------------- #

def _dataset_dict(spec) -> dict:
    d = spec.to_dict()
    d["installed"] = bool(spec.import_name and is_installed(spec.import_name))
    d["local_root_configured"] = bool(settings.dataset_root)
    d["pip_install_command"] = f"pip install {spec.pip_package}" if spec.pip_package else None
    return d


@router.get("/datasets", summary="List external physics-simulation datasets")
def list_datasets():
    return {"count": len(DATASETS), "datasets": [_dataset_dict(d) for d in DATASETS]}


@router.get("/datasets/{slug}", summary="Get a single external dataset")
def get_dataset(slug: str):
    spec = DATASETS_BY_SLUG.get(slug)
    if not spec:
        raise HTTPException(404, f"Unknown dataset slug: {slug}")
    return _dataset_dict(spec)


class ProbeRequest(BaseModel):
    subset: str
    split: str = "train"
    stream: bool = False


@router.post("/datasets/{slug}/probe", summary="Load one real sample of a dataset subset")
def probe(slug: str, body: ProbeRequest):
    spec = DATASETS_BY_SLUG.get(slug)
    if not spec:
        raise HTTPException(404, f"Unknown dataset slug: {slug}")
    if body.subset not in {s.name for s in spec.subsets}:
        raise HTTPException(400, f"Unknown subset for {slug}: {body.subset}")
    if body.split not in spec.splits:
        raise HTTPException(400, f"Unknown split: {body.split} (expected one of {', '.join(spec.splits)})")
    job = submit("probe", lambda: probe_dataset(spec, body.subset, body.split, stream=body.stream,
                                                dataset_root=settings.dataset_root).to_dict())
    job = await_job(job, timeout=settings.sync_wait_seconds)
    return job_to_dict(job)

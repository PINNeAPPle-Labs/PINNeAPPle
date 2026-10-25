"""PINNeAPPle Arena: train PINNeAPPle architectures head to head on canonical physics problems, rank them on a
leaderboard, and browse a catalog of third-party physics-ML libraries run through their own packages.

Two sections share this FastAPI app and its React frontend (``apps/pinn_arena/frontend``, built into ``static/``):

- PINNeAPPle's own architectures: problems, experiments (training + held-out evaluation), leaderboard. The
  problems, reference data, losses, adapters and training protocol are ``pinneapple_arena.pointwise``.
- The external catalog (``/api/external-catalog``): references to 95 third-party libraries and the glue that runs
  them, ``pinneapple_arena.external``. Results carry an honest status; nothing is fabricated.

Run:  uvicorn pinn_arena.api:app --port 8130   (from apps/pinn_arena; PYTHONPATH must include apps/_shared)

Environment (all optional; the README lists them all):
  PINN_ARENA_USER / PINN_ARENA_PASSWORD(_HASH)   HTTP Basic login on everything except /health (appkit)
  PINN_ARENA_DATA_DIR                            SQLite database, uploaded datasets, trained weights
  PINN_ARENA_MAX_EPOCHS / _MAX_ARCHITECTURES / _MAX_COLLOCATION / _MAX_RUNNING / _FREE_EXPERIMENTS   limits
  GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET / GITHUB_CALLBACK_URL / ARENA_JWT_SECRET / ARENA_FRONTEND_URL   GitHub sign-in
  EXTERNAL_CATALOG_*                             external catalog settings (pinn_arena/external/config.py)
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from pinn_arena import settings
from pinn_arena.auth import router as auth_router
from pinn_arena.db import init_db
from pinn_arena.external.api import router as external_catalog_router
from pinn_arena.routes.architectures import router as architectures_router
from pinn_arena.routes.datasets import router as datasets_router
from pinn_arena.routes.experiments import router as experiments_router
from pinn_arena.routes.geometries import router as geometries_router
from pinn_arena.routes.leaderboard import router as leaderboard_router
from pinn_arena.routes.problems import router as problems_router

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
STATIC = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    from pinneapple_arena.external import MODELS
    from pinneapple_arena.pointwise import (
        ARCHITECTURES_CATALOG,
        GEOMETRIES_CATALOG,
        PROBLEMS_CATALOG,
    )
    logger.info("Catalogs: %d problems, %d architectures, %d geometries, %d external libraries",
                len(PROBLEMS_CATALOG), len(ARCHITECTURES_CATALOG), len(GEOMETRIES_CATALOG), len(MODELS))
    yield


app = FastAPI(
    title="PINNeAPPle Arena",
    version=VERSION,
    redirect_slashes=False,
    lifespan=_lifespan,
    description=(
        "Train PINNeAPPle architectures on canonical physics problems under one protocol (reference data, optional "
        "PDE residual, held-out evaluation), rank them on a leaderboard, and run a catalog of third-party "
        "physics-ML libraries through their own packages. Every result reports what really ran."
    ),
)
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix=settings.PREFIX)
rt = install_runtime(app, prefix=settings.PREFIX, name=settings.NAME,
                     units={"experiments": settings.FREE_EXPERIMENTS}, max_jobs=1)

for _router in (problems_router, geometries_router, architectures_router, experiments_router, auth_router,
                datasets_router, leaderboard_router, external_catalog_router):
    app.include_router(_router, prefix="/api")


def _check(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


@app.get("/health", tags=["meta"])
def health():
    """Liveness (no login needed)."""
    return {"status": "ok", "app": "pinn_arena", "version": VERSION}


@app.get("/api/health", tags=["meta"])
def api_health():
    """Status, and whether PINNeAPPle and PyTorch are importable."""
    return {"status": "ok", "api_version": VERSION, "pinneapple_available": _check("pinneapple_neural"),
            "torch_available": _check("torch")}


@app.get("/api/capabilities", tags=["meta"])
def capabilities():
    """Catalog sizes and the compute device."""
    from pinneapple_arena.external import MODELS
    from pinneapple_arena.pointwise import (
        ARCHITECTURES_CATALOG,
        GEOMETRIES_CATALOG,
        PROBLEMS_CATALOG,
        device_info,
    )
    info = device_info()
    return {
        "n_problems": len(PROBLEMS_CATALOG),
        "n_architectures": len(ARCHITECTURES_CATALOG),
        "n_geometries": len(GEOMETRIES_CATALOG),
        "n_external_catalog_models": len(MODELS),
        "torch_available": True,
        "pinneapple_available": True,
        "physics_loss_available": True,
        "gpu_available": info.get("gpu_available", False),
        "device_info": info,
    }


class _SPA(StaticFiles):
    """The built React app; client-side routes (``/leaderboard``, ...) get ``index.html``."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or path.startswith("api/"):
                raise
            return await super().get_response("index.html", scope)


if (STATIC / "index.html").is_file():
    app.mount("/", _SPA(directory=str(STATIC), html=True), name="spa")
else:
    @app.get("/", tags=["meta"])
    def root():
        """Without a built frontend (``npm run build`` in apps/pinn_arena/frontend): where the API lives."""
        return {"message": "PINNeAPPle Arena API", "docs": "/docs", "health": "/health",
                "frontend": "not built: see apps/pinn_arena/README.md"}

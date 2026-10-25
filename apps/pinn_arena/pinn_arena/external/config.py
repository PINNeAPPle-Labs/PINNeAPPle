"""Settings of the external catalog routes (``EXTERNAL_CATALOG_*`` environment variables, all optional)."""

from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # Public base URL used to render "use as API" curl/python snippets in the
    # UI. Set this to the real deployed URL in production.
    # Unset: the base URL of the request.
    public_base_url: str = os.environ.get("EXTERNAL_CATALOG_PUBLIC_BASE_URL", "")

    # Whether /api/external-catalog/models/{slug}/install may run `pip install` on the server (and then only for
    # the app's admin: the PINN_ARENA_USER login or PINN_ARENA_ADMIN_TOKEN). Off by default: arbitrary package
    # installs on a shared server are a real risk; enable it for a trusted local environment only.
    allow_server_side_install: bool = _bool_env("EXTERNAL_CATALOG_ALLOW_SERVER_SIDE_INSTALL", False)

    # How long a synchronous /infer or /finetune request waits for the job
    # to finish before handing back a job_id to poll instead.
    sync_wait_seconds: float = float(os.environ.get("EXTERNAL_CATALOG_SYNC_WAIT_SECONDS", "20"))

    # Hard cap on how many models a single /benchmark call may run at once.
    max_benchmark_models: int = int(os.environ.get("EXTERNAL_CATALOG_MAX_BENCHMARK_MODELS", "40"))

    # Local directory holding downloaded datasets (e.g. the `--base-path` given
    # to `the-well-download`). When set, dataset probes read from here instead
    # of needing remote streaming.
    dataset_root: str | None = os.environ.get("EXTERNAL_CATALOG_DATASET_ROOT") or None


settings = Settings()

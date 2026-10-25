"""
Benchmark / comparison harness — the successor to the standalone pipeline.py
smoke-test script, now callable from the API so the frontend's Benchmark and
Compare views can run it on demand against any subset of the catalog.
"""

from __future__ import annotations

from pinneapple_arena.external.models_data import MODELS, MODELS_BY_SLUG
from pinneapple_arena.external.result import STATUS_OK, ModelResult
from pinneapple_arena.external.runners.registry import call_infer


def select_slugs(slugs: list[str] | None = None, category: str | None = None,
                  domain: str | None = None) -> list[str]:
    if slugs:
        unknown = [s for s in slugs if s not in MODELS_BY_SLUG]
        if unknown:
            raise KeyError(f"Unknown model slug(s): {', '.join(unknown)}")
        return slugs
    picked = MODELS
    if category:
        picked = [m for m in picked if m.arch == category]
    if domain:
        picked = [m for m in picked if domain in m.domains]
    return [m.slug for m in picked]


def run_benchmark(slugs: list[str], include_heavy: bool = False) -> dict:
    results: list[ModelResult] = [call_infer(slug, include_heavy=include_heavy) for slug in slugs]

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    ok_results = [r for r in results if r.status == STATUS_OK]
    timings = [r.metrics.get("inference_seconds") for r in ok_results if r.metrics.get("inference_seconds") is not None]

    summary = {
        "total": len(results),
        "by_status": by_status,
        "real_ok_count": len(ok_results),
        "avg_inference_seconds": round(sum(timings) / len(timings), 4) if timings else None,
        "max_inference_seconds": round(max(timings), 4) if timings else None,
    }
    return {"summary": summary, "results": [r.to_dict() for r in results]}


def run_compare(slugs: list[str], include_heavy: bool = False) -> dict:
    specs = [MODELS_BY_SLUG[s].to_dict() for s in slugs]
    bench = run_benchmark(slugs, include_heavy=include_heavy)
    return {"specs": specs, "results": bench["results"], "summary": bench["summary"]}

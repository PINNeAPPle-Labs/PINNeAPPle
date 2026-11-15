"""Deterministic arXiv publication trends from records or the existing live source.

Coverage is the number of fetched records, not the number of all papers in arXiv;
neither result counts nor partial boundary months imply field-wide growth.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import re
from typing import Iterable

_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?((?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?)$", re.I)


def canonical_arxiv_id(value: str) -> str:
    value = (value or "").strip().split("?")[0]
    match = _ID.search(value)
    if not match:
        raise ValueError("not a recognized arXiv identifier")
    return re.sub(r"v\d+$", "", match.group(1), flags=re.I)


def monthly_trends(papers: Iterable, *, exclude_boundary_months: bool = False) -> dict:
    """Return month counts, unique paper count, and explicit sampling caveats."""
    records: dict[str, str] = {}
    for paper in papers:
        data = paper if isinstance(paper, dict) else vars(paper)
        key = canonical_arxiv_id(str(data.get("arxiv_id") or data.get("link") or data.get("url") or ""))
        published = str(data.get("published") or data.get("data_publicacao") or "")[:10]
        try:
            day = date.fromisoformat(published)
        except ValueError as exc:
            raise ValueError(f"invalid publication date for {key}: {published}") from exc
        if key in records:
            records[key] = min(records[key], day.isoformat())
        else:
            records[key] = day.isoformat()
    months = Counter(d[:7] for d in records.values())
    if not months:
        return {"total_unique_papers": 0, "months": [], "scope": "retrieved_sample",
                "boundary_months_excluded": exclude_boundary_months}
    start, end = min(months), max(months)
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    series = []
    for i in range(start_year * 12 + start_month - 1, end_year * 12 + end_month):
        year, month0 = divmod(i, 12)
        key = f"{year:04d}-{month0 + 1:02d}"
        if exclude_boundary_months and key in (start, end):
            continue
        series.append({"month": key, "count": months.get(key, 0)})
    return {"total_unique_papers": len(records), "months": series,
            "scope": "retrieved_sample", "boundary_months_excluded": exclude_boundary_months,
            "coverage_warning": "Paginated/latest-N searches are truncated samples; boundary months may be incomplete."}


def search_arxiv_trends(query: str, *, k: int = 100, exclude_boundary_months: bool = True) -> dict:
    """Use PINNeAPPle's existing real arXiv client (never a fabricated snapshot)."""
    if not query.strip() or not 1 <= k <= 200:
        raise ValueError("query must be nonempty and k in [1, 200]")
    from pinneapple_tools.hpo_experiments.sources import ArxivSource
    papers = ArxivSource().search(query, k=k)
    return {"query": query, "requested_limit": k,
            **monthly_trends(papers, exclude_boundary_months=exclude_boundary_months)}

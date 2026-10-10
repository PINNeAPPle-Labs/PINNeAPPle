"""Experiment curation: which runs are good enough to become a product, a paper, marketing or training data.

Every run is assessed on ten dimensions computed from what the lab already stored (checks, metrics, files, code
snapshot, datasets, the experiment's description and docs) plus an optional human review. The vocabulary and the
rules are those of ``pinneapple_veriphysics.applicability``: a dimension with no evidence is NOT_RUN (never a neutral
pass), evidence is VERIFIED (a check measured it in this run), INFERRED (indirect: metrics, names, files) or
UNSUPPORTED (nothing), and no number is invented here -- the assessment only classifies what exists.

Dimensions (score 0-1, or None when not run)
    quantitative: validation, reference, baseline, physics, generalization, uncertainty, reproducibility, data
    qualitative:  assets (figures, movies, renders, 3-D viewer), documentation (description, docs, papers),
                  review (human: novelty, clarity, visual appeal, a one-line story, approvals)

Tiers (gates, not a weighted sum: a strong figure never hides a failed check)
    A  flagship      ready to become product, paper or marketing once a human review approves it
    B  solid         validated against at least one independent piece of evidence; demos, use cases, datasets
    C  exploratory   ran, but the evidence is thin (no checks, or only sanity checks)
    D  not usable    failed validation or crashed

Readiness for each use (product, paper, marketing, training data) lists what is missing.

    from pinneapple_lab.curation import curate
    report = curate(LabStore("lab"))              # lab/curation.json and lab/CURATION.md
    python -m pinneapple_lab curate
    python -m pinneapple_lab review kepler_law --story "..." --novelty 4 --approve paper,marketing
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

try:                                                   # the shared vocabulary of the verification layer
    from pinneapple_veriphysics.applicability import (
        FAIL,
        INFERRED,
        INFO,
        NOT_RUN,
        PASS,
        UNSUPPORTED,
        VERIFIED,
    )
except Exception:                                      # noqa: BLE001 - keep the lab importable on its own
    PASS, FAIL, NOT_RUN, INFO = "PASS", "FAIL", "NOT_RUN", "INFO"
    VERIFIED, INFERRED, UNSUPPORTED = "VERIFIED", "INFERRED", "UNSUPPORTED"

TIERS = {"A": "flagship", "B": "solid", "C": "exploratory", "D": "not usable"}
WEIGHTS = {"validation": 15, "reference": 15, "baseline": 10, "physics": 10, "generalization": 10,
           "uncertainty": 5, "reproducibility": 15, "data": 5, "assets": 8, "documentation": 7}
ML_TAGS = {"surrogate", "physics-ai", "fno", "forecasting", "forecast", "machine-learning", "neural-operator"}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_SANITY = r"^finite|_finite|finite_|exits_cleanly|\bnan\b|statistically|converged|accretes"
_GENERAL = (r"unseen|held.?out|holdout|\bood\b|out_of_distribution|extrapol|generaliz|test_|forecast_beyond|"
            r"optimum_prediction_verified|beyond_observation")
_BASELINE = (r"beats|baseline|persistence|climatology|nearest|better_than|outperform|improvement_beats|vs_linear|"
             r"over_naive|than_band")
_REFERENCE = (r"exact|reference|published|measured|known|true_|recovered|within|rel_error|rel_l2|_vs_|barrowman|"
              r"analytic|refit|nusselt|polhamus|only_the_true|accuracy_for|_error$|^st_")
_PHYSICS = (r"conserv|divergence|budget|balance|steady|symmetr|monoton|stable|physical|positive|energy|mass|momentum|"
            r"regime|onset|constraint|shock|unstart|range|strouhal|drag_in|lift_increases|upstream|dominates")


def infer_check_kind(check: dict[str, Any]) -> str:
    """What a check is evidence of (see ``RunContext.check(kind=...)``): the name decides first, then an explicit
    reference value, then the detail text."""
    name = str(check.get("name", "")).lower()
    detail = str(check.get("detail", "")).lower()
    if check.get("kind"):
        return check["kind"]
    if re.search(_SANITY, name):
        return "sanity"
    if re.search(_GENERAL, name) or re.search(_GENERAL, detail):
        return "generalization"
    if re.search(_BASELINE, name):
        return "baseline"
    if check.get("reference") is not None or re.search(_REFERENCE, name):
        return "reference"
    if re.search(_PHYSICS, name):
        return "physics"
    if re.search(_REFERENCE, detail):
        return "reference"
    if re.search(_PHYSICS, detail):
        return "physics"
    return "sanity"


@dataclass
class Dimension:
    name: str
    score: float | None                 # 0..1, None = NOT_RUN
    status: str                         # PASS / FAIL / NOT_RUN / INFO
    tier: str                           # VERIFIED / INFERRED / UNSUPPORTED
    evidence: list[str] = field(default_factory=list)


@dataclass
class Assessment:
    run_id: str
    experiment: str
    status: str
    tier: str
    score: float | None                 # weighted mean over the dimensions that ran (0..100)
    coverage: float                     # fraction of the weight that had evidence
    dimensions: dict[str, Dimension]
    readiness: dict[str, dict[str, Any]]
    gaps: list[str]
    reviewed: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tier_name"] = TIERS[self.tier]
        return d


# ---------------------------------------------------------------------- loading
def _load(path: str, default: Any) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def load_reviews(root: str) -> dict[str, Any]:
    return _load(os.path.join(root, "reviews.json"), {})


def save_review(root: str, experiment: str, **fields) -> dict[str, Any]:
    """Record a human review of an experiment (merged into ``<lab>/reviews.json``)."""
    reviews = load_reviews(root)
    rec = reviews.get(experiment, {})
    rec.update({k: v for k, v in fields.items() if v is not None})
    rec["date"] = time.strftime("%Y-%m-%d")
    reviews[experiment] = rec
    with open(os.path.join(root, "reviews.json"), "w") as f:
        json.dump(reviews, f, indent=1, sort_keys=True)
    return rec


def _docs_mentions(name: str) -> list[str]:
    hits = []
    for p in glob.glob(os.path.join(REPO, "docs", "**", "*.md"), recursive=True) + [os.path.join(REPO, "README.md")]:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                if f"`{name}`" in f.read():
                    hits.append(os.path.relpath(p, REPO))
        except OSError:
            pass
    return hits


# ---------------------------------------------------------------------- dimensions
def _checks_dimension(name: str, checks: list[dict], kinds: set[str]) -> Dimension:
    sel = [c for c in checks if c.get("kind", infer_check_kind(c)) in kinds]
    if not sel:
        return Dimension(name, None, NOT_RUN, UNSUPPORTED, [f"no {'/'.join(sorted(kinds))} check"])
    ok = sum(bool(c["passed"]) for c in sel)
    ev = [f"{'PASS' if c['passed'] else 'FAIL'} {c['name']}" + (f" ({c['detail']})" if c.get("detail") else "")
          for c in sel]
    return Dimension(name, ok / len(sel), PASS if ok == len(sel) else FAIL, VERIFIED, ev)


def _metric_names(metrics: dict[str, Any], pattern: str) -> list[str]:
    return [k for k, v in metrics.items() if re.search(pattern, k.lower()) and isinstance(v, (int, float))]


def assess_run(store, row: dict[str, Any], *, meta: dict[str, Any] | None = None,
               reviews: dict[str, Any] | None = None, siblings: list[dict[str, Any]] | None = None) -> Assessment:
    """Assess one run (a row of ``LabStore.runs()``)."""
    d = store.run_dir(row["experiment"], row["run_id"])
    rec = _load(os.path.join(d, "run.json"), {})
    checks = _load(os.path.join(d, "validation.json"), [])
    for c in checks:
        c.setdefault("kind", infer_check_kind(c))
    metrics = row.get("metrics") or {}
    meta = meta or {}
    tags = set(meta.get("tags", []))
    dims: dict[str, Dimension] = {}

    # quantitative --------------------------------------------------------------------------------------
    if checks:
        ok = sum(bool(c["passed"]) for c in checks)
        dims["validation"] = Dimension("validation", ok / len(checks), PASS if ok == len(checks) else FAIL, VERIFIED,
                                       [f"{ok}/{len(checks)} checks pass"])
    else:
        dims["validation"] = Dimension("validation", None, NOT_RUN, UNSUPPORTED, ["the run records no check"])
    dims["reference"] = _checks_dimension("reference", checks, {"reference"})
    dims["baseline"] = _checks_dimension("baseline", checks, {"baseline"})
    dims["physics"] = _checks_dimension("physics", checks, {"physics"})
    gen = _checks_dimension("generalization", checks, {"generalization"})
    if gen.score is None:
        m = _metric_names(metrics, r"unseen|held|test|ood|extrapol|forecast")
        if m:
            gen = Dimension("generalization", 0.5, INFO, INFERRED, [f"metrics on unseen data, no check: {', '.join(m[:4])}"])
    dims["generalization"] = gen
    unc = _metric_names(metrics, r"sigma|std|ci95|ensemble|uncert|calib|spread|rms_fluct")
    dims["uncertainty"] = (Dimension("uncertainty", 1.0, INFO, INFERRED, [f"reported: {', '.join(unc[:4])}"]) if unc
                           else Dimension("uncertainty", None, NOT_RUN, UNSUPPORTED, ["no uncertainty reported"]))
    code = rec.get("code", {}) or {}
    rep, ev = [], []
    rep.append(1.0 if code.get("files") else 0.0)
    ev.append(f"code snapshot: {len(code.get('files', []))} files" if code.get("files") else "no code snapshot")
    rep.append(1.0 if rec.get("git") else 0.0)
    ev.append(f"git {rec.get('git')}" if rec.get("git") else "no git commit")
    rep.append(1.0 if rec.get("params") is not None else 0.0)
    rep.append(1.0 if (not code.get("uncommitted_diff")) or os.path.exists(os.path.join(d, code["uncommitted_diff"]))
               else 0.0)
    if code.get("uncommitted_diff"):
        ev.append("ran from a dirty tree (diff saved)")
    if siblings:                                      # repeated with other seeds: do the key metrics agree?
        key = next((k for k in metrics if isinstance(metrics[k], (int, float))), None)
        vals = [s["metrics"].get(key) for s in siblings if isinstance(s.get("metrics", {}).get(key), (int, float))]
        if key and len(vals) >= 2:
            import statistics
            m = statistics.mean(vals)
            cv = statistics.pstdev(vals) / abs(m) if m else 0.0
            rep.append(1.0 if cv < 0.05 else 0.5)
            ev.append(f"{len(vals)} repeats, {key} varies {100 * cv:.1f} %")
    dims["reproducibility"] = Dimension("reproducibility", sum(rep) / len(rep), INFO, VERIFIED if rep[0] else INFERRED, ev)
    ds = rec.get("datasets", {}) or {}
    if ds:
        n = sum(int(v.get("n_samples", 0)) for v in ds.values())
        cards = [_load(os.path.join(d, "datasets", k, "card.json"), {}) for k in ds]
        documented = sum(bool(c.get("description")) for c in cards) / max(len(cards), 1)
        dims["data"] = Dimension("data", min(1.0, 0.5 * (n > 0) + 0.5 * documented), INFO, VERIFIED,
                                 [f"{len(ds)} datasets, {n} samples, {int(100 * documented)} % with a card description"])
    else:
        dims["data"] = Dimension("data", None, NOT_RUN, UNSUPPORTED, ["no dataset"])

    # qualitative ---------------------------------------------------------------------------------------
    figs = glob.glob(os.path.join(d, "figures", "*"))
    movies = [f for f in figs if f.endswith((".gif", ".mp4", ".webm"))]
    beauty = [f for f in figs if "beauty" in os.path.basename(f) or "render" in os.path.basename(f)]
    viewer = os.path.exists(os.path.join(d, "viewer", "index.html"))
    a = 0.0
    a += 0.4 if figs else 0.0
    a += 0.2 if len(figs) >= 3 else 0.0
    a += 0.2 if movies else 0.0
    a += 0.2 if (beauty or viewer) else 0.0
    dims["assets"] = Dimension("assets", min(a, 1.0) if figs or viewer else None, INFO if figs else NOT_RUN,
                               VERIFIED if figs else UNSUPPORTED,
                               [f"{len(figs)} figures, {len(movies)} movies" + (", photoreal render" if beauty else "")
                                + (", 3-D viewer" if viewer else "")])
    desc = meta.get("description", "")
    docs = meta.get("docs", [])
    src = _load(os.path.join(d, "inputs", "source.json"), {})
    papers = src.get("papers", []) if isinstance(src, dict) else []
    doc = 0.4 * (len(desc) >= 80) + 0.3 * bool(docs) + 0.3 * bool(papers or meta.get("references"))
    dims["documentation"] = Dimension("documentation", doc, INFO, VERIFIED if docs else INFERRED,
                                      [f"description {len(desc)} chars", f"docs: {', '.join(docs) or 'none'}",
                                       f"papers: {len(papers)}"])
    rv = (reviews or {}).get(row["experiment"])
    if rv:
        parts = [rv[k] / 5 for k in ("novelty", "clarity", "visual_appeal") if isinstance(rv.get(k), (int, float))]
        dims["review"] = Dimension("review", sum(parts) / len(parts) if parts else None, INFO, VERIFIED,
                                   [f"{k}: {rv[k]}" for k in ("reviewer", "novelty", "clarity", "visual_appeal",
                                                              "story", "approved_for", "limitations") if rv.get(k)])
    else:
        dims["review"] = Dimension("review", None, NOT_RUN, UNSUPPORTED, ["no human review yet"])

    # score and coverage ------------------------------------------------------------------------------
    tot = sum(WEIGHTS.values())
    ran = {k: w for k, w in WEIGHTS.items() if dims[k].score is not None}
    score = (100 * sum(dims[k].score * w for k, w in ran.items()) / sum(ran.values())) if ran else None
    coverage = sum(ran.values()) / tot

    # tier gates ---------------------------------------------------------------------------------------
    def ok(k, thr=1.0):
        return dims[k].score is not None and dims[k].score >= thr and dims[k].status != FAIL

    is_ml = bool(tags & ML_TAGS)
    gaps: list[str] = []
    if row["status"] in ("failed", "failed_validation"):
        tier = "D"
        gaps.append("fix the failing checks or the crash" if row["status"] == "failed_validation" else "the run crashed")
    else:
        independent = ok("reference") or ok("baseline") or ok("physics")
        solid = ok("validation") and independent and dims["reproducibility"].score >= 0.6
        flagship = (solid and ok("reference") and (ok("baseline") or ok("physics")) and
                    dims["reproducibility"].score >= 0.75 and (dims["assets"].score or 0) >= 0.6 and
                    dims["documentation"].score >= 0.6 and (not is_ml or ok("generalization")))
        tier = "A" if flagship else "B" if solid else "C"
        if not ok("validation"):
            gaps.append("record validation checks")
        if not ok("reference"):
            gaps.append("compare with an independent reference (exact solution, experiment, published value)")
        if not (ok("baseline") or ok("physics")):
            gaps.append("show it beats a baseline or satisfies a physical law it was not trained on")
        if is_ml and not ok("generalization"):
            gaps.append("test on unseen inputs (held-out designs, out-of-distribution cases)")
        if (dims["assets"].score or 0) < 0.6:
            gaps.append("add figures (a movie, a render or a 3-D view for communication)")
        if dims["documentation"].score < 0.6:
            gaps.append("document it (docs page, references)")
        if dims["reproducibility"].score < 0.75:
            gaps.append("rerun from a committed tree with a code snapshot")

    # readiness per use ----------------------------------------------------------------------------------
    approved = set((rv or {}).get("approved_for", []) if isinstance((rv or {}).get("approved_for"), list) else
                   str((rv or {}).get("approved_for", "")).split(","))
    approved = {a.strip() for a in approved if a and a.strip()}

    def need(cond, msg, lst):
        if not cond:
            lst.append(msg)

    readiness = {}
    for use in ("product", "paper", "marketing", "training_data"):
        miss: list[str] = []
        if tier == "D":
            miss.append("not usable until it validates")
        if use == "product":
            need(ok("generalization"), "generalization evidence", miss)
            need(dims["uncertainty"].score is not None, "uncertainty estimate", miss)
            need(ok("reference") or ok("physics"), "reference or physics check", miss)
            need(dims["reproducibility"].score >= 0.75, "reproducibility", miss)
        elif use == "paper":
            need(ok("reference"), "comparison with a reference", miss)
            need(ok("baseline") or ok("physics"), "baseline or physics check", miss)
            need(dims["reproducibility"].score >= 0.75, "reproducibility", miss)
            need(isinstance((rv or {}).get("novelty"), (int, float)) and rv["novelty"] >= 3, "novelty review >= 3/5", miss)
        elif use == "marketing":
            need((dims["assets"].score or 0) >= 0.8, "strong visuals (movie, render or 3-D view)", miss)
            need(ok("reference") or ok("baseline"), "a validated headline number", miss)
            need(bool((rv or {}).get("story")), "a one-line story (review)", miss)
        else:
            need(dims["data"].score is not None and dims["data"].score >= 0.75, "a documented dataset", miss)
            need(ok("validation"), "validated runs", miss)
        readiness[use] = {"ready": not miss, "approved": use in approved, "missing": miss}
    return Assessment(row["run_id"], row["experiment"], row["status"], tier, None if score is None else round(score, 1),
                      round(coverage, 2), dims, readiness, gaps, reviewed=bool(rv))


def assess_experiment(runs: list[Assessment]) -> dict[str, Any]:
    """Best run, tier counts and the consistency of an experiment's runs."""
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    best = sorted(runs, key=lambda a: (order[a.tier], -(a.score or 0)))[0]
    counts = {t: sum(a.tier == t for a in runs) for t in TIERS}
    usable = sum(a.tier in ("A", "B") for a in runs) / len(runs)
    tier = best.tier
    if len(runs) >= 3 and usable < 0.5 and tier in ("A", "B"):
        tier = "B" if tier == "A" else "C"           # one good run among many failing ones is not a result
    return {"tier": tier, "tier_name": TIERS[tier], "best_run": best.run_id, "best_score": best.score,
            "runs": len(runs), "tier_counts": counts, "usable_fraction": round(usable, 2),
            "readiness": best.readiness, "gaps": best.gaps, "reviewed": best.reviewed}


def curate(store, path_json: str | None = None, path_md: str | None = None) -> dict[str, Any]:
    """Assess every run and experiment; write ``curation.json`` and ``CURATION.md`` in the lab folder."""
    from .spec import get
    reviews = load_reviews(store.root)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "experiments": {}, "runs": {}}
    for exp in sorted(store.status()):
        try:
            cls = get(exp)
            meta = {"description": cls.description, "tags": list(cls.tags),
                    "references": list(getattr(cls, "references", []) or [])}
        except KeyError:
            meta = {"description": "", "tags": []}
        meta["docs"] = _docs_mentions(exp)
        rows = [r for r in store.runs(exp) if r["status"] != "running"]          # in progress: not graded yet
        by_params: dict[str, list] = {}
        for r in rows:
            key = json.dumps({k: v for k, v in r["params"].items() if k != "seed"}, sort_keys=True, default=str)
            by_params.setdefault(key, []).append(r)
        assessed = []
        for r in rows:
            key = json.dumps({k: v for k, v in r["params"].items() if k != "seed"}, sort_keys=True, default=str)
            sib = by_params[key] if len(by_params[key]) > 1 else None
            a = assess_run(store, r, meta=meta, reviews=reviews, siblings=sib)
            assessed.append(a)
            out["runs"][a.run_id] = a.to_dict()
        if assessed:
            out["experiments"][exp] = assess_experiment(assessed)
    path_json = path_json or os.path.join(store.root, "curation.json")
    with open(path_json, "w") as f:
        json.dump(out, f, indent=1, default=str)
    _write_md(out, path_md or os.path.join(store.root, "CURATION.md"))
    return out


def _write_md(out: dict[str, Any], path: str) -> None:
    L = ["# PINNeAPPle Lab curation", "",
         f"Generated {out['generated']}. Tiers: **A flagship** (product / paper / marketing once reviewed), "
         "**B solid** (demos, use cases, datasets), **C exploratory**, **D not usable**. Gates, not averages: see "
         "`pinneapple_lab/curation.py`.", "",
         "| experiment | tier | best score | runs (A/B/C/D) | usable | product | paper | marketing | data | reviewed |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    for exp, e in sorted(out["experiments"].items(), key=lambda kv: (order[kv[1]["tier"]], -(kv[1]["best_score"] or 0))):
        tc = e["tier_counts"]
        rd = e["readiness"]
        mark = {u: "approved" if rd[u]["approved"] else "ready" if rd[u]["ready"] else "-" for u in rd}
        L.append(f"| `{exp}` | **{e['tier']}** {e['tier_name']} | {e['best_score']} | {tc['A']}/{tc['B']}/{tc['C']}/{tc['D']} | "
                 f"{int(100 * e['usable_fraction'])} % | {mark['product']} | {mark['paper']} | {mark['marketing']} | "
                 f"{mark['training_data']} | {'yes' if e['reviewed'] else 'no'} |")
    L += ["", "## What each experiment needs next", ""]
    for exp, e in sorted(out["experiments"].items()):
        if e["gaps"]:
            L.append(f"- `{exp}` ({e['tier']}): " + "; ".join(e["gaps"]))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")

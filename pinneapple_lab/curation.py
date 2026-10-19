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

Tiers A-F (gates, not a weighted sum: a strong figure never hides a failed check; the first rule that holds wins)
    F  not usable    the run crashed, a divergence check failed (non-finite, not converged), a physics check failed
                     (conservation, symmetry, equilibrium), or every reference check failed with no other
                     independent check passing (it contradicts the reference)
    E  known issues  any other failing check: a reference check failed while another independent check passed
                     (large error), unseen inputs failed (outside the validity range), a baseline was not beaten
    A  flagship      ready to become product, paper or marketing once a human review approves it
    B  solid         validated against at least one independent piece of evidence; demos, use cases, datasets
    C  plausible     every check passes and the trust card's model, physics or benchmark stage is answered, but
                     the B gates are not met (only verification against a code that shares the inputs, or not
                     reproducible)
    D  exploratory   ran and is consistent (no failing check), but no evidence stage is answered (no checks, or only
                     sanity checks)

Until 2026-10 there were four tiers: A and B are unchanged, the old C (exploratory) is now C or D, the old D (not
usable) is now E or F. :data:`TIER_HISTORY` records the mapping. The tier of an E or F item comes with its main
problem and the next step (``headline["hint"]``), which the seal prints.

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

from . import next_steps as _ns

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

TIERS = {"A": "flagship", "B": "solid", "C": "plausible", "D": "exploratory", "E": "known problems",
         "F": "not usable"}
TIER_ORDER = {t: i for i, t in enumerate(TIERS)}
TIER_HISTORY = {"A": ("A",), "B": ("B",), "C": ("C", "D"), "D": ("E", "F")}   # old tier -> the letters it splits into
INDEPENDENT = ("reference", "baseline", "physics", "generalization")
WEIGHTS = {"validation": 15, "reference": 15, "baseline": 10, "physics": 10, "generalization": 10,
           "uncertainty": 5, "reproducibility": 15, "data": 5, "assets": 8, "documentation": 7}
ML_TAGS = {"surrogate", "physics-ai", "fno", "forecasting", "forecast", "machine-learning", "neural-operator", "rom",
           "reduced-order-model", "deeponet", "gnn", "pinn"}
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
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    headline: dict[str, Any] = field(default_factory=dict)
    next_steps: list[dict[str, Any]] = field(default_factory=list)    # pinneapple_lab.next_steps.Gap.to_dict()

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
    lim = meta.get("limitations", [])
    if isinstance(lim, dict):
        lim = lim.get(str(row["params"].get(meta.get("case_param", ""), "")), []) if meta.get("case_param") else []
    evidence = evidence_stages(d, checks, metrics, dims, is_ml, list(lim or []))
    gaps: list[str] = []
    issue = problem(row["status"], checks, rec.get("error"))
    if issue:
        tier = issue["tier"]
        gaps.append("the run crashed" if issue["rule"] == "crashed" else "fix the failing checks or the crash")
    else:
        independent = ok("reference") or ok("baseline") or ok("physics") or ok("generalization")
        solid = ok("validation") and independent and dims["reproducibility"].score >= 0.6
        flagship = (solid and ok("reference") and (ok("baseline") or ok("physics")) and
                    dims["reproducibility"].score >= 0.75 and (dims["assets"].score or 0) >= 0.6 and
                    dims["documentation"].score >= 0.6 and (not is_ml or ok("generalization")))
        partly = ok("validation") and any(evidence[k]["status"] == _ANSWERED for k in ("model", "physics", "benchmark"))
        tier = "A" if flagship else "B" if solid else "C" if partly else "D"
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
    nctx = _ns.context_for(row["experiment"], row.get("params") or {},
                           os.path.join(os.path.basename(store.root), os.path.relpath(d, store.root)), tags, is_ml,
                           meta.get("params"))
    steps = path_to_next(tier, issue, checks, dims, evidence, ok, is_ml, nctx, reviewed=bool((reviews or {}).get(
        row["experiment"])))

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
        if tier in ("E", "F"):
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
    stages = {**evidence, "decision": _decision_stage(tier, readiness, list(lim or []))}
    stages = {k: {"title": STAGES[k][0], "question": STAGES[k][1], **v} for k, v in stages.items()}
    return Assessment(row["run_id"], row["experiment"], row["status"], tier, None if score is None else round(score, 1),
                      round(coverage, 2), dims, readiness, gaps, reviewed=bool(rv), stages=stages,
                      headline=problem_headline(issue) if issue else headline(checks, tier, meta.get("check_labels")),
                      next_steps=[g.to_dict() for g in steps])


# ---------------------------------------------------------------------- the path to the next letter
def path_to_next(tier: str, issue: dict[str, Any] | None, checks: list[dict[str, Any]], dims: dict[str, Dimension],
                 evidence: dict[str, dict[str, Any]], ok, is_ml: bool, ctx: dict[str, Any], *,
                 reviewed: bool = False) -> list[Any]:
    """The gaps between ``tier`` and the next letter up (for A: what would strengthen it), each with its PINNeAPPle
    actions (``pinneapple_lab.next_steps``). Computed from the same evidence and thresholds as the letter."""
    G = _ns.make_gap
    out = []
    if tier in ("E", "F"):
        if issue["rule"] == "crashed":
            out.append(G("fix_crashed", "E", ctx, why=(issue["text"], issue["text_pt"])))
        for c in checks:
            if c.get("passed") or (issue["rule"] == "crashed" and c.get("name") == "exits_cleanly"):
                continue                                  # a crash's own "exits cleanly" check is the crash
            k = c.get("kind") or infer_check_kind(c)
            if k == "sanity" and re.search(_DIVERGED, str(c.get("name", "")).lower()):
                t = "fix_diverged"
            elif k == "physics":
                t = "fix_physics"
            elif k == "reference":
                t = "fix_contradicts" if issue["rule"] == "contradicts" else "fix_reference"
            else:
                t = {"generalization": "fix_generalization", "baseline": "fix_baseline"}.get(k, "fix_check")
            out.append(G(t, "E" if t in _ns.F_CAUSING else "D", ctx, check=c))
        out.sort(key=lambda g: g.to != "E")
        return out
    score = lambda k: dims[k].score or 0.0  # noqa: E731

    def dim_gap(t, to, k, thr):
        m = {"value": round(score(k), 2), "threshold": thr, "op": "score"}
        en, pt = _ns.GAP_TYPES[t]["short"]
        return G(t, to, ctx, meas=m, why=(f"{k} {_ns.measured_text(m)}: {en}", f"{k} {_ns.measured_text(m, 'pt')}: {pt}"))

    if tier == "D":
        open_ = [k for k in ("model", "physics", "benchmark") if evidence[k]["status"] != _ANSWERED]
        pt = {"model": "modelo", "physics": "física", "benchmark": "benchmark"}
        out.append(G("no_evidence", "C", ctx, why=(
            f"no passing check answers the trust card's {', '.join(open_)} stages",
            f"nenhuma verificação responde às etapas {', '.join(pt[k] for k in open_)} do cartão")))
        return out
    if tier == "C":
        if not (ok("reference") or ok("baseline") or ok("physics") or ok("generalization")):
            out.append(G("no_independent", "B", ctx, why=(
                "no reference, baseline, physics or unseen-data check passes (verification against a code that shares "
                "the inputs does not count)", "nenhuma verificação de referência, baseline, física ou dados não vistos "
                "passa (verificar contra um código com as mesmas entradas não conta)")))
        if score("reproducibility") < 0.6:
            out.append(dim_gap("reproducibility", "B", "reproducibility", 0.6))
        return out
    if tier == "B":
        if not ok("reference"):
            out.append(G("no_reference", "A", ctx))
        if not (ok("baseline") or ok("physics")):
            out.append(G("no_baseline_or_physics", "A", ctx))
        if is_ml and not ok("generalization"):
            out.append(G("generalization", "A", ctx))
        if score("reproducibility") < 0.75:
            out.append(dim_gap("reproducibility", "A", "reproducibility", 0.75))
        if score("assets") < 0.6:
            out.append(dim_gap("assets", "A", "assets", 0.6))
        if score("documentation") < 0.6:
            out.append(dim_gap("documentation", "A", "documentation", 0.6))
        return out
    refs = [c for c in checks if c.get("passed") and c.get("kind") == "reference"]
    if len(refs) < 2:
        out.append(G("second_reference", _ns.STRENGTHEN, ctx, why=(
            f"{len(refs)} passing independent reference check", f"{len(refs)} verificação de referência independente")))
    if evidence["uncertainty"]["status"] != _ANSWERED:
        out.append(G("uncertainty", _ns.STRENGTHEN, ctx, why=(
            f"uncertainty stage {evidence['uncertainty']['status']}: no check quantifies it",
            f"etapa de incerteza {evidence['uncertainty']['status']}: nenhuma verificação a quantifica")))
    if not ok("baseline"):
        out.append(G("benchmark_row", _ns.STRENGTHEN, ctx))
    if not reviewed:
        out.append(G("review", _ns.STRENGTHEN, ctx))
    return out


def next_steps(item: str, lab: Any = None) -> list[Any]:
    """The path to the next letter of a curated item (``experiment`` or ``experiment/case``), read from the curation
    record: a list of :class:`pinneapple_lab.next_steps.Gap`, each with its suggested actions. ``lab`` as for
    :func:`pinneapple_lab.seal.load_curation`."""
    from .seal import load_curation
    rec = load_curation(lab).get("items", {}).get(item)
    if rec is None:
        raise KeyError(f"{item!r} is not in the curation")
    return [_ns.Gap.from_dict(g) for g in rec.get("next_steps", [])]


# ---------------------------------------------------------------------- E and F: what is wrong, and the next step
_DIVERGED = r"finite|(^|_)nan(_|$)|exits_cleanly|converg|diverg"
# rule -> (English, Portuguese) next step; the seal prints it after the letter it leads to
HINTS = {
    "crashed": ("make the run finish without an error", "fazer a execução terminar sem erro"),
    "diverged": ("make the result finite and converged", "tornar o resultado finito e convergido"),
    "physics": ("make the physics check pass", "fazer a verificação física passar"),
    "contradicts": ("agree with the reference, or pass an independent physics check",
                    "concordar com a referência ou passar numa verificação física independente"),
    "reference": ("bring the error vs the reference within its tolerance",
                  "trazer o erro face à referência para dentro da tolerância"),
    "generalization": ("pass on the unseen inputs", "passar nos dados não vistos"),
    "baseline": ("beat the baseline", "superar o baseline"),
    "other": ("fix the failing check", "corrigir a verificação que falha"),
}


def problem(status: str, checks: list[dict[str, Any]], error: str | None = None) -> dict[str, Any] | None:
    """The E or F rule a run falls under, or None when it finished and every check passed. The first rule that holds
    wins: crashed, diverged (a failing sanity check on finiteness or convergence), physics (a failing physics
    check), contradicts (every reference check failed and no other independent check passed) give F; any other
    failing check gives E (by its kind: reference, generalization, baseline, else other). ``checks`` carry their
    ``kind`` (see :func:`infer_check_kind`)."""
    failing = [c for c in checks if not c.get("passed")]
    if status == "failed":
        last = next((ln.strip() for ln in reversed(str(error or "").strip().splitlines()) if ln.strip()), "")
        last = _short(re.sub(r"\S*/(\S+)", r"\1", last), 64)                # the error's last line, paths cut
        return {"tier": "F", "rule": "crashed", "check": None, "failing": len(failing),
                "text": f"crashed · {last}" if last else "the run crashed",
                "text_pt": f"falhou · {last}" if last else "a execução falhou"}
    if not failing:
        return None
    kind = lambda c: c.get("kind") or infer_check_kind(c)  # noqa: E731
    rule, main, tier = None, None, "E"
    div = [c for c in failing if kind(c) == "sanity" and re.search(_DIVERGED, str(c.get("name", "")).lower())]
    phys = [c for c in failing if kind(c) == "physics"]
    refs = [c for c in checks if kind(c) == "reference"]
    if div:
        rule, main, tier = "diverged", div[0], "F"
    elif phys:
        rule, main, tier = "physics", phys[0], "F"
    elif refs and not any(c.get("passed") for c in refs) and \
            not any(c.get("passed") and kind(c) in INDEPENDENT for c in checks):
        rule, main, tier = "contradicts", next(c for c in failing if kind(c) == "reference"), "F"
    else:
        for k in ("reference", "generalization", "baseline"):
            hit = [c for c in failing if kind(c) == k]
            if hit:
                rule, main = k, hit[0]
                break
        else:
            rule, main = "other", failing[0]
    more = f" (+{len(failing) - 1} more)" if len(failing) > 1 else ""
    line = check_line(main, 90) + more                 # the seal wraps it onto two lines
    return {"tier": tier, "rule": rule, "check": main, "failing": len(failing), "text": line, "text_pt": line}


def problem_headline(issue: dict[str, Any]) -> dict[str, Any]:
    """The headline of an E or F item: the main failing check (``text``, ``text_pt``), the rule and the next step to
    the next tier up (``hint`` / ``hint_pt``, also in ``missing``)."""
    nxt = {"F": "E", "E": "D"}[issue["tier"]]
    en, pt = HINTS[issue["rule"]]
    if issue["rule"] == "other" and issue["failing"] > 1:
        en, pt = f"fix the {issue['failing']} failing checks", f"corrigir as {issue['failing']} verificações que falham"
    c = issue.get("check") or {}
    return {"kind": "failing", "text": issue["text"], "text_pt": issue["text_pt"], "check": c.get("name", ""),
            "rule": issue["rule"], "next_tier": nxt, "hint": f"{nxt}: {en}", "hint_pt": f"{nxt}: {pt}",
            "missing": f"{nxt}: {en}"}


# ---------------------------------------------------------------------- the one-line evidence of an item
def _short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n - 1].rsplit(" ", 1)[0].rstrip(",;:-")
    return cut + "\u2026"


def check_line(check: dict[str, Any], max_chars: int = 64) -> str:
    """One line naming what a check compared against: its ``label`` when the experiment gave one, else the check's
    name (underscores to spaces) and, unless the name already has a colon, the first clause of its detail."""
    if check.get("label"):
        return _short(check["label"], max_chars)
    name = str(check.get("name", "")).replace("_", " ").strip()
    if ":" in name:                                    # a name that already says what it compared with
        return _short(name, max_chars)
    det = re.split(r"[:;]", str(check.get("detail") or ""), maxsplit=1)[0].strip()
    if det and len(name) + 3 < max_chars:
        return name + " \u00b7 " + _short(det, max_chars - len(name) - 3)
    return _short(name, max_chars)


def headline(checks: list[dict[str, Any]], tier: str, labels: dict[str, Any] | None = None) -> dict[str, Any]:
    """The evidence a seal or a showcase tile may print, taken from the run's passing checks only (nothing written by
    hand): ``kind`` "reference" (an independent reference the result was compared with), "benchmark" (a physical
    law, a baseline or unseen data, no independent reference) or "" (nothing passed), ``text`` the line, ``check``
    the check's name and ``missing`` what the item still lacks for the next claim.

    ``labels`` (``Experiment.check_labels``) maps a check name to its short display line,
    ``{"en": ..., "pt": ..., "short": ..., "short_pt": ..., "also": [other check names the line also claims]}``
    (``short`` for small tiles); it becomes ``label`` / ``label_pt`` / ``label_short`` / ``label_short_pt`` only when
    that check and every check in ``also`` passed, so a label never says more than the run showed. A ``label`` recorded on the check itself (``ctx.check(label=...)``) is used when the experiment has none."""
    passing = [c for c in checks if c.get("passed")]
    ref = [c for c in passing if c.get("kind") == "reference"]
    other = [c for c in passing if c.get("kind") in ("physics", "baseline", "generalization")] or \
        [c for c in passing if c.get("kind") == "verification"]                 # C: the solver verified, no more
    if ref:
        c, kind, missing = ref[0], "reference", "" if tier == "A" else "flagship gates (see gaps)"
    elif other:
        c, kind = other[0], "benchmark" if other[0].get("kind") != "verification" else "verification"
        missing = "independent reference" if kind == "benchmark" else "an independent check (reference, baseline or physics)"
    else:
        return {"kind": "", "text": "", "check": "", "missing": "a passing check"}
    out = {"kind": kind, "text": check_line(c), "check": c.get("name", ""), "missing": missing}
    lab = (labels or {}).get(c.get("name", ""))
    names = {x.get("name") for x in passing}
    if isinstance(lab, dict) and all(a in names for a in lab.get("also", [])):
        out["label"] = lab.get("en", "")
        for k, dst in (("pt", "label_pt"), ("short", "label_short"), ("short_pt", "label_short_pt")):
            if lab.get(k):
                out[dst] = lab[k]
    elif c.get("label"):
        out["label"] = c["label"]
    return out


# ---------------------------------------------------------------------- the six questions of a trustworthy result
STAGES = {
    "data_geometry": ("Data and geometry", "Was the physical problem represented correctly?"),
    "model": ("Model", "Can the model (or solver) represent the relevant dynamics?"),
    "physics": ("Physical constraints", "Are the equations and the boundary conditions respected?"),
    "benchmark": ("Benchmark", "How does the result compare with a solver or reference data?"),
    "uncertainty": ("Uncertainty", "Where may the prediction not be reliable?"),
    "decision": ("Engineering decision", "Is the result adequate for the intended use?"),
}
_DATA_CHECK = r"mesh|converg|geometr|domain|finite|inside|stay|overlap|input|resolution|grid|bounded|in_the_tank"
_MODEL_CHECK = r"projection|capacity|rank|energy_captured|mode_pair|modes_come|pod_|unseen|generaliz|held"
_ANSWERED, _PARTIAL, _OPEN, _FAILED = "answered", "partial", "open", "failed"


def _stage(status: str, evidence: list[str], nxt: str = "") -> dict[str, Any]:
    return {"status": status, "evidence": evidence, "next": nxt}


def trust_stages(d: str, checks: list[dict], metrics: dict[str, Any], dims: dict[str, Dimension], tier: str,
                 readiness: dict[str, dict[str, Any]], meta: dict[str, Any], is_ml: bool,
                 limitations: list[str]) -> dict[str, dict[str, Any]]:
    """The six questions a reviewer asks of a physics / Physics-AI result, each answered from the run's own
    evidence: answered (a passing check), partial (indirect evidence), open (nothing yet) or failed (a failing
    check). Nothing is inferred beyond what the run stored."""
    out = {**evidence_stages(d, checks, metrics, dims, is_ml, limitations),
           "decision": _decision_stage(tier, readiness, limitations)}
    return {k: {"title": STAGES[k][0], "question": STAGES[k][1], **v} for k, v in out.items()}


def _decision_stage(tier: str, readiness: dict[str, dict[str, Any]], limitations: list[str]) -> dict[str, Any]:
    ready = [u for u, r in readiness.items() if r.get("ready")]
    if tier in ("E", "F"):
        return _stage(_FAILED, [f"tier {tier} ({TIERS[tier]})"], "not usable until it validates")
    if tier in ("A", "B") and ready and limitations:
        return _stage(_ANSWERED, [f"tier {tier} ({TIERS[tier]}), ready for {', '.join(ready)}",
                                  f"{len(limitations)} limitations bound the use"])
    miss = [m for u, r in readiness.items() for m in r.get("missing", [])]
    return _stage(_PARTIAL if tier in ("A", "B") else _OPEN,
                  [f"tier {tier} ({TIERS[tier]})"] + ([f"ready for {', '.join(ready)}"] if ready else []),
                  (miss[0] if miss else "declare the limitations of the intended use"))


def evidence_stages(d: str, checks: list[dict], metrics: dict[str, Any], dims: dict[str, Dimension], is_ml: bool,
                    limitations: list[str]) -> dict[str, dict[str, Any]]:
    """The first five questions of the trust card (data and geometry, model, physics, benchmark, uncertainty), which
    do not depend on the tier; the C / D rule reads them."""
    def by(pattern=None, kinds=None):
        out = []
        for c in checks:
            if kinds and c.get("kind") not in kinds:
                continue
            if pattern and not re.search(pattern, c["name"], re.I):
                continue
            out.append(c)
        return out

    def summary(cs):
        return [f"{'PASS' if c['passed'] else 'FAIL'} {c['name']}" + (f" ({c.get('detail', '')[:90]})"
                                                                      if c.get("detail") else "") for c in cs[:5]]

    def from_checks(cs, nxt):
        if not cs:
            return None
        if any(not c["passed"] for c in cs):
            return _stage(_FAILED, summary([c for c in cs if not c["passed"]] + [c for c in cs if c["passed"]]),
                          "fix the failing check" + ("s" if sum(not c["passed"] for c in cs) > 1 else ""))
        return _stage(_ANSWERED, summary(cs), nxt)

    out: dict[str, dict[str, Any]] = {}
    # 1. data and geometry: what was simulated, on which mesh, with which inputs
    inputs = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(d, "inputs", "*.json")))
    cs = by(_DATA_CHECK, {"sanity", "physics"}) or by(kinds={"sanity"})
    st = from_checks(cs, "")
    ev = ([f"inputs recorded: {', '.join(inputs[:6])}"] if inputs else []) + (st["evidence"] if st else [])
    if st and st["status"] == _FAILED:
        out["data_geometry"] = _stage(_FAILED, ev, st["next"])
    elif st and inputs:
        out["data_geometry"] = _stage(_ANSWERED, ev)
    elif inputs or st:
        out["data_geometry"] = _stage(_PARTIAL, ev or ["sanity checks only"],
                                      "record the geometry, mesh and inputs" if not inputs else
                                      "add a mesh-convergence or domain check")
    else:
        out["data_geometry"] = _stage(_OPEN, ["no inputs or geometry recorded"], "record the inputs (ctx.input)")
    # 2. model capacity: can it represent the dynamics (unseen data for a learned model, verification for a solver)
    if is_ml:
        st = from_checks(by(_MODEL_CHECK) + by(kinds={"generalization"}), "")
        if st is None:
            m = [k for k in metrics if re.search(r"unseen|held|test|projection|forecast|rank", k, re.I)]
            out["model"] = (_stage(_PARTIAL, [f"metrics without a check: {', '.join(m[:4])}"],
                                   "turn the unseen-data metric into a check") if m else
                            _stage(_OPEN, ["no test on unseen inputs"], "evaluate on held-out cases"))
        else:
            out["model"] = st
    else:
        st = from_checks(by(kinds={"verification"}) + [c for c in by(r"converg|_vs_|exact|analytic|order|recover")
                                                         if c.get("kind") != "verification"]
                         or by(kinds={"reference"}), "")
        out["model"] = st or _stage(_OPEN, ["the solver is not verified on a problem with a known answer"],
                                    "add a verification case (exact solution or mesh convergence)")
    # 3. physical constraints
    out["physics"] = from_checks(by(kinds={"physics"}), "") or _stage(
        _OPEN, ["no conservation, symmetry or boundary-condition check"],
        "check a law the result was not fitted to (mass, energy, divergence, equilibrium)")
    # 4. benchmark
    out["benchmark"] = from_checks(by(kinds={"reference", "baseline"}), "") or _stage(
        _OPEN, ["no comparison with a solver, data or baseline"], "compare with a reference or a simple baseline")
    # 5. uncertainty: quantified, or at least where it is not reliable
    st = from_checks(by(kinds={"uncertainty"}) + by(r"uncert|calib|coverage|ensemble|spread"), "")
    unc = dims.get("uncertainty")
    if st:
        out["uncertainty"] = st
    elif unc is not None and unc.score is not None:
        out["uncertainty"] = _stage(_PARTIAL, unc.evidence + ([f"{len(limitations)} declared limitations"]
                                                              if limitations else []), "calibrate it (coverage check)")
    elif limitations:
        out["uncertainty"] = _stage(_PARTIAL, [f"limitations: {limitations[0][:110]}"] +
                                    ([f"and {len(limitations) - 1} more"] if len(limitations) > 1 else []),
                                    "quantify it (ensemble, GP standard deviation, mesh or seed spread)")
    else:
        out["uncertainty"] = _stage(_OPEN, ["no uncertainty and no declared limitation"],
                                    "declare the limitations and quantify the spread")
    return out


# ---------------------------------------------------------------------- strategy: limitations and the path forward
EFFORT = {"S": 1, "M": 3, "L": 8}          # relative cost: S hours, M a day or two, L a week or more
USES = ("product", "paper", "marketing", "training_data")


def _action(gap: str, tags: set[str], detail: str = "") -> tuple[str, str]:
    """(action, effort) for a missing piece of evidence, specific to the kind of experiment."""
    cfd = tags & {"cfd", "lbm", "openfoam", "hydro"}
    ml = tags & ML_TAGS
    disc = tags & {"discovery", "symbolic-regression"}
    fc = tags & {"forecasting", "forecast", "weather"}
    if gap == "reference":
        if disc:
            return ("Recover the law from a second, independent dataset and compare the coefficients with the textbook "
                    "values", "M")
        if cfd:
            return ("Validate against a published experiment or benchmark value (drag, Strouhal, Nusselt) and run a "
                    "grid-convergence study (coarse / medium / fine)", "M")
        if ml:
            return "Score against the high-fidelity solver on designs the model never saw", "M"
        if tags & {"astrophysics", "astronomy"}:
            return "Compare with an analytic solution (Bondi, equilibrium torus) or a published simulation", "M"
        return "Add a check against an exact solution, a measurement or a published value", "M"
    if gap == "baseline":
        if fc:
            return "Compare with persistence and climatology at every lead time", "S"
        if disc:
            return ("Compare with a naive model (a linear or polynomial fit, the textbook law without the discovered "
                    "terms)", "S")
        if ml:
            return "Compare with simple surrogates (nearest design, linear regression, POD + Gaussian process)", "S"
        if tags & {"pinn", "xtfc", "pde"}:
            return "Compare with a classical solver at equal cost (finite differences, spectral)", "S"
        return "Compare with the simplest method that could do the job", "S"
    if gap == "physics":
        return ("Check a law the model was not trained on: conservation of mass / momentum / energy, symmetry, a "
                "physical bound", "S")
    if gap == "generalization":
        if fc:
            return "Score unseen years / regimes (out-of-distribution) and report the skill horizon", "M"
        return "Hold out designs or an out-of-distribution regime and report the error there", "M"
    if gap == "uncertainty":
        return ("Report an ensemble spread or intervals and check their calibration "
                "(pinneapple_veriphysics.robustness: ensemble_study, calibration_against_reference)", "M")
    if gap == "reproducibility":
        return "Rerun from a committed tree (the code snapshot is automatic) and repeat with three seeds", "S"
    if gap == "assets":
        return ("Add a movie, a photoreal render (pp.viz.render) or the 3-D viewer (pp.viz.web_viewer) of the key "
                "result", "S")
    if gap == "documentation":
        return "Write a docs page and cite the reference papers (Experiment.references)", "S"
    if gap == "review":
        return ("Human review: novelty, clarity, the one-line story "
                "(python -m pinneapple_lab review <experiment> --story ... --novelty ...)", "S")
    if gap == "data":
        return "Save the fields as a dataset with a card (ctx.dataset)", "S"
    if gap == "fix":
        return f"Fix the failing checks or the crash{': ' + detail if detail else ''}", "M"
    return gap, "M"


_MISSING_TO_GAP = {   # readiness "missing" text -> gap key
    "generalization evidence": "generalization", "uncertainty estimate": "uncertainty",
    "reference or physics check": "reference", "reproducibility": "reproducibility",
    "comparison with a reference": "reference", "baseline or physics check": "baseline",
    "novelty review >= 3/5": "review", "strong visuals (movie, render or 3-D view)": "assets",
    "a validated headline number": "reference", "a one-line story (review)": "review",
    "a documented dataset": "data", "validated runs": "fix", "not usable until it validates": "fix",
}


def plan_experiment(runs: list[Assessment], meta: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    """Limitations, the actions that would close the gaps (with effort and what each unlocks) and, per use, the
    closest run and its distance (sum of effort) to being ready."""
    tags = set(meta.get("tags", []))
    order = TIER_ORDER
    per_use: dict[str, Any] = {}
    actions: dict[str, dict[str, Any]] = {}
    for use in USES:
        best = None
        for a in runs:
            r = a.readiness[use]
            gaps = list(dict.fromkeys(_MISSING_TO_GAP.get(m, m) for m in r["missing"]))
            cost = sum(EFFORT[_action(g, tags)[1]] for g in gaps)
            key = (not r["approved"], not r["ready"], cost, order[a.tier], -(a.score or 0))
            if best is None or key < best[0]:
                best = (key, a, gaps, cost, r)
        _, a, gaps, cost, r = best
        per_use[use] = {"status": "approved" if r["approved"] else "ready" if r["ready"] else "gap",
                        "run": a.run_id, "distance": cost, "missing": r["missing"]}
        for g in gaps:
            text, effort = _action(g, tags)
            act = actions.setdefault(g, {"gap": g, "action": text, "effort": effort, "unlocks": []})
            act["unlocks"].append(use)
    # the tier step: what the best run lacks for the next tier
    best = sorted(runs, key=lambda a: (order[a.tier], -(a.score or 0)))[0]
    tier_gap_keys = {"compare with an independent": "reference", "show it beats": "baseline", "test on unseen": "generalization",
                     "add figures": "assets", "document it": "documentation", "rerun from": "reproducibility",
                     "record validation": "reference", "fix the failing": "fix", "the run crashed": "fix"}
    next_tier = {"F": "E", "E": "D", "D": "C", "C": "B", "B": "A", "A": None}[best.tier]
    for g in best.gaps:
        key = next((v for k, v in tier_gap_keys.items() if g.startswith(k)), None)
        if key:
            text, effort = _action(key, tags)
            act = actions.setdefault(key, {"gap": key, "action": text, "effort": effort, "unlocks": []})
            if next_tier and f"tier {next_tier}" not in act["unlocks"]:
                act["unlocks"].append(f"tier {next_tier}")
    plan = sorted(actions.values(), key=lambda x: (-len(x["unlocks"]), EFFORT[x["effort"]]))
    # limitations: declared by the experiment, by the reviewer, and detected
    lim = [f"declared: {x}" for x in meta.get("limitations", [])]
    if review and review.get("limitations"):
        lim.append(f"review: {review['limitations']}")
    failed = [a for a in runs if a.tier in ("E", "F")]
    if failed:
        lim.append(f"detected: {len(failed)} of {len(runs)} runs fail validation or crash (outside the validated range)")
    inferred = [k for k, d in best.dimensions.items() if d.tier == INFERRED and d.score is not None]
    if inferred:
        lim.append(f"detected: evidence only inferred (not measured by a check) for {', '.join(inferred)}")
    if best.coverage < 0.6:
        lim.append(f"detected: only {int(100 * best.coverage)} % of the evidence dimensions have data")
    closest = min((u for u in USES if per_use[u]["status"] == "gap"), key=lambda u: per_use[u]["distance"], default=None)
    return {"limitations": lim, "actions": plan, "uses": per_use, "closest_use": closest, "next_tier": next_tier}


def assess_experiment(runs: list[Assessment]) -> dict[str, Any]:
    """Best run, tier counts and the consistency of an experiment's runs."""
    order = TIER_ORDER
    best = sorted(runs, key=lambda a: (order[a.tier], -(a.score or 0)))[0]
    counts = {t: sum(a.tier == t for a in runs) for t in TIERS}
    usable = sum(a.tier in ("A", "B") for a in runs) / len(runs)
    tier = best.tier
    if len(runs) >= 3 and usable < 0.5 and tier in ("A", "B"):
        tier = "B" if tier == "A" else "C"           # one good run among many failing ones is not a result
    return {"tier": tier, "tier_name": TIERS[tier], "best_run": best.run_id, "best_score": best.score,
            "runs": len(runs), "tier_counts": counts, "usable_fraction": round(usable, 2),
            "readiness": best.readiness, "gaps": best.gaps, "reviewed": best.reviewed, "stages": best.stages,
            "headline": best.headline, "next_steps": _consistency(tier, best, usable)}


def _consistency(tier: str, best: Assessment, usable: float) -> list[dict[str, Any]]:
    """The best run's gaps; when the item was lowered because most of its runs fail, that comes first."""
    if tier == best.tier:
        return best.next_steps
    ctx = _ns.context_for(best.experiment, {}, "", set(), False, {})
    g = _ns.make_gap("consistency", best.tier, ctx, meas={"value": usable, "threshold": 0.5, "op": "score"},
                     why=(f"only {round(100 * usable)} % of its runs are A or B", f"só {round(100 * usable)} % das "
                          "execuções são A ou B"))
    return [g.to_dict()] + best.next_steps


def curate(store, path_json: str | None = None, path_md: str | None = None) -> dict[str, Any]:
    """Assess every run and experiment; write ``curation.json`` and ``CURATION.md`` in the lab folder."""
    from .spec import get
    reviews = load_reviews(store.root)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "experiments": {}, "items": {},
           "runs": {}}
    for exp in sorted(store.status()):
        try:
            cls = get(exp)
            meta = {"description": cls.description, "tags": list(cls.tags),
                    "references": list(getattr(cls, "references", []) or []),
                    "limitations": getattr(cls, "limitations", []) or [],
                    "case_param": getattr(cls, "case_param", "") or "",
                    "check_labels": getattr(cls, "check_labels", {}) or {}, "params": dict(cls.params)}
        except KeyError:
            cls = None
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
            out["runs"][a.run_id] = {k: v for k, v in a.to_dict().items() if k != "next_steps"}   # on items only
        if assessed:
            lim_all = meta.get("limitations", [])
            m0 = {**meta, "limitations": lim_all if isinstance(lim_all, list) else []}
            e = assess_experiment(assessed)
            e["plan"] = plan_experiment(assessed, m0, reviews.get(exp))
            e["description"] = meta.get("description", "")
            e["story"] = (reviews.get(exp) or {}).get("story", "")
            out["experiments"][exp] = e
            # portfolio items: one per case for experiments that bundle distinct cases, else the experiment
            cp = getattr(cls, "case_param", "") if cls is not None else ""
            groups: dict[str, list] = {}
            for a_, r in zip(assessed, rows, strict=True):
                groups.setdefault(str(r["params"].get(cp)) if cp else "", []).append(a_)
            for case, runs_c in groups.items():
                key = f"{exp}/{case}" if case else exp
                lim_c = (lim_all.get(case, []) if isinstance(lim_all, dict) else lim_all)
                rv = reviews.get(key) or (reviews.get(exp) if not case else None)
                item = assess_experiment(runs_c)
                item["plan"] = plan_experiment(runs_c, {**meta, "limitations": lim_c}, rv)
                cdesc = (getattr(cls, "case_descriptions", {}) or {}).get(case) if cls is not None else None
                item.update(experiment=exp, case=case, story=(rv or {}).get("story", ""),
                            description=cdesc or meta.get("description", ""),
                            title=(rv or {}).get("title") or (os.path.splitext(os.path.basename(case))[0].replace("_", " ")
                                                               if case else exp))
                out["items"][key] = item
    path_json = path_json or os.path.join(store.root, "curation.json")
    with open(path_json, "w") as f:
        json.dump(out, f, indent=1, default=str)
    _write_md(out, path_md or os.path.join(store.root, "CURATION.md"))
    return out


def _write_md(out: dict[str, Any], path: str) -> None:
    L = ["# PINNeAPPle Lab curation", "",
         f"Generated {out['generated']}. Tiers: **A flagship** (product / paper / marketing once reviewed), "
         "**B solid** (demos, use cases, datasets), **C plausible** (partly checked), **D exploratory** (not "
         "validated), **E known problems** (a failing check), **F not usable** (crashed, diverged, failed physics or "
         "contradicts the reference). Gates, not averages: see `pinneapple_lab/curation.py`.", "",
         "| experiment | tier | best score | runs (A/B/C/D/E/F) | usable | product | paper | marketing | data | reviewed |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    order = TIER_ORDER
    for exp, e in sorted(out["experiments"].items(), key=lambda kv: (order[kv[1]["tier"]], -(kv[1]["best_score"] or 0))):
        tc = e["tier_counts"]
        rd = e["readiness"]
        mark = {u: "approved" if rd[u]["approved"] else "ready" if rd[u]["ready"] else "-" for u in rd}
        L.append(f"| `{exp}` | **{e['tier']}** {e['tier_name']} | {e['best_score']} | {'/'.join(str(tc.get(t, 0)) for t in TIERS)} | "
                 f"{int(100 * e['usable_fraction'])} % | {mark['product']} | {mark['paper']} | {mark['marketing']} | "
                 f"{mark['training_data']} | {'yes' if e['reviewed'] else 'no'} |")
    mark = {"answered": "✓", "partial": "~", "open": "?", "failed": "✗"}
    L += ["", "## Trust card: six questions per item", "",
          "Data and geometry (was the physical problem represented correctly?), model (can it represent the dynamics?), "
          "physical constraints (equations and boundary conditions respected?), benchmark (against a solver or reference "
          "data?), uncertainty (where may it not be reliable?), engineering decision (adequate for the intended use?). "
          "✓ answered by a passing check, ~ partial evidence, ? open, ✗ a failing check.", "",
          "| item | tier | data & geometry | model | physics | benchmark | uncertainty | decision |",
          "|---|---|---|---|---|---|---|---|"]
    for key, it in sorted(out.get("items", {}).items(), key=lambda kv: (order[kv[1]["tier"]], kv[0])):
        sg = it.get("stages", {})
        L.append(f"| `{key}` | {it['tier']} | " + " | ".join(mark.get(sg.get(k, {}).get("status"), "") for k in STAGES)
                 + " |")
    problems = [(k, it) for k, it in sorted(out.get("items", {}).items()) if it["tier"] in ("E", "F")]
    if problems:
        L += ["", "## E and F: the main problem and the next step", "",
              "| item | tier | main problem | next step |", "|---|---|---|---|"]
        for key, it in sorted(problems, key=lambda kv: (order[kv[1]["tier"]], kv[0])):
            hl = it.get("headline", {})
            L.append(f"| `{key}` | {it['tier']} | {hl.get('text', '')} | {hl.get('hint', '')} |")
    L += ["", "## What each experiment needs next", ""]
    for exp, e in sorted(out["experiments"].items()):
        if e["gaps"]:
            L.append(f"- `{exp}` ({e['tier']}): " + "; ".join(e["gaps"]))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")

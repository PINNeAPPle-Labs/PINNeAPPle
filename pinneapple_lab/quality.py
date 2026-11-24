"""The experiment quality checklist and its badge (#923).

An industrial reader has to be able to see, without reading a README, how much of an experiment's protocol is on
record. This module answers that with **nine items, each met or not, each with the evidence**: baseline, held-out
split, repeats and intervals, equal budget, ablation, convergence, provenance, verdict and what was *not* measured.

It is a checklist, not a score. The letter stays with :mod:`pinneapple_lab.curation` (A-F, gates over the evidence
dimensions); this module never re-grades, never weights and never invents a number -- every item is computed from
files that already exist (the ``experiment.json`` checklist, the metrics' ``how``, the params, ``provenance`` and the
folder's markdown). Where nothing exists to read, the item is ``not assessed`` and says why, instead of counting as a
failure.

The scale is the same one curation and ``pinneapple_data.readiness`` use: a status that is not a function of a
number, and an evidence line per item. That is why the seal of this package is still ``pinneapple_lab.seal`` /
``curation`` and the dataset gate is still ``pinneapple_data.readiness``: the badge below is their checklist view.

    >>> from pinneapple_lab import quality
    >>> c = quality.assess("examples/base_workflows/equation_discovery")     # doctest: +SKIP
    >>> c.badge()                                                          # doctest: +SKIP
    'QUALITY 5/8 · missing: ablation, convergence, not measured'
    >>> c.missing                                                          # doctest: +SKIP
    ['ablation', 'convergence', 'not measured']

What a manifest may declare (``experiment.json``, optional and ignored by older readers, so old manifests stay
valid)::

    "quality": {"schema": "pinneapple.quality.declaration/1",
                "not_applicable": {"convergence": "no solver: the data are measured, not computed"},
                "not_measured": ["wall time on the reference hardware", "the Re range above 5e5"],
                "reviewed_by": "Yan Barros", "date": "2026-11-24"}

``not_applicable`` switches an item off with a stated reason (a declared reason is the evidence); ``not_measured``
satisfies item nine, which is the one item a folder cannot prove by having files.

The report is written as ``lab/QUALITY.md`` (one row per folder) and attached to the Hub catalogue entry, and
``docs/quality_backlog.md`` lists the gaps sorted by how visible each experiment is.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .manifest import FILENAME, REPO, Manifest, load

__all__ = ["ITEMS", "Item", "Checklist", "DECLARATION", "SCHEMA", "STATUSES", "assess", "assess_all", "badge",
           "backlog", "declaration_of", "render_markdown", "visibility"]

SCHEMA = "pinneapple.quality.checklist/1"
DECLARATION = "pinneapple.quality.declaration/1"
STATUSES = ("met", "missing", "not applicable", "not assessed")
MET, MISSING, NA, UNASSESSED = STATUSES

#: The nine items of the issue, in the order the issue lists them.
ITEMS = ("reference", "holdout", "repeats", "budget", "ablation", "convergence", "provenance", "verdict",
         "not_measured")

TITLES = {
    "reference": "Reference or baseline",
    "holdout": "Held-out or extrapolation split",
    "repeats": "Repeats (seeds or bootstrap) with intervals",
    "budget": "Equal budget across the methods compared",
    "ablation": "Ablation (a removal or a variant)",
    "convergence": "Convergence (mesh or time-step study)",
    "provenance": "Provenance and the reproduce command",
    "verdict": "Verdict computed from results, losses included",
    "not_measured": "An explicit list of what was not measured",
}

#: The label a badge prints -- ``missing: ablation, convergence``, never a sentence.
SHORT = {"reference": "reference", "holdout": "held-out", "repeats": "repeats", "budget": "budget",
         "ablation": "ablation", "convergence": "convergence", "provenance": "provenance", "verdict": "verdict",
         "not_measured": "not measured"}

FIXES = {
    "reference": "compare with an exact solution, a solver, measured data or a published value, and cite it",
    "holdout": "hold out a split (or extrapolate past the training range) and check on it",
    "repeats": "repeat with several seeds or a bootstrap and put an interval on the headline number",
    "budget": "record the training or run budget of every method being compared",
    "ablation": "add one removal or variant, so the claim \"X helps\" has a control",
    "convergence": "refine the mesh or the time step and show the result stops moving",
    "provenance": "record the git commit, the inputs' hashes and the command that reproduces the folder",
    "verdict": "write the checks as results (pass/fail with a detail), keeping the ones that failed",
    "not_measured": "list what the experiment does not measure, in the manifest's quality block or the README",
}

# the words a folder uses for each item; kept next to the item they serve so a reviewer can change one without
# hunting through a 300-line detector
_BUDGET = re.compile(r"budget|epochs?|steps?|n_iter|iterations?|n_collocation|n_train|hidden|width|depth|params\b|"
                     r"n_samples", re.I)
_BUDGET_METRIC = re.compile(_BUDGET.pattern + r"|flops|wall|seconds|cost|time per", re.I)
_SEEDS = re.compile(r"seeds?\b|n_seeds|bootstrap|n_boot|ensemble|n_restarts|n_runs|repeats?|folds?", re.I)
_INTERVAL = re.compile(r"interval|confidence|\bci\b|ci_?9?5|±|\bstd\b|stdev|error bar|"
                       r"\[\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\]", re.I)
_ABLATION = re.compile(r"ablat|variant|\bwithout\b|\w_off\b|disabled|remov\w*|control run|switched off|"
                       r"no[-_ ]physics|physics_off|no[-_ ]drag|no[-_ ]magnus", re.I)
_CONVERGENCE = re.compile(r"mesh (?:study|refin|independ)|refin\w*|richardson|time.?step|timestep|\bdt\b|"
                          r"grid (?:study|refin|independ)|resolution (?:study|refin)|h.?refin|asymptotic range|"
                          r"cell size", re.I)
_NOT_MEASURED = re.compile(r"not measured|n[ãa]o medid|what was not|not assessed|n[ãa]o avaliad|limitations?|"
                           r"caveats?|out of scope|fora de escopo", re.I)
_IMPROVEMENT = re.compile(r"beats?|better|improv|outperform|wins?\b|gain|reduces?\b|faster|cheaper|helps?", re.I)
_SOLVER = re.compile(r"openfoam|calculix|su2|solver|simulation|numerical|finite|fdm|fem|fvm|lbm|mesh|dx\b|\bdt\b|"
                     r"n_cells|n_cell|nx\b|ny\b|cfl", re.I)
_EVIDENCE_KINDS = ("reference", "analytic", "baseline", "physics", "generalization", "verification")


@dataclass
class Item:
    """One line of the checklist: the status, the files that said so, and how to close it."""
    id: str
    status: str
    evidence: list[str] = field(default_factory=list)
    fix: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.id}: status {self.status!r} is not one of {STATUSES}")

    @property
    def title(self) -> str:
        return TITLES.get(self.id, self.id)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "status": self.status, "evidence": list(self.evidence),
                "fix": self.fix if self.status == MISSING else ""}


@dataclass
class Checklist:
    """The nine items of one folder (or one run), the counts, and the badge text."""
    subject: str
    items: list[Item]
    quality: dict[str, Any] = field(default_factory=dict)      # the manifest's declaration, when it has one
    where: str = ""                                            # the folder, relative to the checkout

    def by_id(self, item_id: str) -> Item:
        return next(i for i in self.items if i.id == item_id)

    @property
    def met(self) -> list[str]:
        return [i.id for i in self.items if i.status == MET]

    @property
    def missing(self) -> list[str]:
        return [i.id for i in self.items if i.status == MISSING]

    @property
    def not_applicable(self) -> list[str]:
        return [i.id for i in self.items if i.status == NA]

    @property
    def not_assessed(self) -> list[str]:
        return [i.id for i in self.items if i.status == UNASSESSED]

    @property
    def applicable(self) -> int:
        return sum(1 for i in self.items if i.status in (MET, MISSING))

    def badge(self) -> str:
        return badge(self)

    def explain(self) -> str:
        lines = [f"{self.subject}: {self.badge()}"]
        for i in self.items:
            line = f"  [{i.status:>14}] {i.id}: {i.title}"
            if i.evidence:
                line += f" -- {'; '.join(i.evidence[:2])}"
            if i.status == MISSING and i.fix:
                line += f" (fix: {i.fix})"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "subject": self.subject, "where": self.where,
                "met": len(self.met), "applicable": self.applicable, "badge": self.badge(),
                "missing": self.missing, "not_applicable": self.not_applicable,
                "not_assessed": self.not_assessed, "items": [i.to_dict() for i in self.items],
                "declaration": self.quality}

    def to_markdown(self) -> str:
        return render_markdown([self], title=self.subject)


def badge(c: Checklist) -> str:
    """``QUALITY 5/8 · missing: ablation, convergence`` -- what a README line prints. Zero applicable items is
    ``not assessed`` and nothing else: a folder with no evidence is not a folder that failed."""
    if c.applicable == 0:
        return "not assessed"
    text = f"QUALITY {len(c.met)}/{c.applicable}"
    if c.missing:
        text += " · missing: " + ", ".join(short_name(i).lower() for i in c.missing)
    return text


def short_name(item_id: str) -> str:
    return SHORT.get(item_id, item_id)


# ---------------------------------------------------------------------------------------------- reading the folder
def declaration_of(m: Manifest) -> dict[str, Any]:
    """The manifest's optional ``quality`` block, validated: an unknown item id or schema is ignored with a note
    rather than raising, because a newer manifest must stay readable by an older package (#923)."""
    raw = getattr(m, "quality", None) or {}
    if not isinstance(raw, Mapping):
        return {"schema": DECLARATION, "problems": [f"quality block is {type(raw).__name__}, not an object"]}
    out = dict(raw)
    problems = []
    if raw.get("schema") not in (None, DECLARATION):
        problems.append(f"declaration schema {raw.get('schema')!r} is newer than {DECLARATION!r}: read as not assessed")
        return {"schema": DECLARATION, "problems": problems}
    out["schema"] = DECLARATION
    na = raw.get("not_applicable") or {}
    if isinstance(na, (list, tuple)):
        na = {str(k): "" for k in na}
    unknown = sorted(set(na) - set(ITEMS))
    if unknown:
        problems.append(f"not_applicable names items that do not exist: {unknown}")
        na = {k: v for k, v in na.items() if k in ITEMS}
    out["not_applicable"] = dict(na)
    nm = raw.get("not_measured") or []
    out["not_measured"] = [str(x) for x in nm] if isinstance(nm, (list, tuple)) else [str(nm)]
    out["problems"] = problems
    return out


def _checks(m: Manifest) -> list[dict[str, Any]]:
    """The record's checks with a kind, using the same inference the curation uses, so both read a folder alike."""
    from .curation import infer_check_kind
    out = []
    for c in m.checks or []:
        if not isinstance(c, Mapping) or "name" not in c:
            continue
        d = dict(c)
        if not d.get("kind"):
            d["kind"] = infer_check_kind(d)
        out.append(d)
    return out


def _text(c: Mapping[str, Any]) -> str:
    return " ".join(str(c.get(k, "")) for k in ("name", "detail", "reference", "kind"))


def _metrics(m: Manifest) -> list[dict[str, Any]]:
    return [x for x in (m.metrics or []) if isinstance(x, Mapping) and x.get("name")]


def _folders_markdown(base: str) -> list[tuple[str, str]]:
    """``(name, text)`` of the markdown next to the manifest: the README first, the rest by name. Only the top level
    of the folder, so a big ``results/`` tree is not read 177 times."""
    out = []
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for name in ["README.md"] + [n for n in names if n.lower().endswith(".md") and n != "README.md"]:
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                out.append((name, f.read()))
        except OSError:
            continue
    return out


def _is_comparison(m: Manifest, checks: Sequence[Mapping[str, Any]]) -> bool:
    """Is there more than one thing being compared? Two methods, or a baseline check. Deliberately narrow: a
    loose "compared with" anywhere in a metric turns every folder into a comparison and makes the budget item
    meaningless."""
    if len([x for x in (m.methods or []) if str(x).strip()]) >= 2:
        return True
    return any(c.get("kind") == "baseline" for c in checks)


def _is_solver(m: Manifest) -> bool:
    blob = " ".join([*(m.modules or []), *(m.methods or []), json.dumps(m.params or {}, default=str)])
    if _SOLVER.search(blob):
        return True
    return any(_SOLVER.search(f"{x.get('name', '')}") for x in _metrics(m))


# ---------------------------------------------------------------------------------------------- the nine items
def assess(subject: str | Manifest, *, base: str | None = None) -> Checklist:
    """The checklist of one folder (a path, a manifest, or a run folder with a ``manifest``).

    ``base`` is the folder the manifest's relative paths and markdown live in; it is derived from a path argument and
    defaults to the checkout's ``examples/<id>`` for a bare :class:`Manifest`.
    """
    if isinstance(subject, Manifest):
        m, path = subject, base or os.path.join(REPO, *(subject.id.split("/")))
    else:
        path = str(subject)
        m = load(path)
    if os.path.isfile(path) or path.endswith(FILENAME):
        base = os.path.dirname(os.path.abspath(path))
    else:
        base = os.path.abspath(path)
    return _assess_manifest(m, base)


def _assess_manifest(m: Manifest, base: str) -> Checklist:
    q = declaration_of(m)
    checks = _checks(m)
    metrics = _metrics(m)
    params = {str(k): v for k, v in (m.params or {}).items()}
    prov = m.provenance or {}
    markdown = _folders_markdown(base)
    items: list[Item] = []

    def declared(item_id: str) -> bool:
        return item_id in (q.get("not_applicable") or {})

    def na(item_id: str) -> Item | None:
        if not declared(item_id):
            return None
        reason = str((q.get("not_applicable") or {}).get(item_id) or "").strip()
        return Item(item_id, NA, [f"declared not applicable: {reason}" if reason
                                  else "declared not applicable in the manifest (no reason given)"])

    def no_evidence(item_id: str) -> Item:
        return Item(item_id, UNASSESSED, ["the record has no check and no metric to read"], FIXES[item_id])

    has_record = bool(checks or metrics)

    # 1 -- reference or baseline ---------------------------------------------------------------------
    refs = [c for c in checks if c.get("kind") in ("reference", "analytic", "baseline")]
    names = [str(c["name"]) for c in refs]
    sourced = [str(c["name"]) for c in refs if c.get("reference") not in (None, "", [])
               or str(c.get("detail", "")).strip()]
    if (i := na("reference")):
        items.append(i)
    elif refs and (sourced or m.references):
        items.append(Item("reference", MET, [f"{len(refs)} reference or baseline check(s): {', '.join(names[:3])}"]
                          + ([f"cites {', '.join(map(str, m.references[:2]))}"] if m.references else [])))
    elif refs:
        items.append(Item("reference", MISSING, [f"{len(refs)} reference check(s): {', '.join(names[:3])}",
                                                 "neither the check nor references[] says what it is"],
                          FIXES["reference"]))
    elif checks or metrics:
        items.append(Item("reference", MISSING, ["no reference and no baseline check: nothing to compare with"],
                          FIXES["reference"]))
    else:
        items.append(no_evidence("reference"))

    # 2 -- held-out or extrapolation split -------------------------------------------------------------
    held = [c for c in checks if c.get("kind") == "generalization"] or \
           [x for x in metrics if re.search(r"unseen|held.?out|holdout|\bood\b|extrapol|test_|generaliz|forecast",
                                            str(x.get("name", "")), re.I)]
    if (i := na("holdout")):
        items.append(i)
    elif held:
        names = [str(x.get("name")) for x in held[:3]]
        items.append(Item("holdout", MET, [f"{len(held)} held-out or extrapolation result(s): {', '.join(names)}"]))
    elif checks or metrics:
        items.append(Item("holdout", MISSING, ["every number is on the data the result was fitted or tuned on"],
                          FIXES["holdout"]))
    else:
        items.append(no_evidence("holdout"))

    # 3 -- repeats and intervals -----------------------------------------------------------------------
    seed_params = [k for k in params if _SEEDS.search(k)]
    seed_metrics = [str(x.get("name")) for x in metrics if _SEEDS.search(f"{x.get('name')} {x.get('how', '')}")]
    intervals = [str(x.get("name")) for x in metrics if _INTERVAL.search(f"{x.get('how', '')}")]
    if (i := na("repeats")):
        items.append(i)
    elif seed_params or seed_metrics or intervals:
        items.append(Item("repeats", MET, [x for x in (
            f"seeds or repeats recorded: {', '.join(seed_params[:3])}" if seed_params else "",
            f"repeated over: {', '.join(seed_metrics[:3])}" if seed_metrics else "",
            f"intervals on: {', '.join(intervals[:3])}" if intervals else "") if x]))
    elif checks or metrics:
        items.append(Item("repeats", MISSING, ["a single run per configuration and no interval on the headline number"],
                          FIXES["repeats"]))
    else:
        items.append(no_evidence("repeats"))

    # 4 -- equal budget (comparisons only) -------------------------------------------------------------
    if (i := na("budget")):
        items.append(i)
    elif not has_record:
        items.append(no_evidence("budget"))
    elif not _is_comparison(m, checks):
        items.append(Item("budget", NA, ["nothing is compared: one method, no baseline"]))
    else:
        budget = [k for k in params if _BUDGET.search(k)]
        cost = [str(x.get("name")) for x in metrics if _BUDGET_METRIC.search(f"{x.get('name')} {x.get('unit', '')}")]
        if budget or cost:
            items.append(Item("budget", MET, [f"budget recorded: {', '.join((budget or cost)[:3])}"]))
        else:
            items.append(Item("budget", MISSING, ["methods are compared but no budget per method is recorded"],
                              FIXES["budget"]))

    # 5 -- ablation (only where the claim is "X helps") -------------------------------------------------
    claim = [c for c in checks if c.get("kind") == "baseline" or _IMPROVEMENT.search(_text(c))] or \
            [x for x in metrics if _IMPROVEMENT.search(str(x.get("name", "")))]
    ablated = [c["name"] for c in checks if _ABLATION.search(_text(c))] + \
              [str(x.get("name")) for x in metrics if _ABLATION.search(str(x.get("name", "")))] + \
              [k for k in params if _ABLATION.search(k)]
    if (i := na("ablation")):
        items.append(i)
    elif not has_record:
        items.append(no_evidence("ablation"))
    elif not (claim or len([x for x in (m.methods or []) if x]) >= 2):
        items.append(Item("ablation", NA, ["the experiment claims no improvement, so there is nothing to ablate"]))
    elif ablated:
        items.append(Item("ablation", MET, [f"ablation or variant recorded: {', '.join(map(str, ablated[:3]))}"]))
    else:
        what = f"claims an improvement ({str(claim[0].get('name'))[:40]})" if claim else \
            f"compares {len([x for x in (m.methods or []) if x])} methods"
        items.append(Item("ablation", MISSING, [f"{what} with no removal or variant behind it"], FIXES["ablation"]))

    # 6 -- convergence (numerical results only) --------------------------------------------------------
    conv = [c["name"] for c in checks if _CONVERGENCE.search(_text(c))] + \
           [k for k in params if _CONVERGENCE.search(k)] + \
           [str(x.get("name")) for x in metrics if _CONVERGENCE.search(str(x.get("name", "")))]
    if (i := na("convergence")):
        items.append(i)
    elif not has_record:
        items.append(no_evidence("convergence"))
    elif not _is_solver(m):
        items.append(Item("convergence", NA, ["no mesh or time step in this experiment (data or analytic result)"]))
    elif conv:
        items.append(Item("convergence", MET, [f"study recorded: {', '.join(map(str, conv[:3]))}"]))
    else:
        items.append(Item("convergence", MISSING, ["a numerical result with no mesh or time-step study"],
                          FIXES["convergence"]))

    # 7 -- provenance --------------------------------------------------------------------------------
    git, created, env = prov.get("git"), prov.get("created"), prov.get("environment") or {}
    command = m.links.get("script") or (m.scripts[0] if m.scripts else "")
    if (i := na("provenance")):
        items.append(i)
    evidence = [x for x in (f"git {git}" if git else "", f"written {created}" if created else "",
                            f"python {env.get('python') or '?'} on {env.get('platform') or '?'}" if env else "",
                            f"reproduce: {command}" if command else "") if x]
    if git and command:
        items.append(Item("provenance", MET, evidence))
    else:
        lacks = [x for x, ok in (("the git commit", git), ("the reproduce command", command)) if not ok]
        items.append(Item("provenance", MISSING, evidence + [f"missing {', '.join(lacks)}"], FIXES["provenance"]))

    # 8 -- verdict computed from results, losses included ----------------------------------------------
    if (i := na("verdict")):
        items.append(i)
    elif not checks:
        items.append(Item("verdict", UNASSESSED, ["the record has no check: nothing to compute a verdict from"],
                          FIXES["verdict"]))
    else:
        passed = sum(1 for c in checks if c.get("passed"))
        failed = [str(c["name"]) for c in checks if not c.get("passed")]
        decided = all(isinstance(c.get("passed"), bool) for c in checks)
        items.append(Item("verdict", MET if decided else MISSING,
                          [f"{passed}/{len(checks)} checks pass" +
                           (f"; the failing ones are kept: {', '.join(failed[:3])}" if failed else "")],
                          "" if decided else FIXES["verdict"]))

    # 9 -- what was not measured -----------------------------------------------------------------------
    declared_nm = list(q.get("not_measured") or [])
    sections = [(name, text) for name, text in markdown
                if re.search(r"^#{1,6}\s*.*(" + _NOT_MEASURED.pattern + r")", text, re.I | re.M)]
    notes = str(getattr(m, "notes", "") or "")
    if declared_nm:
        items.append(Item("not_measured", MET, [f"declared: {', '.join(declared_nm[:3])}"]))
    elif sections:
        items.append(Item("not_measured", MET, [f"section in {sections[0][0]}: "
                                                f"{_heading(sections[0][1]) or 'a not-measured section'}"]))
    elif notes and _NOT_MEASURED.search(notes):
        items.append(Item("not_measured", MET, [f"manifest notes: {notes[:60]}"]))
    else:
        items.append(Item("not_measured", MISSING, ["no list of what the experiment does not measure"],
                          FIXES["not_measured"]))

    where = os.path.relpath(base, REPO) if os.path.isabs(base) else base
    return Checklist(subject=m.id, items=items, quality=q, where=where)


def _heading(text: str) -> str:
    for line in text.splitlines():
        if line.lstrip().startswith("#") and _NOT_MEASURED.search(line):
            return line.strip("# ").strip()[:70]
    return ""


# ---------------------------------------------------------------------------------------------- the whole repo
def visibility(m: Manifest, docs: Iterable[tuple[str, str]]) -> int:
    """How visible an experiment is: the guides and pages that name it, plus the folders and pages that link its
    guide. Read from a single pass over the markdown of the checkout (``docs(…)``), so the backlog sorts by what a
    reader meets first, not by folder name."""
    needles = [m.id, os.path.basename(m.id), m.title]
    guide = str((m.links or {}).get("guide") or "")
    n = 0
    for _name, text in docs:
        low = text.lower()
        if any(x and str(x).lower() in low for x in needles) or (guide and guide.lower() in low):
            n += 1
    return n


def _docs_texts(root: str | None = None) -> list[tuple[str, str]]:
    root = root or REPO
    out = []
    for rel in ("README.md", "OVERVIEW.md", "OVERVIEW_EN.md", "START_HERE.md", "docs"):
        p = os.path.join(root, rel)
        files = [p] if os.path.isfile(p) else []
        if os.path.isdir(p):
            for dirpath, _dirs, names in os.walk(p):
                files += [os.path.join(dirpath, x) for x in names if x.endswith(".md")]
        for f in files:
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    out.append((os.path.relpath(f, root), fh.read()))
            except OSError:
                continue
    return out


def assess_all(examples: str | None = None, *, docs: bool = True) -> list[tuple[Checklist, int]]:
    """Every ``examples/*`` folder with its checklist and its visibility, most visible first."""
    from .manifest import scan
    texts = _docs_texts() if docs else []
    out = []
    for path, m in scan([examples or os.path.join(REPO, "examples")], lab=""):
        c = assess(m, base=os.path.dirname(path))
        out.append((c, visibility(m, texts)))
    return sorted(out, key=lambda t: (-t[1], t[0].subject))


def backlog(rows: Sequence[tuple[Checklist, int]], *, limit: int | None = None,
            min_visible: int = 0) -> str:
    """The gaps per example, sorted by how visible the example is (the backlog file of the issue)."""
    rows = sorted(rows, key=lambda t: (-t[1], t[0].subject))
    rows = [r for r in rows if r[1] >= min_visible and r[0].missing]
    rows = rows[:limit] if limit else rows
    lines = ["# Quality backlog of the examples", "",
             "Generated by `python -m pinneapple_lab quality --backlog docs/quality_backlog.md` (#923). Each row is an",
             "`examples/*` folder, its badge and the checklist items it does not meet; sorted by how visible the",
             "example is (the guides and pages that name it). The letter is the curation's; this is the checklist",
             "beside it, and it is not a gate -- `python -m pinneapple_lab manifests --check` stays the gate.", "",
             "| visibility | experiment | badge | missing | fix |", "|---|---|---|---|---|"]
    for c, vis in rows:
        fixes = "; ".join(f"{short_name(i)}: {c.by_id(i).fix}" for i in c.missing[:2])
        lines.append(f"| {vis} | `{c.subject}` | {c.badge()} | {', '.join(short_name(i) for i in c.missing)} | "
                     f"{fixes} |")
    if not rows:
        lines.append("| - | - | - | every example meets its applicable items | - |")
    return "\n".join(lines) + "\n"


def render_markdown(checklists: Sequence[Checklist], *, title: str = "Experiment quality checklist",
                    wider: Mapping[str, int] | None = None) -> str:
    """One markdown page: the counts, the legend, and one row per checklist (``lab/QUALITY.md``)."""
    tot = {s: sum(1 for c in checklists for i in c.items if i.status == s) for s in STATUSES}
    lines = [f"# {title}", "",
             f"{len(checklists)} folders · " + " · ".join(f"{tot[s]} {s}" for s in STATUSES),
             "", "The nine items of #923, checked from the files of each folder (manifest, checks, params, metrics,",
             "provenance, markdown). A checklist, not a score: the letter is the curation's, and no item is a gate.",
             "", "| experiment | badge | " + " | ".join(TITLES[i].split(" (")[0] for i in ITEMS) + " |",
             "|---|---|" + "---|" * len(ITEMS)]
    mark = {MET: "yes", MISSING: "**no**", NA: "n/a", UNASSESSED: "?"}
    for c in checklists:
        vis = f" ({wider[c.subject]})" if wider and c.subject in wider else ""
        cells = [mark[next((i.status for i in c.items if i.id == item_id), UNASSESSED)] for item_id in ITEMS]
        lines.append(f"| `{c.subject}`{vis} | {c.badge()} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

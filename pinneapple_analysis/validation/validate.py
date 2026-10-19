"""``pp.validate``: one call, one validation report for a Physics AI model (#696).

``validate(model, problem, data=None, reference=None, level="standard", query=None)`` picks the checks that apply to
the model and the problem from the existing modules, runs them, and returns a :class:`ValidationReport`: per-check
evidence and a pass / warn / fail status, a credibility level per factor (``schema.FACTORS``) and a deploy decision.
A check that cannot apply is reported as not applicable with the reason.

Models: a zoo model (``pp.zoo.load(...)`` or its name), a ``torch.nn.Module`` (PyTorch, PhysicsNeMo), an ONNX file
(``.onnx``, run with onnxruntime) or any callable. Problems: a zoo model's own card (``problem=None``), ``"darcy_2d"``
(or ``{"pde": "darcy", "f": 1.0}``: field operators ``a -> u`` on the cell-centred grid of
``operator_benchmarks.darcy_2d``), or a small spec ``{"valid_range": {...}, "relations": [...]}`` for a tabular
callable.

What runs (``level``):

=================  ==============================================================================================
``quick``          applicability (validity range, training envelope), solution verification (guaranteed residual
                   bound, flux balance), reference error, the zoo gate, the method's catalog status
``standard``       + spectral fidelity, metamorphic relations, the data audit, the calibrated error bound
``thorough``       + robustness to input perturbations; fault injection (#697) and rollout checks (#701) when they
                   apply
=================  ==============================================================================================

The modules it composes: ``uncertainty.residual_bound`` (#699), ``validation.spectral`` (#702),
``validation.metamorphic`` and ``pinneapple_hub.zoo.relations`` (#698), ``pinneapple_data.audit`` (#700),
``pinneapple_hub.zoo.validate`` (the zoo gate), ``physics_audit.assumptions.Envelope`` (applicability) and
``pinneapple_catalog`` (method status).
"""
from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import schema
from .schema import FAIL, NA, PASS, WARN

__all__ = ["ValidationCheck", "ValidationReport", "validate", "DEFAULT_TOLERANCES", "as_field_model", "pp_validate"]

#: ``energy_bound``: relative guaranteed error bound above which the prediction is not trusted (at 1 the bound allows
#: an error as large as the solution itself; the bound is loose, about 10 x the actual error of a good FNO: 0.40 for a
#: 4 % error in distribution, 4.6 under a coefficient-contrast shift); ``flux``: relative imbalance of the boundary flux; ``reference``:
#: mean relative L2 error against the reference; ``spectral``: |high-wavenumber bias|; ``symmetry``: relative
#: violation of a symmetry of the problem; ``noise``: allowed ratio of the model's sensitivity to the solver's.
DEFAULT_TOLERANCES: dict[str, float] = {"energy_bound": 1.0, "flux": 0.1, "reference": 0.1, "spectral": 0.25,
                                         "symmetry": 0.1, "noise": 3.0, "alpha": 0.1}
LEVELS = ("quick", "standard", "thorough")


# ---------------------------------------------------------------------------------------------- report
@dataclass
class ValidationCheck:
    id: str
    status: str
    summary: str
    value: Any = None
    threshold: Any = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in schema.STATUSES:
            raise ValueError(f"status {self.status!r} not in {schema.STATUSES}")
        schema.spec_of(self.id)                                  # unknown ids are refused

    @property
    def spec(self) -> schema.CheckSpec:
        return schema.spec_of(self.id)

    @property
    def factor(self) -> str:
        return self.spec.factor

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "factor": self.factor, "status": self.status, "summary": self.summary,
                "value": _json(self.value), "threshold": _json(self.threshold), "critical": self.spec.critical,
                "evidence": _json(self.evidence)}


def _json(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json(x.tolist())
    if isinstance(x, np.generic):
        return _json(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


@dataclass
class ValidationReport:
    model: str
    problem: str
    level: str
    checks: list[ValidationCheck]
    provenance: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------- decisions
    @property
    def decision(self) -> str:
        """``fail`` when a critical check fails, ``warn`` when a non-critical one fails or any warns, else ``pass``."""
        if any(c.status == FAIL and c.spec.critical for c in self.checks):
            return FAIL
        if any(c.status in (FAIL, WARN) for c in self.checks):
            return WARN
        return PASS

    @property
    def verdict(self) -> str:
        """The deploy verdict in one line, e.g. ``fail: out of validity range``."""
        bad = [c for c in self.checks if c.status == FAIL and c.spec.critical] or \
              [c for c in self.checks if c.status in (FAIL, WARN)]
        if not bad:
            return "pass: every applicable check passed"
        reasons = []
        for c in bad:
            r = "out of validity range" if c.factor == "applicability" else c.spec.title
            if r not in reasons:
                reasons.append(r)
        return f"{self.decision}: " + "; ".join(reasons)

    def credibility(self) -> dict[str, dict[str, Any]]:
        """Credibility level per factor (``schema.LEVELS``) with the checks behind it."""
        out = {}
        for f in schema.FACTORS:
            cs = [c for c in self.checks if c.factor == f and c.status != NA]
            if not cs:
                lvl = 0
            elif any(c.status == FAIL and c.spec.critical for c in cs):
                lvl = 1
            elif any(c.status in (FAIL, WARN) for c in cs):          # a non-critical failure asks for review
                lvl = 2
            else:
                lvl = max(c.spec.strength for c in cs)
            out[f] = {"level": lvl, "meaning": schema.LEVELS[lvl], "checks": [c.id for c in cs]}
        return out

    def __getitem__(self, check_id: str) -> ValidationCheck:
        return next(c for c in self.checks if c.id == check_id)

    def get(self, check_id: str) -> ValidationCheck | None:
        return next((c for c in self.checks if c.id == check_id), None)

    # -------------------------------------------------------------- outputs
    def to_dict(self) -> dict[str, Any]:
        return {"schema": "validation-report/v1", "model": self.model, "problem": self.problem, "level": self.level,
                "decision": self.decision, "trust_decision": schema.DECISIONS[self.decision],
                "verdict": self.verdict, "credibility": self.credibility(),
                "checks": [c.to_dict() for c in self.checks], "provenance": _json(self.provenance)}

    def to_json(self, path=None) -> str:
        s = json.dumps(self.to_dict(), indent=1)
        if path is not None:
            from pathlib import Path
            Path(path).write_text(s + "\n")
        return s

    def __str__(self) -> str:
        lines = [f"validation of {self.model} on {self.problem} ({self.level}): {self.verdict}"]
        for f, v in self.credibility().items():
            lines.append(f"  {f:22s} level {v['level']}  ({v['meaning']})")
        for c in self.checks:
            lines.append(f"  {c.status.upper():15s} {c.id}: {c.summary}")
        return "\n".join(lines)

    def to_html(self) -> str:
        cred = "".join(f"<tr><td>{f}</td><td>{v['level']}</td><td>{html.escape(v['meaning'])}</td></tr>"
                       for f, v in self.credibility().items())
        rows = "".join(f"<tr class='{c.status}'><td>{html.escape(c.id)}</td><td>{c.status}</td>"
                       f"<td>{html.escape(c.summary)}</td></tr>" for c in self.checks)
        return (f"<!doctype html><meta charset='utf-8'><title>Validation: {html.escape(self.model)}</title>"
                "<style>body{font-family:sans-serif;max-width:60em;margin:2em auto;padding:0 1em}"
                "td,th{border-bottom:1px solid #ccc;padding:.3em;text-align:left;vertical-align:top}"
                ".fail td:nth-child(2){color:#b00}.warn td:nth-child(2){color:#a60}.pass td:nth-child(2){color:#070}"
                "</style>"
                f"<h1>{html.escape(self.model)}</h1><p>{html.escape(self.problem)}, level {self.level}: "
                f"<b>{html.escape(self.verdict)}</b></p><h2>Credibility</h2><table>{cred}</table>"
                f"<h2>Checks</h2><table><tr><th>Check</th><th>Status</th><th>Result</th></tr>{rows}</table>")

    def to_trust_report(self):
        """The same evidence as a :class:`pinneapple_analysis.trust.TrustReport` (the trust card's decision)."""
        from pinneapple_analysis.trust.trust_report import (
            CONTRADICTS,
            NOT_RUN,
            SUPPORTS,
            Check,
            TrustReport,
        )
        st = {PASS: SUPPORTS, WARN: SUPPORTS, FAIL: CONTRADICTS, NA: NOT_RUN}
        checks = [Check(c.id, st[c.status], critical=c.spec.critical,
                        value=c.value if isinstance(c.value, (int, float)) else None,
                        limit=c.threshold if isinstance(c.threshold, (int, float)) else None,
                        source="pp.validate", reason=c.summary) for c in self.checks]
        return TrustReport(checks, model=self.model, details={"validation_decision": self.decision})

    def to_check_report(self):
        """A ``pinneapple check`` report (SARIF, JUnit, GitHub annotations through ``pinneapple_check.report``):
        one outcome per check that has a rule; warnings keep their rule's severity but pass."""
        from pinneapple_check.result import Location, Outcome, Report
        from pinneapple_check.rules import RULES
        rep = Report(targets=[{"name": self.model, "kind": "model", "path": self.model, "physics": self.problem,
                               "declared": True}])
        for c in self.checks:
            if c.status == NA or not c.spec.rule:
                continue
            rep.outcomes.append(Outcome(c.spec.rule, self.model, c.id, c.status != FAIL, c.summary,
                                        Location(self.provenance.get("source", "pp.validate")), c.value,
                                        c.threshold, RULES[c.spec.rule].severity))
        rep.notes = [f"{c.id}: not applicable ({c.summary})" for c in self.checks if c.status == NA]
        return rep

    def render(self, fmt: str) -> str:
        """``json``, ``html``, ``text``, or a ``pinneapple check`` format (``sarif``, ``junit``, ``github``)."""
        if fmt == "json":
            return self.to_json()
        if fmt == "html":
            return self.to_html()
        if fmt == "text":
            return str(self)
        from pinneapple_check.report import render
        return render(self.to_check_report(), fmt)


# ---------------------------------------------------------------------------------------------- models
def _is_zoo(model) -> bool:
    return hasattr(model, "card") and hasattr(model, "predict") and hasattr(model, "in_range")


def as_field_model(model) -> Callable[[np.ndarray], np.ndarray]:
    """A field operator ``(B, n, n) -> (B, n, n)`` in NumPy from a torch module (PyTorch, PhysicsNeMo; outputs with a
    ``.y`` are unwrapped), an ONNX file or a callable. Channel axes of size one are added or removed as needed."""
    if isinstance(model, str) and model.lower().endswith(".onnx"):
        import onnxruntime as ort
        sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        rank = len(sess.get_inputs()[0].shape)

        def run_onnx(a):
            x = np.ascontiguousarray(a, np.float32)
            x = x[:, None] if rank == 4 and x.ndim == 3 else x
            return np.asarray(sess.run(None, {name: x})[0], float).reshape(np.shape(a))
        run_onnx.name = model
        return run_onnx
    try:
        import torch
        is_torch = isinstance(model, torch.nn.Module)
    except ImportError:
        is_torch = False
    if is_torch:
        model.eval()

        def run_torch(a):
            x = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
            with torch.no_grad():
                try:
                    y = model(x)
                except (RuntimeError, ValueError):                # expects a channel axis (PhysicsNeMo: ValueError)
                    y = model(x[:, None])
            y = getattr(y, "y", y)
            return y.detach().cpu().numpy().astype(float).reshape(np.shape(a))
        run_torch.name = type(model).__name__
        return run_torch
    if callable(model):
        return lambda a: np.asarray(model(np.asarray(a)), float).reshape(np.shape(a))
    raise TypeError(f"cannot run {type(model).__name__} as a field model")


def _describe(x: Any) -> Any:
    """``x`` as JSON for the provenance: callables by name, arrays by shape, other objects by type."""
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, Mapping):
        return {str(k): _describe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_describe(v) for v in x]
    if isinstance(x, np.ndarray):
        return f"array{list(x.shape)}"
    if callable(x):
        return f"<function {getattr(x, '__name__', type(x).__name__)}>"
    return getattr(x, "name", None) or type(x).__name__


def _name(model) -> str:
    if _is_zoo(model):
        return f"{model.name} v{model.card.version}"
    if isinstance(model, str):
        return model
    return getattr(model, "name", None) or type(model).__name__


# ---------------------------------------------------------------------------------------------- helpers
def _status(ok: bool, warn: bool = False) -> str:
    return PASS if ok else WARN if warn else FAIL


def _rel_l2(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    p, t = np.asarray(p, float), np.asarray(t, float)
    return np.linalg.norm((p - t).reshape(len(t), -1), axis=1) / np.linalg.norm(t.reshape(len(t), -1), axis=1)


def _method_status(method: str | None) -> ValidationCheck:
    if not method:
        return ValidationCheck("method.status", NA, "no catalog method id given (validate(..., method='S2'))")
    from pinneapple_catalog import methods
    st = methods.method_status().get(method)
    if st is None:
        return ValidationCheck("method.status", FAIL, f"{method} is not in the method catalog")
    ok = st.get("status") == "validated"
    return ValidationCheck("method.status", _status(ok, warn=True), f"{method}: {st.get('status')}",
                           value=st.get("status"), threshold="validated", evidence=dict(st))


def _field_features(a: np.ndarray) -> dict[str, float]:
    from pinneapple_analysis.benchmark.validation import _features
    f = _features(a)
    f.update(minimum=float(np.min(a)), maximum=float(np.max(a)))
    return f


def _envelope_check(train_a: np.ndarray | None, query: np.ndarray) -> ValidationCheck:
    if train_a is None:
        return ValidationCheck("applicability.envelope", NA, "no training inputs given (data={'train': ...}) to "
                               "define the envelope")
    from pinneapple_physics.physics_audit.assumptions import Envelope
    env = Envelope.from_samples([_field_features(a) for a in train_a])
    res = [env.check(_field_features(q)) for q in query]
    out = [i for i, r in enumerate(res) if not r["inside"]]
    ranges = {k: [round(v[0], 4), round(v[1], 4)] for k, v in env.ranges.items()}
    if out:
        f = _field_features(query[out[0]])
        worst = [k for k, (lo, hi) in env.ranges.items() if not lo <= f[k] <= hi]
        detail = ", ".join(f"{k} = {f[k]:.4g} (trained on [{env.ranges[k][0]:.4g}, {env.ranges[k][1]:.4g}])"
                           for k in worst)
        return ValidationCheck("applicability.envelope", FAIL, f"out of validity range: {len(out)} of {len(query)} "
                               f"queries outside the training envelope; query {out[0]}: {detail}",
                               value=len(out) / len(query), threshold=0.0,
                               evidence={"outside": out[:50], "ranges": ranges})
    return ValidationCheck("applicability.envelope", PASS, f"all {len(query)} queries inside the training envelope "
                           f"(amplitude, spectral centroid, min, max)", value=0.0, threshold=0.0,
                           evidence={"ranges": ranges})


def _audit_checks(checks, source: str = "") -> list[ValidationCheck]:
    """Checks of a dataset audit card (``pinneapple_data.audit``) under the ``data.*`` ids; units are left to the
    dataset's own card."""
    fam = {"physics": "data.physics", "constraint": "data.physics", "diversity": "data.diversity",
           "leakage": "data.leakage", "finite": "data.finite"}
    out = []
    for c in checks:
        head = c["id"].split(".")[0].split(":")[0]
        if head not in fam:
            continue
        if head in ("leakage", "finite"):
            cid, extra = fam[head], ""
        elif head == "constraint":                            # a stable id; the expression stays in the summary
            expr = c["id"].split(":", 1)[1].strip()
            cid, extra = f"data.physics.constraint.{re.sub(r'[^a-z0-9]+', '-', expr.lower()).strip('-')}", \
                f"{expr}: "
        else:
            cid, extra = f"{fam[head]}.{c['id'].split('.', 1)[-1]}", ""
        out.append(ValidationCheck(cid, c["status"], (f"{source}: " if source else "") + extra + c["summary"],
                                   value=c.get("value"), threshold=c.get("threshold")))
    return out


# ---------------------------------------------------------------------------------------------- Darcy
def _darcy_checks(f, data, reference, query, level, tol, rhs, seed) -> list[ValidationCheck]:
    from pinneapple_analysis.uncertainty.residual_bound import (
        ResidualErrorCalibrator,
        darcy_problem,
        energy_error_bound,
    )
    data = {k: dict(v) for k, v in (data or {}).items()}
    train_a = data.get("train", {}).get("a")
    test = data.get("test")
    if reference is not None:                                   # reference = (a, u) pairs
        test = {"a": np.asarray(reference[0]), "u": np.asarray(reference[1])}
    Q = np.asarray(query, float) if query is not None else (test["a"] if test else None)
    if Q is None:
        raise ValueError("give query inputs (query=...) or a test split (data={'test': {'a', 'u'}})")
    if Q.ndim == 2:
        Q = Q[None]
    C: list[ValidationCheck] = [_envelope_check(train_a, Q)]
    # ---- solution verification on the queries (no ground truth)
    pq = f(Q)
    bounds, flux = [], []
    n = Q.shape[-1]
    for a, u in zip(Q, pq, strict=True):
        P = darcy_problem(a, rhs)
        bounds.append(energy_error_bound(P, u)["relative"])
        out = sum(float((2 * a[s] * u[s]).sum()) for s in ((0, slice(None)), (-1, slice(None)), (slice(None), 0),
                                                          (slice(None), -1)))
        src = float(np.broadcast_to(np.asarray(rhs, float), (n, n)).sum() / n ** 2)
        flux.append(abs(out - src) / abs(src))
    bounds, flux = np.asarray(bounds), np.asarray(flux)
    mb = float(np.max(bounds))
    C.append(ValidationCheck("residual.energy_bound", _status(mb <= tol["energy_bound"]),
                             f"guaranteed relative energy-norm error bound: median {np.median(bounds):.3g}, max "
                             f"{mb:.3g} over {len(Q)} queries (from the residual alone)", value=mb,
                             threshold=tol["energy_bound"], evidence={"per_query": bounds.tolist()[:200]}))
    mf = float(np.max(flux))
    C.append(ValidationCheck("conservation.flux", _status(mf <= tol["flux"]), f"net boundary flux against the source: "
                             f"max imbalance {mf:.3g} over {len(Q)} queries", value=mf, threshold=tol["flux"],
                             evidence={"per_query": flux.tolist()[:200]}))
    C.append(ValidationCheck("boundary.values", NA, "cell-centred unknowns: the zero boundary value enters through "
                             "the boundary flux (conservation.flux) and the residual"))
    # ---- reference
    pt = None
    if test is not None:
        pt = f(test["a"])
        e = _rel_l2(pt, test["u"])
        C.append(ValidationCheck("reference.rel_l2", _status(float(e.mean()) <= tol["reference"]),
                                 f"relative L2 error against {len(e)} reference solutions: mean {e.mean():.3g}, "
                                 f"max {e.max():.3g}", value=float(e.mean()), threshold=tol["reference"],
                                 evidence={"median": float(np.median(e)), "max": float(e.max())}))
    else:
        C.append(ValidationCheck("reference.rel_l2", NA, "no reference solutions given"))
    if level == "quick":
        return C
    if test is not None:
        from .spectral import spectral_comparison
        sp = spectral_comparison(pt, test["u"], 2, periodic=False)
        C.append(ValidationCheck("reference.spectral", _status(abs(sp.hf_bias) <= tol["spectral"]),
                                 f"high-wavenumber energy {sp.band_ratio['high']:.3g} x the reference's (bias "
                                 f"{sp.hf_bias:+.3g})", value=sp.hf_bias, threshold=tol["spectral"],
                                 evidence={"band_ratio": sp.band_ratio}))
    else:
        C.append(ValidationCheck("reference.spectral", NA, "no reference solutions given"))
    # ---- metamorphic: the square's symmetries (f symmetric) map solutions to solutions
    from .metamorphic import Symmetry, check_relations
    F = np.broadcast_to(np.asarray(rhs, float), (n, n))
    rels = []                                                   # only the symmetries the source itself has
    if np.allclose(F, np.flip(F, 0)):
        rels.append(Symmetry("reflect-x", tol["symmetry"], transform_in=lambda x: np.flip(x, 1),
                             transform_out=lambda x, y: np.flip(y, 1)))
    if np.allclose(F, F.T):
        rels.append(Symmetry("transpose", tol["symmetry"], transform_in=lambda x: np.swapaxes(x, 1, 2),
                             transform_out=lambda x, y: np.swapaxes(y, 1, 2)))
    if rels:
        Qs = Q[: min(len(Q), 8)]
        rep = check_relations(f, rels, lambda rng, k: Qs, n=len(Qs), seed=seed)
        for r in rep.results:
            C.append(ValidationCheck(f"metamorphic.symmetry.{r.name}", _status(r.passed),
                                     f"worst relative violation {r.worst:.3g} over {r.n_tested} queries",
                                     value=r.worst, threshold=r.tol))
    else:
        C.append(ValidationCheck("metamorphic", NA, "the source has neither symmetry: no symmetry to test"))
    # ---- data audit
    if data:
        from dataclasses import replace

        from pinneapple_data.audit import DARCY, audit_dataset, darcy_consistency
        phys = replace(DARCY, check=lambda s: darcy_consistency(s["a"], s["u"], rhs))     # the problem's own source
        card = audit_dataset(data, physics=phys, max_physics_samples=16, seed=seed)
        C += _audit_checks([{"id": c.id, "status": c.status, "summary": c.summary, "value": c.value,
                             "threshold": c.threshold} for c in card.checks])
    else:
        C.append(ValidationCheck("data.physics", NA, "no training data given"))
    # ---- calibrated error bound (needs reference pairs: half calibrates, half measures coverage)
    if test is not None and len(test["a"]) >= 40:
        res = np.asarray([energy_error_bound(darcy_problem(a, rhs), u)["relative"]
                          for a, u in zip(test["a"], pt, strict=True)])
        err = _rel_l2(pt, test["u"])
        from pinneapple_analysis.uncertainty.residual_bound import coverage_study
        study = coverage_study(res, err, alpha=tol["alpha"], n_splits=200, seed=seed)
        cal = ResidualErrorCalibrator(tol["alpha"]).calibrate(res, err)
        qb = cal.bound(bounds)
        inside = C[0].status != FAIL
        ok = study["mean"] >= 1 - tol["alpha"] - 0.03
        C.append(ValidationCheck("uncertainty.residual_bound", _status(ok),
                                 f"{100 * (1 - tol['alpha']):.0f} % error bound from the residual: mean coverage "
                                 f"{100 * study['mean']:.1f} % over {study['n_splits']} calibration / test splits of "
                                 f"{len(res)} reference pairs; queries: median bound {np.median(qb):.3g} relative L2"
                                 + ("" if inside else " (not valid: queries outside the envelope)"),
                                 value=study["mean"], threshold=1 - tol["alpha"],
                                 evidence={"coverage_study": study, "query_bounds": qb.tolist()[:200]}))
    else:
        C.append(ValidationCheck("uncertainty.residual_bound", NA, "needs at least 40 reference pairs to calibrate"))
    if level != "thorough":
        return C
    # ---- robustness: sensitivity to a 1 % multiplicative perturbation, against the solver's own
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import darcy_2d
    rng = np.random.default_rng(seed)
    Qs = Q[: min(len(Q), 4)]
    pert = Qs * (1 + 0.01 * rng.standard_normal(Qs.shape))
    dm = _rel_l2(f(pert), f(Qs))
    ds = _rel_l2(np.stack([darcy_2d(a, rhs) for a in pert]), np.stack([darcy_2d(a, rhs) for a in Qs]))
    ratio = float(np.max(dm / np.maximum(ds, 1e-12)))
    C.append(ValidationCheck("robustness.input_noise", _status(ratio <= tol["noise"]),
                             f"1 % input noise moves the prediction {ratio:.3g} x as much as it moves the solution",
                             value=ratio, threshold=tol["noise"]))
    C.append(_fault_injection())
    C.append(ValidationCheck("rollout", NA, "a steady operator has no rollout"))
    return C


def _rollout_checks(step, cfg: Mapping[str, Any], data: Mapping[str, Any], level: str) -> list[ValidationCheck]:
    """``problem={"rollout": {"ndim", "dt", "conserved", "invariants", "chaotic", "steps", "burn_in"}}``, ``data={"u0",
    "reference", "long_u0", "long_reference"}``: the model is a stepper ``u -> u_next`` (``validation.rollout``)."""
    from .rollout import rollout, validate_rollout
    ref = data.get("reference")
    long_ref = data.get("long_reference") if level != "quick" else None
    steps = int(cfg.get("steps", (len(ref) - 1) if ref is not None else 100))
    pred = rollout(step, data["u0"], steps)
    long_pred = rollout(step, data.get("long_u0", data["u0"]), len(long_ref) - 1) if long_ref is not None else None
    kw = dict(dt=float(cfg.get("dt", 1.0)), conserved=cfg.get("conserved", ()), chaotic=bool(cfg.get("chaotic")),
              periodic=bool(cfg.get("periodic", True)), tolerances=cfg.get("tolerances"))
    if cfg.get("invariants") is not None:                     # {name: f(traj, ndim)}, e.g. a particle system's energy
        kw["invariants"] = cfg["invariants"]
    rep = validate_rollout(pred, int(cfg["ndim"]), ref, **kw)
    checks = rep.checks
    if long_pred is not None:
        st = validate_rollout(long_pred, int(cfg["ndim"]), long_ref=long_ref, burn_in=int(cfg.get("burn_in", 0)), **kw)
        checks += [c for c in st.checks if c["id"] == "rollout.statistics"]
    else:
        checks.append({"id": "rollout.statistics", "status": NA,
                       "summary": "no long reference run (data['long_reference'])" if level != "quick" else
                       "long-run statistics run from level 'standard'"})
    return [ValidationCheck(c["id"], c["status"], c["summary"], value=c.get("value"), threshold=c.get("threshold"))
            for c in checks]


def _generative_checks(sampler, cfg: Mapping[str, Any], data: Mapping[str, Any], level: str, seed: int
                       ) -> list[ValidationCheck]:
    """``problem={"generative": {"physics": "darcy", "f": 1.0, "n_samples": 32}}``, ``data={"train": {...}}``: a
    generative model given as a sampler ``n -> {column: (n, ...)}``. Its samples are audited like a dataset
    (``pinneapple_data.audit``): every sample against the PDE (``data.physics.*``), diversity and mode collapse
    (``data.diversity.*``), and memorisation as leakage, generated samples that copy training samples
    (``data.leakage``)."""
    from dataclasses import replace

    from pinneapple_data.audit import BUILTIN_PHYSICS, DARCY, audit_dataset, darcy_consistency
    n = int(cfg.get("n_samples", 32 if level == "quick" else 64))
    gen = {k: np.asarray(v, float) for k, v in sampler(n).items()}
    phys = cfg.get("physics")
    if phys == "darcy":
        rhs = cfg.get("f", 1.0)
        phys = replace(DARCY, check=lambda s: darcy_consistency(s["a"], s["u"], rhs))
    elif isinstance(phys, str):
        phys = BUILTIN_PHYSICS[phys]
    splits = {"test": gen}
    if (data or {}).get("train") is not None:
        splits["train"] = {k: np.asarray(v, float) for k, v in data["train"].items()}
    card = audit_dataset(splits, name="generated samples", physics=phys, max_physics_samples=n, seed=seed,
                         duplicate_splits=("test",))
    return _audit_checks([{"id": c.id, "status": c.status, "summary": c.summary, "value": c.value,
                           "threshold": c.threshold} for c in card.checks], source=f"{n} generated samples")


def _pinn_checks(model, spec: Mapping[str, Any], data: Mapping[str, Any], level: str, seed: int
                 ) -> list[ValidationCheck]:
    """``problem={"pinn": "burgers_1d" | PINNProblem, "tolerances": {...}}``, ``data={"train_points": X}``: a trained
    PINN on fresh points (``validation.pinn``)."""
    from .pinn import validate_pinn
    rep = validate_pinn(model, spec["pinn"], n_points=1024 if level == "quick" else 4096,
                        train_points=data.get("train_points"), tolerances=spec.get("tolerances"), seed=seed)
    return [ValidationCheck(c["id"], c["status"], c["summary"], value=c.get("value"), threshold=c.get("threshold"))
            for c in rep.checks]


def _fault_injection() -> ValidationCheck:
    try:
        from . import fault_injection  # noqa: F401  (#697)
    except ImportError:
        return ValidationCheck("robustness.fault_injection", NA, "fault-injection suite not installed (#697)")
    return ValidationCheck("robustness.fault_injection", NA, "run validation.fault_injection for the detection matrix")


# ---------------------------------------------------------------------------------------------- zoo
def _zoo_checks(model, query, level, seed) -> list[ValidationCheck]:
    from pinneapple_hub import zoo
    C: list[ValidationCheck] = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = zoo.validate(model, inputs=query)
    st = {"supports": PASS, "contradicts": FAIL, "not_run": NA}
    physics_i = 0
    for c in tr.checks:
        if c.name.startswith("card metrics"):
            cid = "reference.card_metrics"
        elif c.name.startswith("held-out"):
            cid = "reference.acceptance." + c.name.split()[1]
        elif c.name.startswith("query inputs"):
            cid = "applicability.range"
        else:
            physics_i += 1
            cid = f"physics.model_checks.{physics_i}"
        summary = c.name if cid != "applicability.range" or c.status != "contradicts" else \
            f"out of validity range: {100 * tr.details.get('fraction_out_of_range', 0):.0f} % of the queries outside " \
            f"the card's validity range"
        C.append(ValidationCheck(cid, st[c.status], summary + (f" ({c.reason})" if c.reason else ""), value=c.value,
                                 threshold=c.limit))
    if query is None:
        C.append(ValidationCheck("applicability.range", NA, "no query inputs given"))
    X, _ = model.reference("test")
    outputs = set(model.output_names) | set(model._forward(np.asarray(X[:1], float)))
    if any(o.endswith("_std") or "upper95" in o for o in outputs):
        C.append(ValidationCheck("uncertainty.spread", PASS, "the model returns a predictive spread or bound"))
    else:
        C.append(ValidationCheck("uncertainty.spread", NA, "the model returns no spread"))
    if level == "quick":
        return C
    from pinneapple_hub.zoo.relations import ZOO_RELATIONS, check_zoo_relations
    if model.name in ZOO_RELATIONS:
        rep = check_zoo_relations(model, n=64, seed=seed)
        for r in rep.results:
            C.append(ValidationCheck(f"metamorphic.{r.kind}.{r.name}", _status(r.passed),
                                     f"worst violation {r.worst:.3g} over {r.n_tested} samples", value=r.worst,
                                     threshold=r.tol))
    else:
        C.append(ValidationCheck("metamorphic", NA, f"no relations declared for {model.name}"))
    from pinneapple_hub.zoo import datasets
    card = datasets.audit_card(model.card.data_lineage.get("dataset", ""))
    if card is None:
        C.append(ValidationCheck("data.physics", NA, "the training dataset has no stored audit card"))
    else:
        C += _audit_checks(card["checks"], source=f"stored audit card of {card['dataset']}")
    return C


# ---------------------------------------------------------------------------------------------- generic tabular
def _tabular_checks(f, spec, data, query, level, tol, seed) -> list[ValidationCheck]:
    C: list[ValidationCheck] = []
    box = spec.get("valid_range") or {}
    if query is not None and box:
        q = {k: np.asarray(v, float) for k, v in query.items()}
        out = np.zeros(len(next(iter(q.values()))), bool)
        for k, (lo, hi) in box.items():
            if k in q:
                out |= (q[k] < lo) | (q[k] > hi)
        C.append(ValidationCheck("applicability.range", FAIL if out.any() else PASS,
                                 f"out of validity range: {int(out.sum())} of {len(out)} queries outside {box}"
                                 if out.any() else f"all {len(out)} queries inside the validity range",
                                 value=float(out.mean()), threshold=0.0))
    else:
        C.append(ValidationCheck("applicability.range", NA, "needs a valid_range and query inputs"))
    ref = (data or {}).get("test")
    if ref is not None and spec.get("outputs"):
        y = f({k: v for k, v in ref.items() if k not in spec["outputs"]})
        errs = [float(np.linalg.norm(np.asarray(y[o]) - ref[o]) / np.linalg.norm(ref[o])) for o in spec["outputs"]]
        C.append(ValidationCheck("reference.rel_l2", _status(max(errs) <= tol["reference"]),
                                 "relative L2 error per output: " + ", ".join(
                                     f"{o} {e:.3g}" for o, e in zip(spec["outputs"], errs, strict=True)),
                                 value=max(errs), threshold=tol["reference"]))
    else:
        C.append(ValidationCheck("reference.rel_l2", NA, "no reference data given"))
    for cid in ("residual.pde", "conservation.flux"):
        C.append(ValidationCheck(cid, NA, "no PDE stated for a tabular model"))
    if level != "quick" and spec.get("relations") and box:
        from .metamorphic import box_sampler, check_relations, relations_from_spec
        rep = check_relations(f, relations_from_spec(spec["relations"]), box_sampler(box), n=64, seed=seed)
        for r in rep.results:
            C.append(ValidationCheck(f"metamorphic.{r.kind}.{r.name}", _status(r.passed),
                                     f"worst violation {r.worst:.3g} over {r.n_tested} samples", value=r.worst,
                                     threshold=r.tol))
    return C


# ---------------------------------------------------------------------------------------------- entry point
def validate(model, problem=None, data: Mapping[str, Mapping[str, np.ndarray]] | None = None,
             reference: Sequence[np.ndarray] | None = None, level: str = "standard", *,
             query: Any = None, method: str | None = None, tolerances: Mapping[str, float] | None = None,
             seed: int = 0) -> ValidationReport:
    """Validate ``model`` on ``problem``.

    ``data``: ``{"train": {...}, "test": {...}}`` (Darcy: arrays ``a`` and ``u`` of shape ``(n, ny, nx)``);
    ``reference``: ``(inputs, outputs)`` pairs to compare with (instead of ``data["test"]``); ``query``: the inputs
    the model will be used on, checked without ground truth (validity range, residual, calibrated bound);
    ``method``: the model's catalog id (``pinneapple_catalog``) for code verification."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    if isinstance(model, str) and not model.lower().endswith(".onnx"):
        from pinneapple_hub import zoo
        model = zoo.load(model)
    import importlib.metadata as md
    try:
        version = md.version("pinneapple")
    except md.PackageNotFoundError:
        version = "unknown"
    from .report_artifact import hash_data, hash_model
    prov: dict[str, Any] = {"library_version": version, "level": level, "source": "pp.validate",
                            "model_hash": hash_model(model), "data_hash": hash_data(data, reference, query),
                            "physics": {"problem": _describe(problem),
                                        "tolerances": tol, "seed": seed}}
    checks: list[ValidationCheck] = [_method_status(method)]
    if problem is None and _is_zoo(model):
        pname = f"zoo card of {model.name}"
        checks += _zoo_checks(model, query, level, seed)
        prov["model_card"] = {"name": model.name, "version": model.card.version}
        prov["validity_range"] = {k: v for k, v in model.card.validity_range.items()
                                  if isinstance(v, (list, tuple)) and len(v) == 2}
    else:
        spec = {"pde": "darcy"} if problem in ("darcy", "darcy_2d") else dict(problem or {}) \
            if isinstance(problem, Mapping) else _from_physical_problem(problem)
        if "generative" in spec:
            pname = spec.get("name", "generative model")
            checks += _generative_checks(model, spec["generative"], data or {}, level, seed)
        elif "pinn" in spec:
            checks += _pinn_checks(model, spec, data or {}, level, seed)
            pname = f"PINN, {getattr(spec['pinn'], 'name', spec['pinn'])}"
            prov["physics"]["problem"] = _describe({**spec, "pinn": getattr(spec["pinn"], "name", spec["pinn"])})
        elif "rollout" in spec:
            pname = spec.get("name", "autoregressive rollout")
            checks += _rollout_checks(model, spec["rollout"], data or {}, level)
        elif spec.get("pde") == "darcy":
            pname = "Darcy 2-D, -div(a grad u) = f, u = 0 (cell-centred, darcy_2d)"
            checks += _darcy_checks(as_field_model(model), data, reference, query, level, tol, spec.get("f", 1.0),
                                    seed)
        else:
            pname = spec.get("name", "tabular model")
            f = model if callable(model) else as_field_model(model)
            checks += _tabular_checks(f, spec, data, query, level, tol, seed)
    if level == "thorough" and not any(c.id.startswith("robustness") for c in checks):
        checks.append(ValidationCheck("robustness.input_noise", NA, "no perturbation study for this problem"))
        checks.append(_fault_injection())
    return ValidationReport(_name(model), pname, level, checks, prov)


def _from_physical_problem(problem) -> dict[str, Any]:
    """A ``PhysicalProblem`` (or preset name): its domain becomes the validity range of the coordinate inputs."""
    if isinstance(problem, str):
        import pinneapple as pp
        problem = pp.PhysicalProblem.from_preset(problem)
    dom = getattr(problem, "domain_bounds", None) or getattr(problem, "bounds", None) or {}
    return {"name": getattr(problem, "name", type(problem).__name__),
            "valid_range": {k: tuple(v) for k, v in dict(dom).items()}}


class _PPValidate:
    """``pp.validate``: call it to validate a model (:func:`validate`); its attributes are those of
    ``pinneapple_analysis.validation`` (``pp.validate.PhysicsValidator`` as before)."""

    def __call__(self, *args, **kwargs) -> ValidationReport:
        return validate(*args, **kwargs)

    def __getattr__(self, name: str):
        import importlib
        return getattr(importlib.import_module("pinneapple_analysis.validation"), name)

    def __dir__(self):
        import importlib
        return sorted(set(dir(importlib.import_module("pinneapple_analysis.validation"))) | {"__call__"})

    __doc__ = validate.__doc__


pp_validate = _PPValidate()

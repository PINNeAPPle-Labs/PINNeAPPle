"""Physics learning verification: has a model learned the physics, or only a low loss?

A low physics loss (the residual on the model's own collocation points) and a low held-out error (on the training
distribution) do not show that a model responds to causes as the physics does. This verifier tests the *learned
relations* themselves, independently of the training loss:

``intervention:*``
    Change one cause (power, source location, a conductivity, the cooling, a contact resistance the model was
    never given) and compare the model's response with the reference solver's response to the same change, at
    informative points and on the peak temperature: magnitude and sign. The expected responses are solved, not
    hand-written rules.
``residual``, ``interface_flux``, ``boundary:*``, ``energy_balance``
    The governing equation inside each layer, continuity of heat flux across each interface, the boundary
    conditions, and the global heat balance -- on the model's own field, at points chosen by the verifier.
``symmetry``, ``scaling``
    The mirror image of the source gives the mirror field; twice the power gives twice the temperature rise
    (linear conduction).
``neighbour_consistency``
    Inside a layer the field is harmonic, so its value equals the mean over a small circle around the point
    (mean-value property): a stencil-free consistency test between a point and its neighbours.
``sufficiency``
    For each informative point, does the model's *input* set contain every cause that changes the physics there?
    The reference sensitivity of the temperature at the point to each physical parameter (over a plausible
    variation) is compared with the parameters the model takes; points where an omitted cause matters are listed
    -- no training could make the model right there.

Informative points (:func:`select_points`) are chosen from the physics instead of sweeping the domain: the largest
reference responses to the interventions, the interfaces, the source, the boundaries, spread out by a farthest-point
rule. :func:`baselines` computes what the usual metrics would say (the PINN-style physics loss at random collocation
points, the held-out error on the training distribution), so a report can show what each catches.

Knowledge built into an architecture is told from learned knowledge by re-running the structural checks on the
model with *untrained* weights (``Model.reinit``): a property that holds for random weights is guaranteed by the
architecture; one that holds only after training is learned (and here demonstrated); otherwise not demonstrated.

The case (geometry, parameters, reference) is :class:`.physics_learning_cases.Section`; the result is a
:class:`VerificationReport` with a :class:`pinneapple_analysis.trust.TrustReport`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .physics_learning_cases import DEFAULT, PARAMS, TRAIN_BOX, WIDTH, Model, Section, training_distribution

#: the default interventions: (name, change of theta, whether the change leaves the training box)
INTERVENTIONS = (
    ("power x2", {"P": 2.0}, False),
    ("source moved", {"x_s": 0.62}, False),
    ("source outside the box", {"x_s": 0.88}, True),
    ("TIM conductivity x2", {"k_tim": 8.0}, False),
    ("TIM conductivity /2", {"k_tim": 2.0}, False),
    ("cooling x2", {"h": 2.0}, False),
    ("contact resistance added", {"R_c": 1.0e-5}, True),
)
#: plausible variation of each physical parameter, for the sufficiency check (multiplier, or absolute for R_c)
VARIATION = {"P": 1.2, "x_s": 0.05, "k_tim": 1.5, "h": 1.3, "R_c": 5e-6, "k_die": 1.1, "k_lid": 1.1,
             "sigma": 1.2}
TOL = {"intervention": 0.10, "intervention_abs": 0.01, "residual": 0.05, "boundary": 0.05,
       "energy": 0.01, "symmetry": 0.01, "scaling": 0.01, "mean_value": 0.01, "sufficiency": 0.02}


def _apply(theta: dict, change: dict) -> dict:
    out = dict(theta)
    for k, v in change.items():
        if k in ("P", "h"):
            out[k] = theta[k] * v
        else:
            out[k] = v
    return out


@dataclass
class Finding:
    check: str
    passed: bool
    value: float
    limit: float
    detail: str = ""
    provenance: str = ""             # "architecture" | "learned" | "not demonstrated" | ""
    points: list = field(default_factory=list)

    def line(self) -> str:
        p = f" [{self.provenance}]" if self.provenance else ""
        return f"{'PASS' if self.passed else 'FAIL'}  {self.check:<32} {self.value:10.3g} (limit {self.limit:.3g}){p}"


@dataclass
class VerificationReport:
    model: str
    findings: list[Finding]
    points: np.ndarray
    reasons: list[str]
    insufficient: list[dict]
    trust: Any = None

    @property
    def failed(self) -> list[Finding]:
        """Failed checks within the model's scope (interventions on causes it does not take are reported, and
        listed by ``out_of_scope``, but are not defects of the model)."""
        return [f for f in self.findings if not f.passed and f.provenance != "out of scope"]

    @property
    def out_of_scope(self) -> list[Finding]:
        return [f for f in self.findings if f.provenance == "out of scope"]

    @property
    def detected(self) -> bool:
        return bool(self.failed)

    def to_markdown(self) -> str:
        lines = [f"### {self.model}", "", "| check | result | value | limit | provenance | detail |",
                 "|---|---|---|---|---|---|"]
        for f in self.findings:
            lines.append(f"| {f.check} | {'PASS' if f.passed else 'FAIL'} | {f.value:.3g} | {f.limit:.3g} | "
                         f"{f.provenance} | {f.detail} |")
        if self.insufficient:
            lines += ["", f"Points where the model's inputs cannot represent the physics: {len(self.insufficient)}"]
        if self.trust is not None:
            lines += ["", f"Decision: **{self.trust.decisao}**"]
        return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- derivatives
def _grad(model, X, th, hx=4e-6, hz=1e-6):
    e1, e2 = np.array([hx, 0.0]), np.array([0.0, hz])
    fx = (model(X + e1, th) - model(X - e1, th)) / (2 * hx)
    fz = (model(X + e2, th) - model(X - e2, th)) / (2 * hz)
    return fx, fz


def _laplacian(model, X, th, hx=8e-6, hz=2e-6):
    e1, e2 = np.array([hx, 0.0]), np.array([0.0, hz])
    T0 = model(X, th)
    return ((model(X + e1, th) - 2 * T0 + model(X - e1, th)) / hx ** 2
            + (model(X + e2, th) - 2 * T0 + model(X - e2, th)) / hz ** 2)


def _dz_one_sided(model, X, th, side: int, h=1e-6):
    """Second-order one-sided d/dz at X from below (side=-1) or above (side=+1)."""
    e = np.array([0.0, h * side])
    return side * (-3 * model(X, th) + 4 * model(X + e, th) - model(X + 2 * e, th)) / (2 * h)


# ------------------------------------------------------------------------------------------- point selection
def select_points(case: Section, theta: dict, k: int = 48, seed: int = 0, interventions=INTERVENTIONS,
                  pool: int = 3000) -> tuple[np.ndarray, list[str]]:
    """``k`` informative points: scored by the reference's own response to every intervention, plus quotas near
    each interface, the source and the boundaries; spread by farthest-point sampling within each group."""
    rng = np.random.default_rng(seed)
    H = case.height
    zi = case.interfaces
    margin = 3e-6
    P = np.column_stack([rng.uniform(margin, WIDTH - margin, pool), rng.uniform(margin, H - margin, pool)])
    dist_if = np.min(np.abs(P[:, 1:2] - zi[None, :]), 1)
    P = P[dist_if > 2 * margin]
    base = case.evaluate(theta, P)
    score = np.zeros(len(P))
    for _, ch, _ in interventions:
        d = np.abs(case.evaluate(_apply(theta, ch), P) - base)
        score = np.maximum(score, d / max(d.max(), 1e-300))

    def fps(cand, n):
        if not len(cand) or n <= 0:
            return []
        Xs = P[cand] / np.array([WIDTH, H])
        chosen = [int(cand[np.argmax(score[cand])])]
        dmin = np.linalg.norm(Xs - P[chosen[0]] / np.array([WIDTH, H]), axis=1)
        while len(chosen) < min(n, len(cand)):
            i = int(np.argmax(dmin * (0.2 + score[cand])))
            chosen.append(int(cand[i]))
            dmin = np.minimum(dmin, np.linalg.norm(Xs - P[cand[i]] / np.array([WIDTH, H]), axis=1))
        return chosen
    groups = {
        "response to interventions": np.argsort(-score)[: max(4 * k, 200)],
        "near an interface": np.nonzero(dist_if[dist_if > 2 * margin] < 15e-6)[0],
        "near the source": np.nonzero((P[:, 1] < 20e-6) & (np.abs(P[:, 0] - theta["x_s"] * WIDTH) < 0.15 * WIDTH))[0],
        "near a boundary": np.nonzero((P[:, 0] < 0.05 * WIDTH) | (P[:, 0] > 0.95 * WIDTH) | (P[:, 1] > H - 20e-6))[0],
    }
    quota = {"response to interventions": k // 2, "near an interface": k // 6, "near the source": k // 6,
             "near a boundary": k - k // 2 - 2 * (k // 6)}
    idx, reasons = [], []
    for g, cand in groups.items():
        for i in fps(np.setdiff1d(cand, idx), quota[g]):
            idx.append(i)
            reasons.append(g)
    return P[idx], reasons


def random_points(case: Section, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, WIDTH, k), rng.uniform(0, case.height, k)])


# ------------------------------------------------------------------------------------------------ the checks
def _scales(case: Section, theta: dict) -> dict:
    T = case.solve(theta)
    q_mean = theta["P"] / WIDTH                                      # W/m^2 through the stack, on average
    return {"T": float(T.max()), "q": q_mean, "vol": q_mean / LAYER_MIN}


LAYER_MIN = 30e-6


def _fv(case: Section, th: dict) -> dict:
    """The case's finite-volume operator for ``th`` (cached): matrix, source, cell centres, Robin conductances."""
    from pinneapple_condense import fv
    key = ("fv",) + tuple(sorted(th.items()))
    hit = case._cache.get(key)
    if hit is not None:
        return hit
    x, zc = case.centres()
    nzt = len(zc)
    k = np.broadcast_to(case.conductivity(zc, th)[None, None, :], (case.nx, 1, nzt)).copy()
    dz = np.diff(case.z_edges())
    R = np.zeros(nzt - 1)
    R[case.nz[0] - 1] = th["R_c"]
    sysm = fv.assemble(k, dz, R, case.dx, th["h"], 0.0, dy=1.0)
    f = np.zeros((case.nx, nzt))
    f[:, 0] = case.source(x, th) * case.dx
    Xc = np.column_stack([np.repeat(x, nzt), np.tile(zc, case.nx)])
    out = {"K": sysm.K.tocsr(), "f": f.ravel(), "X": Xc, "nz": nzt, "g_top": sysm.g_top.ravel()}
    case._cache[key] = out
    return out


def _cell_rows(case: Section, X: np.ndarray) -> np.ndarray:
    ze = case.z_edges()
    i = np.clip((X[:, 0] / case.dx).astype(int), 0, case.nx - 1)
    k = np.clip(np.searchsorted(ze, X[:, 1], side="right") - 1, 0, len(ze) - 2)
    return i * (len(ze) - 1) + k


def _probe_rows(case: Section, n: int = 19) -> dict[str, np.ndarray]:
    """Cells on both sides of each interface, and on each boundary, at ``n`` lateral or vertical positions."""
    nzt = sum(case.nz)
    ii = np.linspace(1, case.nx - 2, n).astype(int)
    rows = {}
    kk = [case.nz[0] - 1, case.nz[0], case.nz[0] + case.nz[1] - 1, case.nz[0] + case.nz[1]]
    rows["interface_flux"] = np.concatenate([ii * nzt + k for k in kk])
    rows["boundary:source_flux"] = ii * nzt
    rows["boundary:robin_top"] = ii * nzt + nzt - 1
    ks = np.linspace(0, nzt - 1, 15).astype(int)
    rows["boundary:adiabatic_sides"] = np.concatenate([ks, (case.nx - 1) * nzt + ks])
    return rows


def fv_residual(model: Model, case: Section, th: dict, rows: np.ndarray) -> np.ndarray:
    """Finite-volume residual of the model's own values on the given cells: ``K T_model - f`` on those rows only
    (the model is evaluated at the cell centres of each row's stencil), in W/m, relative to the larger of the mean
    heat per column and the cell's own source. Zero for the reference; truncation-sized for a smooth exact field;
    large where the model breaks the equation, an interface condition or a boundary condition."""
    F = _fv(case, th)
    K = F["K"][rows]
    cols = np.unique(K.indices)
    T = np.zeros(F["K"].shape[0])
    T[cols] = model(F["X"][cols], th)
    r = K @ T - F["f"][rows]
    scale = np.maximum(th["P"] / case.nx, np.abs(F["f"][rows]))
    return np.abs(r) / scale


def structural_checks(model: Model, case: Section, theta: dict, X: np.ndarray, probes: bool = True) -> list[Finding]:
    """Checks on the model's own field that need no reference values: the discrete residual of the governing
    equation inside the layers, across the interfaces and on each boundary (on the cells of the informative points
    and of a sparse set of probes, never the whole grid), the global heat balance, mean-value consistency between a
    point and its neighbours, symmetry and scaling."""
    th = {**DEFAULT, **theta}
    sc = _scales(case, th)
    out = []
    probes = _probe_rows(case) if probes else {}
    special = np.concatenate(list(probes.values())) if probes else np.zeros(0, int)
    own = np.setdiff1d(_cell_rows(case, X), special)
    r = fv_residual(model, case, th, own) if len(own) else np.zeros(1)
    out.append(Finding("residual", float(r.max()) <= TOL["residual"], float(r.max()), TOL["residual"],
                       f"discrete residual on the {len(own)} cells of the informative points / heat per column"))
    for name, rows in probes.items():
        v = float(fv_residual(model, case, th, rows).max())
        out.append(Finding(name, v <= TOL["boundary"], v, TOL["boundary"],
                           f"discrete residual on {len(rows)} probe cells / heat per column"))
    # the boundary condition itself on the continuous field: k dT/dx at x = 0 and x = L (one-sided, 0.5 um)
    zs = np.linspace(0.02, 0.98, 15) * case.height
    g = []
    for x0, sgn in ((0.0, 1.0), (WIDTH, -1.0)):
        Xb = np.column_stack([np.full(len(zs), x0), zs])
        e = np.array([0.5e-6 * sgn, 0.0])
        g.append(np.abs(case.conductivity(zs, th) * sgn * (-3 * model(Xb, th) + 4 * model(Xb + e, th)
                                                         - model(Xb + 2 * e, th)) / 1e-6) / sc["q"])
    gv = float(np.max(g))
    out.append(Finding("boundary:side_gradient", gv <= TOL["boundary"], gv, TOL["boundary"],
                       "max |k dT/dx| at x = 0, L / (P/L)"))
    # energy balance: Robin outflow of the top cells (the model's values there) over the power
    F = _fv(case, th)
    top = np.arange(case.nx) * F["nz"] + F["nz"] - 1
    eb = abs(float(F["g_top"] @ model(F["X"][top], th)) / th["P"] - 1)
    out.append(Finding("energy_balance", eb <= TOL["energy"], eb, TOL["energy"], "|heat out of the top / P - 1|"))
    # mean-value property inside the layers (continuous field)
    zi = case.interfaces
    H = case.height
    far = (np.min(np.abs(X[:, 1:2] - zi[None, :]), 1) > 6e-6) & (X[:, 1] > 6e-6) & (X[:, 1] < H - 6e-6) & \
          (X[:, 0] > 2e-5) & (X[:, 0] < WIDTH - 2e-5)
    if far.any():
        rr = 3e-6
        ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ring = np.mean([model(X[far] + rr * np.array([np.cos(a), np.sin(a)]), th) for a in ang], 0)
        mv = np.abs(model(X[far], th) - ring) / sc["T"]
        out.append(Finding("neighbour_consistency", float(mv.max()) <= TOL["mean_value"], float(mv.max()),
                           TOL["mean_value"], "|T(c) - mean over a 3 um circle| / Tmax"))
    Xm = np.column_stack([WIDTH - X[:, 0], X[:, 1]])
    sym = float(np.abs(model(X, th) - model(Xm, {**th, "x_s": 1 - th["x_s"]})).max() / sc["T"])
    out.append(Finding("symmetry", sym <= TOL["symmetry"], sym, TOL["symmetry"],
                       "max |T(x; x_s) - T(L - x; 1 - x_s)| / Tmax"))
    if "P" in model.inputs:
        t2 = {**th, "P": 2 * th["P"]}
        s2 = float(np.abs(model(X, t2) - 2 * model(X, th)).max() / (2 * sc["T"]))
        out.append(Finding("scaling", s2 <= TOL["scaling"], s2, TOL["scaling"], "max |T(2P) - 2 T(P)| / (2 Tmax)"))
    return out


def intervention_checks(model: Model, case: Section, theta: dict, X: np.ndarray,
                        interventions=INTERVENTIONS) -> list[Finding]:
    """Response of the model to each intervention against the reference response, at ``X`` and on the peak."""
    th = {**DEFAULT, **theta}
    xs = np.linspace(0.01, 0.99, 199) * WIDTH
    J = np.column_stack([xs, np.full(len(xs), 1e-7)])                # junction line (bottom of the die)
    out = []
    for name, ch, outside in interventions:
        t2 = _apply(th, ch)
        dref = case.evaluate(t2, X) - case.evaluate(th, X)
        dmod = model(X, t2) - model(X, th)
        pref = case.tmax(t2) - case.tmax(th)
        pmod = float(model(J, t2).max() - model(J, th).max())
        scale = max(np.abs(dref).max(), abs(pref), 1e-12)
        err = max(float(np.abs(dmod - dref).max()), abs(pmod - pref)) / scale
        small = TOL["intervention_abs"] * case.tmax(th)
        sign_ok = bool(np.sign(pmod) == np.sign(pref) or abs(pref) < small)
        missing = [k for k in ch if k not in model.inputs]
        ok = err <= TOL["intervention"] and sign_ok
        det = f"peak response model {pmod:+.3g} K vs reference {pref:+.3g} K"
        if missing:
            det += f"; the model does not take {', '.join(missing)}"
        if outside:
            det += "; outside the training box"
        f = Finding(f"intervention:{name}", ok, err, TOL["intervention"], det)
        if missing:                       # a cause the model was never given: out of its scope, not a defect
            f.provenance = "out of scope"
        out.append(f)
    return out


def sufficiency(model: Model, case: Section, theta: dict, X: np.ndarray) -> tuple[Finding, list[dict]]:
    """For each point, the physical parameters outside the model's inputs whose plausible variation changes the
    reference temperature there by more than ``TOL['sufficiency']`` x the local temperature rise."""
    th = {**DEFAULT, **theta}
    base = case.evaluate(th, X)
    bad = []
    sens = {}
    for p in PARAMS:
        if p in model.inputs:
            continue
        v = VARIATION[p]
        t2 = {**th, p: th[p] + v} if p in ("R_c", "x_s") else {**th, p: th[p] * v}
        sens[p] = np.abs(case.evaluate(t2, X) - base) / np.maximum(np.abs(base), 1e-12)
    for i in range(len(X)):
        miss = {p: float(s[i]) for p, s in sens.items() if s[i] > TOL["sufficiency"]}
        if miss:
            bad.append({"point": X[i].tolist(), "missing": miss})
    frac = len(bad) / max(len(X), 1)
    return Finding("sufficiency", not bad, frac, 0.0,
                   f"{len(bad)} of {len(X)} points need a cause the model does not take"), bad


def _provenance(model: Model, case: Section, theta: dict, X: np.ndarray, findings: list[Finding]) -> None:
    if model.reinit is None:
        return
    raw = {f.check: f for f in structural_checks(model.reinit(), case, theta, X)}
    for f in findings:
        if f.check in raw:
            if raw[f.check].passed:
                f.provenance = "architecture"
            elif f.passed:
                f.provenance = "learned"
            else:
                f.provenance = "not demonstrated"


def verify(model: Model, case: Section | None = None, theta: dict | None = None, *, k: int = 48, seed: int = 0,
           points: np.ndarray | None = None, probes: bool = True) -> VerificationReport:
    """All checks of the module docstring on ``model`` at ``theta`` (default: the nominal point). ``points``:
    use these instead of :func:`select_points`; ``probes=False``: no interface and boundary probe cells (the
    random-point ablation)."""
    from pinneapple_analysis.trust import Check, TrustReport
    case = case or Section()
    th = {**DEFAULT, **(theta or {})}
    if points is None:
        X, reasons = select_points(case, th, k, seed)
    else:
        X, reasons = np.asarray(points, float), ["given"] * len(points)
    findings = intervention_checks(model, case, th, X) + structural_checks(model, case, th, X, probes)
    suf, bad = sufficiency(model, case, th, X)
    if not suf.passed:
        suf.provenance = "out of scope"   # a limit of the model's inputs, listed per point; not a training defect
    findings.append(suf)
    _provenance(model, case, th, X, findings)
    critical = ("intervention", "energy_balance", "interface_flux")
    checks = [Check(f.check, "not_run" if f.provenance == "out of scope" else
                    ("supports" if f.passed else "contradicts"),
                    critical=f.check.startswith(critical), value=f.value, limit=f.limit,
                    source="pinneapple_analysis.validation.physics_learning",
                    reason=f.detail) for f in findings]
    rep = TrustReport(checks, model=model.name, details={"points": len(X)})
    return VerificationReport(model.name, findings, X, reasons, bad, rep)


# ---------------------------------------------------------------------------------------------- the baselines
def baselines(model: Model, case: Section | None = None, *, n_theta: int = 12, n_points: int = 400,
              seed: int = 1) -> dict:
    """What the usual metrics say, on the training distribution (:func:`.physics_learning_cases.
    training_distribution`): the relative L2 error at random points (held-out error), and a physics loss -- the
    mean squared discrete residual (as :func:`fv_residual`) at random cells, which is what a PINN's loss averages."""
    case = case or Section()
    rng = np.random.default_rng(seed)
    thetas = training_distribution(n_theta, rng)
    errs, losses = [], []
    H = case.height
    n_cells = case.nx * sum(case.nz)
    for th in thetas:
        th = {**DEFAULT, **th}
        X = np.column_stack([rng.uniform(0, WIDTH, n_points), rng.uniform(0, H, n_points)])
        ref = case.evaluate(th, X)
        errs.append(np.linalg.norm(model(X, th) - ref) / np.linalg.norm(ref))
        rows = rng.choice(n_cells, n_points, replace=False)
        losses.append(float(np.mean(fv_residual(model, case, th, rows) ** 2)))
    return {"heldout_rel_l2": float(np.mean(errs)), "physics_loss": float(np.mean(losses))}


#: what the usual metrics would accept (documented choices): 1 % held-out relative L2, and a physics loss of 1e-2
#: (an RMS discrete residual of 10 % of the mean heat per column)
BASELINE_LIMITS = {"heldout_rel_l2": 0.01, "physics_loss": 1e-2}


def detection_study(severities=(0.25, 0.5, 1.0, 2.0), budgets=(4, 8, 16, 48), *, seed: int = 0,
                    extra_models: list[Model] | None = None) -> dict:
    """Detection of the verifier against the usual metrics: every defective model of
    :func:`.physics_learning_cases.defective_models` at each severity, the controls of
    :func:`.physics_learning_cases.good_models` (false alarms), the verifier with informative points against the
    same checks on random points at equal budgets, and the baselines with :data:`BASELINE_LIMITS`."""
    import time

    from .physics_learning_cases import defective_models, good_models
    case = Section()
    rows = []

    def run(m, sev, label):
        t0 = time.perf_counter()
        r = verify(m, case, k=48, seed=seed)
        tv = time.perf_counter() - t0
        b = baselines(m, case)
        row = {"model": m.name, "group": label, "defect": m.defect, "severity": sev, "verifier": r.detected,
               "verifier_s": tv, "decision": r.trust.decisao, "failed": [f.check for f in r.failed],
               "heldout_rel_l2": b["heldout_rel_l2"], "physics_loss": b["physics_loss"],
               "flag_heldout": b["heldout_rel_l2"] > BASELINE_LIMITS["heldout_rel_l2"],
               "flag_loss": b["physics_loss"] > BASELINE_LIMITS["physics_loss"],
               "out_of_scope": [f.check for f in r.out_of_scope], "insufficient_points": len(r.insufficient)}
        for kb in budgets:
            ri = verify(m, case, k=kb, seed=seed)
            rr = verify(m, case, points=random_points(case, kb, seed=seed + 7), probes=False)
            row[f"informative_{kb}"] = ri.detected
            row[f"random_{kb}"] = rr.detected
        rows.append(row)

    for m in good_models(case, amplitudes=(0.001, 0.003), modes=(1, 2, 3)):
        run(m, 0.0, "control")
    for sev in severities:
        for m in defective_models(case, sev):
            if m.defect in ("out of distribution", "spurious correlation") and sev != 1.0:
                continue                                                 # no size to scale
            run(m, sev, "defect")
    for m in extra_models or []:
        run(m, None, "learned")
    summary = {}
    for g in ("control", "defect"):
        rr = [r for r in rows if r["group"] == g]
        if rr:
            summary[g] = {k: sum(r[k] for r in rr) / len(rr)
                          for k in ["verifier", "flag_heldout", "flag_loss"]
                          + [f"informative_{b}" for b in budgets] + [f"random_{b}" for b in budgets]}
            summary[g]["n"] = len(rr)
    return {"rows": rows, "summary": summary, "budgets": list(budgets), "limits": BASELINE_LIMITS, "tol": TOL}

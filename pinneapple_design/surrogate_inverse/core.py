"""The machinery shared by every case: a small MLP surrogate, gradient descent on the design through the frozen
surrogate, and a verification card that re-checks the answer against the ground truth.

The method (also called "backprop into the design"): sample the design space, label every sample with the ground
truth model, train an MLP on those labels, freeze its weights and treat the design variables as the only trainable
tensor. The surrogate is differentiable, so Adam can move the design along the gradient of the predicted objective
plus constraint penalties. The surrogate's answer is never reported on its own say-so: :class:`VerificationCard`
re-runs the ground truth on the final design and grades the gap.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from pinneapple_analysis.trust import Check, TrustReport

PASS, WARNING, FAIL = "PASS", "WARNING", "FAIL"


class MLPSurrogate(nn.Module):
    """``tanh`` MLP with its inputs mapped from their bounds to [-1, 1] and its output standardised.

    ``log_output=True`` fits ``log(y)``: right for a positive output that spans orders of magnitude (a thermal
    resistance, a release rate, a stress) and keeps the prediction positive. ``forward`` takes and returns physical
    units, so the frozen surrogate can sit inside a loss written in physical units.
    """

    def __init__(self, bounds: Sequence[tuple[float, float]], hidden: int = 48, n_layers: int = 3,
                 log_output: bool = False):
        super().__init__()
        lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        hi = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        if torch.any(hi <= lo):
            raise ValueError("every bound needs low < high")
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_std", torch.ones(()))
        self.log_output = bool(log_output)
        layers: list[nn.Module] = []
        width = len(bounds)
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    @property
    def n_inputs(self) -> int:
        return int(self.lo.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0
        out = self.net(z).squeeze(-1) * self.y_std + self.y_mean
        return torch.exp(out) if self.log_output else out


@dataclass
class SurrogateFit:
    """A trained surrogate with its held-out errors (relative, on rows it never trained on).

    Attributes
    ----------
    val_mean_rel_err : float [-]
        Mean relative error on held-out rows.
    val_p95_rel_err : float [-]
        95th percentile.
    val_max_rel_err : float [-]
        Largest.
    n_train : int [-]
        Training rows.
    n_val : int [-]
        Held-out rows.
    epochs : int [-]
        Epochs.
    seconds : float [s]
        Training time.
    """
    model: MLPSurrogate
    val_mean_rel_err: float
    val_p95_rel_err: float
    val_max_rel_err: float
    n_train: int
    n_val: int
    epochs: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {"val_mean_rel_err": self.val_mean_rel_err, "val_p95_rel_err": self.val_p95_rel_err,
                "val_max_rel_err": self.val_max_rel_err, "n_train": self.n_train, "n_val": self.n_val,
                "epochs": self.epochs, "seconds": self.seconds}


def fit_surrogate(X: torch.Tensor, y: torch.Tensor, bounds: Sequence[tuple[float, float]], *, hidden: int = 48,
                  n_layers: int = 3, epochs: int = 1500, lr: float = 3e-3, seed: int = 0, log_output: bool = False,
                  val_fraction: float = 0.1, rel_floor: float = 1e-12) -> SurrogateFit:
    """Full-batch Adam with a cosine learning-rate decay on ``(X, y)``; the last ``val_fraction`` of the rows (the
    data are expected in random order) is held out and scored as ``|pred - y| / max(|y|, rel_floor)``."""
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[1] != len(bounds):
        raise ValueError(f"X must be (n, {len(bounds)}) with one label per row; got {tuple(X.shape)}, {tuple(y.shape)}")
    if log_output and torch.any(y <= 0):
        raise ValueError("log_output needs strictly positive labels")
    n_val = max(1, int(round(len(y) * val_fraction)))
    if n_val >= len(y):
        raise ValueError("not enough rows for a held-out split")
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    model = MLPSurrogate(bounds, hidden=hidden, n_layers=n_layers, log_output=log_output)
    x_tr, y_tr, x_va, y_va = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
    target = torch.log(y_tr) if log_output else y_tr
    model.y_mean.fill_(float(target.mean()))
    model.y_std.fill_(float(target.std()) or 1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
    for _ in range(epochs):
        opt.zero_grad()
        pred = model.net(2.0 * (x_tr - model.lo) / (model.hi - model.lo) - 1.0).squeeze(-1)
        loss = torch.mean((pred - (target - model.y_mean) / model.y_std) ** 2)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        err = (model(x_va) - y_va).abs() / y_va.abs().clamp_min(rel_floor)
    return SurrogateFit(model=model, val_mean_rel_err=float(err.mean()), val_p95_rel_err=float(err.quantile(0.95)),
                        val_max_rel_err=float(err.max()), n_train=len(y_tr), n_val=n_val, epochs=epochs,
                        seconds=time.perf_counter() - t0)


def freeze(model: nn.Module) -> nn.Module:
    """Stop gradients into the weights: from here on only the design moves."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def _to_raw(value: float, lo: float, hi: float) -> float:
    frac = min(max((value - lo) / (hi - lo), 1e-4), 1 - 1e-4)
    return math.log(frac / (1 - frac))


def descend(loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor], x0: Mapping[str, float], *,
            bounds: Mapping[str, tuple[float, float]], steps: int = 400, lr: float | Mapping[str, float] = 0.01,
            reparam: str = "clamp",
            project: Callable[[dict[str, torch.Tensor]], None] | None = None) -> dict[str, float]:
    """Adam on the design variables ``x0`` to minimise ``loss_fn(values)``.

    ``reparam="clamp"`` optimises the physical values and clamps them to ``bounds`` after every step, then calls
    ``project(values)`` (in place, without gradients) for constraints that couple variables. ``reparam="sigmoid"``
    optimises an unconstrained tensor per variable mapped through ``low + (high - low) * sigmoid(raw)``, which puts
    variables of very different magnitude on the same footing (one learning rate moves all of them).
    """
    if reparam not in ("clamp", "sigmoid"):
        raise ValueError("reparam must be 'clamp' or 'sigmoid'")
    names = list(x0)
    if reparam == "clamp":
        raw = {k: torch.tensor(float(x0[k]), requires_grad=True) for k in names}
    else:
        raw = {k: torch.tensor(_to_raw(float(x0[k]), *bounds[k]), requires_grad=True) for k in names}
    groups = [{"params": [raw[k]], "lr": float(lr[k] if isinstance(lr, Mapping) else lr)} for k in names]
    opt = torch.optim.Adam(groups)

    def values() -> dict[str, torch.Tensor]:
        if reparam == "clamp":
            return dict(raw)
        return {k: bounds[k][0] + (bounds[k][1] - bounds[k][0]) * torch.sigmoid(raw[k]) for k in names}

    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(values())
        loss.backward()
        opt.step()
        if reparam == "clamp":
            with torch.no_grad():
                for k in names:
                    raw[k].clamp_(*bounds[k])
                if project is not None:
                    project(raw)
    with torch.no_grad():
        return {k: float(v) for k, v in values().items()}


# -- verification ------------------------------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    verdict: str                     # PASS | WARNING | FAIL
    detail: str
    value: float | None = None
    limit: float | None = None


def banded(name: str, error: float, pass_limit: float, warn_limit: float, *, pass_text: str, warn_text: str,
           fail_text: str) -> CheckResult:
    """PASS when ``error <= pass_limit``, WARNING up to ``warn_limit``, FAIL above."""
    if error <= pass_limit:
        return CheckResult(name, PASS, pass_text, error, pass_limit)
    if error <= warn_limit:
        return CheckResult(name, WARNING, warn_text, error, pass_limit)
    return CheckResult(name, FAIL, fail_text, error, pass_limit)


def relative_error(predicted: float, reference: float, floor: float) -> float:
    """Relative error of ``predicted`` against ``reference``, with ``floor`` guarding a reference near zero.

    Returns
    -------
    err : float [-]
        ``|predicted - reference| / max(|reference|, floor)``.
    """
    return abs(predicted - reference) / max(abs(reference), floor)


def constraint_checks(violations: Mapping[str, float], ok_text: str = "Within physical/manufacturing limit.",
                      skip: Sequence[str] = ()) -> list[CheckResult]:
    """One check per signed violation (``<= 0`` is satisfied)."""
    out = []
    for name, amount in violations.items():
        if name in skip:
            continue
        if amount <= 0:
            out.append(CheckResult(f"constraint_{name}", PASS, ok_text, amount, 0.0))
        else:
            out.append(CheckResult(f"constraint_{name}", FAIL, f"Violation of {amount:.5g} (SI units) above the limit.",
                                   amount, 0.0))
    return out


@dataclass
class VerificationCard:
    """The final design re-checked against the ground truth (never the surrogate grading itself).

    ``covered`` and ``not_covered`` name what this card does and does not check, so its coverage is explicit:
    ``len(covered) / (len(covered) + len(not_covered))``. ``values`` holds the case's numbers (real and surrogate
    outputs, errors)."""
    checks: list[CheckResult]
    values: dict[str, float]
    covered: list[str]
    not_covered: list[str]

    @property
    def overall_verdict(self) -> str:
        verdicts = {c.verdict for c in self.checks}
        return FAIL if FAIL in verdicts else WARNING if WARNING in verdicts else PASS

    @property
    def trust_score(self) -> int:
        return round(100 * sum(c.verdict == PASS for c in self.checks) / len(self.checks)) if self.checks else 0

    @property
    def coverage(self) -> float:
        n = len(self.covered) + len(self.not_covered)
        return len(self.covered) / n if n else 0.0

    def __getattr__(self, name: str) -> Any:            # card.real_temperature_c etc. read from ``values``
        values = self.__dict__.get("values", {})
        if name in values:
            return values[name]
        raise AttributeError(name)

    def trust_report(self, model: str = "") -> TrustReport:
        """The same checks as a :class:`pinneapple_analysis.trust.TrustReport` (FAIL is critical, WARNING is a
        non-critical contradiction), with every ``not_covered`` item as a check that did not run."""
        checks = [Check(c.name, "supports" if c.verdict == PASS else "contradicts", critical=c.verdict == FAIL,
                        value=c.value, limit=c.limit, reason=c.detail if c.verdict != PASS else "")
                  for c in self.checks]
        checks += [Check(item, "not_run", reason="outside what this model checks") for item in self.not_covered]
        return TrustReport(checks=checks, model=model)

    def to_dict(self) -> dict[str, Any]:
        tr = self.trust_report()
        return {
            "checks": [{"name": c.name, "verdict": c.verdict, "detail": c.detail} for c in self.checks],
            **self.values,
            "overall_verdict": self.overall_verdict,
            "trust_score": self.trust_score,
            "coverage": self.coverage,
            "coverage_note": ("Covers " + ", ".join(self.covered) + ". Does NOT cover " + ", ".join(self.not_covered)
                              + "."),
            "trust": {"decision": tr.decisao, "score": tr.score, "checks_run": tr.cobertura, "reasons": tr.motivos},
        }


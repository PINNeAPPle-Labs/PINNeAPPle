"""Exceptional-point tuning: can a control parameter move the EP of a PT-symmetric dimer along a desired law?

Reference: "A No-Go Theorem for Exceptional-Point Tuning in PT-Symmetric Dimers: Pedagogical Structure and
Device-Design Constraints", Zenodo record 23287622 (Project Prometheus). For the reciprocal dimer the EP sits at
gamma_EP = |b(lam)|, so with a polynomial coupling b(lam) = J + c1 lam + ... + cn lam^n (lam real, cn != 0)

    gamma_EP^2(lam) = |b(lam)|^2,  a real polynomial of degree 2n with leading coefficient |cn|^2 > 0,

which can never equal the linear target J^2 + lam J <G> (degree 1) for a nontrivial perturbation: the no-go
theorem. The paper's classification and the three ways around it:

* phase-only coupling b = J exp(i phi(lam)): the EP does not move;
* reciprocal polynomial coupling: quadratic (or higher) corrections are unavoidable;
* Route A, non-polynomial coupling b = sqrt(J^2 + lam J <G>): exact linear tuning, PT symmetric;
* Route B, state-dependent feedback: a polynomial feedback map can have a non-polynomial fixed point (conjecture in
  the paper; :func:`feedback_coupling` shows one such map, which is not a proof);
* Route C, non-reciprocal hopping b_ab = J, b_ba = J + lam <G>: exact linear tuning, PT symmetry broken.

:func:`analyze_ep_tuning` checks any coupling law symbolically (sympy): the EP law, its degree, the residual against
the target, the class of the perturbation and which of the theorem's four assumptions (A PT symmetry, B dimension 2,
C polynomial coupling, D reciprocity) hold. :func:`theorem_profile` states the same assumptions for
:mod:`pinneapple_physics.physics_audit`, so a device design is audited like any model: when one assumption fails the
theorem no longer constrains it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["EPTuningReport", "analyze_ep_tuning", "classify_coupling", "coupling_classes", "route_a_sqrt_coupling",
           "route_c_asymmetric", "feedback_coupling", "theorem_profile", "design_features", "THEOREM_ASSUMPTIONS"]


@dataclass
class EPTuningReport:
    ep_law: Any                              # sympy expression of gamma_EP^2(lam)
    target: Any
    residual: Any                            # ep_law - target, expanded
    exact: bool                              # the coupling realises the target law exactly
    degree: int | None                       # degree of ep_law in lam (None if not polynomial)
    perturbation_class: str
    assumptions: dict[str, bool]             # A..D
    theorem_applies: bool
    routes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        import sympy as sp

        out = [f"gamma_EP^2(lam) = {sp.simplify(self.ep_law)}",
               f"target           = {self.target}",
               f"class: {self.perturbation_class}; exact target realised: {'YES' if self.exact else 'NO'}"
               + (f"; degree in lam: {self.degree}" if self.degree is not None else "")]
        if not self.exact:
            out.append(f"residual (law - target) = {self.residual}")
        names = {"A": "PT symmetry", "B": "two-dimensional", "C": "polynomial coupling", "D": "reciprocal coupling"}
        out.append("assumptions: " + ", ".join(f"{k} {names[k]}: {'holds' if v else 'fails'}"
                                               for k, v in self.assumptions.items()))
        out.append("no-go theorem " + ("APPLIES: exact linear tuning is impossible in this class" if self.theorem_applies
                                       else "does not apply (an assumption fails)"))
        out += [f"route: {r}" for r in self.routes] + [f"note: {n}" for n in self.notes]
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())


def _symbols():
    import sympy as sp
    return sp.Symbol("lam", real=True), sp.Symbol("J", positive=True), sp.Symbol("G", real=True)


def classify_coupling(b_ab, b_ba=None, lam=None) -> str:
    """``"constant"``, ``"phase-only"``, ``"reciprocal polynomial"``, ``"reciprocal non-polynomial"`` or
    ``"non-reciprocal"`` (``b_ba`` given and not the conjugate of ``b_ab``)."""
    import sympy as sp

    lam = lam if lam is not None else _symbols()[0]
    if b_ba is not None and sp.simplify(b_ba - sp.conjugate(b_ab)) != 0:
        return "non-reciprocal"
    if not sp.sympify(b_ab).has(lam):
        return "constant"
    if sp.simplify(sp.diff(sp.Abs(b_ab) ** 2, lam)) == 0 or sp.simplify(sp.diff(b_ab * sp.conjugate(b_ab), lam)) == 0:
        return "phase-only"
    return "reciprocal polynomial" if sp.sympify(b_ab).is_polynomial(lam) else "reciprocal non-polynomial"


def analyze_ep_tuning(b_ab, b_ba=None, target=None, lam=None, dimension: int = 2) -> EPTuningReport:
    """Symbolic check of an EP tuning law. ``b_ab``, ``b_ba``: sympy expressions in ``lam`` (default symbol ``lam``,
    real); ``b_ba=None`` means reciprocal (``conj(b_ab)``). ``target``: desired gamma_EP^2(lam), default
    ``J**2 + lam*J*G`` with the module's symbols (``_symbols()``: ``lam`` real, ``J`` > 0, ``G`` real)."""
    import sympy as sp

    s_lam, J, G = _symbols()
    lam = lam if lam is not None else s_lam
    target = J ** 2 + lam * J * G if target is None else target
    b_ab = sp.sympify(b_ab)
    reciprocal = b_ba is None or sp.simplify(sp.sympify(b_ba) - sp.conjugate(b_ab)) == 0
    b_ba_e = sp.conjugate(b_ab) if b_ba is None else sp.sympify(b_ba)
    law = b_ab * b_ba_e
    if reciprocal:                     # real radicand x, on the domain x >= 0 where the coupling is real: conj(sqrt(x)) = sqrt(x)
        law = law.replace(lambda e: isinstance(e, sp.conjugate) and isinstance(e.args[0], sp.Pow)
                          and e.args[0].exp == sp.Rational(1, 2) and e.args[0].base.is_real is True
                          and not e.args[0].base.has(sp.I), lambda e: e.args[0])
    law = sp.expand(sp.simplify(law))
    residual = sp.simplify(sp.expand(law - target))
    exact = residual == 0
    try:
        poly = sp.Poly(law, lam)
        degree = poly.degree()
    except sp.PolynomialError:
        degree = None
    cls = classify_coupling(b_ab, None if reciprocal else b_ba_e, lam)
    polynomial = bool(b_ab.is_polynomial(lam) and b_ba_e.is_polynomial(lam))
    assumptions = {"A": reciprocal, "B": dimension == 2, "C": polynomial, "D": reciprocal}
    applies = all(assumptions.values()) and cls not in ("constant", "phase-only")
    routes, notes = [], []
    if cls == "phase-only":
        notes.append("phase-only coupling: |b| does not depend on lam, so the EP does not move")
    if applies and not exact:
        notes.append("a nontrivial polynomial b(lam) gives |b(lam)|^2 of degree 2 x deg(b) with a positive leading "
                     "coefficient: the linear target is out of reach (no-go theorem)")
        routes += ["A: non-polynomial coupling b = sqrt(J^2 + lam J <G>) (keeps PT symmetry)",
                   "B: state-dependent self-consistent feedback (conjecture in the paper)",
                   "C: non-reciprocal hopping b_ab = J, b_ba = J + lam <G> (breaks PT symmetry)"]
    if exact and not reciprocal:
        notes.append("exact linear tuning obtained by breaking reciprocity, hence PT symmetry (route C)")
    if exact and reciprocal and not polynomial:
        notes.append("exact linear tuning obtained with a non-polynomial coupling (route A)")
    if dimension != 2:
        notes.append("the theorem is proved for dimers only; for chains see PTChain (the trimer's EP3 also sits "
                     "at gamma^2 = 2|b|^2, quadratic in b)")
    return EPTuningReport(law, target, residual, exact, degree, cls, assumptions, applies, routes, notes)


def coupling_classes() -> list[tuple[str, str, Any, Any]]:
    """The paper's four coupling classes as sympy laws in the module's symbols (``_symbols()``), one tuple
    ``(key, title, b_ab, b_ba)`` each (``b_ba`` None: reciprocal), ready for :func:`analyze_ep_tuning`:

    * ``reciprocal_linear``: b = J + c lam (the no-go case, ``c`` a free complex symbol);
    * ``phase_only``: b = J exp(i lam) (the EP does not move);
    * ``route_a``: b = sqrt(J^2 + lam J G) (exact, PT symmetric);
    * ``route_c``: b_ab = J, b_ba = J + lam G (exact, non-reciprocal)."""
    import sympy as sp

    lam, J, G = _symbols()
    return [("reciprocal_linear", "reciprocal linear coupling b = J + c lam", J + sp.Symbol("c") * lam, None),
            ("phase_only", "phase-only coupling b = J exp(i lam)", J * sp.exp(sp.I * lam), None),
            ("route_a", "route A: b = sqrt(J^2 + lam J G)", sp.sqrt(J ** 2 + lam * J * G), None),
            ("route_c", "route C: b_ab = J, b_ba = J + lam G", J, J + lam * G)]


def route_a_sqrt_coupling(J: float, G: float) -> Callable[[float], complex]:
    """b(lam) = sqrt(J^2 + lam J G): gamma_EP^2 = J^2 + lam J G exactly, PT symmetric (defined while the radicand
    is >= 0; beyond that the coupling becomes imaginary and the EP leaves the real gamma axis)."""
    return lambda lam: complex(np.sqrt(complex(J ** 2 + lam * J * G)))


def route_c_asymmetric(J: float, G: float) -> tuple[Callable[[float], complex], Callable[[float], complex]]:
    """(b_ab, b_ba) = (J, J + lam G): gamma_EP^2 = J (J + lam G) = J^2 + lam J G exactly, but H12 != conj(H21)."""
    return (lambda lam: complex(J)), (lambda lam: complex(J + lam * G))


def feedback_coupling(J: float, G: float, lam: float, b0: float | None = None, iters: int = 60,
                      tol: float = 1e-14) -> tuple[float, int]:
    """Route B illustration: the coupling is updated from a measurement of itself,
    b <- (b + (J^2 + lam J G) / b) / 2, a polynomial map in (b, 1/b, lam). Its fixed point is
    b* = sqrt(J^2 + lam J G), the route-A coupling. This shows that a feedback loop *can* produce the
    non-polynomial dependence the theorem rules out for static polynomial couplings; it does not prove the paper's
    Conjecture 1 for any physical feedback mechanism. Returns (b*, iterations used)."""
    target = J ** 2 + lam * J * G
    if target <= 0:
        raise ValueError("J^2 + lam J G must be positive for a real fixed point")
    b = float(b0 if b0 is not None else J)
    for k in range(1, iters + 1):
        nb = 0.5 * (b + target / b)
        if abs(nb - b) <= tol * abs(nb):
            return nb, k
        b = nb
    return b, iters


# ------------------------------------------------------------------------------------------------ hypothesis map
def _theorem_assumptions():
    from pinneapple_physics.physics_audit import Assumption, Condition

    return (
        Assumption("pt_symmetry", "A: PT symmetric Hamiltonian, P H* P = H",
                   (Condition("pt_violation", hi=1e-8, band=1e-9, log=False),), "PT-symmetry breaking",
                   "non-reciprocal (gain-biased) hopping"),
        Assumption("two_level", "B: two-dimensional Hamiltonian (dimer)",
                   (Condition("dimension", lo=1.5, hi=2.5, band=0.2, log=False),), "multi-mode coupling",
                   "higher-order exceptional points"),
        Assumption("polynomial_coupling", "C: coupling polynomial in the control parameter",
                   (Condition("non_polynomial", hi=0.5, band=0.2, log=False),), "non-polynomial control law",
                   "non-polynomial or state-dependent coupling"),
        Assumption("reciprocal_coupling", "D: reciprocal interaction, H12 = conj(H21)",
                   (Condition("non_reciprocity", hi=1e-8, band=1e-9, log=False),), "non-reciprocal transport",
                   "non-reciprocal hopping"),
    )


THEOREM_ASSUMPTIONS = ("pt_symmetry", "two_level", "polynomial_coupling", "reciprocal_coupling")


def theorem_profile():
    """The theorem as a :class:`~pinneapple_physics.physics_audit.ModelProfile`: auditing a device design against it
    says whether the no-go result constrains that design (all four assumptions VALID) or not (one VIOLATED)."""
    from pinneapple_physics.physics_audit import ModelProfile

    return ModelProfile("no_go_ep_linear_tuning", "theorem", ("PT-symmetric dimer Hamiltonian",),
                        _theorem_assumptions(), notes="Zenodo 23287622")


def design_features(H: np.ndarray, polynomial: bool) -> dict[str, float]:
    """Features of a device Hamiltonian (at one operating point) for :func:`theorem_profile`'s audit."""
    from .pt_dimer import pt_violation

    H = np.asarray(H, dtype=complex)
    nr = float(abs(H[0, 1] - np.conj(H[1, 0])) / max(abs(H[0, 1]), abs(H[1, 0]), 1e-300)) if H.shape[0] >= 2 else 0.0
    return {"pt_violation": pt_violation(H), "dimension": float(H.shape[0]),
            "non_polynomial": 0.0 if polynomial else 1.0, "non_reciprocity": nr}

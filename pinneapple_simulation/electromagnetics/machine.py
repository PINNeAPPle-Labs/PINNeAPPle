"""Radial-flux machine with surface permanent magnets (or none: a reluctance rotor), 2D A_z FEM.

The cross-section is meshed with :func:`~pinneapple_simulation.electromagnetics.mesh.polar_mesh`:
rotor rings turn with the rotor and a one-layer moving band in the middle of the air gap is
re-triangulated for each rotor angle, so rotor and stator elements keep their indices and shapes at
every angle (smooth co-energy, rotor-frame quantities for losses, a fixed design-variable set for
topology optimization). Slots, teeth and magnets are resolved on the polar grid (their edges snap to
the nearest grid line); ``n_theta`` sets the angular resolution.

    from pinneapple_simulation.electromagnetics.machine import SPMMachine
    m = SPMMachine()                                   # 4 poles, 24 slots, illustrative data
    sol = m.solve(theta=0.0, currents=(0.0, 0.0, 0.0))  # no load
    T = m.torque(sol)                                  # Maxwell stress (Arkkio), N m
    Tc = m.torque_coenergy(0.01, (0.0, 0.0, 0.0))      # virtual work, N m
    psi = m.flux_linkage(sol)                          # Wb-turn per phase

Conventions: angles in radians, rotor angle ``theta`` counter-clockwise, torque on the rotor positive
counter-clockwise; ``currents`` are the instantaneous phase currents (A). Phase ``k`` of a balanced
supply is ``I cos(p theta + phi0 + pi/2 + gamma - 2 pi k / 3)``, where ``phi0`` is the electrical angle
of the magnet flux linkage at ``theta = 0`` (from :meth:`SPMMachine.pm_flux_angle`), so ``gamma = 0`` puts
the current on the q-axis (in phase with the back-EMF) and ``gamma > 0`` advances it towards -d.

Default dimensions and materials are illustrative (not a commercial design); the iron curve is
:func:`~pinneapple_simulation.electromagnetics.materials.illustrative_steel`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .fem import Geometry, MagnetostaticSolution, solve_magnetostatic
from .materials import NU0, BHCurve, BrauerCurve, LinearMaterial, illustrative_steel
from .mesh import Mesh2D, Ring, polar_mesh
from .postprocess import (
    CoilSide,
    IronLossModel,
    element_flux_density_history,
    flux_linkage,
    iron_loss,
    magnet_eddy_loss_resistance_limited,
    torque_maxwell_stress,
    torque_maxwell_stress_gradient,
)

__all__ = ["SPMMachine", "MachineSolution", "belt_winding", "ScaledCurve"]

TWO_PI = 2.0 * math.pi


def belt_winding(slots: int, pole_pairs: int, phases: int = 3) -> list[tuple[int, int]]:
    """Single-layer 60-degree phase-belt layout: (phase, sign) per slot, slot ``k`` centred at
    ``k * 2 pi / slots``. Phase axes follow the positive rotation (A, then B 120 electrical degrees
    ahead), so a counter-clockwise rotor sees the sequence A, B, C."""
    gamma = pole_pairs * TWO_PI / slots  # slot pitch, electrical
    q = slots / (2 * phases * pole_pairs)
    shift = -math.pi / (2 * phases) + (max(round(q), 1) - 1) * gamma / 2.0
    # belts in +phi order: A+, C-, B+, A-, C+, B- (for 3 phases)
    order = [(0, 1), (2, -1), (1, 1), (0, -1), (2, 1), (1, -1)]
    out = []
    for k in range(slots):
        a = (k * gamma - shift) % TWO_PI
        out.append(order[int(a // (math.pi / phases)) % 6])
    return out


class ScaledCurve:
    """Density-interpolated reluctivity ``nu = nu0 + w (nu_iron(s) - nu0)`` with per-element weights ``w``
    (topology optimization of iron). ``curve`` is a B-H curve object or a linear ``mu_r``."""

    def __init__(self, curve, w: np.ndarray):
        self.curve = curve
        self.w = np.asarray(w, float)

    def _iron(self, s):
        if isinstance(self.curve, (int, float)):
            nu = NU0 / self.curve
            return nu + 0 * s, 0 * s, 0.5 * nu * s
        return self.curve.nu(s), self.curve.dnu(s), self.curve.energy(s)

    def nu(self, s):
        nu, _, _ = self._iron(s)
        return NU0 + self.w * (nu - NU0)

    def dnu(self, s):
        _, d, _ = self._iron(s)
        return self.w * d

    def energy(self, s):
        _, _, e = self._iron(s)
        return 0.5 * NU0 * s + self.w * (e - 0.5 * NU0 * s)

    def nu_iron(self, s):
        return self._iron(s)[0]


class _Combined:
    def __init__(self, forms, weights):
        self.forms, self.weights = forms, weights

    def value(self, A):
        return sum(w * f.value(A) for f, w in zip(self.forms, self.weights, strict=True))

    def gradient(self, A):
        return sum(w * f.gradient(A) for f, w in zip(self.forms, self.weights, strict=True))


@dataclass
class MachineSolution:
    theta: float
    currents: tuple
    mesh: Mesh2D
    sol: MagnetostaticSolution
    J: np.ndarray

    @property
    def A(self):
        return self.sol.A

    @property
    def B(self):
        return self.sol.B


@dataclass
class SPMMachine:
    """Surface-PM (``magnet_thickness > 0``) or reluctance rotor inside a slotted stator."""

    pole_pairs: int = 2
    slots: int = 24
    r_rotor: float = 0.040  # rotor iron outer radius (magnets sit on it)
    magnet_thickness: float = 0.004
    magnet_arc: float = 0.8  # fraction of the pole pitch
    magnetization: str = "parallel"  # "parallel" or "radial"
    Br: float = 1.2
    mu_r_magnet: float = 1.05
    magnet_sigma: float = 6.7e5  # S/m, order of sintered NdFeB
    airgap: float = 0.001
    tooth_tip: float = 0.001
    slot_opening: float = 0.2  # fraction of the slot pitch
    slot_width: float = 0.5  # fraction of the slot pitch (slot body)
    slot_depth: float = 0.015
    r_out: float = 0.085
    length: float = 0.05
    turns_per_slot: float = 20.0
    winding: list | None = None  # (phase, sign) per slot; default belt_winding
    iron: object = field(default_factory=illustrative_steel)  # BHCurve, BrauerCurve or LinearMaterial
    rotor_iron: object | None = None  # default: same as iron
    n_theta: int = 360
    rotor_core_layers: int = 5
    gap_layers: int = 2  # air layers on each side of the moving band
    rotor_core_inner: float = 0.0  # radius of a non-magnetic shaft (0: iron to the centre)
    rotor_design: np.ndarray | None = None  # densities of the rotor core elements (topology optimization)
    design_penalty: float = 3.0

    def __post_init__(self):
        if self.n_theta % (2 * self.pole_pairs) or (self.slots and self.n_theta % self.slots):
            raise ValueError("n_theta must be a multiple of 2*pole_pairs and of slots")
        if self.winding is None and self.slots:
            self.winding = belt_winding(self.slots, self.pole_pairs)
        self._rings = self._make_rings()
        self._names = None
        self._phi0 = None

    # ------------------------------------------------------------------ geometry
    @property
    def r_magnet(self):
        return self.r_rotor + self.magnet_thickness

    @property
    def r_bore(self):
        return self.r_magnet + self.airgap

    @property
    def band(self):
        """(r1, r2): the air annulus used by the Maxwell-stress torque."""
        return self.r_magnet, self.r_bore

    def _make_rings(self) -> list[Ring]:
        n, p2 = self.n_theta, 2 * self.pole_pairs
        rings = [Ring(0.0, 1)]
        K = self.rotor_core_layers
        r0 = self.rotor_core_inner
        if r0 > 0.0:  # non-magnetic shaft, coarse
            for k in (1, 2):
                rings.append(Ring(r0 * k / 2, max(p2 * 4, 2 * p2 * round(n * k / (8 * p2))), rotating=True))
        for k in range(1, K + 1):
            r = r0 + (self.r_rotor - r0) * k / K
            nk = n if r0 > 0 else max(p2 * 4, 2 * p2 * round(n * k / (2 * K * p2)))
            rings.append(Ring(r, min(nk, n), rotating=True))
        nm = max(2, round(self.magnet_thickness / (self.airgap / 2))) if self.magnet_thickness > 0 else 0
        for k in range(1, nm + 1):
            rings.append(Ring(self.r_rotor + self.magnet_thickness * k / nm, n, rotating=True))
        g = self.airgap
        k = self.gap_layers
        for j in range(1, k + 1):  # rotor side; the band joins ring k (rotating) and ring k+1
            rings.append(Ring(self.r_magnet + g * j / (2 * k + 1), n, rotating=True))
        for j in range(k + 1, 2 * k + 2):
            rings.append(Ring(self.r_magnet + g * j / (2 * k + 1), n))
        if self.slots:
            rings.append(Ring(self.r_bore + self.tooth_tip / 2, n))
            rings.append(Ring(self.r_bore + self.tooth_tip, n))
            ns = max(3, round(self.slot_depth / (2 * g)))
            r_t = self.r_bore + self.tooth_tip
            for k in range(1, ns + 1):
                rings.append(Ring(r_t + self.slot_depth * k / ns, n))
        r_y = rings[-1].r
        ny = max(2, round((self.r_out - r_y) / (3 * g)))
        for k in range(1, ny + 1):
            rings.append(Ring(r_y + (self.r_out - r_y) * k / ny, n))
        return rings

    def _classify(self, layer: int, r: float, phi: float) -> str:
        if r < self.rotor_core_inner and layer <= 1:
            return "shaft"
        if r < self.r_rotor:
            return "rotor_core"
        if r < self.r_magnet:
            tau = math.pi / self.pole_pairs
            k = int(round(phi / tau)) % (2 * self.pole_pairs)
            d = (phi - k * tau + math.pi) % TWO_PI - math.pi
            return f"magnet{k}" if abs(d) < self.magnet_arc * tau / 2 else "rotor_air"
        if r < self.r_bore:
            return "gap"
        if not self.slots:
            return "stator_iron"
        tau = TWO_PI / self.slots
        k = int(round(phi / tau)) % self.slots
        d = (phi - k * tau + math.pi) % TWO_PI - math.pi
        if r < self.r_bore + self.tooth_tip:
            return "slot_opening" if abs(d) < self.slot_opening * tau / 2 else "stator_iron"
        if r < self.r_bore + self.tooth_tip + self.slot_depth:
            return f"slot{k}" if abs(d) < self.slot_width * tau / 2 else "stator_iron"
        return "stator_iron"

    def region_names(self) -> list[str]:
        names = ["shaft", "rotor_core", "rotor_air", "gap", "slot_opening", "stator_iron"]
        names += [f"magnet{k}" for k in range(2 * self.pole_pairs)]
        names += [f"slot{k}" for k in range(self.slots)]
        return names

    def mesh(self, theta: float = 0.0) -> Mesh2D:
        return polar_mesh(self._rings, self._classify, theta=theta, region_names=self.region_names())

    def rotor_mask(self, mesh: Mesh2D) -> np.ndarray:
        rot = np.array([r.rotating for r in self._rings])
        return rot[mesh.layer] & rot[mesh.layer + 1]

    def coil_sides(self) -> list[CoilSide]:
        return [CoilSide(f"slot{k}", ph, self.turns_per_slot, sg) for k, (ph, sg) in enumerate(self.winding or [])]

    # ------------------------------------------------------------------ materials and sources
    def _magnet_br(self, mesh: Mesh2D, theta: float) -> np.ndarray:
        Br = np.zeros((mesh.n_elements, 2))
        c = mesh.centroids()
        tau = math.pi / self.pole_pairs
        for k in range(2 * self.pole_pairs):
            m = mesh.mask(f"magnet{k}")
            if not m.any():
                continue
            sgn = 1.0 if k % 2 == 0 else -1.0
            if self.magnetization == "radial":
                ang = np.arctan2(c[m, 1], c[m, 0])
            else:
                ang = np.full(m.sum(), k * tau + theta)
            Br[m, 0] = sgn * self.Br * np.cos(ang)
            Br[m, 1] = sgn * self.Br * np.sin(ang)
        return Br

    def current_density(self, mesh: Mesh2D, geometry: Geometry, currents: Sequence[float]) -> np.ndarray:
        J = np.zeros(mesh.n_elements)
        for cs in self.coil_sides():
            m = mesh.mask(cs.region)
            S = geometry.area[m].sum()
            if S > 0:
                J[m] = cs.sign * cs.turns * currents[cs.phase] / S
        return J

    def _materials(self, mesh: Mesh2D, design: np.ndarray | None):
        nu = np.full(mesh.n_elements, NU0)
        nonlinear = []
        nu[mesh.mask_prefix("magnet")] = NU0 / self.mu_r_magnet
        stator = mesh.mask("stator_iron")
        rotor = mesh.mask("rotor_core")
        rotor_iron = self.rotor_iron if self.rotor_iron is not None else self.iron
        for mask, mat in ((stator, self.iron), (rotor, rotor_iron)):
            if mask is rotor and design is not None:
                continue
            if isinstance(mat, LinearMaterial):
                nu[mask] = mat.nu
            elif isinstance(mat, (BHCurve, BrauerCurve)) or hasattr(mat, "dnu"):
                nonlinear.append((mask, mat))
            else:
                nu[mask] = NU0 / float(mat)
        if design is not None:
            w = np.asarray(design, float) ** self.design_penalty
            mat = rotor_iron.mu_r if isinstance(rotor_iron, LinearMaterial) else rotor_iron
            nonlinear.append((rotor, ScaledCurve(mat, w)))
        return nu, nonlinear

    # ------------------------------------------------------------------ solve
    def solve(self, theta: float = 0.0, currents: Sequence[float] = (0.0, 0.0, 0.0),
              design: np.ndarray | None = None, A0=None, keep_tangent: bool = False,
              tol: float = 1e-9) -> MachineSolution:
        mesh = self.mesh(theta)
        g = Geometry.of(mesh)
        design = self.rotor_design if design is None else design
        nu, nonlinear = self._materials(mesh, design)
        J = self.current_density(mesh, g, currents) if self.slots else np.zeros(mesh.n_elements)
        Br = self._magnet_br(mesh, theta)
        sol = solve_magnetostatic(mesh, nu, J=J, Br=Br, nonlinear=nonlinear, geometry=g, A0=A0,
                                  keep_tangent=keep_tangent, tol=tol)
        return MachineSolution(theta, tuple(currents), mesh, sol, J)

    def gap_annuli(self) -> list[tuple[float, float]]:
        """Air annuli used by the Maxwell-stress torque: the gap layers on the rotor side and on the
        stator side of the moving band (the band itself is excluded: its elements change shape with
        the rotor angle and add an angle-periodic error)."""
        k, g, r0 = self.gap_layers, self.airgap, self.r_magnet
        return [(r0, r0 + g * k / (2 * k + 1)), (r0 + g * (k + 1) / (2 * k + 1), self.r_bore)]

    def torque(self, ms: MachineSolution) -> float:
        """Maxwell-stress (Arkkio) torque, N m: the length-weighted mean of the Arkkio integrals over
        the two annuli of :meth:`gap_annuli` (each one alone is a valid estimate)."""
        c = ms.mesh.centroids()
        rc = np.hypot(c[:, 0], c[:, 1])
        gap = ms.mesh.mask("gap")
        num = den = 0.0
        for r1, r2 in self.gap_annuli():
            el = np.where(gap & (rc > r1) & (rc < r2))[0]
            num += torque_maxwell_stress(ms.mesh, ms.B, r1, r2, self.length, elements=el) * (r2 - r1)
            den += r2 - r1
        return num / den

    def torque_form(self, ms: MachineSolution):
        """The torque of :meth:`torque` as a quadratic form of A: ``(value(A), gradient(A))`` object."""
        c = ms.mesh.centroids()
        rc = np.hypot(c[:, 0], c[:, 1])
        gap = ms.mesh.mask("gap")
        forms, w = [], []
        ann = self.gap_annuli()
        den = sum(r2 - r1 for r1, r2 in ann)
        for r1, r2 in ann:
            el = np.where(gap & (rc > r1) & (rc < r2))[0]
            forms.append(torque_maxwell_stress_gradient(ms.mesh, ms.sol.geometry, r1, r2, el, self.length))
            w.append((r2 - r1) / den)
        return _Combined(forms, w)

    def design_elements(self, mesh: Mesh2D | None = None) -> np.ndarray:
        """Indices of the rotor-core elements (the design domain of :attr:`rotor_design`)."""
        mesh = mesh or self.mesh(0.0)
        return np.where(mesh.mask("rotor_core"))[0]

    def torque_and_design_gradient(self, theta: float, currents: Sequence[float], design: np.ndarray,
                                   A0=None):
        """Torque (N m) and dT/d(design) by the adjoint method (Newton tangent is symmetric).

        Residual ``R = P^T (K(nu(rho, |B|^2)) A - f)``; ``K_t lambda = P^T dT/dA``;
        ``dT/drho_e = -area_e (dnu_e/drho_e) grad(lambda_e) . grad(A_e)`` with
        ``nu = nu0 + rho^p (nu_iron(|B|^2) - nu0)``.
        """
        import scipy.sparse.linalg as spla

        ms = self.solve(theta, currents, design=design, A0=A0, keep_tangent=True, tol=1e-11)
        form = self.torque_form(ms)
        T = form.value(ms.A)
        dTdA = form.gradient(ms.A)
        P = ms.sol.P
        lam = P @ spla.spsolve(ms.sol.tangent, P.T @ dTdA)
        mesh, g = ms.mesh, ms.sol.geometry
        el = np.where(mesh.mask("rotor_core"))[0]
        B2 = (ms.B[el] ** 2).sum(1)
        rotor_iron = self.rotor_iron if self.rotor_iron is not None else self.iron
        mat = rotor_iron.mu_r if isinstance(rotor_iron, LinearMaterial) else rotor_iron
        nu_fe = ScaledCurve(mat, np.ones(len(el))).nu_iron(B2)
        rho = np.asarray(design, float)
        dnu = self.design_penalty * rho ** (self.design_penalty - 1) * (nu_fe - NU0)
        a, lm = ms.A[mesh.tris[el]], lam[mesh.tris[el]]
        gA = np.column_stack([(a * g.gx[el]).sum(1), (a * g.gy[el]).sum(1)])
        gL = np.column_stack([(lm * g.gx[el]).sum(1), (lm * g.gy[el]).sum(1)])
        dT = -g.area[el] * dnu * (gA * gL).sum(1)
        return T, dT, ms

    def coenergy(self, theta: float, currents: Sequence[float], design=None) -> float:
        """Co-energy (J), up to an angle-independent constant."""
        return self.length * self.solve(theta, currents, design).sol.coenergy

    def torque_coenergy(self, theta: float, currents: Sequence[float], design=None,
                        steps: int = 1) -> float:
        """Virtual-work torque ``dW'/dtheta`` at constant currents, N m.

        Fourth-order central differences of the co-energy at ``theta +- k h`` (k = 1, 2), with ``h`` =
        ``steps`` angular grid steps: at those angles the moving band has exactly the same element
        shapes (rotated), so the differences contain no band-distortion error."""
        h = steps * TWO_PI / self.n_theta
        W = [self.coenergy(theta + k * h, currents, design) for k in (-2, -1, 1, 2)]
        return (W[0] - 8 * W[1] + 8 * W[2] - W[3]) / (12 * h)

    def flux_linkage(self, ms: MachineSolution) -> np.ndarray:
        return flux_linkage(ms.mesh, ms.sol.geometry, ms.A, self.coil_sides(), 3, self.length)

    # ------------------------------------------------------------------ operating points
    def pm_flux_angle(self) -> float:
        """Electrical angle ``phi0`` of the no-load flux-linkage space vector at ``theta = 0``."""
        if self._phi0 is None:
            psi = self.flux_linkage(self.solve(0.0, (0.0, 0.0, 0.0)))
            a = np.exp(2j * math.pi / 3)
            self._phi0 = float(np.angle((2 / 3) * (psi[0] + a * psi[1] + a * a * psi[2])))
        return self._phi0

    def phase_currents(self, theta: float, I_peak: float, gamma: float = 0.0, phi0: float | None = None):
        phi0 = self.pm_flux_angle() if phi0 is None else phi0
        x = self.pole_pairs * theta + phi0 + math.pi / 2 + gamma
        return tuple(I_peak * math.cos(x - 2 * math.pi * k / 3) for k in range(3))

    def _sweep(self, thetas, currents_of):
        """Solve at each angle, warm-starting Newton from the previous angle (same node numbering)."""
        out, A0 = [], None
        for t in thetas:
            ms = self.solve(t, currents_of(t), A0=A0)
            A0 = ms.A
            out.append(ms)
        return out

    def cogging(self, thetas: Sequence[float]) -> np.ndarray:
        """No-load (cogging) torque at each rotor angle, N m."""
        return np.array([self.torque(ms) for ms in self._sweep(thetas, lambda t: (0.0, 0.0, 0.0))])

    def cogging_period(self) -> float:
        """Mechanical period of the cogging torque: 2 pi / lcm(slots, 2 p)."""
        return TWO_PI / math.lcm(self.slots, 2 * self.pole_pairs)

    def no_load_flux(self, thetas: Sequence[float]) -> np.ndarray:
        """Magnet flux linkage per phase (Wb-turn), shape (angles, 3)."""
        return np.array([self.flux_linkage(ms) for ms in self._sweep(thetas, lambda t: (0.0, 0.0, 0.0))])

    def torque_vs_current_angle(self, I_peak: float, gammas: Sequence[float], theta: float = 0.0):
        out, A0 = [], None
        for gm in gammas:
            ms = self.solve(theta, self.phase_currents(theta, I_peak, gm), A0=A0)
            A0 = ms.A
            out.append(self.torque(ms))
        return np.array(out)

    def period_sweep(self, I_peak: float, gamma: float, steps: int, theta0: float = 0.0):
        """Solutions at ``steps`` rotor angles over one electrical period with synchronous currents."""
        thetas = theta0 + (TWO_PI / self.pole_pairs) * np.arange(steps) / steps
        return thetas, self._sweep(thetas, lambda t: self.phase_currents(t, I_peak, gamma))

    def losses(self, rpm: float, I_peak: float, gamma: float, model: IronLossModel, steps: int = 36,
               n_harmonics: int | None = None) -> dict:
        """Iron loss (stator, rotor) and resistance-limited magnet eddy loss over one electrical period."""
        thetas, sols = self.period_sweep(I_peak, gamma, steps)
        mesh = sols[0].mesh
        g = sols[0].sol.geometry
        rot = self.rotor_mask(mesh)
        Bh = element_flux_density_history([s.B for s in sols], thetas, rot)
        f = rpm / 60.0 * self.pole_pairs
        iron_st = mesh.mask("stator_iron")
        iron_rt = mesh.mask("rotor_core")
        Pst, _ = iron_loss(Bh, f, model, g.area, iron_st, self.length, n_harmonics)
        Prt, _ = iron_loss(Bh, f, model, g.area, iron_rt, self.length, n_harmonics)
        A_hist = np.array([s.A for s in sols])
        pieces = [mesh.mask(f"magnet{k}") for k in range(2 * self.pole_pairs)]
        pieces = [p for p in pieces if p.any()]
        Pm = magnet_eddy_loss_resistance_limited(mesh, g, A_hist, 1.0 / f, self.magnet_sigma, pieces,
                                                 self.length) if pieces else 0.0
        torque = np.array([self.torque(s) for s in sols])
        return {"stator_iron_W": Pst, "rotor_iron_W": Prt, "magnet_eddy_W": Pm, "frequency_Hz": f,
                "torque_mean_Nm": float(torque.mean()), "torque": torque, "thetas": thetas}

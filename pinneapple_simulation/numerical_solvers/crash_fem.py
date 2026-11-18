"""Explicit structural crash dynamics on 8-node hexahedra (pole impact, barrier impact, axial crush).

Written from the published methods — no code from OpenRadioss, LS-DYNA or CalculiX is used:

- central-difference explicit time integration with lumped mass, the scheme every explicit crash code runs, at the
  critical time step of the mesh (``dt = cfl * h / c_d``, ``c_d`` the dilatational wave speed of the element);
- 2 x 2 x 2 Gauss integration instead of one-point integration with hourglass control: the eight integration points
  supply the anti-hourglass stiffness themselves, so the element has nothing to tune and cannot hourglass;
- **corotational kinematics**: every element carries the rotation ``R`` of the polar decomposition of its centre
  deformation gradient; the previous stress and strain are rotated with ``R_new R_old^T`` and the strain measure is
  taken in the rotated frame (``eps = 0.5 (R^T F + F^T R) - I``), so a part can fold and rotate without the
  small-strain measure invalidating the result — which is what a crash simulation is made of;
- J2 (von Mises) plasticity with linear isotropic hardening, radial return at every integration point, element
  erosion at a critical equivalent plastic strain (and on element inversion), with the eroded element's mass removed
  from its nodes;
- penalty contact against a **rigid obstacle**: a moving wall (the barrier of a full-frontal impact) or a moving
  cylinder (the pole of a pole impact), with Coulomb friction, the elastic contact energy kept and the reaction
  force recorded;
- the energies (kinetic, internal, plastic work, contact) and the crashworthiness quantities a design sweep ranks by:
  peak wall force, peak deceleration, peak internal energy, peak plastic work, kinetic-energy absorption
  ``eta = 1 - E_kin(T) / E_kin(0)`` and the impact duration above a threshold of the peak force.

Envelope: the corotational strain measure holds for moderate stretches (elastic strains are small, plastic ones may
be large, but no single element should be stretched by more than a few tens of percent); the material is isotropic
hardening without damage, and erosion stands in for failure. Elements that would break that envelope (relative
length change above ``ENVELOPE``) set ``CrashResult.out_of_envelope`` — the run still returns, but its numbers are
outside the verified range of this model.

    from pinneapple_simulation.numerical_solvers.crash_fem import Material, RigidPole, CrashSolver
    from pinneapple_simulation.numerical_solvers.solid_fem import box_mesh

    mesh = box_mesh(0.30, 0.06, 0.04, 30, 6, 4)                     # a 300 mm bar, 5 mm elements
    mat = Material(rho=7.8e3, E=2.1e11, nu=0.3, yield_stress=6.0e8, hardening=4.0e8, erosion_strain=0.4)
    pole = RigidPole(radius=0.10, x0=0.32)                          # rigid cylinder just ahead of the bar
    solver = CrashSolver(mesh, material=mat, contact=pole, initial_velocity=(30.0, 0.0, 0.0), end_time=0.02)
    res = solver.run()
    print(res.qoi())                                                # peak_wall_force, ke_absorption, ...
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .solid_fem import HexMesh, elasticity_matrix, von_mises

_GP = 1.0 / math.sqrt(3.0)
_GAUSS = np.array([[a, b, c] for a in (-_GP, _GP) for b in (-_GP, _GP) for c in (-_GP, _GP)])   # 2 x 2 x 2
_XI = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)     # hexahedron order of HexMesh
ENVELOPE = 0.35                      # relative length change beyond which the strain measure is out of its range
ENERGY_TOL = 0.05                    # imbalance against the external work that has to hold two frames in a row
ENERGY_TRIP = 0.25                   # imbalance against the external work that one frame is enough to trip

__all__ = ["Material", "RigidPole", "RigidWall", "CrashSolver", "CrashResult", "qoi_from_history",
           "contact_force_on_pole"]


# ------------------------------------------------------------------------------------------------- material
@dataclass(frozen=True)
class Material:
    """Isotropic-hardening J2 material — the phenomenological model of a crash alloy (DP600, DP1000, mild steel).

    ``yield_stress`` is the initial yield stress, ``hardening`` the slope of the yield stress against the equivalent
    plastic strain (linear isotropic hardening; 0 is perfectly plastic), ``erosion_strain`` the equivalent plastic
    strain at which the element is removed (0 keeps every element)."""
    rho: float                   # density [kg/m^3]
    E: float                     # Young's modulus [Pa]
    nu: float                    # Poisson ratio [-]
    yield_stress: float          # initial yield stress [Pa]
    hardening: float = 0.0       # d(yield stress)/d(equivalent plastic strain) [Pa]
    erosion_strain: float = 0.0  # equivalent plastic strain at erosion [-], 0 = never

    def __post_init__(self):
        if self.rho <= 0 or self.E <= 0 or self.yield_stress <= 0:
            raise ValueError("rho, E and yield_stress must be positive")
        if not -1.0 < self.nu < 0.5:
            raise ValueError("nu must lie in (-1, 0.5)")
        if self.erosion_strain < 0 or self.hardening < 0:
            raise ValueError("hardening and erosion_strain must be >= 0")

    def wave_speed(self) -> float:
        """Dilatational (P) wave speed, ``sqrt((lambda + 2 mu) / rho)`` [m/s]; it sets the stable time step."""
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        return math.sqrt((lam + 2 * mu) / self.rho)


# ------------------------------------------------------------------------------------------------- contact
@dataclass(frozen=True)
class RigidPole:
    """Rigid cylinder, axis along z, centre ``(x0 + vx t, y0)``, radius ``radius``.

    A pole impact on a body that moves is the same problem seen from the pole: keep ``vx = 0`` and give the body
    ``initial_velocity``. ``z_range`` limits the pole to a band of z (the height it spans); ``None`` spans it all."""
    radius: float
    x0: float = 0.0
    y0: float = 0.0
    vx: float = 0.0
    z_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class RigidWall:
    """Rigid plane at ``x = x0 + vx t``, the body on its +x side (the barrier of a full-frontal impact)."""
    x0: float
    vx: float = 0.0


def _signed_distance(obstacle, x: np.ndarray, t: float) -> np.ndarray:
    """Signed distance to the obstacle at time ``t``: >= 0 outside it."""
    if isinstance(obstacle, RigidWall):
        return (obstacle.x0 + obstacle.vx * t) - x[:, 0]
    if isinstance(obstacle, RigidPole):
        xp = obstacle.x0 + obstacle.vx * t
        d = np.hypot(x[:, 0] - xp, x[:, 1] - obstacle.y0) - obstacle.radius
        if obstacle.z_range is not None:
            lo, hi = obstacle.z_range
            d = np.where((x[:, 2] >= lo) & (x[:, 2] <= hi), d, 1e9)
        return d
    raise TypeError(f"unknown obstacle {obstacle!r}")


def _outward_normal(obstacle, x: np.ndarray) -> np.ndarray:
    """Unit normal of the obstacle pointing away from it, at the points ``x``."""
    n = np.zeros_like(x)
    if isinstance(obstacle, RigidWall):
        n[:, 0] = -1.0                                                     # towards the body, which sits at x < wall
        return n
    xp = obstacle.x0
    dx, dy = x[:, 0] - xp, x[:, 1] - obstacle.y0
    r = np.maximum(np.hypot(dx, dy), 1e-30)
    n[:, 0], n[:, 1] = dx / r, dy / r
    return n


# ------------------------------------------------------------------------------------------------- result
def qoi_from_history(times: np.ndarray, contact_force: np.ndarray, energies: dict[str, np.ndarray],
                     threshold: float = 0.03, velocity: np.ndarray | None = None) -> dict:
    """The crashworthiness quantities of an impact, from recorded histories.

    ``peak_wall_force`` [N]: largest contact force along x. ``peak_internal_energy`` [J] and ``peak_plastic_work``
    [J]: their maxima. ``ke_absorption`` [-]: ``1 - E_kin(T) / E_kin(0)``. ``impact_duration`` [s]: the time the
    pole force stays above ``threshold`` (3 % by default) of its peak. ``peak_deceleration`` [m/s^2]: the largest
    rate of change of the mean node velocity when ``velocity`` is given."""
    t = np.asarray(times, float)
    fx = np.asarray(contact_force, float)[:, 0] if np.asarray(contact_force).ndim > 1 else np.asarray(
        contact_force, float)
    peak_force = float(np.abs(fx).max())
    ke = np.asarray(energies["kinetic"], float)
    ext = np.asarray(energies.get("external", np.zeros(1)), float)
    above = np.abs(fx) >= threshold * peak_force if peak_force > 0 else np.zeros(len(fx), bool)
    duration = float(t[np.nonzero(above)[0][-1]] - t[np.nonzero(above)[0][0]]) if above.any() else 0.0
    acc = 0.0
    if velocity is not None and len(t) > 2 and np.asarray(velocity).ndim == 3:
        vmean = np.asarray(velocity, float).mean(axis=1)
        acc = float(np.abs(np.gradient(vmean, t, axis=0)).max())
    return {"peak_wall_force": peak_force,
            "peak_deceleration": acc,
            "peak_internal_energy": float(np.asarray(energies["internal"], float).max()),
            "peak_plastic_work": float(np.asarray(energies["plastic"], float).max()),
            "ke_absorption": float(1.0 - ke[-1] / ke[0]) if ke[0] > 0 else
            (float(1.0 - ke[-1] / max(ext[-1], 1e-30)) if ext.size and ext[-1] > 0 else 0.0),
            "impact_duration": duration}


@dataclass
class CrashResult:
    """Frames, histories and design quantities of a crash run."""
    mesh: HexMesh
    parts: np.ndarray                    # (m,) part id of every element
    times: np.ndarray                    # (F,) s
    displacement: np.ndarray             # (F, n, 3) m
    velocity: np.ndarray                 # (F, n, 3) m/s
    von_mises: np.ndarray                # (F, m) Pa, the largest integration point of the element
    plastic_strain: np.ndarray           # (F, m) -, the largest integration point of the element
    eroded: np.ndarray                   # (F, m) bool
    contact_force: np.ndarray            # (F, 3) N, on the body (the obstacle feels the opposite)
    energies: dict[str, np.ndarray]      # kinetic, elastic, internal, plastic, contact [J] per frame
    out_of_envelope: bool = False
    envelope_reason: str | None = None    # "stretch" or "energy_balance": why the run left the verified range
    dt: float = 0.0
    time_history: np.ndarray | None = None        # (S,) s, every time step (the force pulse needs it)
    force_history: np.ndarray | None = None       # (S,) N, the contact force along x at every step
    info: dict | None = None                      # whatever the case builder attaches (design, mesh, fixed nodes)
    vx_history: np.ndarray | None = None          # (S,) m/s, mean axial velocity of the free nodes per step

    @property
    def n_frames(self) -> int:
        return int(len(self.times))

    def qoi(self, threshold: float = 0.03) -> dict:
        """See :func:`qoi_from_history`, evaluated on the per-step force history (a contact pulse is far shorter
        than the frame interval of the stored fields)."""
        times, force = ((self.time_history, self.force_history) if self.force_history is not None
                        else (self.times, self.contact_force))
        q = qoi_from_history(times, force, self.energies, threshold=threshold)
        if self.vx_history is not None and self.time_history is not None and len(self.vx_history) > 2:
            # the measured deceleration is a filtered one: a raw central difference on an impulsive contact reads the
            # single-step spike of the penalty spring, which is a numerical artefact, not what an accelerometer in the
            # vehicle would record (the SAE CFC 60 filter is the usual choice; a moving mean of a few steps stands in)
            vx = np.asarray(self.vx_history, float)
            w = max(1, min(len(vx) // 20, 10))
            if w > 1:
                vx = np.convolve(vx, np.ones(w) / w, mode="same")
            acc = np.gradient(vx, np.asarray(self.time_history, float))
            q["peak_deceleration"] = float(np.abs(acc[w:-w] if len(acc) > 2 * w + 2 else acc).max())
            return q
        if self.velocity is not None and len(self.times) > 2:
            vmean = np.asarray(self.velocity, float).mean(axis=1)
            q["peak_deceleration"] = float(np.abs(np.gradient(vmean, self.times, axis=0)).max())
        return q

    def energy_balance(self) -> np.ndarray:
        """Energy in the body per frame (kinetic + internal + contact), [J].

        For a free body or an impact with no external work done during the stored frames this is the quantity that
        has to stay constant; with an obstacle driving the body it grows by the work of the contact force, which
        :meth:`energies` keeps as ``external`` (``external + constant = kinetic + internal + contact``)."""
        return (np.asarray(self.energies["kinetic"], float) + np.asarray(self.energies["internal"], float)
                + np.asarray(self.energies["contact"], float))


def contact_force_on_pole(result: CrashResult) -> np.ndarray:
    """The force the obstacle feels (the opposite of the one on the body), (F, 3) N."""
    return -np.asarray(result.contact_force, float)


# ------------------------------------------------------------------------------------------------- kinematics
def _dN_dxi(xi: np.ndarray) -> np.ndarray:
    """dN_a/dxi_k at the natural point ``xi`` (8, 3)."""
    d = np.empty((8, 3))
    d[:, 0] = _XI[:, 0] * (1 + _XI[:, 1] * xi[1]) * (1 + _XI[:, 2] * xi[2]) / 8
    d[:, 1] = _XI[:, 1] * (1 + _XI[:, 0] * xi[0]) * (1 + _XI[:, 2] * xi[2]) / 8
    d[:, 2] = _XI[:, 2] * (1 + _XI[:, 0] * xi[0]) * (1 + _XI[:, 1] * xi[1]) / 8
    return d


def _B_matrix(dN_dx: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix (..., 6, 24) from shape-function gradients (..., 8, 3); Voigt with engineering
    shear (xx, yy, zz, xy, yz, zx)."""
    sh = dN_dx.shape[:-2]
    B = np.zeros(sh + (6, 24))
    gx, gy, gz = dN_dx[..., 0], dN_dx[..., 1], dN_dx[..., 2]
    B[..., 0, 0::3], B[..., 1, 1::3], B[..., 2, 2::3] = gx, gy, gz
    B[..., 3, 0::3], B[..., 3, 1::3] = gy, gx
    B[..., 4, 1::3], B[..., 4, 2::3] = gz, gy
    B[..., 5, 0::3], B[..., 5, 2::3] = gz, gx
    return B


def _voigt_to_tensor(s: np.ndarray) -> np.ndarray:
    """Voigt stresses (..., 6) -> (..., 3, 3); the shear entries are the stress components."""
    t = np.empty(s.shape[:-1] + (3, 3))
    t[..., 0, 0], t[..., 1, 1], t[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    t[..., 0, 1] = t[..., 1, 0] = s[..., 3]
    t[..., 1, 2] = t[..., 2, 1] = s[..., 4]
    t[..., 0, 2] = t[..., 2, 0] = s[..., 5]
    return t


def _tensor_to_voigt(t: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> Voigt (xx, yy, zz, xy, yz, zx)."""
    out = np.empty(t.shape[:-2] + (6,))
    out[..., 0], out[..., 1], out[..., 2] = t[..., 0, 0], t[..., 1, 1], t[..., 2, 2]
    out[..., 3], out[..., 4], out[..., 5] = t[..., 0, 1], t[..., 1, 2], t[..., 0, 2]
    return out


def _strain_voigt(grad: np.ndarray) -> np.ndarray:
    """Symmetric part of a displacement gradient (..., 3, 3) -> Voigt strain with engineering shear."""
    t = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    out = _tensor_to_voigt(t)
    out[..., 3:] *= 2.0
    return out


def _voigt_strain_to_tensor(s: np.ndarray) -> np.ndarray:
    """Voigt strain with engineering shear (..., 6) -> (..., 3, 3) tensor strain (shear halved)."""
    t = s.copy()
    t[..., 3:] *= 0.5
    return _voigt_to_tensor(t)


def _tensor_strain_to_voigt(t: np.ndarray) -> np.ndarray:
    """(..., 3, 3) tensor strain -> Voigt strain with engineering shear."""
    out = _tensor_to_voigt(t)
    out[..., 3:] *= 2.0
    return out


# ------------------------------------------------------------------------------------------------- solver
class CrashSolver:
    """Explicit nonlinear dynamics of a hexahedral assembly against a rigid obstacle (see the module docstring).

    Parameters
    ----------
    mesh:
        :class:`~pinneapple_simulation.numerical_solvers.solid_fem.HexMesh`; ``mesh.shape`` (nx, ny, nz) is 0 for
        a mesh a code built itself, which is fine — only the connectivity is used here.
    material:
        A :class:`Material` for the whole mesh, or ``{part_id: Material}`` together with ``parts``.
    parts:
        (m,) part id of every element — the semantic part labels a part-aware surrogate also wants; default 0.
    contact:
        :class:`RigidPole` or :class:`RigidWall`, or ``None`` (nothing in the way).
    fixed:
        Node ids held at rest (the clamped end of the assembly); ``None`` for a free body.
    initial_velocity:
        (3,) for the whole body or (n, 3) per node [m/s].
    cfl:
        Fraction of the critical time step (0.9 by default; explicit dynamics is stable for cfl <= 1).
    penalty:
        Contact stiffness as a multiple of ``m_node / dt^2`` (a penalty that stiffens the contact more than the
        node's own natural frequency is the usual choice, and 1.0 is stable for the central-difference scheme).
    friction:
        Coulomb coefficient of the contact, 0 by default.
    end_time, frames, record_velocity:
        End of the simulated physical time, how many frames to keep over it, and whether to store the velocity
        history (needed for ``peak_deceleration``).
    """

    def __init__(self, mesh: HexMesh, *, material: Material | dict[int, Material], parts=None,
                 contact: RigidPole | RigidWall | None = None, fixed=None, initial_velocity=0.0,
                 cfl: float = 0.9, penalty: float = 1.0, friction: float = 0.0,
                 end_time: float | None = None, frames: int = 41, record_velocity: bool = True,
                 dt: float | None = None):
        self.mesh = mesh
        self.elements = np.asarray(mesh.elements, int)
        self.n = len(mesh.nodes)
        self.m = len(self.elements)
        self.parts = np.zeros(self.m, int) if parts is None else np.asarray(parts, int).ravel()
        if self.parts.shape != (self.m,):
            raise ValueError("parts must give one id per element")
        if isinstance(material, dict):
            missing = sorted(set(self.parts.tolist()) - set(material))
            if missing:
                raise ValueError(f"no material for part(s) {missing}")
            mats = [material[int(p)] for p in self.parts]
        else:
            mats = [material] * self.m
        self.contact = contact
        self.t = 0.0
        self.cfl, self.penalty_scale, self.friction = float(cfl), float(penalty), float(friction)
        self.record_velocity = bool(record_velocity)

        # material arrays, one entry per element
        self.rho_e = np.array([q.rho for q in mats], float)
        self.sy_e = np.array([q.yield_stress for q in mats], float)
        self.H_e = np.array([q.hardening for q in mats], float)
        self.ee_e = np.array([q.erosion_strain for q in mats], float)
        self.D_e = np.array([elasticity_matrix(q.E, q.nu) for q in mats], float)               # (m, 6, 6)
        self.mu_e = np.array([q.E / (2 * (1 + q.nu)) for q in mats], float)
        self.wave_e = np.array([q.wave_speed() for q in mats], float)

        # reference configuration: gradients, volume, B matrices at the 8 Gauss points
        self.x_e = np.asarray(mesh.nodes, float)[self.elements]                                # (m, 8, 3)
        dN_dxi = np.stack([_dN_dxi(g) for g in _GAUSS])                                        # (8gp, 8, 3)
        J = np.einsum("gai,eaj->egij", dN_dxi, self.x_e)                                       # (m, 8gp, 3, 3)
        detJ = np.linalg.det(J)
        if (detJ <= 0).any():
            raise ValueError("the mesh has an inverted or degenerate element (det J <= 0)")
        self.gp_w = detJ                                                                       # (m, 8gp) dV at each gp
        self.volume_e = detJ.sum(1)                                                            # (m,)
        self.dN_dx_gp = np.einsum("gai,egij->egaj", dN_dxi, np.linalg.inv(J))                   # (m, 8gp, 8, 3)
        self.B_e = _B_matrix(self.dN_dx_gp)                                                     # (m, 8gp, 6, 24)
        self.dN_dx_mean = self.dN_dx_gp.mean(1)                                                # (m, 8, 3), centre
        self.mass_e = self.rho_e * self.volume_e

        # lumped nodal mass and the inverse that roots the update
        self.node_mass = np.zeros(self.n)
        np.add.at(self.node_mass, self.elements.ravel(), np.repeat(self.mass_e[:, None] / 8.0, 8, 1).ravel())
        if fixed is None:
            self.fixed = np.zeros(self.n, bool)
        else:
            arr = np.asarray(fixed)
            if arr.dtype == bool and arr.size == self.n:
                self.fixed = arr.ravel().copy()
            else:
                self.fixed = np.zeros(self.n, bool)
                self.fixed[arr.ravel().astype(int)] = True        # node ids, not a mask
        self._update_inv_mass()

        # stable time step: the exact highest eigenfrequency of this element with its lumped mass, not an estimate
        # (the textbook h / c_d estimate over-rates the smallest hexahedron with lumped mass by a factor ~2, which
        # makes a central-difference run diverge while looking "Courant-safe")
        h = np.cbrt(self.volume_e)
        estimate = self.cfl * np.min(h / self.wave_e)
        omega_max = self._max_eigenfrequency()
        dt_crit = math.inf if omega_max == 0 else 2.0 / omega_max
        self.dt = float(dt) if dt is not None else float(min(self.cfl * dt_crit, dt_crit))
        self.dt_estimate = float(estimate)
        self.penalty_k = self.penalty_scale * self.node_mass / self.dt ** 2

        v0 = np.asarray(initial_velocity, float)
        if v0.size == 1:                                                        # a scalar speed along x
            v0 = np.array([float(v0.ravel()[0]), 0.0, 0.0])
        self.v = (np.repeat(v0.reshape(1, 3), self.n, 0) if v0.size == 3 else v0.copy())
        if self.v.shape != (self.n, 3):
            raise ValueError("initial_velocity must be a scalar, (3,) or (n, 3)")
        self.v[self.fixed] = 0.0
        self.u = np.zeros((self.n, 3))

        # state at the integration points
        self.R_e = np.repeat(np.eye(3)[None], self.m, 0)                                       # (m, 3, 3)
        self.sig = np.zeros((self.m, 8, 6))                                                    # (m, 8gp, 6)
        self.eps_p = np.zeros((self.m, 8))                                                     # (m, 8gp) eq plastic
        self.eps_p_t = np.zeros((self.m, 8, 6))                                                # its tensor
        self.eps_prev = np.zeros((self.m, 8, 6))                                               # corotated strain
        self.eroded = np.zeros(self.m, bool)
        self.out_of_envelope = False
        self.envelope_reason: str | None = None

        self.end_time = float(end_time) if end_time is not None else None
        self.frames = max(2, int(frames))
        self._plastic_work = 0.0
        self._contact_energy = 0.0
        self._external_work = 0.0
        self._force_hist: list[np.ndarray] = []
        self._time_hist: list[float] = []
        self._vx_hist: list[float] = []

    # ------------------------------------------------------------------------------------ housekeeping
    def _update_inv_mass(self) -> None:
        self.inv_mass = np.where(self.node_mass > 0, 1.0 / np.maximum(self.node_mass, 1e-30), 0.0)
        self.inv_mass[self.fixed] = 0.0

    def element_stiffness(self) -> np.ndarray:
        """(m, 24, 24) linear elastic stiffness of every element — used for the critical time step."""
        B3 = self.B_e.reshape(self.m, 8, 6, 24)
        return np.einsum("egci,egcd,egdj,eg->eij", B3, self.D_e[:, None], B3, self.gp_w, optimize=True)

    def _max_eigenfrequency(self) -> float:
        """Highest eigenfrequency of any element carrying its own share of the lumped mass [rad/s].

        Computed from the generalised problem ``K_e phi = omega^2 M_e phi`` with the diagonal ``M_e``, which is
        what the central-difference scheme actually sees; shared nodes only make the true masses larger, so this is
        a conservative upper bound."""
        K = self.element_stiffness()
        m_share = np.repeat(np.repeat((self.mass_e / 8.0)[:, None], 8, axis=1), 3, axis=1)      # (m, 24)
        if (m_share <= 0).any():
            return 0.0
        d = 1.0 / np.sqrt(m_share)
        scaled = K * d[:, :, None] * d[:, None, :]
        lam = np.linalg.eigvalsh(scaled).max(axis=1)
        return float(np.sqrt(max(float(np.max(lam)), 0.0)))

    def _kinetics(self) -> np.ndarray:
        """Deformation gradient of every element at its centre, (m, 3, 3)."""
        du = self.u[self.elements]                                                             # (m, 8, 3)
        grad = np.einsum("eaj,eai->eji", du, self.dN_dx_mean)                                   # du_i / dX_j
        return np.eye(3)[None] + grad

    def _strain_at_gauss_points(self) -> tuple[np.ndarray, np.ndarray]:
        """(corotational strain at the 8 Gauss points (m, 8, 6), deformation gradient of the centre)."""
        du = self.u[self.elements]
        grad_gp = np.einsum("eai,egaj->egij", du, self.dN_dx_gp, optimize=True)                                # du_i / dX_j
        F_gp = np.eye(3)[None, None] + grad_gp
        F_c = np.eye(3)[None] + np.einsum("eai,eaj->eij", du, self.dN_dx_mean)
        R = _polar_rotation(F_c)
        RtF = np.einsum("eji,egjk->egik", R, F_gp)                                             # R^T F
        eps = _strain_voigt(RtF - np.eye(3)[None, None])                                        # (m, 8gp, 6)
        self._R_last, self._F_last = R, F_c
        return eps, F_c


    # ------------------------------------------------------------------------------------ one step
    def step(self) -> dict:
        """Advance one central-difference step and return its instantaneous scalars."""
        eps, F_c = self._strain_at_gauss_points()
        R = self._R_last
        R_rel = np.einsum("eij,ekj->eik", R, self.R_e)                                          # R_new R_old^T
        sig_prev = _tensor_to_voigt(np.einsum("eij,egjk,elk->egil", R_rel,
                                              _voigt_to_tensor(self.sig), R_rel))
        eps_prev = _tensor_strain_to_voigt(np.einsum("eij,egjk,elk->egil", R_rel,
                                                     _voigt_strain_to_tensor(self.eps_prev), R_rel))
        d_eps = eps - eps_prev                                                                  # (m, 8gp, 6)
        sig_trial = sig_prev + np.einsum("egij,egj->egi", self.D_e[:, None], d_eps)
        eps_p_t_prev = _tensor_strain_to_voigt(np.einsum("eij,egjk,elk->egil", R_rel,
                                                         _voigt_strain_to_tensor(self.eps_p_t), R_rel))
        sig, eps_p, eps_p_t, d_wp = self._return_map(sig_trial, self.eps_p, eps_p_t_prev)
        # internal forces: f_a = R sum_gp B^T sigma dV, at the 8 nodes of the element
        B5 = self.B_e.reshape(self.m, 8, 6, 8, 3)                                               # (m,gp,comp,node,dof)
        f_gp = np.einsum("egc,egcad,eg->ead", sig, B5, self.gp_w, optimize=True)                # (m, 8, 3)
        f_el = np.einsum("eij,eaj->eai", R, f_gp)                                               # global frame
        f_el[self.eroded] = 0.0
        f_node = np.zeros((self.n, 3))
        np.add.at(f_node, self.elements.ravel(), f_el.reshape(-1, 3))
        # contact
        f_contact, contact_energy = self._contact()
        self._force = f_contact.sum(0)
        self._force_hist.append(self._force.copy())
        self._time_hist.append(self.t)
        self._vx_hist.append(float(self.v[~self.fixed, 0].mean()) if (~self.fixed).any() else 0.0)
        # central difference update: M a = f_ext - f_int
        a = (f_contact - f_node) * self.inv_mass[:, None]
        self.v = self.v + self.dt * a
        du = self.dt * self.v
        self.u = self.u + du
        self.t += self.dt
        # energies and bookkeeping
        self._plastic_work += d_wp
        self._contact_energy = contact_energy        # stored in the penalty springs now, not spent
        self._external_work += float(np.sum(f_contact * du))
        self.R_e = R
        self.sig = sig
        self.eps_p = eps_p
        self.eps_p_t = eps_p_t
        self.eps_prev = eps
        self._erode(F_c)
        self._check_envelope(F_c)
        kinetic = 0.5 * float(np.sum(self.node_mass[:, None] * self.v ** 2))
        return {"t": self.t, "kinetic": kinetic, "force": f_contact.sum(0)}

    def energies(self) -> dict:
        """kinetic [J], elastic [J], plastic work [J], internal (elastic + plastic) [J], contact [J].

        The elastic part is the strain energy of the *current* configuration, ``0.5 sigma : (eps - eps_p) V`` over the
        integration points, so ``kinetic + internal`` is conserved by the scheme (and is the check to run on a run)."""
        eps, _ = self._strain_at_gauss_points()
        kinetic = 0.5 * float(np.sum(self.node_mass[:, None] * self.v ** 2))
        elastic = float(np.sum(0.5 * np.einsum("egc,egc,eg->", self.sig, eps - self.eps_p_t, self.gp_w)))
        elastic = max(elastic, 0.0)
        return {"kinetic": kinetic, "elastic": elastic, "plastic": self._plastic_work,
                "internal": elastic + self._plastic_work, "contact": self._contact_energy,
                "external": self._external_work}

    def _return_map(self, sig_trial: np.ndarray, eps_p: np.ndarray, eps_p_t: np.ndarray):
        """Radial return to the von Mises surface with linear isotropic hardening, at every integration point.

        Returns the new stress, the new equivalent plastic strain, its tensor (Voigt, engineering shear) and the
        plastic work of the increment."""
        dev = sig_trial.copy()
        mean = sig_trial[..., :3].mean(-1, keepdims=True)
        dev[..., :3] -= mean
        q = np.sqrt(1.5 * (dev[..., :3] ** 2).sum(-1) + 3.0 * (dev[..., 3:] ** 2).sum(-1))       # von Mises
        sy = self.sy_e[:, None] + self.H_e[:, None] * eps_p
        active = q > sy
        d_gamma = np.where(active, (q - sy) / (3.0 * self.mu_e[:, None] + self.H_e[:, None]), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            n_dev = np.where(active[..., None], 1.5 * dev / np.maximum(q, 1e-30)[..., None], 0.0)
        sig = sig_trial - np.where(active[..., None], 2.0 * self.mu_e[:, None, None] * d_gamma[..., None] * n_dev, 0.0)
        d_eps_p = d_gamma[..., None] * n_dev
        d_eps_p[..., 3:] *= 2.0                                 # engineering shear of the plastic strain increment
        eps_p_new = eps_p + d_gamma
        sy_new = self.sy_e[:, None] + self.H_e[:, None] * eps_p_new
        d_wp = float(np.sum(0.5 * (sy + sy_new) * d_gamma * self.gp_w))
        return sig, eps_p_new, eps_p_t + d_eps_p, d_wp

    def _contact(self) -> tuple[np.ndarray, float]:
        """Penalty contact with the rigid obstacle: (force on the nodes (n,3), elastic contact energy)."""
        if self.contact is None:
            return np.zeros((self.n, 3)), 0.0
        x_new = self.mesh.nodes + self.u
        dist = _signed_distance(self.contact, x_new, self.t)
        pen = np.maximum(-dist, 0.0)
        pen[self.fixed] = 0.0
        if not pen.any():
            return np.zeros((self.n, 3)), 0.0
        n = _outward_normal(self.contact, x_new)
        force = (self.penalty_k * pen)[:, None] * n
        if self.friction:
            vt = self.v - (self.v * n).sum(1)[:, None] * n
            mag = np.linalg.norm(vt, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                fric = -np.where(mag[:, None] > 1e-12,
                                 (self.friction * np.abs(self.penalty_k * pen))[:, None] * vt
                                 / np.maximum(mag, 1e-30)[:, None], 0.0)
            force = force + fric
        return force, 0.5 * float(np.sum(self.penalty_k * pen ** 2))

    def _erode(self, F_c: np.ndarray) -> None:
        """Remove the elements that reached their erosion strain or inverted (their mass leaves their nodes)."""
        detF = np.linalg.det(F_c)
        kill = ((self.ee_e > 0) & (self.eps_p.max(1) >= self.ee_e)) | (detF < 0.25)
        kill &= ~self.eroded
        if not kill.any():
            return
        self.eroded |= kill
        self.sig[kill] = 0.0
        lost = np.repeat(self.mass_e[kill][:, None] / 8.0, 8, 1)
        np.subtract.at(self.node_mass, self.elements[kill].ravel(), lost.ravel())
        self.node_mass = np.maximum(self.node_mass, 0.0)
        self._update_inv_mass()

    def _check_envelope(self, F_c: np.ndarray) -> None:
        if self.out_of_envelope:
            return
        keep = ~self.eroded
        if not keep.any():
            return
        stretch = np.abs(np.cbrt(np.abs(np.linalg.det(F_c[keep]))) - 1.0)
        if float(stretch.max()) > ENVELOPE:
            self.out_of_envelope = True
            self.envelope_reason = "stretch"

    def _check_energy_balance(self, frame: dict) -> None:
        """Flag a run whose kinetic + internal + contact energy stops matching the work the obstacle put in.

        The scheme conserves that balance (lumped mass, central difference and eight integration points are all
        energy-consistent), so a run that loses it has left the regime this model was verified in — whatever the
        element stretches say, and however plausible the force pulse looks.

        What the balance looks like, measured: in the campaign's window (pole impact, 8e-4 s, eleven frames) it drifts
        from 0.02 % to 1.0 % of the work as the crush develops; the thin-wall design that starts eroding jumps from
        1.0 % to 25 % between two frames and to 367 % by the end. A single frame can lag by one step's work where the
        pulse begins, so :data:`ENERGY_TOL` has to hold for two frames in a row, while :data:`ENERGY_TRIP` trips at
        once. A non-finite state trips it as well."""
        if self.out_of_envelope:
            return
        ext = float(frame["external"])
        got = float(frame["kinetic"]) + float(frame["internal"]) + float(frame["contact"])
        if not math.isfinite(got):
            self.out_of_envelope = True
            self.envelope_reason = "energy_balance"
            return
        self._ext_max = max(getattr(self, "_ext_max", 0.0), abs(ext))
        # before the obstacle has done a measurable part of its work the balance is a ratio of two near-zero
        # numbers; the check starts once 5 % of the run's work is in
        if self._ext_max <= 0.0 or ext < 0.05 * self._ext_max:
            return
        imbalance = abs(got - ext)
        over = imbalance > ENERGY_TOL * self._ext_max
        self._imbalance_frames = (getattr(self, "_imbalance_frames", 0) + 1) if over else 0
        if self._imbalance_frames >= 2 or imbalance > ENERGY_TRIP * self._ext_max:
            self.out_of_envelope = True
            self.envelope_reason = "energy_balance"

    # ------------------------------------------------------------------------------------ run
    def run(self) -> CrashResult:
        """Run to ``end_time`` and return every frame, history and the design quantities."""
        if self.end_time is None:
            raise ValueError("end_time is required to run(); set it in the constructor")
        n_steps = max(2, int(round(self.end_time / self.dt)))
        n_out = max(2, int(self.frames))
        if n_steps + 1 < n_out:
            raise ValueError(f"end_time={self.end_time} covers {n_steps + 1} steps at dt={self.dt}, fewer than the "
                             f"{n_out} frames requested: the frames have to be distinct")
        # Exactly ``frames`` frames, evenly spaced from 0 to the end time. The count must not depend on the design:
        # every run of a campaign has its own dt, hence its own step count, and a dataset whose geometries carry
        # different frame numbers cannot be batched (and reads as one frame per geometry if the axis silently goes).
        out_steps = set(np.unique(np.rint(np.linspace(0, n_steps, n_out))).astype(int).tolist())
        self._ext_max = 0.0
        self._imbalance_frames = 0
        fr: list[dict] = []
        for k in range(n_steps + 1):
            if k in out_steps:
                fr.append(self._frame())
                self._check_energy_balance(fr[-1])
            if k < n_steps:
                self.step()
        return CrashResult(
            mesh=self.mesh, parts=self.parts.copy(),
            times=np.array([f["t"] for f in fr]),
            displacement=np.stack([f["u"] for f in fr]),
            velocity=np.stack([f["v"] for f in fr]),
            von_mises=np.stack([f["vm"] for f in fr]),
            plastic_strain=np.stack([f["eps_p"] for f in fr]),
            eroded=np.stack([f["eroded"] for f in fr]),
            contact_force=np.stack([f["force"] for f in fr]),
            energies={k: np.array([f[k] for f in fr]) for k in ("kinetic", "elastic", "internal", "plastic",
                                                                "contact", "external")},
            out_of_envelope=self.out_of_envelope, envelope_reason=self.envelope_reason, dt=self.dt,
            time_history=np.array(self._time_hist), force_history=np.array([f[0] for f in self._force_hist]),
            vx_history=np.array(self._vx_hist))

    def _frame(self) -> dict:
        vm = von_mises(self.sig).max(1) if self.m else np.zeros(0)
        e = self.energies()
        return {"t": self.t, "u": self.u.copy(),
                "v": self.v.copy() if self.record_velocity else np.zeros_like(self.u),
                "vm": vm, "eps_p": self.eps_p.max(1), "eroded": self.eroded.copy(),
                "kinetic": e["kinetic"], "elastic": e["elastic"], "internal": e["internal"],
                "plastic": e["plastic"], "contact": e["contact"], "external": e["external"],
                "force": getattr(self, "_force", np.zeros(3)).copy()}


def _polar_rotation(F: np.ndarray) -> np.ndarray:
    """Rotation of the polar decomposition of a batch of 3 x 3 matrices, (..., 3, 3)."""
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    bad = np.linalg.det(R) < 0
    if bad.any():
        U = np.where(bad[..., None, None], U * np.array([1.0, 1.0, -1.0]).reshape(3, 1), U)
        R = U @ Vt
    return R

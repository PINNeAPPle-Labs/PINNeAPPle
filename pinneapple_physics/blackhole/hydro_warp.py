"""NVIDIA Warp kernels for the axisymmetric accretion solver (``RIAFConfig(backend="warp")``).

The scheme is the one in :mod:`pinneapple_physics.blackhole.hydro`, which stays the reference: MUSCL-minmod
reconstruction of the primitives, HLL fluxes, the two azimuthal viscous stresses, the exact geometric sources and
SSP-RK2. What changes is how it is executed. The reference evaluates every face as a separate tensor operation, so on
a 128 x 64 grid a step is a few hundred small kernels dominated by launch overhead (about 10 ms on one CPU core).
Here **one fused kernel per stage** does the work of a whole stage of one cell: it reconstructs the four faces of that
cell, computes their HLL fluxes and the viscous corrections, takes the divergence and adds the sources. Two small
kernels complete the stage: one writes the mass and angular-momentum flux through ``r_in`` and ``r_out`` (the
quantities ``AccretionFlow.budget`` closes on) and one turns the conserved state back into primitives and fills the
ghost cells (the reference's ``cons_to_prim_i`` + ``_bc``).

Three things follow from the kernels being fully local:

* the state stays in the same buffers the reference uses -- ``flow.W`` is a zero-copy Warp view of the torch tensor
  (DLPack), so both backends read and write the same numbers;
* a Warp device that is a GPU runs the same kernels unchanged, which is where the speed-up the issue asks for comes
  from (a consumer card computes float64 at about 1/32 of float32, so the port is float64 only: the tests require
  agreement with the reference to round-off, and a float32 kernel cannot give it);
* ``wp.Tape`` can differentiate a step with respect to ``alpha`` (the viscosity enters through a scalar the tape
  knows), which is the route to calibration and inverse problems (issue #410). The gradient check in
  ``tests/test_hydro_warp.py`` compares it against a finite difference of the same observable.

Absence of the package is not an error for the library: this module imports without ``warp-lang`` installed,
:func:`warp_status` says what is missing and how to install it, and only a request for the backend raises.

    pip install warp-lang                 # or: pip install "pinneapple[warp]"
    from pinneapple_physics.blackhole import RIAFConfig, AccretionFlow, torus_state
    flow = AccretionFlow(RIAFConfig(nr=128, ntheta=64, backend="warp"))
    flow.set_primitives(torus_state(flow))
    flow.run(t_end=100.0, every=10.0)

``pinneapple_tools.compute_backends`` is the general Warp entry point of the library (zero-copy tensors, ``warp.fem``
weak forms and a differentiable rigid-body step); this module is the same idea applied to a finite-volume solver.
"""
import math
import time
from typing import Any

import numpy as np

__all__ = [
    "INSTALL_HINT",
    "WARP_PORTS",
    "WarpStepper",
    "bench",
    "format_table",
    "require_warp",
    "solver_status",
    "tape_accreted_mass",
    "warp_available",
    "warp_device",
    "warp_status",
    "warp_stepper",
]

NG = 2  # ghost layers, as in hydro.AccretionFlow
INSTALL_HINT = 'pip install warp-lang          # or: pip install "pinneapple[warp]"'

_WARP: Any = None
_WARP_ERROR: str | None = None


# ---------------------------------------------------------------------------------------------------------------
# Lazy import: the library imports without warp-lang installed
# ---------------------------------------------------------------------------------------------------------------
def _wp():
    """Import Warp once, quiet, initialised. Raises :class:`ImportError` with the install line when absent."""
    global _WARP, _WARP_ERROR
    if _WARP is not None:
        return _WARP
    try:
        import warp as wp
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        _WARP_ERROR = f"{type(exc).__name__}: {exc}"
        raise ImportError(
            "the Warp backend of the accretion solver needs NVIDIA Warp: " + INSTALL_HINT
        ) from exc
    wp.config.quiet = True
    wp.config.log_level = getattr(wp, "LOG_WARNING", wp.config.log_level)
    probe = getattr(wp, "is_initialized", None)
    try:
        if not callable(probe) or not probe():
            wp.init()
    except Exception:            # pragma: no cover - a Warp build without a usable device still imports
        wp.init()
    _WARP = wp
    return wp


def warp_available() -> bool:
    """``True`` when ``warp`` imports here."""
    try:
        _wp()
        return True
    except Exception:
        return False


def warp_device() -> str | None:
    """The device Warp kernels run on: ``"cuda:0"`` on an NVIDIA GPU, otherwise ``"cpu"``; ``None`` without Warp."""
    if not warp_available():
        return None
    wp = _wp()
    device = wp.get_device()
    return f"cuda:{device.ordinal}" if device.is_cuda else "cpu"


def warp_status() -> dict[str, Any]:
    """Whether the Warp backend can run here, what is missing, and the install line for it.

    The same shape the other optional backends of the library report: a dict you can print or log, never an
    exception. ``available=False`` always comes with a non-empty ``missing`` and the ``install`` line to fix it.
    """
    status: dict[str, Any] = {
        "backend": "warp",
        "available": False,
        "missing": "warp-lang",
        "install": INSTALL_HINT,
        "device": None,
        "version": None,
        "reason": "",
    }
    try:
        wp = _wp()
    except Exception as exc:
        status["reason"] = f"{exc} ({_WARP_ERROR or 'import failed'})"
        return status
    device = wp.get_device()
    try:
        cuda_build = len(wp.get_cuda_devices()) > 0
    except Exception:            # pragma: no cover - a build without the CUDA runtime
        cuda_build = False
    status.update(
        available=True,
        missing=None,
        device=f"cuda:{device.ordinal}" if device.is_cuda else "cpu",
        version=getattr(wp, "__version__", "unknown"),
        reason=(
            "kernels run on the GPU"
            if device.is_cuda
            else "Warp runs on the CPU here (what the equivalence tests use; a CUDA build of warp-lang selects the "
                 "GPU automatically on an NVIDIA machine)"
        ),
    )
    status["cuda_build"] = bool(cuda_build)
    return status


def require_warp(what: str = "the Warp backend") -> None:
    """Raise :class:`ImportError` with the install line when Warp is not importable."""
    try:
        _wp()
    except ImportError as exc:
        raise ImportError(f"{what} needs NVIDIA Warp: {INSTALL_HINT}") from exc


# ---------------------------------------------------------------------------------------------------------------
# The seam: which solvers have a Warp path, and what a new one has to implement
# ---------------------------------------------------------------------------------------------------------------
#: Issue #410 keeps the PyTorch solver as the reference and adds the kernels behind it one solver at a time. This
#: table is the list the tests hold to, so "ported" cannot silently mean "a bit ported": a solver is either
#: executed by Warp kernels with an equivalence test against its PyTorch reference, or it is not, with a reason.
WARP_PORTS: dict[str, dict[str, Any]] = {
    "blackhole.hydro": {
        "ported": True,
        "reference": "pinneapple_physics.blackhole.hydro.AccretionFlow",
        "equivalence_test": "tests/test_hydro_warp.py::test_warp_matches_the_torch_reference",
        "note": "one fused kernel per SSP-RK2 stage (reconstruction + HLL + viscous stresses + divergence + sources)",
    },
    "compressible_fv": {
        "ported": False,
        "reference": "pinneapple_simulation.numerical_solvers.compressible_fv.CompressibleFV",
        "reason": (
            "not ported yet. The seam is this module: a stepper class with dt()/step(dt) that keeps the reference's "
            "buffers, one kernel with the same arithmetic as CompressibleFV.rhs for the HLLC flux and the viscous "
            "terms, and an entry here whose equivalence test compares the two to round-off, in the shape of "
            "WarpStepper. The 2-D/3-D switch, the six boundary-condition kinds and the CFL + viscous time step are "
            "the parts that need their own tests before it can be called ported."
        ),
    },
    "shallow_water_fv": {
        "ported": False,
        "reference": "pinneapple_simulation.numerical_solvers.shallow_water_fv.ShallowWater2D",
        "reason": (
            "not ported yet, for the same seam as compressible_fv. The extra work here is the solver's wall and gate "
            "masking (mirrored ghost states at solid faces) and its wet/dry wave speeds, which are value-dependent "
            "branches: they are differentiable piecewise, and the dry branch needs a test of its own before the port "
            "can be trusted."
        ),
    },
}


def solver_status(solver: str) -> dict[str, Any]:
    """The entry of :data:`WARP_PORTS` for ``solver``, raising with the list when the name is unknown."""
    if solver not in WARP_PORTS:
        raise KeyError(f"unknown solver {solver!r}; known: {', '.join(sorted(WARP_PORTS))}")
    return WARP_PORTS[solver]


def not_ported(solver: str) -> None:
    """Raise :class:`NotImplementedError` naming the seam, for a solver whose Warp path is not written yet."""
    entry = solver_status(solver)
    raise NotImplementedError(f"the Warp path of {solver!r} is not written yet: {entry['reason']}")


# ---------------------------------------------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------------------------------------------
_KERNELS: dict[str, Any] | None = None


def _build_kernels() -> dict[str, Any]:
    """Define the kernels once, on first use (Warp compiles Python source, so they cannot be built at import).

    Everything inside is float64 and explicit about it: ``wp.float64(...)`` around every literal, because a bare
    Python float is a float32 constant in Warp and mixing the two is an error (not a silent conversion). Every
    expression is written in the order ``AccretionFlow.rhs`` writes it -- same operands, same order -- so the two
    backends differ only by the last bits of the sum, which is what "matches to round-off" means in the tests.
    """
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    wp = _wp()

    @wp.func
    def _minmod(a: wp.float64, b: wp.float64) -> wp.float64:
        """hydro._minmod: sign(a) min(|a|, |b|) where a b > 0, zero otherwise (the same value, without the sign)."""
        if a * b <= wp.float64(0.0):
            return wp.float64(0.0)
        if wp.abs(a) < wp.abs(b):
            return a
        return b

    @wp.func
    def _recon_l(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64:
        """Left state at the face to the right of cell b: b + 0.5 minmod(b - a, c - b)."""
        return b + wp.float64(0.5) * _minmod(b - a, c - b)

    @wp.func
    def _recon_r(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64:
        """Right state at the face to the left of cell b: b - 0.5 minmod(b - a, c - b)."""
        return b - wp.float64(0.5) * _minmod(b - a, c - b)

    @wp.func
    def _nu(rho: wp.float64, p: wp.float64, r: wp.float64, gamma: wp.float64, alpha: wp.float64,
            visc: wp.int32) -> wp.float64:
        """Kinematic viscosity of the alpha prescription: 1 = SS (alpha cs^2 / Omega_K), 2 = ST (alpha r^1/2)."""
        if visc == 1:
            return alpha * gamma * p / rho * (r - wp.float64(2.0)) * wp.sqrt(r)
        if visc == 2:
            return alpha * wp.sqrt(r)
        return wp.float64(0.0)

    @wp.func
    def _rn(W: wp.array3d(dtype=wp.float64), p: wp.int32, q: wp.int32, r: wp.array1d(dtype=wp.float64),
            gamma: wp.float64, alpha: wp.float64, visc: wp.int32) -> wp.float64:
        """rho * nu in a cell: what the viscous stress is built from."""
        return W[0, p, q] * _nu(W[0, p, q], W[4, p, q], r[p], gamma, alpha, visc)

    @wp.func
    def _hll(rhol: wp.float64, vrl: wp.float64, vtl: wp.float64, vpl: wp.float64, pl: wp.float64,
             rhor: wp.float64, vrr: wp.float64, vtr: wp.float64, vpr: wp.float64, pr: wp.float64,
             axis: wp.int32, Rf: wp.float64, gamma: wp.float64
             ) -> tuple[wp.float64, wp.float64, wp.float64, wp.float64, wp.float64]:
        """HLL flux of (rho, rho vr, rho vth, rho l, E) through a face normal to r (axis 0) or theta (axis 1)."""
        vnl = vrl
        vnr = vrr
        if axis == 1:
            vnl = vtl
            vnr = vtr
        ll = vpl * Rf
        lr = vpr * Rf
        El = pl / (gamma - wp.float64(1.0)) + wp.float64(0.5) * rhol * (vrl * vrl + vtl * vtl + vpl * vpl)
        Er = pr / (gamma - wp.float64(1.0)) + wp.float64(0.5) * rhor * (vrr * vrr + vtr * vtr + vpr * vpr)
        cl = wp.sqrt(gamma * pl / rhol)
        cr = wp.sqrt(gamma * pr / rhor)
        SL = wp.min(vnl - cl, vnr - cr)
        SR = wp.max(vnl + cl, vnr + cr)
        SL = wp.min(SL, wp.float64(0.0))
        SR = wp.max(SR, wp.float64(0.0))
        # left and right conserved states and fluxes
        UL0, UL1, UL2, UL3, UL4 = rhol, rhol * vrl, rhol * vtl, rhol * ll, El
        UR0, UR1, UR2, UR3, UR4 = rhor, rhor * vrr, rhor * vtr, rhor * lr, Er
        FL0 = rhol * vnl
        FL1 = rhol * vrl * vnl
        FL2 = rhol * vtl * vnl
        FR0 = rhor * vnr
        FR1 = rhor * vrr * vnr
        FR2 = rhor * vtr * vnr
        if axis == 0:
            FL1 = FL1 + pl
            FR1 = FR1 + pr
        else:
            FL2 = FL2 + pl
            FR2 = FR2 + pr
        FL3 = rhol * ll * vnl
        FL4 = (El + pl) * vnl
        FR3 = rhor * lr * vnr
        FR4 = (Er + pr) * vnr
        den = SR - SL + wp.float64(1e-300)
        return ((SR * FL0 - SL * FR0 + SL * SR * (UR0 - UL0)) / den,
                (SR * FL1 - SL * FR1 + SL * SR * (UR1 - UL1)) / den,
                (SR * FL2 - SL * FR2 + SL * SR * (UR2 - UL2)) / den,
                (SR * FL3 - SL * FR3 + SL * SR * (UR3 - UL3)) / den,
                (SR * FL4 - SL * FR4 + SL * SR * (UR4 - UL4)) / den)

    @wp.func
    def _face_r(W: wp.array3d(dtype=wp.float64), P: wp.int32, q: wp.int32,
                r: wp.array1d(dtype=wp.float64), th: wp.array1d(dtype=wp.float64),
                rf: wp.array1d(dtype=wp.float64), gamma: wp.float64, alpha: wp.float64, visc: wp.int32
                ) -> tuple[wp.float64, wp.float64, wp.float64, wp.float64, wp.float64]:
        """Flux through the r face between cells P and P + 1, at cell column q (reconstruction, HLL, T_rphi)."""
        wl0 = _recon_l(W[0, P - 1, q], W[0, P, q], W[0, P + 1, q])
        wl1 = _recon_l(W[1, P - 1, q], W[1, P, q], W[1, P + 1, q])
        wl2 = _recon_l(W[2, P - 1, q], W[2, P, q], W[2, P + 1, q])
        wl3 = _recon_l(W[3, P - 1, q], W[3, P, q], W[3, P + 1, q])
        wl4 = _recon_l(W[4, P - 1, q], W[4, P, q], W[4, P + 1, q])
        wr0 = _recon_r(W[0, P, q], W[0, P + 1, q], W[0, P + 2, q])
        wr1 = _recon_r(W[1, P, q], W[1, P + 1, q], W[1, P + 2, q])
        wr2 = _recon_r(W[2, P, q], W[2, P + 1, q], W[2, P + 2, q])
        wr3 = _recon_r(W[3, P, q], W[3, P + 1, q], W[3, P + 2, q])
        wr4 = _recon_r(W[4, P, q], W[4, P + 1, q], W[4, P + 2, q])
        s = wp.sin(th[q])
        Rf = rf[P + 1] * s
        f0, f1, f2, f3, f4 = _hll(wl0, wl1, wl2, wl3, wl4, wr0, wr1, wr2, wr3, wr4, 0, Rf, gamma)
        if visc != 0 and alpha > wp.float64(0.0):
            # the two azimuthal stresses are the only ones kept: T_rphi enters angular momentum and energy
            Rm = wp.abs(r[P] * s)
            Rp = wp.abs(r[P + 1] * s)
            Om_m = W[3, P, q] / wp.max(Rm, wp.float64(1e-12))
            Om_p = W[3, P + 1, q] / wp.max(Rp, wp.float64(1e-12))
            rn = wp.float64(0.5) * (_rn(W, P, q, r, gamma, alpha, visc) + _rn(W, P + 1, q, r, gamma, alpha, visc))
            T = rn * Rf * (Om_p - Om_m) / (r[P + 1] - r[P])
            f3 = f3 - Rf * T
            f4 = f4 - wp.float64(0.5) * (W[3, P, q] + W[3, P + 1, q]) * T
        return f0, f1, f2, f3, f4

    @wp.func
    def _face_t(W: wp.array3d(dtype=wp.float64), p: wp.int32, Q: wp.int32,
                r: wp.array1d(dtype=wp.float64), th: wp.array1d(dtype=wp.float64),
                thf: wp.array1d(dtype=wp.float64), gamma: wp.float64, alpha: wp.float64, visc: wp.int32
                ) -> tuple[wp.float64, wp.float64, wp.float64, wp.float64, wp.float64]:
        """Flux through the theta face between cells Q and Q + 1, at cell row p (reconstruction, HLL, T_thetaphi)."""
        wl0 = _recon_l(W[0, p, Q - 1], W[0, p, Q], W[0, p, Q + 1])
        wl1 = _recon_l(W[1, p, Q - 1], W[1, p, Q], W[1, p, Q + 1])
        wl2 = _recon_l(W[2, p, Q - 1], W[2, p, Q], W[2, p, Q + 1])
        wl3 = _recon_l(W[3, p, Q - 1], W[3, p, Q], W[3, p, Q + 1])
        wl4 = _recon_l(W[4, p, Q - 1], W[4, p, Q], W[4, p, Q + 1])
        wr0 = _recon_r(W[0, p, Q], W[0, p, Q + 1], W[0, p, Q + 2])
        wr1 = _recon_r(W[1, p, Q], W[1, p, Q + 1], W[1, p, Q + 2])
        wr2 = _recon_r(W[2, p, Q], W[2, p, Q + 1], W[2, p, Q + 2])
        wr3 = _recon_r(W[3, p, Q], W[3, p, Q + 1], W[3, p, Q + 2])
        wr4 = _recon_r(W[4, p, Q], W[4, p, Q + 1], W[4, p, Q + 2])
        sf = wp.sin(thf[Q + 1])
        Rf = r[p] * sf
        f0, f1, f2, f3, f4 = _hll(wl0, wl1, wl2, wl3, wl4, wr0, wr1, wr2, wr3, wr4, 1, Rf, gamma)
        if visc != 0 and alpha > wp.float64(0.0):
            Rm = wp.abs(r[p] * wp.sin(th[Q]))
            Rp = wp.abs(r[p] * wp.sin(th[Q + 1]))
            Om_m = W[3, p, Q] / wp.max(Rm, wp.float64(1e-12))
            Om_p = W[3, p, Q + 1] / wp.max(Rp, wp.float64(1e-12))
            rn = wp.float64(0.5) * (_rn(W, p, Q, r, gamma, alpha, visc) + _rn(W, p, Q + 1, r, gamma, alpha, visc))
            T = rn * sf * (Om_p - Om_m) / (th[Q + 1] - th[Q])
            f3 = f3 - Rf * T
            f4 = f4 - wp.float64(0.5) * (W[3, p, Q] + W[3, p, Q + 1]) * T
        return f0, f1, f2, f3, f4

    @wp.func
    def _rhs_cell(W: wp.array3d(dtype=wp.float64), p: wp.int32, q: wp.int32,
                  r: wp.array1d(dtype=wp.float64), th: wp.array1d(dtype=wp.float64),
                  rf: wp.array1d(dtype=wp.float64), thf: wp.array1d(dtype=wp.float64),
                  area_r: wp.array2d(dtype=wp.float64), area_t: wp.array2d(dtype=wp.float64),
                  vol: wp.array2d(dtype=wp.float64), inv_r: wp.array1d(dtype=wp.float64),
                  geo_t: wp.array1d(dtype=wp.float64), gamma: wp.float64, alpha: wp.float64, visc: wp.int32
                  ) -> tuple[wp.float64, wp.float64, wp.float64, wp.float64, wp.float64]:
        """dU/dt of one interior cell (p, q): the four faces, their divergence, and the geometric sources."""
        fl0, fl1, fl2, fl3, fl4 = _face_r(W, p - 1, q, r, th, rf, gamma, alpha, visc)
        fr0, fr1, fr2, fr3, fr4 = _face_r(W, p, q, r, th, rf, gamma, alpha, visc)
        gb0, gb1, gb2, gb3, gb4 = _face_t(W, p, q - 1, r, th, thf, gamma, alpha, visc)
        gt0, gt1, gt2, gt3, gt4 = _face_t(W, p, q, r, th, thf, gamma, alpha, visc)
        V = vol[p, q]
        Arl, Arr = area_r[p, q], area_r[p + 1, q]
        Atl, Ath = area_t[p, q], area_t[p, q + 1]
        d0 = -((Arr * fr0 - Arl * fl0) + (Ath * gt0 - Atl * gb0)) / V
        d1 = -((Arr * fr1 - Arl * fl1) + (Ath * gt1 - Atl * gb1)) / V
        d2 = -((Arr * fr2 - Arl * fl2) + (Ath * gt2 - Atl * gb2)) / V
        d3 = -((Arr * fr3 - Arl * fl3) + (Ath * gt3 - Atl * gb3)) / V
        d4 = -((Arr * fr4 - Arl * fl4) + (Ath * gt4 - Atl * gb4)) / V
        # sources: the spherical geometric terms and the Paczynski-Wiita gravity, exact in the same sense
        rho = W[0, p, q]
        vr = W[1, p, q]
        vth = W[2, p, q]
        vph = W[3, p, q]
        pr = W[4, p, q]
        ir = inv_r[p]
        cot = wp.cos(th[q]) / wp.sin(th[q])
        dr = r[p] - wp.float64(2.0)
        gphi = wp.float64(1.0) / (dr * dr)
        d1 = d1 + ir * (rho * (vth * vth + vph * vph) + wp.float64(2.0) * pr) - rho * gphi
        d2 = d2 + ir * (pr * geo_t[q] + rho * vph * vph * cot - rho * vr * vth)
        d4 = d4 + (-rho * vr * gphi)
        return d0, d1, d2, d3, d4

    @wp.kernel
    def stage1(W: wp.array3d(dtype=wp.float64), r: wp.array1d(dtype=wp.float64),
               th: wp.array1d(dtype=wp.float64), rf: wp.array1d(dtype=wp.float64),
               thf: wp.array1d(dtype=wp.float64), area_r: wp.array2d(dtype=wp.float64),
               area_t: wp.array2d(dtype=wp.float64), vol: wp.array2d(dtype=wp.float64),
               inv_r: wp.array1d(dtype=wp.float64), geo_t: wp.array1d(dtype=wp.float64),
               gamma: wp.float64, alpha: wp.array1d(dtype=wp.float64), visc: wp.int32, dt: wp.float64,
               U0: wp.array3d(dtype=wp.float64), Un: wp.array3d(dtype=wp.float64)):
        """First SSP-RK2 stage: U0 = prim_to_cons(W) and Un = U0 + dt * rhs(W), one cell per thread."""
        i, j = wp.tid()
        p = i + 2
        q = j + 2
        R = r[p] * wp.sin(th[q])
        rho = W[0, p, q]
        vr = W[1, p, q]
        vth = W[2, p, q]
        vph = W[3, p, q]
        pr = W[4, p, q]
        U0[0, i, j] = rho
        U0[1, i, j] = rho * vr
        U0[2, i, j] = rho * vth
        U0[3, i, j] = rho * vph * R
        U0[4, i, j] = pr / (gamma - wp.float64(1.0)) + wp.float64(0.5) * rho * (vr * vr + vth * vth + vph * vph)
        d0, d1, d2, d3, d4 = _rhs_cell(W, p, q, r, th, rf, thf, area_r, area_t, vol, inv_r, geo_t, gamma,
                                       alpha[0], visc)
        Un[0, i, j] = U0[0, i, j] + dt * d0
        Un[1, i, j] = U0[1, i, j] + dt * d1
        Un[2, i, j] = U0[2, i, j] + dt * d2
        Un[3, i, j] = U0[3, i, j] + dt * d3
        Un[4, i, j] = U0[4, i, j] + dt * d4

    @wp.kernel
    def stage2(W: wp.array3d(dtype=wp.float64), r: wp.array1d(dtype=wp.float64),
               th: wp.array1d(dtype=wp.float64), rf: wp.array1d(dtype=wp.float64),
               thf: wp.array1d(dtype=wp.float64), area_r: wp.array2d(dtype=wp.float64),
               area_t: wp.array2d(dtype=wp.float64), vol: wp.array2d(dtype=wp.float64),
               inv_r: wp.array1d(dtype=wp.float64), geo_t: wp.array1d(dtype=wp.float64),
               gamma: wp.float64, alpha: wp.array1d(dtype=wp.float64), visc: wp.int32, dt: wp.float64,
               U0: wp.array3d(dtype=wp.float64), U1: wp.array3d(dtype=wp.float64),
               Un: wp.array3d(dtype=wp.float64)):
        """Second SSP-RK2 stage: Un = 0.5 (U0 + U1 + dt * rhs(W)) -- out of place, so U1 survives for the tape."""
        i, j = wp.tid()
        p = i + 2
        q = j + 2
        d0, d1, d2, d3, d4 = _rhs_cell(W, p, q, r, th, rf, thf, area_r, area_t, vol, inv_r, geo_t, gamma,
                                       alpha[0], visc)
        Un[0, i, j] = wp.float64(0.5) * (U0[0, i, j] + U1[0, i, j] + dt * d0)
        Un[1, i, j] = wp.float64(0.5) * (U0[1, i, j] + U1[1, i, j] + dt * d1)
        Un[2, i, j] = wp.float64(0.5) * (U0[2, i, j] + U1[2, i, j] + dt * d2)
        Un[3, i, j] = wp.float64(0.5) * (U0[3, i, j] + U1[3, i, j] + dt * d3)
        Un[4, i, j] = wp.float64(0.5) * (U0[4, i, j] + U1[4, i, j] + dt * d4)

    @wp.kernel
    def budget(W: wp.array3d(dtype=wp.float64), r: wp.array1d(dtype=wp.float64),
               th: wp.array1d(dtype=wp.float64), rf: wp.array1d(dtype=wp.float64),
               area_r: wp.array2d(dtype=wp.float64), gamma: wp.float64,
               alpha: wp.array1d(dtype=wp.float64), visc: wp.int32, nr: wp.int32, nrows: wp.int32,
               fl: wp.array2d(dtype=wp.float64)):
        """Mass and angular-momentum flux through r_in (row 0 of fl) and r_out (row 2), per theta face."""
        j = wp.tid()
        q = j + 2
        fl[0, j] = wp.float64(0.0)
        fl[1, j] = wp.float64(0.0)
        fl[2, j] = wp.float64(0.0)
        fl[3, j] = wp.float64(0.0)
        fi0, fi1, fi2, fi3, fi4 = _face_r(W, 1, q, r, th, rf, gamma, alpha[0], visc)
        fo0, fo1, fo2, fo3, fo4 = _face_r(W, nrows - 3, q, r, th, rf, gamma, alpha[0], visc)
        Ain = area_r[2, q]
        Aout = area_r[2 + nr, q]
        fl[0, j] = -Ain * fi0                       # mass_in  (into the domain)
        fl[1, j] = Aout * fo0                       # mass_out (out of it)
        fl[2, j] = -Ain * fi3                       # angmom_in
        fl[3, j] = Aout * fo3                       # angmom_out

    @wp.func
    def _cons_to_prim(U: wp.array3d(dtype=wp.float64), ii: wp.int32, jj: wp.int32, r: wp.array1d(dtype=wp.float64),
                      th: wp.array1d(dtype=wp.float64), p: wp.int32, q: wp.int32, gamma: wp.float64,
                      rho_floor: wp.float64, p_floor: wp.float64, v_max: wp.float64, t_cap: wp.float64
                      ) -> tuple[wp.float64, wp.float64, wp.float64, wp.float64, wp.float64, wp.int32]:
        """``AccretionFlow.cons_to_prim_i``: the caps (speed below v_max in units of c, c_s^2 <= t_cap |Phi|)."""
        R = r[p] * wp.sin(th[q])
        rho = wp.max(U[0, ii, jj], rho_floor)
        vr = U[1, ii, jj] / rho
        vth = U[2, ii, jj] / rho
        vph = U[3, ii, jj] / (rho * R)
        vmag = wp.sqrt(vr * vr + vth * vth + vph * vph)
        scale = wp.min(v_max / wp.max(vmag, wp.float64(1e-300)), wp.float64(1.0))
        vr = vr * scale
        vth = vth * scale
        vph = vph * scale
        ek = wp.float64(0.5) * rho * (vr * vr + vth * vth + vph * vph)
        pr = wp.max((gamma - wp.float64(1.0)) * (U[4, ii, jj] - ek), p_floor)
        ncap = 0
        if scale < wp.float64(1.0):
            ncap = 1
        if t_cap > wp.float64(0.0):
            pmax = t_cap * rho / (gamma * (r[p] - wp.float64(2.0)))
            if pr > pmax:
                pr = pmax
                ncap = ncap + 1
        return rho, vr, vth, vph, pr, ncap

    @wp.kernel
    def convert(U: wp.array3d(dtype=wp.float64), r: wp.array1d(dtype=wp.float64),
                th: wp.array1d(dtype=wp.float64), outer: wp.array3d(dtype=wp.float64), has_outer: wp.int32,
                gamma: wp.float64, rho_floor: wp.float64, p_floor: wp.float64, v_max: wp.float64,
                t_cap: wp.float64, N: wp.int32, M: wp.int32, caps: wp.array2d(dtype=wp.int32),
                W: wp.array3d(dtype=wp.float64)):
        """Conserved -> primitives on the interior (with the two safety caps) and the ghost cells, as ``_bc``.

        A ghost cell is written from the *conserved* state of the interior cell it mirrors, never by reading ``W``:
        that cell is being rewritten by this same launch, so reading it would be a race (it was, at 3e-2 on v_phi
        before this became a function call). ``has_outer`` (``outer_bc="fixed"``) instead reads the held state,
        which the reference keeps as a plain tensor.
        """
        p, q = wp.tid()
        g = 2
        interior = p >= g and p < N - g and q >= g and q < M - g
        if interior:
            rho, vr, vth, vph, pr, ncap = _cons_to_prim(U, p - g, q - g, r, th, p, q, gamma, rho_floor, p_floor,
                                                        v_max, t_cap)
            caps[p, q] = ncap
            W[0, p, q] = rho
            W[1, p, q] = vr
            W[2, p, q] = vth
            W[3, p, q] = vph
            W[4, p, q] = pr
        else:
            caps[p, q] = 0
            # rows: zero gradient at r_in, held at r_out for outer_bc="fixed"; columns: reflected at the poles,
            # where v_theta flips sign
            sr = p
            if p < g:
                sr = g
            if p >= N - g:
                sr = N - g - 1
            sc = q
            if q < g:
                sc = 2 * g - 1 - q
            if q >= M - g:
                sc = M - g - 1 - (q - (M - g))
            if has_outer != 0 and p >= N - g:
                W[0, p, q] = outer[0, p - (N - g), q]
                W[1, p, q] = outer[1, p - (N - g), q]
                W[2, p, q] = outer[2, p - (N - g), q]
                W[3, p, q] = outer[3, p - (N - g), q]
                W[4, p, q] = outer[4, p - (N - g), q]
            else:
                rho, vr, vth, vph, pr, _ = _cons_to_prim(U, sr - g, sc - g, r, th, sr, sc, gamma, rho_floor,
                                                         p_floor, v_max, t_cap)
                if p < g:                                   # r_in: no inflow
                    vr = wp.min(vr, wp.float64(0.0))
                else:
                    if p >= N - g:                          # r_out without a held state: no inflow either
                        vr = wp.max(vr, wp.float64(0.0))
                if q < g or q >= M - g:                     # poles: the reflected v_theta
                    vth = -vth
                W[0, p, q] = rho
                W[1, p, q] = vr
                W[2, p, q] = vth
                W[3, p, q] = vph
                W[4, p, q] = pr

    @wp.kernel
    def accumulate(fl1: wp.array2d(dtype=wp.float64), fl2: wp.array2d(dtype=wp.float64), nt: wp.int32,
                   scale: wp.float64, prev: wp.array1d(dtype=wp.float64), out: wp.array1d(dtype=wp.float64)):
        """out = prev + scale * (stage-1 + stage-2 mass flux): the observable the tape differentiates."""
        s = wp.float64(0.0)
        for j in range(nt):
            s = s + fl1[0, j] + fl2[0, j]
        out[0] = prev[0] + scale * s

    _KERNELS = {"stage1": stage1, "stage2": stage2, "budget": budget, "convert": convert,
                "accumulate": accumulate}
    return _KERNELS


# ---------------------------------------------------------------------------------------------------------------
# The stepper: the accretion solver, executed by the kernels above
# ---------------------------------------------------------------------------------------------------------------
class WarpStepper:
    """Holds the geometry of an :class:`~pinneapple_physics.blackhole.hydro.AccretionFlow` in Warp arrays.

    ``flow.W`` is a zero-copy view: a step writes the new primitives where the reference keeps them, so
    ``flow.primitives()``, ``flow.budget()`` and ``flow.run()`` work unchanged.

    ``track=True`` is the tape mode, and it changes two things, both of them because of how Warp builds an adjoint:
    the derivative of a kernel is evaluated **at the values its inputs have when the backward pass runs**, so an
    input that a later launch overwrites gives a wrong gradient (measured: 18 % on the angular momentum after one
    step, before this was fixed). In track mode the state is therefore a private Warp array and every stage gets
    fresh buffers, so no buffer is ever written after another kernel has read it -- ``flow.W`` is not touched, which
    is also why :func:`tape_accreted_mass` takes a flow and leaves it where it was.
    """

    def __init__(self, flow, *, track: bool = False):
        import torch

        wp = _wp()
        c = flow.cfg
        if c.dtype != torch.float64:
            raise ValueError("backend='warp' needs float64 (RIAFConfig(dtype=torch.float64), the default)")
        want_cuda = str(torch.device(c.device)) != "cpu"
        device = warp_device()
        if want_cuda and (device is None or not device.startswith("cuda")):
            raise ValueError(
                f"the flow is on {c.device!r} and Warp is on {device!r}: put both on the CPU or both on the GPU "
                "(install a CUDA build of warp-lang for the GPU)"
            )
        self.flow = flow
        self.cfg = c
        self.track = track
        self.wp = wp
        self.device = device
        self.kernels = _build_kernels()
        nr, nt, g = c.nr, c.ntheta, NG
        self.nr, self.nt = nr, nt
        self.N, self.M = nr + 2 * g, nt + 2 * g
        self.shape_u = (5, nr, nt)

        def geom(t):
            if hasattr(t, "detach"):                      # a torch tensor (the geometry the reference keeps)
                t = t.detach().cpu().numpy()
            return wp.array(np.ascontiguousarray(np.asarray(t, dtype=np.float64)), dtype=wp.float64,
                            device=device)

        self.r = geom(flow.r[:, 0])
        self.th = geom(flow.th[0])
        self.rf = geom(np.asarray(flow.rf_np))
        self.thf = geom(np.asarray(flow.thf_np))
        self.area_r = geom(flow.area_r)
        self.area_t = geom(flow.area_t)
        self.vol = geom(flow.vol)
        self.inv_r = geom(flow.inv_r[:, 0])
        self.geo_t = geom(flow.geo_t[0])
        self.visc = {"none": 0, "SS": 1, "ST": 2}[c.viscosity]
        self.W = self._state_array()
        self.alpha = wp.array(np.array([float(c.alpha)], dtype=np.float64), dtype=wp.float64, device=device,
                              requires_grad=track)
        self.U0 = self._zeros(self.shape_u)
        self.U1 = self._zeros(self.shape_u)
        self.U2 = self._zeros(self.shape_u)
        self.fl1 = self._zeros((4, nt))
        self.fl2 = self._zeros((4, nt))
        self.caps = self._zeros((self.N, self.M), dtype=wp.int32)
        self.dummy_outer = wp.zeros((5, g, self.M), dtype=wp.float64, device=device)
        self._outer_t = None
        self._outer = self.dummy_outer
        self._has_outer = 0

    def _zeros(self, shape, *, dtype=None):
        wp = self.wp
        dtype = wp.float64 if dtype is None else dtype
        return wp.zeros(shape, dtype=dtype, device=self.device,
                        requires_grad=self.track and dtype == wp.float64)

    def _state_array(self):
        """The state: a zero-copy view of ``flow.W``, or (tape mode) a private Warp array holding the same values."""
        if not self.track:
            return self.wp.from_torch(self.flow.W)
        a = np.ascontiguousarray(np.asarray(self.flow.W.detach().cpu().numpy(), dtype=np.float64))
        return self.wp.array(a, dtype=self.wp.float64, device=self.device, requires_grad=True)

    # ---------------------------------------------------------------- state
    def sync_outer(self) -> None:
        """Pick up ``flow._outer`` (``outer_bc="fixed"``), which the reference rebuilds on ``set_primitives``."""
        outer = getattr(self.flow, "_outer", None)
        if outer is None:
            self._outer, self._has_outer = self.dummy_outer, 0
            self._outer_t = None
            return
        if outer is not self._outer_t:
            a = np.ascontiguousarray(np.asarray(outer.detach().cpu().numpy(), dtype=np.float64))
            self._outer = self.wp.array(a, dtype=self.wp.float64, device=self.device)
            self._outer_t = outer
        self._has_outer = 1

    def refresh_state(self) -> None:
        """Rebuild the state from ``flow.W`` (a caller that replaced it, rather than writing into it)."""
        self.W = self._state_array()

    def alpha_grad(self) -> float | None:
        """d(observable)/d(alpha) left by the last ``tape.backward`` on this stepper's alpha."""
        if self.alpha.grad is None:
            return None
        return float(self.alpha.grad.numpy()[0])

    # ---------------------------------------------------------------- time stepping
    def dt(self) -> float:
        """The time step of the reference, unchanged (same CFL and viscous reduction, so both pick the same dt)."""
        return float(self.flow.dt())

    def step(self, dt: float) -> tuple[list[float], list[float], int]:
        """One SSP-RK2 step; returns the two boundary flux sets and the number of capped cells (as the reference)."""
        wp, k = self.wp, self.kernels
        c = self.cfg
        self.sync_outer()
        geom = [self.r, self.th, self.rf, self.thf, self.area_r, self.area_t, self.vol, self.inv_r, self.geo_t]
        dim2 = (self.nr, self.nt)
        # tape mode: everything a later launch would overwrite gets its own buffer (see the class docstring)
        self.U0 = self._zeros(self.shape_u) if self.track else self.U0
        self.U1 = self._zeros(self.shape_u) if self.track else self.U1
        self.fl1 = self._zeros((4, self.nt)) if self.track else self.fl1
        self.fl2 = self._zeros((4, self.nt)) if self.track else self.fl2
        U2 = self._zeros(self.shape_u) if self.track else self.U2
        common = [self.W, *geom]
        # stage 1: U0 = prim_to_cons(W), U1 = U0 + dt rhs(W); the boundary fluxes of the same W
        wp.launch(k["stage1"], dim=dim2, inputs=[*common, c.gamma, self.alpha, self.visc, dt],
                  outputs=[self.U0, self.U1])
        wp.launch(k["budget"], dim=(self.nt,),
                  inputs=[self.W, self.r, self.th, self.rf, self.area_r, c.gamma, self.alpha, self.visc,
                          self.nr, self.N],
                  outputs=[self.fl1])
        self._to_primitives(self.U1)
        # stage 2: U2 = 0.5 (U0 + U1 + dt rhs(W)) with W rebuilt from U1, then W rebuilt from U2
        wp.launch(k["stage2"], dim=dim2,
                  inputs=[self.W, *geom, c.gamma, self.alpha, self.visc, dt, self.U0, self.U1], outputs=[U2])
        wp.launch(k["budget"], dim=(self.nt,),
                  inputs=[self.W, self.r, self.th, self.rf, self.area_r, c.gamma, self.alpha, self.visc,
                          self.nr, self.N],
                  outputs=[self.fl2])
        self._to_primitives(U2)
        self.U2 = U2
        capped = int(self.caps.numpy().sum())
        f1 = [float(x) for x in self.fl1.numpy()[:4].sum(axis=1)]
        f2 = [float(x) for x in self.fl2.numpy()[:4].sum(axis=1)]
        return f1, f2, capped

    def _to_primitives(self, U) -> None:
        """The reference's ``cons_to_prim_i`` plus ``_bc``, as one kernel over the whole grid (ghosts included)."""
        c = self.cfg
        wp = self.wp
        t_cap = -1.0 if c.t_cap is None else float(c.t_cap)
        caps = self._zeros((self.N, self.M), dtype=wp.int32) if self.track else self.caps
        out = self._zeros((5, self.N, self.M)) if self.track else self.W
        wp.launch(self.kernels["convert"], dim=(self.N, self.M),
                  inputs=[U, self.r, self.th, self._outer, self._has_outer, c.gamma, c.rho_floor, c.p_floor,
                          c.v_max, t_cap, self.N, self.M],
                  outputs=[caps, out])
        self.caps = caps
        if self.track:
            self.W = out


def warp_stepper(flow, *, track: bool = False) -> WarpStepper:
    """The Warp stepper of ``flow`` -- the seam a new solver plugs into (see :data:`WARP_PORTS`)."""
    require_warp()
    stepper = WarpStepper(flow, track=track)
    stepper.refresh_state()
    return stepper


# ---------------------------------------------------------------------------------------------------------------
# Differentiability through Warp's tape
# ---------------------------------------------------------------------------------------------------------------
def tape_accreted_mass(flow, steps: int, *, dt: float) -> dict[str, float | None]:
    """Mass through ``r_in`` over ``steps`` steps, and its derivative w.r.t. ``cfg.alpha``, from ``wp.Tape``.

    The observable is the same one :func:`pinneapple_physics.blackhole.hydro.accreted_mass` returns for the
    reference: ``dt x (stage-1 + stage-2 flux through r_in) x 2 pi``, summed over the steps. It is accumulated into a
    new one-element array at every step rather than into a pair of them, because Warp's adjoint reads the values the
    arrays hold when the backward pass runs: a buffer reused two steps later would be read with the wrong numbers.

    The flow is a pure input: its state is copied, never written, so the value after the call is the one it had
    before. Integrate with :func:`tape_accreted_mass` the way ``total.backward()`` is used for the reference, i.e.
    compare the returned ``dmass_dalpha`` with a finite difference of the reference's ``accreted_mass``.
    """
    wp = _wp()
    k = _build_kernels()
    stepper = WarpStepper(flow, track=True)
    two_pi = 2.0 * math.pi
    prev = wp.zeros(1, dtype=wp.float64, device=stepper.device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        for _ in range(steps):
            stepper.step(dt)
            out = wp.zeros(1, dtype=wp.float64, device=stepper.device, requires_grad=True)
            wp.launch(k["accumulate"], dim=1,
                      inputs=[stepper.fl1, stepper.fl2, stepper.nt, 0.5 * dt * two_pi, prev], outputs=[out])
            prev = out
    mass = float(prev.numpy()[0])
    tape.backward(prev)
    return {"mass": mass, "dmass_dalpha": stepper.alpha_grad(), "steps": steps, "dt": dt,
            "device": warp_device()}


# ---------------------------------------------------------------------------------------------------------------
# Measurement harness: what a step costs, per step and per simulated time
# ---------------------------------------------------------------------------------------------------------------
def bench(grids: tuple[tuple[int, int], ...] = ((64, 32), (128, 64)), steps: int = 20, warmup: int = 3,
          backends: tuple[str, ...] = ("torch", "warp"), device: str = "cpu",
          alpha: float = 0.1, repeat: int = 3) -> list[dict[str, Any]]:
    """Time one backend over one grid, for the reference and for the kernels, on CPU and (when present) GPU.

    Each row is one (backend, grid, device) measurement: the **median** of ``repeat`` timed runs of ``steps`` steps
    after ``warmup`` steps (a laptop with something else running gives a step time that varies by a factor of three
    between runs, so a single sample is not worth quoting), the time per step, the simulated time covered and the
    cost per unit of simulated time (the number a run to 5 x 10^5 GM/c^3 is planned with), plus the largest relative
    difference of the primitive fields against the reference run on the same grid after the same number of steps --
    the number that says whether the speed-up is the same physics. ``t_end`` is the simulated time after all the
    steps of the row, warm-up and repeats included.

    A backend or device that cannot run here gives a row with ``"skipped"`` and the reason, never an exception:
    a table that silently drops a column reads as if that column had been measured.
    """
    import torch

    from .hydro import AccretionFlow, RIAFConfig, torus_state

    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    for nr, nt in grids:
        reference = None
        for backend in backends:
            row: dict[str, Any] = {"backend": backend, "device": device, "grid": f"{nr} x {nt}", "steps": steps,
                                   "repeat": repeat, "skipped": None}
            if backend == "warp":
                status = warp_status()
                if not status["available"]:
                    row.update(skipped=status["reason"], steps=0)
                    rows.append(row)
                    continue
                if device != "cpu" and not status["device"].startswith("cuda"):
                    row.update(skipped=f"Warp runs on {status['device']} here (no CUDA build of warp-lang)", steps=0)
                    rows.append(row)
                    continue
            if device != "cpu" and not torch.cuda.is_available():
                row.update(skipped="this machine has no CUDA device for PyTorch", steps=0)
                rows.append(row)
                continue
            cfg = RIAFConfig(nr=nr, ntheta=nt, alpha=alpha, backend=backend, device=device)
            flow = AccretionFlow(cfg)
            flow.set_primitives(torus_state(flow))
            if backend == "warp":
                flow._warp()                       # compile before the clock starts
            for _ in range(warmup):
                flow.step()
            per_step = []
            for _ in range(max(1, repeat)):
                t0 = time.perf_counter()
                for _ in range(steps):
                    flow.step()
                per_step.append((time.perf_counter() - t0) / steps)
            step_s = float(np.median(per_step))
            prim = flow.primitives()
            n_steps = warmup + max(1, repeat) * steps
            dt_mean = flow.t / max(n_steps, 1)
            row.update(wall_s=step_s * steps, ms_per_step=1e3 * step_s, t_end=flow.t, mdot=flow.mdot_in,
                       ms_per_sim_time=1e3 * step_s / max(dt_mean, 1e-30))
            if reference is None:
                reference = prim
                row["rel_error"] = 0.0
            else:
                scale = np.abs(reference).max(axis=(1, 2))
                row["rel_error"] = float(np.max(np.abs(prim - reference) / (scale[:, None, None] + 1e-300)))
            rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    """The rows of :func:`bench` as the Markdown table the docs quote.

    "vs reference" is the first backend measured on that grid and device -- ``bench`` puts the PyTorch reference
    first -- so a row's number is the speed-up over the reference and not over itself.
    """
    head = ("| backend | device | grid | steps | ms/step | simulated time | s per unit time | vs reference |"
            " max rel. difference |\n|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    base: dict[tuple[str, str], float] = {}
    for r in rows:
        if r.get("skipped"):
            lines.append(f"| {r['backend']} | {r['device']} | {r['grid']} | - | - | - | - | - | "
                         f"not measured: {r['skipped']} |")
            continue
        key = (r["grid"], r["device"])
        base.setdefault(key, r["ms_per_step"])
        speed = base[key] / r["ms_per_step"] if r["ms_per_step"] else float("nan")
        lines.append(f"| {r['backend']} | {r['device']} | {r['grid']} | {r['steps']} | {r['ms_per_step']:.2f} | "
                     f"{r['t_end']:.1f} | {r['ms_per_sim_time'] / 1e3:.3g} | {speed:.2f}x | "
                     f"{r['rel_error']:.2e} |")
    return head + "\n".join(lines) + "\n"


if __name__ == "__main__":                      # python -m pinneapple_physics.blackhole.hydro_warp
    print(format_table(bench()))

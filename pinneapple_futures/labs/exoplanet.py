"""Exoplanet laboratory: the physics of a hypothetical world.

The question this experiment answers is the one the Futures Lab exists for: *what changes if the world changes?*
Three pieces of ordinary physics are enough to say something exact about it.

**Radiative equilibrium (0-D).** A planet absorbs the starlight it intercepts and re-emits it. Over the whole
sphere, with the flux redistributed everywhere::

    S (1 - a) pi R^2 = 4 pi R^2 sigma T_eq^4     ->     T_eq = [S (1 - a) / (4 sigma)]^(1/4)

and, with no redistribution at all (a tidally locked world without an atmosphere, which re-emits from its
day side only)::

    T_day = [S (1 - a) / (2 sigma)]^(1/4)

Both are exact consequences of the Stefan-Boltzmann law. The run does not trust the algebra: it finds the root of
``sigma T^4 (4 red) = S (1-a)`` by bisection and compares it with the closed form, so a mistake on either side
shows up as a failed comparison.

**Latitudinal transport (1-D).** With a diffusive heat transport the annual-mean energy balance is::

    D d/dx[(1 - x^2) dT/dx] - (A + B T) + S(x) (1 - a) = 0,     x = sin(latitude)

which has a closed-form solution: expanding the absorbed flux in Legendre polynomials, ``S(1-a) = sum s_n P_n``,
the temperature is ``T = sum T_n P_n`` with::

    T_n = (s_n - A delta_n0) / (B + D n (n + 1))

that is, a global mean ``T_0 = (s_0 - A)/B`` fixed by the energy balance alone, and a pole-to-equator contrast that
the transport ``D`` damps. The diffusion form on a sphere and its Legendre solution follow the energy-balance model
of North (1975), *Theory of energy-balance climate models*, Journal of the Atmospheric Sciences 32(11), 2033-2043.
``A``, ``B`` and the shape of ``S(x)`` are linearisation choices of this experiment, so they are carried as design
inputs: every reference here is analytic, and the conclusions do not rest on their value.

**Hydrostatic column and scale height.** With an isothermal atmosphere the momentum balance gives::

    H = R T / (M g),      p(z) = p0 exp(-z / H)

which is where gravity enters -- the one place in this experiment where it does. A heavier world has a thinner
atmosphere for the same temperature. The equilibrium temperature itself does not contain ``g`` at all, and the
experiment says so instead of pretending otherwise.
"""
from __future__ import annotations

import numpy as np

from ..core import SpeculativeExperiment, register
from ..numerics import refinement, tridiag
from ..schema import Claim, Hypothesis, Input, Proposal, Scenario

#: Stefan-Boltzmann constant, W m^-2 K^-4 (CODATA 2022).
SIGMA_SB = 5.670374419e-8
#: molar gas constant, J mol^-1 K^-1 (CODATA 2022, exact by the definition of the kelvin).
R_GAS = 8.314462618
#: mean molar mass of dry air, kg/mol (ISO 2533 standard atmosphere).
M_AIR = 0.0289644
#: standard sea-level pressure and temperature (ISO 2533).
P_ISO, T_ISO = 101325.0, 288.15
#: standard-atmosphere pressure at 1000 m, Pa (ISO 2533:1975, table of the standard atmosphere).
P_ISO_1000 = 89875.0

__all__ = ["ExoplanetLaboratory", "super_earth", "SIGMA_SB", "R_GAS", "scale_height",
           "equilibrium_temperature", "solve_equilibrium"]


def equilibrium_temperature(insolation: float, albedo: float, *, redistribution: float = 1.0) -> float:
    """``[S (1 - a) / (4 red sigma)]^(1/4)``: ``red = 1`` spreads the flux over the whole sphere, ``red = 0.5``
    re-emits from one hemisphere only."""
    if not 0.0 <= albedo < 1.0:
        raise ValueError("the albedo is a fraction of the incident light, in [0, 1)")
    if insolation <= 0:
        raise ValueError("a star delivers a positive flux")
    if not 0.0 < redistribution <= 1.0:
        raise ValueError("the redistributed fraction is in (0, 1]")
    return float((insolation * (1.0 - albedo) / (4.0 * redistribution * SIGMA_SB)) ** 0.25)


def solve_equilibrium(insolation: float, albedo: float, *, redistribution: float = 1.0,
                      lo: float = 1.0, hi: float = 5000.0, iterations: int = 200) -> float:
    """The same temperature, found as the root of ``4 red sigma T^4 = S (1 - a)`` by bisection.

    A second, independent route to the number :func:`equilibrium_temperature` returns; the experiment compares
    the two.
    """
    target = insolation * (1.0 - albedo) / (4.0 * redistribution)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if SIGMA_SB * mid**4 < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def scale_height(temperature: float, gravity: float, molar_mass: float = M_AIR) -> float:
    """``H = R T / (M g)`` in metres."""
    if temperature <= 0 or gravity <= 0 or molar_mass <= 0:
        raise ValueError("temperature, gravity and molar mass must be positive")
    return R_GAS * temperature / (molar_mass * gravity)


def p2(x: np.ndarray) -> np.ndarray:
    """The second Legendre polynomial, the shape of the annual-mean absorbed flux here."""
    return 0.5 * (3.0 * x * x - 1.0)


def faces_of(x: np.ndarray) -> np.ndarray:
    """``1 - x^2`` on the faces between consecutive nodes (the area weight of the sphere in the diffusion term)."""
    mid = 0.5 * (x[:-1] + x[1:])
    return 1.0 - mid**2


def _area_mean(t: np.ndarray) -> float:
    """The area-weighted mean over the sphere: uniform in ``x = sin(latitude)``, with half cells at the poles."""
    w = np.ones(t.size)
    w[0] = w[-1] = 0.5
    return float(np.sum(w * t) / np.sum(w))


def _rk4_column(h_scale: float, p0: float, steps: int, span: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``dp/dz = -p/H`` with classical RK4; returns the nodes and the pressure."""
    z = np.linspace(0.0, span * h_scale, steps + 1)
    dz = z[1] - z[0]
    p = np.empty(steps + 1)
    p[0] = p0

    def f(val: float) -> float:
        return -val / h_scale

    for i in range(steps):
        k1 = f(p[i])
        k2 = f(p[i] + 0.5 * dz * k1)
        k3 = f(p[i] + 0.5 * dz * k2)
        k4 = f(p[i] + dz * k3)
        p[i + 1] = p[i] + dz / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z, p


@register
class ExoplanetLaboratory(SpeculativeExperiment):
    name = "futures.exoplanet_laboratory"
    title = "Exoplanet laboratory: the physics of a hypothetical world"
    area = "planetary_systems"
    topic = "radiative balance, latitudinal transport and the hydrostatic column"
    level = "research"
    kind = "theory"
    dims = ("0D", "1D")
    scenario_params = ("gravity", "ambient_temperature")
    answers = ("gravity", "insolation", "albedo", "atmosphere", "surface_temperature")
    description = ("Radiative equilibrium found twice, a diffusive energy-balance model against its Legendre "
                   "solution, and the scale height of an atmosphere -- exact references throughout, so that a "
                   "change of the world's parameters can be told apart from a mistake in the numerics.")
    references = ("North, G. R. (1975). Theory of energy-balance climate models. J. Atmos. Sci. 32(11), 2033-2043.",
                  "Stefan-Boltzmann law, sigma = 5.670374419e-8 W m^-2 K^-4 (CODATA 2022).",
                  "ISO 2533:1975 standard atmosphere (sea-level pressure and the 1000 m pressure).")
    params = {
        "gravity": 9.80665,           # m/s^2 (also supplied by a scenario)
        "insolation": 1361.0,         # W/m^2, total solar irradiance at 1 au (satellite radiometry)
        "albedo": 0.30,               # planetary albedo
        "molar_mass": M_AIR,          # kg/mol
        "ambient_temperature": T_ISO,  # K, the temperature that sets the scale height
        "transport": 0.6,             # W m^-2 K^-1, the diffusion coefficient D
        "olr_a": 203.3,               # W/m^2, the constant term of the linearised outgoing longwave radiation
        "olr_b": 2.09,                # W m^-2 K^-1, its slope
        "insolation_p2": 0.482,       # amplitude of the P2 term of the annual-mean absorbed flux
        "n_lat": 181,                 # nodes in x = sin(latitude)
    }
    quick = {"n_lat": 61}

    # ------------------------------------------------------------------ scenario
    def world(self) -> Scenario:
        """The Earth-like default world, with every number and its provenance."""
        return Scenario(
            "earth_like", "Earth-like default",
            "The reference world of the laboratory: the insolation and albedo of the Earth, an isothermal "
            "atmosphere at the standard sea-level temperature.",
            gravity=9.80665, ambient_temperature=T_ISO, ambient_pressure=P_ISO,
            inputs=(Input("insolation", 1361.0, "W/m^2", "measured",
                          "total solar irradiance at 1 au (satellite radiometry, TSIS)"),
                    Input("albedo", 0.30, "-", "measured", "planetary albedo of the Earth"),
                    Input("olr_a", 203.3, "W/m^2", "design_input",
                          "constant of the linearised outgoing longwave radiation: a model choice of this "
                          "experiment, not a measurement; every reference here is analytic"),
                    Input("olr_b", 2.09, "W/m^2/K", "design_input",
                          "slope of the linearised outgoing longwave radiation: a model choice of this experiment"),
                    Input("transport", 0.6, "W/m^2/K", "design_input",
                          "diffusion coefficient of the transport term: a model choice, swept in the study"),
                    Input("insolation_p2", 0.482, "-", "design_input",
                          "amplitude of the P2 term of the annual-mean absorbed flux: a shape choice, and the "
                          "global mean does not depend on it (checked)")),
            claims=(Claim("the planet radiates as a black body of emissivity one", "sigma T^4", "established",
                          "Stefan-Boltzmann law"),
                    Claim("the outgoing longwave radiation is linear in the surface temperature", "A + B T",
                          "extrapolated", "a first-order expansion, valid near the calibration temperature only"),
                    Claim("the annual-mean heat transport is diffusive", "D d/dx[(1-x^2) dT/dx]",
                          "extrapolated", "North (1975): a closure assumption, not a conservation law")))

    # ------------------------------------------------------------------ run
    def _run(self, res, dim: str, p: dict) -> None:
        g = float(p["gravity"])
        s0, a = float(p["insolation"]), float(p["albedo"])
        molar, t_surf = float(p["molar_mass"]), float(p["ambient_temperature"])

        # ---------------------------------------------------------------- radiative equilibrium (both dims)
        t_eq = solve_equilibrium(s0, a, redistribution=1.0)
        t_day = solve_equilibrium(s0, a, redistribution=0.5)
        res.value("T_eq_redistributed", t_eq, "K", "root of sigma T^4 = S(1-a)/4, solved by bisection")
        res.value("T_eq_dayside", t_day, "K", "the same balance over the day hemisphere only")
        res.compare("radiative equilibrium temperature", t_eq, equilibrium_temperature(s0, a), tol=1e-10,
                    kind="closed_form", unit="K",
                    source="T = [S(1-a)/(4 sigma)]^(1/4), Stefan-Boltzmann balance over the whole sphere")
        res.compare("day-side equilibrium temperature", t_day, equilibrium_temperature(s0, a, redistribution=0.5),
                    tol=1e-10, kind="closed_form", unit="K",
                    source="T = [S(1-a)/(2 sigma)]^(1/4), one hemisphere re-emitting")
        res.check("a world without redistribution is hotter on its day side", 1.0 < t_day / t_eq < 1.2,
                  value=float(t_day / t_eq), limit=2.0 ** 0.25,
                  detail="the ratio is 2^(1/4): the same flux leaving over half the area", source="bound")

        # ---------------------------------------------------------------- hydrostatic column
        h_scale = scale_height(t_surf, g, molar)
        res.value("scale_height", h_scale, "m", "H = R T / (M g)")
        z, p_rk4 = _rk4_column(h_scale, P_ISO, 501)
        err_col = float(np.max(np.abs(p_rk4 - P_ISO * np.exp(-z / h_scale))) / P_ISO)
        res.check("hydrostatic column integrates to p0 exp(-z/H)", err_col < 1e-9, value=err_col, limit=1e-9,
                  detail="RK4 of dp/dz = -p/H over 0..5H against the closed form (max relative)", source="analytic")
        col_err = []
        for steps in (25, 51, 101, 201):
            zz, pp = _rk4_column(h_scale, P_ISO, steps)
            col_err.append(float(np.max(np.abs(pp - P_ISO * np.exp(-zz / h_scale))) / P_ISO))
        orders = [float(np.log(col_err[i] / col_err[i + 1]) / np.log(2.0)) for i in range(len(col_err) - 1)]
        res.check("the column integrator is fourth order", min(orders) > 3.5,
                  value=float(min(orders)), limit=3.5,
                  detail=f"observed orders {', '.join(f'{o:.2f}' for o in orders)} as the step halves",
                  source="convergence")

        h_half = scale_height(t_surf, 0.5 * g, molar)
        res.compare("scale height at half the gravity", h_half / h_scale, 2.0, tol=1e-14, kind="limit",
                    source="H = R T/(M g) is proportional to 1/g (hydrostatic balance)")

        # the isothermal model against the standard atmosphere it is a simplification of
        p_1km = P_ISO * np.exp(-1000.0 / scale_height(T_ISO, 9.80665, M_AIR))
        res.bench("isothermal model against the ISO 2533 standard atmosphere",
                  abs(p_1km - P_ISO_1000) / P_ISO_1000 < 1e-2, value=p_1km, limit=P_ISO_1000,
                  detail=(f"p(1000 m) = {p_1km:.0f} Pa against the tabulated {P_ISO_1000:.0f} Pa; the "
                          "difference is the -6.5 K/km lapse rate this isothermal model leaves out"))
        res.value("p_1000m_isothermal", p_1km, "Pa", "H from the standard sea-level temperature")

        if dim == "0D":
            self._notes(res)
            return

        # ---------------------------------------------------------------- latitudinal energy balance
        n = int(p["n_lat"])
        d, aa, bb = float(p["transport"]), float(p["olr_a"]), float(p["olr_b"])
        shape = float(p["insolation_p2"])
        x, t = self._solve_profile(n, s0, a, aa, bb, d, shape)
        t_exact = self._legendre_profile(x, s0, a, aa, bb, d, shape)
        mean_t, mean_exact = _area_mean(t), _area_mean(t_exact)
        s2c = -shape * s0 / 4.0 * (1.0 - a)

        res.value("T_global_mean", mean_t, "K", "area-weighted mean of the latitudinal solution")
        res.value("T_pole_minus_equator", float(t[0] - t[n // 2]), "K", "pole minus equator of the solution")
        res.compare("global mean temperature", mean_t, mean_exact, tol=2e-3, kind="closed_form", unit="K",
                    source="area-weighted mean of T_0 + T_2 P_2 with T_0 = (s_0 - A)/B (North 1975)",
                    how="the residual is the discretisation error of the grid, and the refinement ladder shows it "
                        "falling at second order")
        res.compare("pole-to-equator contrast", float(t[0] - t[n // 2]),
                    float(t_exact[0] - t_exact[n // 2]), tol=1e-2, kind="closed_form", unit="K",
                    source="P_2(-1) - P_2(0) = 1.5, so the contrast is 1.5 T_2 with T_2 = s_2/(B + 6D)")

        err = float(np.max(np.abs(t - t_exact)) / np.ptp(t_exact))
        res.check("pole-to-equator profile matches the Legendre solution", err < 1e-2, value=err, limit=1e-2,
                  detail="max |T_FD - T_analytic| relative to the pole-to-equator range of the solution",
                  source="analytic")

        absorbed = s0 / 4.0 * (1.0 - a) * (1.0 - shape * p2(x))
        w = np.ones(n)
        w[0] = w[-1] = 0.5
        balance = float(np.sum(w * (aa + bb * t)) - np.sum(w * absorbed))
        res.check("global radiative balance", abs(balance) / abs(np.sum(w * absorbed)) < 1e-12, value=balance,
                  limit=0.0, detail="area-weighted outgoing longwave radiation minus absorbed flux (W/m^2)",
                  source="conservation")

        sizes = [n // 4, n // 2, n]

        def profile_error(nn: int) -> float:
            xx = np.linspace(-1.0, 1.0, nn)
            exact = self._legendre_profile(xx, s0, a, aa, bb, d, shape)
            return float(np.max(np.abs(self._solve_profile(nn, s0, a, aa, bb, d, shape)[1] - exact)) / np.ptp(exact))

        ladder = refinement(profile_error, sizes)
        orders = ladder["orders"]
        res.check("second-order convergence under refinement", bool(orders) and min(orders) > 1.5,
                  value=float(min(orders)) if orders else None, limit=1.5,
                  detail=f"observed orders {', '.join(f'{o:.2f}' for o in orders)} at n = {sizes}: the max norm "
                         "at the poles approaches two from below", source="convergence")

        # ---------------------------------------------------------------- sensitivity of the contrast
        def contrast(dd_: float) -> float:
            return 1.5 * s2c / (bb + 6.0 * dd_)

        dd = 1e-4 * d
        res.compare("d(pole contrast)/d(transport)", (contrast(d + dd) - contrast(d - dd)) / (2.0 * dd),
                    -9.0 * s2c / (bb + 6.0 * d) ** 2, tol=1e-6, kind="closed_form", unit="K^2 m^2/W",
                    source="d/dD [1.5 s_2/(B+6D)] = -9 s_2/(B+6D)^2, differentiated in closed form")
        res.check("a stronger transport damps the pole-to-equator contrast", s2c < 0 < bb + 6.0 * d,
                  value=float(contrast(d)), limit=0.0,
                  detail="the transport term only appears as B + 6D, so it can only shrink the n = 2 mode",
                  source="sensitivity")

        res.data("profile", x=x, T=t, T_exact=t_exact, absorbed=absorbed)
        self._notes(res)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _solve_profile(n: int, s0: float, a: float, aa: float, bb: float, d: float,
                       shape: float) -> tuple[np.ndarray, np.ndarray]:
        """The discrete energy balance, in conservative flux form.

        ``(1 - x^2)`` vanishes at both poles, so the discrete diffusion term carries no flux through the boundary
        and conserves globally; that is why the balance check below can be exact rather than approximate.
        """
        x = np.linspace(-1.0, 1.0, n)
        h = x[1] - x[0]
        face = faces_of(x)
        lower = np.zeros(n)
        diag = np.zeros(n)
        upper = np.zeros(n)
        # a finite-volume row: the divergence of the flux integrated over the cell, with the half cells at the two
        # poles weighted by their true share of the area; without that the scheme is only first order at the ends
        wgt = np.ones(n)
        wgt[0] = wgt[-1] = 0.5
        g = d * face / h**2
        lower[1:] = g / wgt[1:]                  # + (1 - x^2) on the face below the node
        upper[:-1] = g / wgt[:-1]                # + (1 - x^2) on the face above it
        diag[:-1] -= g / wgt[:-1]
        diag[1:] -= g / wgt[1:]
        diag -= bb
        absorbed = s0 / 4.0 * (1.0 - a) * (1.0 - shape * p2(x))
        return x, tridiag(lower, diag, upper, aa - absorbed)

    @staticmethod
    def _legendre_profile(x: np.ndarray, s0: float, a: float, aa: float, bb: float, d: float,
                          shape: float) -> np.ndarray:
        """``T_0 + T_2 P_2(x)``: the closed form the discrete solution has to reproduce."""
        s0c = s0 / 4.0 * (1.0 - a)
        s2c = -shape * s0 / 4.0 * (1.0 - a)
        return (s0c - aa) / bb + s2c / (bb + 6.0 * d) * p2(x)

    @staticmethod
    def _notes(res) -> None:
        res.notes.append("gravity enters only through the scale height H = R T/(M g); the radiative balance "
                         "contains no g, and changing g alone cannot change T_eq")
        res.notes.append("the atmosphere is isothermal and the transport is diffusive: both are closures, so the "
                         "latitudinal contrast is a property of the model, not a prediction of a climate")
        res.notes.append("no technological or experimental claim is made: nothing here is built or measured "
                         "except the standard atmosphere the isothermal column is compared with")


# ---------------------------------------------------------------------------------------------- the proposal
def super_earth(gravity: float = 2.5 * 9.80665) -> Proposal:
    """The What If Engine's worked proposal: the same planet at ``gravity``, everything else unchanged."""
    exp = ExoplanetLaboratory()
    sc = exp.world().with_gravity(gravity, source="the speculative change the question asks about")
    return Proposal(
        title=f"A hypothetical world at {gravity / 9.80665:.3g} g",
        question=f"What would the physics of a planet of {gravity:.4g} m/s^2 surface gravity look like?",
        speculation=("A rocky planet identical to the Earth in composition, insolation and albedo, with a heavier "
                     "surface gravity."),
        scenario=sc,
        hypotheses=(Hypothesis("the atmosphere is thinner in proportion to g at the same temperature",
                               phenomenon="hydrostatic column",
                               falsified_by="a scale height that does not follow 1/g",
                               plausibility="established"),
                    Hypothesis("the equilibrium temperature is unchanged by gravity alone",
                               phenomenon="radiative balance",
                               falsified_by="an equilibrium temperature that moves when only g moves",
                               plausibility="established"),
                    Hypothesis("a thinner atmosphere transports less heat, so the pole-to-equator contrast widens",
                               phenomenon="latitudinal transport",
                               falsified_by="a contrast that does not respond to the transport coefficient",
                               plausibility="extrapolated")),
        equations=("T_eq = [S(1-a)/(4 sigma)]^(1/4)",
                   "D d/dx[(1-x^2) dT/dx] - (A + B T) + S(x)(1-a) = 0",
                   "H = R T/(M g)"),
        phenomena=("radiative equilibrium", "latitudinal transport", "hydrostatic column"),
        experiment=ExoplanetLaboratory.name)

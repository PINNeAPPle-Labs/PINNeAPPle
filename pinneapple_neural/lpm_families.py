"""The first five Large Physics Model families: one model per design space, five design spaces.

Each family is a *vertical* with a reference the library already trusts — an exact closed form, or the immersed
finite-element solver checked against the physics. A family declares:

* the geometry it varies and the operating point it varies (``geometry_params`` / ``operating_params``);
* ``points_and_normals`` — the body as a surface point cloud, in metres, which is the geometry the model sees;
* ``field`` — the target at those points, from :attr:`reference`;
* ``ood_probes`` — designs outside the family, so the out-of-distribution flag is measured and not asserted.

| Family | Vertical | Field | Reference |
|---|---|---|---|
| ``pp_fin`` | electronics cooling: extruded-fin heat sink | T on the fins and the base (°C) | 1-D fin theory + the resistance network of :mod:`pinneapple_physics.closed_form.plate_fin_heatsink` (Incropera) |
| ``pp_airfoil`` | UAV, wind and compressor sections | Cp on the surface | Joukowski conformal map, exact potential flow with the Kutta condition (Milne-Thomson; Anderson) |
| ``pp_nozzle`` | propulsion: converging-diverging nozzle | Mach and p/p0 on the wall | quasi-1-D isentropic flow (Sutton & Biblarz ch. 3; NACA 1135) |
| ``pp_exchanger`` | process and HVAC: shell-and-tube exchanger | dimensionless shell and tube temperatures along the pass | Kern shell side, Gnielinski tube side, ε-NTU from the 1-2 energy equations (:mod:`…closed_form.heat_exchanger`) |
| ``pp_bracket`` | structural: L-bracket under load | displacement magnitude (m) and von Mises (Pa) | immersed finite-element elasticity, checked on equilibrium (:mod:`pinneapple_simulation.immersed`) |

What each reference does **not** include is written into its :class:`~.lpm_pipeline.Reference`, and the card of every
run repeats it. A model trained here is a model of its reference: the held-out error measures the fit to that
reference, and calling it agreement with a solver or with a test is the reviewer's job, not this module's.

Four of the five are dimensionless by construction (Cp, Mach, p/p0, θ) because that is how the physics scales; only
the heat sink and the bracket predict dimensional fields, and both say so in their ``field_units``.
"""
from __future__ import annotations

import math

import numpy as np

from .lpm_pipeline import DesignSpec, FieldFamily, Reference

__all__ = ["AIRFOIL", "BRACKET", "EXCHANGER", "FIN", "NOZZLE"]


# ===================================================================================== pp_fin: extruded-fin sink


class FinHeatSink(FieldFamily):
    """An extruded-fin heat sink in still air or in a duct: how hot its fins run at a given power.

    Geometry: the fin count, thickness and pitch, the fin height, the base depth and thickness. Operating point:
    the dissipated power, the ambient temperature and the approach velocity (0 = natural convection).

    Reference: the resistance network of :mod:`pinneapple_physics.closed_form.plate_fin_heatsink` (Elenbaas for
    natural convection, Teertstra for forced flow, fin-array conduction from
    :mod:`pinneapple_physics.closed_form.fin_array_conduction`), and the 1-D fin profile
    ``theta(x) = theta_root cosh(m (Lc - x)) / cosh(m Lc)`` built on the same ``m`` and ``Lc`` — the module computes
    both from one call so the scalar and the profile cannot diverge.
    """

    name = "pp_fin"
    title = "Extruded-fin heat sink: fin temperature over the design space"
    domain = "thermal / electronics cooling"
    dim = 2
    field_names = ("t_fin_c",)
    field_units = ("degC",)
    field_scale = (100.0,)
    characteristic_length_m = 0.05
    n_points = 96
    tolerance = 0.05
    geometry_params = {"n_fins": (6.0, 18.0), "fin_thickness_m": (0.001, 0.0025),
                       "fin_pitch_m": (0.005, 0.010), "fin_height_m": (0.010, 0.045),
                       "base_depth_m": (0.030, 0.080), "base_thickness_m": (0.003, 0.010)}
    operating_params = {"power_w": (5.0, 80.0), "t_ambient_c": (15.0, 45.0), "air_velocity_m_s": (0.0, 4.0)}
    reference = Reference(
        name="1-D fin theory on the fin-array resistance network",
        citation=("Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, ch. 3 (fins); Elenbaas (1942) and "
                  "Teertstra et al. (1999) for the convective coefficient; pinneapple_physics.closed_form."
                  "plate_fin_heatsink and .fin_array_conduction"),
        assumptions=("steady state, 1-D conduction along each fin with an adiabatic-tip corrected length",
                     "radiation neglected (the module says the part runs cooler than predicted in still air)",
                     "forced flow assumed ducted: all air passes between the fins",
                     "air properties from the module's own correlations, one iteration to convergence"),
    )

    # -- drawing -------------------------------------------------------------------------------------------------
    def sample_geometry(self, rng: np.random.Generator) -> dict[str, float]:
        """A geometry the reference accepts: the fin gap and the height/gap ratio stay inside the module's limits."""
        for _ in range(200):
            n = float(int(round(rng.uniform(*self.geometry_params["n_fins"]))))
            t = float(rng.uniform(*self.geometry_params["fin_thickness_m"]))
            pitch = float(rng.uniform(*self.geometry_params["fin_pitch_m"]))
            height = float(rng.uniform(*self.geometry_params["fin_height_m"]))
            gap = pitch - t
            if gap < 1.5e-3 or height / gap > 20.0:
                continue
            return {"n_fins": n, "fin_thickness_m": t, "fin_pitch_m": pitch, "fin_height_m": height,
                    "base_depth_m": float(rng.uniform(*self.geometry_params["base_depth_m"])),
                    "base_thickness_m": float(rng.uniform(*self.geometry_params["base_thickness_m"])),
                    "base_width_m": n * pitch}
        raise RuntimeError("pp_fin: could not draw a geometry inside the reference's own limits")

    def sample_operating(self, rng: np.random.Generator) -> dict[str, float]:
        """Still air (the natural-convection branch) or a forced flow well inside the forced correlation's range.

        The reference switches correlation at zero velocity; a velocity just above it is a *forced* draw with an
        almost-zero coefficient, which is numerically finite and physically meaningless. So the draw is bimodal and
        never lands in that sliver — :meth:`validate` rejects any that still falls outside the correlation.
        """
        still = bool(rng.random() < 0.35)
        return {"power_w": float(rng.uniform(*self.operating_params["power_w"])),
                "t_ambient_c": float(rng.uniform(*self.operating_params["t_ambient_c"])),
                "air_velocity_m_s": 0.0 if still else float(rng.uniform(0.8, 4.0))}

    def validate(self, spec: DesignSpec) -> str:
        """Reject a design whose correlations were used outside the range the reference documents."""
        from pinneapple_physics.closed_form import plate_fin_heatsink as pf

        net = self._network(spec)
        dimensionless = net["audit"].get("dimensionless", {})
        if net["audit"]["mode"] == "forced":
            re_star = dimensionless.get("re_star")
            if re_star is not None and not (0.26 <= float(re_star) <= 175.0):
                return f"channel Re* {float(re_star):.3g} outside the Teertstra range 0.26-175"
        else:
            elenbaas = dimensionless.get("elenbaas")
            if elenbaas is not None and not (0.1 <= float(elenbaas) <= 1e4):
                return f"Elenbaas number {float(elenbaas):.3g} outside 0.1-1e4"
        sink, op = net["sink"], net["op"]
        errors = pf.geometry_errors(sink, op)
        if errors:
            return "; ".join(errors)
        return ""

    # -- reference -----------------------------------------------------------------------------------------------
    def _network(self, spec: DesignSpec) -> dict[str, object]:
        """The heat sink, its operating point and everything both the profile and the base temperature need."""
        from pinneapple_physics.closed_form import fin_array_conduction as fa
        from pinneapple_physics.closed_form import plate_fin_heatsink as pf

        g = spec.geometry
        sink = pf.HeatSinkGeometry(base_width=g["base_width_m"], base_depth=g["base_depth_m"],
                                   base_thickness=g["base_thickness_m"], n_fins=int(g["n_fins"]),
                                   fin_thickness=g["fin_thickness_m"], fin_height=g["fin_height_m"])
        op = pf.OperatingPoint(power_w=spec.operating["power_w"], t_ambient_c=spec.operating["t_ambient_c"],
                               air_velocity_m_s=spec.operating["air_velocity_m_s"],
                               source_width=0.6 * g["base_width_m"], source_depth=0.6 * g["base_depth_m"])
        out = pf.evaluate(sink, op)
        k = pf.MATERIALS[sink.material]["k"]
        params = fa.fin_array_profile_params(
            k_material_w_mk=k, h_conv_w_m2k=float(out["h_w_m2k"]), n_fins=sink.n_fins,
            fin_thickness_m=sink.fin_thickness, fin_length_m=sink.fin_height,
            transverse_extent_m=sink.base_depth, spacing_extent_m=sink.base_width,
            base_thickness_m=sink.base_thickness, power_w=op.power_w)
        return {"sink": sink, "op": op, "params": params, "audit": out, "fa": fa}

    #: stations along each fin, on both faces (the tip is added separately)
    _LEVELS = 3

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """Both faces of every fin (sampled along the height) plus the base plane between them, in a 2-D section."""
        points, normals, on_fin = self._points(spec)
        return points, normals

    def _points(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The point cloud and, per point, whether it sits on a fin (the rest are on the base plane)."""
        g = spec.geometry
        n, t, height = int(g["n_fins"]), g["fin_thickness_m"], g["fin_height_m"]
        pitch = g["fin_pitch_m"]
        width = n * pitch
        # three stations, none exactly at the root (the fin equation is clamped there, so it would add no signal)
        levels = np.clip(np.array([0.15, 0.65, 1.0]) * height, 0.0, height)
        points, normals, on_fin = [], [], []
        for i in range(n):
            y0 = -0.5 * width + (i + 0.5) * pitch
            for x in levels:
                points += [[x - 0.5 * height, y0 - 0.5 * t], [x - 0.5 * height, y0 + 0.5 * t]]
                normals += [[0.0, -1.0], [0.0, 1.0]]
                on_fin += [True, True]
            points.append([0.5 * height, y0])
            normals.append([1.0, 0.0])
            on_fin.append(True)
        # the base plane between fins: it shares the fin-array network, so its temperature is the base temperature
        for i in range(max(n - 1, 1)):
            points.append([-0.5 * height, -0.5 * width + (i + 1) * pitch])
            normals.append([-1.0, 0.0])
            on_fin.append(False)
        return (np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64),
                np.asarray(on_fin, dtype=bool))

    def field(self, spec: DesignSpec) -> np.ndarray:
        net = self._network(spec)
        fa, params = net["fa"], net["params"]
        t_amb, power = spec.operating["t_ambient_c"], spec.operating["power_w"]
        points, _, on_fin = self._points(spec)
        height = spec.geometry["fin_height_m"]
        values = np.asarray([fa.fin_array_temperature_at_position_c(params, t_amb, float(p[0] + 0.5 * height))
                             for p in points], dtype=np.float64)
        # the base-plane points sit on the base, not on a fin: the network's base temperature, same parameters
        values[~on_fin] = fa.fin_array_base_temperature_c(params, t_amb, power)
        return values[:, None]

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        base = {"fin_thickness_m": 0.002, "fin_height_m": 0.030, "base_depth_m": 0.05,
                "base_thickness_m": 0.005, "base_width_m": 0.10}
        op = {"power_w": 30.0, "t_ambient_c": 25.0, "air_velocity_m_s": 2.0}
        return [
            ("a 30-fin sink: 3x the fin count of the trained family",
             DesignSpec({**base, "n_fins": 30.0, "fin_pitch_m": 0.004}, op, label="probe:dense")),
            ("a 15x-power sink: 450 W on a 3 cm fin",
             DesignSpec({**base, "n_fins": 10.0, "fin_pitch_m": 0.010}, {**op, "power_w": 450.0},
                        label="probe:power")),
        ]


# =============================================================================== pp_airfoil: Joukowski potential flow


def _joukowski_surface(x_c: float, y_c: float, radius: float, c: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The mapped body ``z = zeta + c^2/zeta`` of the circle ``|zeta - (x_c + i y_c)| = radius``.

    Returns the points ``(n, 2)`` and the outward unit normals, which map through the derivative of a conformal map
    as ``(dz/dzeta) * e^{i theta}``.
    """
    theta = 2.0 * math.pi * (np.arange(n) + 0.5) / n  # no point exactly on the cusp at theta = 0
    zeta = (x_c + 1j * y_c) + radius * np.exp(1j * theta)
    z = zeta + c * c / zeta
    dz = 1.0 - (c * c) / (zeta * zeta)
    normal = dz * np.exp(1j * theta)
    normal = normal / np.abs(normal)
    return np.stack([z.real, z.imag], axis=1), np.stack([normal.real, normal.imag], axis=1)


def _joukowski_cp(x_c: float, y_c: float, radius: float, c: float, n: int, alpha_deg: float,
                  circulation: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact potential-flow Cp on the mapped body, with the Kutta condition unless ``circulation`` is given.

    ``circulation`` is ``Gamma / (U R)``; with the default, the value that makes the velocity finite at the cusp
    is ``4 pi sin(alpha + beta)`` with ``beta`` the angle of the centre seen from the trailing edge.

    The speed is evaluated in the factored form ``dw/dzeta = dw_c + (zeta - c) B(zeta)`` rather than as the quotient
    ``|dw/dzeta| / |dz/dzeta|``. Both are the same number, but at the trailing edge both numerator and denominator go
    to zero (that is what the Kutta condition does), and the plain quotient loses every digit to cancellation: with
    the factored form the residual ``dw_c`` is a tiny additive term instead of a 0/0, and the Cp near the cusp stops
    depending on how many points the surface was sampled with.
    """
    alpha = math.radians(alpha_deg)
    zeta0 = x_c + 1j * y_c
    beta = math.atan2(y_c, c - x_c)
    if circulation is None:
        circulation = 4.0 * math.pi * math.sin(alpha + beta)
    # ``circulation`` is Gamma / (U R); the vortex term of the complex potential carries Gamma / U = circulation * R
    gamma_u = circulation * radius
    theta = 2.0 * math.pi * (np.arange(n) + 0.5) / n
    zeta = zeta0 + radius * np.exp(1j * theta)
    # B(zeta): what is left of the vortical part after the (zeta - c) the trailing edge forces on both terms
    bracket = (radius * radius * np.exp(1j * alpha) * (c + zeta - 2.0 * zeta0)
               / ((zeta - zeta0) ** 2 * (c - zeta0) ** 2)
               - 1j * gamma_u / (2.0 * math.pi * (zeta - zeta0) * (c - zeta0)))
    # the residual of the Kutta condition: exactly zero when the circulation is the one that cancels it
    residual = (np.exp(-1j * alpha) - (radius * radius) * np.exp(1j * alpha) / (c - zeta0) ** 2
                + 1j * gamma_u / (2.0 * math.pi * (c - zeta0)))
    speed = np.abs(bracket + residual / (zeta - c)) * np.abs(zeta) ** 2 / np.abs(zeta + c)
    return 1.0 - speed * speed, zeta, float(circulation)


class JoukowskiAirfoil(FieldFamily):
    """A family of cambered Joukowski airfoils in incompressible potential flow: the pressure on the section.

    Geometry: the camber and the thickness of the section, both as the position of the generating circle's centre
    in the mapped plane (``y_c`` metres up for camber, ``x_c`` metres aft of the origin for thickness; the chord is
    about 1 m). Operating point: the incidence.

    Reference: the exact conformal map ``z = zeta + c^2/zeta`` applied to a circle, with the Kutta condition fixing
    the circulation. That is exact inviscid potential flow — no boundary layer, no separation, no compressibility —
    and the tests pin two independent checks: the code reproduces the library's analytic ellipse Cp when the
    circulation is zero, and the lift it integrates equals ``rho U Gamma``.
    """

    name = "pp_airfoil"
    title = "Cambered Joukowski airfoil: surface Cp over camber, thickness and incidence"
    domain = "aerodynamics / UAV, wind and compressor sections"
    dim = 2
    field_names = ("cp",)
    field_units = ("-",)
    field_scale = (10.0,)
    characteristic_length_m = 0.5
    n_points = 120
    tolerance = 0.05
    geometry_params = {"thickness_x_c_m": (-0.090, -0.010), "camber_y_c_m": (0.0, 0.060)}
    operating_params = {"alpha_deg": (-4.0, 14.0)}
    reference = Reference(
        name="Joukowski conformal map with the Kutta condition (exact potential flow)",
        citation=("Milne-Thomson, Theoretical Hydrodynamics, 5th ed. (1968), ch. 6 (the Joukowski transformation); "
                  "Anderson, Fundamentals of Aerodynamics, ch. 4 (Kutta condition and Kutta-Joukowski theorem)"),
        assumptions=("incompressible, inviscid, irrotational: no boundary layer, no separation, no drag from "
                     "pressure", "two-dimensional and steady", "a cusped trailing edge, so the Kutta condition "
                     "applies and the circulation is fixed by it"),
    )

    #: the map constant: the trailing edge sits at ``z = 2 c``, which sets a chord of about ``2 c`` metres
    _C: float = 0.5

    def _circle(self, spec: DesignSpec) -> tuple[float, float, float, float]:
        x_c, y_c = spec.geometry["thickness_x_c_m"], spec.geometry["camber_y_c_m"]
        radius = math.hypot(self._C - x_c, y_c)  # the circle passes through zeta = c: a cusped trailing edge
        return x_c, y_c, radius, self._C

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        return _joukowski_surface(*self._circle(spec), self.n_points)

    def field(self, spec: DesignSpec) -> np.ndarray:
        cp, _, _ = _joukowski_cp(*self._circle(spec), self.n_points, spec.operating["alpha_deg"])
        return cp[:, None]

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        return [
            ("a 36% thick section, four times the trained thickness",
             DesignSpec({"thickness_x_c_m": -0.36, "camber_y_c_m": 0.0}, {"alpha_deg": 4.0}, label="probe:thick")),
            ("an 18% camber, three times the trained camber",
             DesignSpec({"thickness_x_c_m": -0.05, "camber_y_c_m": 0.18}, {"alpha_deg": 8.0}, label="probe:camber")),
        ]


# ============================================================================ pp_nozzle: quasi-1-D isentropic nozzle


class ConicalNozzle(FieldFamily):
    """A converging-diverging conical nozzle: Mach and pressure along the wall for a given gas.

    Geometry: the exit and chamber area ratios, the throat radius and the divergence half-angle (which sets the
    length). Operating point: the ratio of specific heats and the gas constant — the two properties the ideal
    1-D solution depends on. Chamber pressure and temperature are **not** parameters: for this reference they
    scale the field exactly, so the field is reported normalised and the card states the reconstruction.

    Reference: quasi-1-D isentropic flow, ``A/A*(M)`` inverted for the Mach number on each branch and
    ``p/p0(M)`` from :mod:`pinneapple_physics.closed_form.compressible`.
    """

    name = "pp_nozzle"
    title = "Converging-diverging nozzle: wall Mach and pressure across the design space"
    domain = "propulsion / rocket and gas-turbine nozzles"
    dim = 2
    field_names = ("mach", "pressure_over_chamber")
    field_units = ("-", "-")
    field_scale = (5.0, 1.0)
    characteristic_length_m = 0.25
    n_points = 100
    tolerance = 0.02
    geometry_params = {"area_ratio_exit": (2.0, 25.0), "chamber_ratio": (2.5, 12.0),
                       "throat_radius_m": (0.020, 0.080), "half_angle_deg": (8.0, 30.0)}
    operating_params = {"gamma": (1.15, 1.30), "r_gas_j_kg_k": (250.0, 400.0)}
    reference = Reference(
        name="quasi-1-D isentropic flow of a calorically perfect gas",
        citation=("Sutton & Biblarz, Rocket Propulsion Elements, 9th ed. (2017), ch. 3; NACA Report 1135 (1953), "
                  "eqs. 43-45 and 80; pinneapple_physics.closed_form.rocket_nozzle and .compressible"),
        assumptions=("one-dimensional section averages, isentropic: no shocks, friction, heat loss or separation",
                     "constant gamma, calorically perfect gas", "flow attached and expanded to the exit area ratio",
                     "the field is p/p_chamber; the physical pressure is that times the chamber pressure"),
    )

    def _contour(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """``x`` along the axis (throat at ``x = 0``, which is always a sample) and the wall radius there."""
        g = spec.geometry
        rt = g["throat_radius_m"]
        r_exit = rt * math.sqrt(g["area_ratio_exit"])
        r_chamber = rt * math.sqrt(g["chamber_ratio"])
        tan_half = math.tan(math.radians(g["half_angle_deg"]))
        length_exit = (r_exit - rt) / tan_half
        length_chamber = (r_chamber - rt) / tan_half
        n_chamber = max(self.n_points // 2, 2)
        x = np.concatenate([np.linspace(-length_chamber, 0.0, n_chamber, endpoint=False),
                            np.linspace(0.0, length_exit, self.n_points - n_chamber + 1)])
        r = np.where(x < 0.0, r_chamber - (x + length_chamber) * tan_half, rt + x * tan_half)
        return x, r

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """Both walls of the axisymmetric section, as a 2-D body: the fluid is inside, so the normals point in."""
        x, r = self._contour(spec)
        tan_half = math.tan(math.radians(spec.geometry["half_angle_deg"]))
        slope = np.where(x < 0.0, -tan_half, tan_half)
        points, normals = [], []
        for xi, ri, si in zip(x, r, slope, strict=True):
            for sign in (1.0, -1.0):
                points.append([xi, sign * ri])
                vector = np.array([si, -sign])
                normals.append(vector / np.linalg.norm(vector))
        return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)

    def field(self, spec: DesignSpec) -> np.ndarray:
        from pinneapple_physics.closed_form.compressible import isentropic_ratios
        from pinneapple_physics.closed_form.rocket_nozzle import mach_from_area_ratio

        gamma = spec.operating["gamma"]
        x, r = self._contour(spec)
        rt = spec.geometry["throat_radius_m"]
        ratio = (r / rt) ** 2
        mach = np.array([1.0 if abs(a - 1.0) < 1e-9 else mach_from_area_ratio(float(a), gamma, supersonic=(xi >= 0.0))
                         for a, xi in zip(ratio, x, strict=True)])
        pressure = 1.0 / isentropic_ratios(mach, gamma)["p0_p"]
        both = np.concatenate([mach, mach])
        return np.stack([both, np.concatenate([pressure, pressure])], axis=1)

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        gas = {"gamma": 1.22, "r_gas_j_kg_k": 320.0}
        return [
            ("an area ratio of 60, 2.4x the trained exit ratio",
             DesignSpec({"area_ratio_exit": 60.0, "chamber_ratio": 6.0, "throat_radius_m": 0.05,
                         "half_angle_deg": 15.0}, gas, label="probe:ratio")),
            ("a gamma of 1.67 (monatomic gas), far outside the combustion-gas family",
             DesignSpec({"area_ratio_exit": 10.0, "chamber_ratio": 6.0, "throat_radius_m": 0.05,
                         "half_angle_deg": 15.0}, {"gamma": 1.67, "r_gas_j_kg_k": 208.0}, label="probe:gamma")),
        ]


# ========================================================================== pp_exchanger: shell-and-tube exchanger


class ShellTubeExchanger(FieldFamily):
    """A 1-2 shell-and-tube exchanger: the two fluid temperatures along the pass, dimensionless.

    Geometry: tube outside diameter, tube count, shell diameter, length and baffle spacing. Operating point: the
    two mass flows. The field is dimensionless, ``theta = (T - T_cold_in) / (T_hot_in - T_cold_in)``, because for
    this reference the inlet temperatures only rescale it; the physical temperature is
    ``T_cold_in + theta (T_hot_in - T_cold_in)`` and the card says so.

    Reference: Kern's method on the shell side, Gnielinski (laminar below Re 2300) on the tube side, UA from the
    two film coefficients, and the ε-NTU of a 1-2 exchanger integrated from its three energy equations. The 1-2
    model of :mod:`pinneapple_physics.closed_form.heat_exchanger` takes the shell stream as ``C_min``, so the
    family draws the shell flow below the tube flow — a design it cannot represent is never handed to it.
    """

    name = "pp_exchanger"
    title = "Shell-and-tube exchanger: dimensionless temperature profiles over its design space"
    domain = "energy / process and HVAC heat exchangers"
    dim = 2
    field_names = ("theta_shell", "theta_tube")
    field_units = ("-", "-")
    field_scale = (1.0, 1.0)
    characteristic_length_m = 1.0
    n_points = 80
    tolerance = 0.02
    geometry_params = {"tube_od_m": (0.010, 0.025), "n_tubes": (20.0, 160.0), "shell_id_m": (0.10, 0.35),
                       "length_m": (0.6, 3.5), "baffle_spacing_m": (0.05, 0.25)}
    operating_params = {"m_dot_shell_kg_s": (0.5, 3.5), "m_dot_tube_kg_s": (4.0, 12.0)}
    #: the tube count, the shell and the baffles are invisible in the tube's own outline, so they are conditioned
    #: explicitly — the field depends on them through the shell-side coefficient, and a model that cannot see them
    #: cannot learn the field
    geometry_features = {"n_tubes": (20.0, 160.0), "shell_over_length": (0.03, 0.60),
                         "baffle_over_length": (0.02, 0.40)}
    reference = Reference(
        name="Kern + Gnielinski film coefficients with the 1-2 exchanger energy equations (epsilon-NTU)",
        citation=("Kern (1950) via pinneapple_physics.closed_form.heat_exchanger.kern_shell; Gnielinski (1976) and "
                  "Incropera ch. 8 for the tube side; Nagle's 1-2 model integrated in "
                  "pinneapple_physics.closed_form.heat_exchanger.one_two_exchanger_ode"),
        assumptions=("water on both sides with properties held at 45 degC (cp 4182, mu 6.5e-4, k 0.64, rho 990)",
                     "1-2 exchanger, shell fluid well mixed across each section, tube passes in counterflow",
                     "constant UA along the exchanger, no fouling, no heat loss to the surroundings",
                     "shell stream is C_min by construction, which is what the 1-2 model assumes"),
    )

    #: water properties held fixed, so the reference is one model and not a property call per design
    _WATER = {"cp": 4182.0, "mu": 6.5e-4, "k": 0.64, "rho": 990.0, "wall_m": 0.001}

    def geometry_feature_values(self, spec: DesignSpec) -> dict[str, float]:
        g = spec.geometry
        return {"n_tubes": g["n_tubes"], "shell_over_length": g["shell_id_m"] / g["length_m"],
                "baffle_over_length": g["baffle_spacing_m"] / g["length_m"]}

    def _reference(self, spec: DesignSpec) -> tuple[float, float, dict[str, float]]:
        """``(NTU, Cr, details)`` for one design, from the two film coefficients and the two capacities."""
        from pinneapple_physics.closed_form import heat_exchanger as hx

        g, op, w = spec.geometry, spec.operating, self._WATER
        d_in = g["tube_od_m"] - 2.0 * w["wall_m"]
        area_flow = g["n_tubes"] * math.pi * d_in ** 2 / 4.0
        u_tube = op["m_dot_tube_kg_s"] / (w["rho"] * area_flow)
        re = w["rho"] * u_tube * d_in / w["mu"]
        pr = w["cp"] * w["mu"] / w["k"]
        nu_tube = hx.gnielinski(re, pr)["nu"]
        h_tube = nu_tube * w["k"] / d_in
        area_tube = g["n_tubes"] * math.pi * d_in * g["length_m"]
        pitch = 1.25 * g["tube_od_m"]
        shell = hx.kern_shell(op["m_dot_shell_kg_s"], w["mu"], w["k"], w["cp"], shell_id=g["shell_id_m"],
                              baffle_spacing=g["baffle_spacing_m"], tube_od=g["tube_od_m"], pitch=pitch)
        h_shell = shell["h"]
        area_shell = g["n_tubes"] * math.pi * g["tube_od_m"] * g["length_m"]
        ua = 1.0 / (1.0 / max(h_tube * area_tube, 1e-12) + 1.0 / max(h_shell * area_shell, 1e-12))
        c_shell = op["m_dot_shell_kg_s"] * w["cp"]
        c_tube = op["m_dot_tube_kg_s"] * w["cp"]
        c_min, c_max = min(c_shell, c_tube), max(c_shell, c_tube)
        details = {"re_tube": re, "nu_tube": nu_tube, "h_tube": h_tube, "h_shell": h_shell,
                   "ua": ua, "c_min": c_min, "c_max": c_max, "pitch": pitch,
                   "re_shell": shell["re"], "nu_shell": shell["nu"]}
        return ua / c_min, c_min / c_max, details

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """The representative tube: its wall along the exchanger length, both sides, dimensionless in x/L."""
        g = spec.geometry
        radius = 0.5 * g["tube_od_m"] / g["length_m"]  # radius over length: the dimensionless tube the model sees
        x = np.linspace(0.0, 1.0, self.n_points // 2)
        points = np.concatenate([np.stack([x, np.full_like(x, radius)], 1),
                                 np.stack([x, np.full_like(x, -radius)], 1)])
        normals = np.concatenate([np.tile([0.0, 1.0], (len(x), 1)), np.tile([0.0, -1.0], (len(x), 1))])
        return points, normals

    def validate(self, spec: DesignSpec) -> str:
        """Reject a design whose film coefficients or energy equations do not come out solvable."""
        from pinneapple_physics.closed_form.heat_exchanger import one_two_exchanger_ode

        try:
            ntu, cr, details = self._reference(spec)
        except Exception as exc:                       # noqa: BLE001 - a rejected draw, with the reason
            return f"the reference could not be evaluated ({type(exc).__name__}: {exc})"
        if not (math.isfinite(ntu) and math.isfinite(cr) and ntu > 0.0 and 0.0 < cr <= 1.0):
            return f"NTU {ntu:g} with Cr {cr:g} is not a solvable exchanger"
        if min(details["h_tube"], details["h_shell"], details["re_tube"]) <= 0.0:
            return "a film coefficient or the tube Reynolds number came out non-positive"
        try:
            one_two_exchanger_ode(ntu, cr, profiles=True)
        except Exception as exc:                       # noqa: BLE001 - the boundary-value solve failed
            return f"the 1-2 energy equations did not converge ({exc})"
        return ""

    def field(self, spec: DesignSpec) -> np.ndarray:
        from pinneapple_physics.closed_form.heat_exchanger import one_two_exchanger_ode

        ntu, cr, _ = self._reference(spec)
        _, profiles = one_two_exchanger_ode(ntu, cr, profiles=True)
        x = profiles["x"]
        shell = np.interp(np.linspace(0.0, 1.0, self.n_points // 2), x, profiles["shell"])
        tube = np.interp(np.linspace(0.0, 1.0, self.n_points // 2), x, profiles["pass2"])
        return np.stack([np.concatenate([shell, shell]), np.concatenate([tube, tube])], axis=1)

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        op = {"m_dot_shell_kg_s": 1.5, "m_dot_tube_kg_s": 6.0}
        return [
            ("a 300-tube shell, twice the trained tube count",
             DesignSpec({"tube_od_m": 0.019, "n_tubes": 300.0, "length_m": 2.0, "shell_id_m": 0.30,
                         "baffle_spacing_m": 0.12}, op, label="probe:tubes")),
            ("a 9 m exchanger, 2.6x the trained length",
             DesignSpec({"tube_od_m": 0.019, "n_tubes": 60.0, "length_m": 9.0, "shell_id_m": 0.20,
                         "baffle_spacing_m": 0.12}, op, label="probe:length")),
        ]


# ==================================================================================== pp_bracket: L-bracket, FEM


class LoadedBracket(FieldFamily):
    """An L-bracket under a tip load: displacement and stress over the design space — the one solver-backed family.

    Geometry: the two leg lengths and the leg thickness. Operating point: the total load, its direction in the
    plane and the Young's modulus, so one family covers aluminium and steel parts.

    Reference: :mod:`pinneapple_simulation.immersed`, the immersed finite-element solver (Q1 background grid,
    cut-cell quadrature, ghost-penalty stabilisation, Nitsche boundary conditions) solved in 2-D plane elasticity.
    The displacement is evaluated at the same surface points the model is given, and every design is accepted only
    if the solver's own equilibrium check closes (``imbalance`` below 1e-8 of the applied load).
    """

    name = "pp_bracket"
    title = "L-bracket under a tip load: displacement and von Mises across the design space"
    domain = "structural / brackets and machine parts in linear elasticity"
    dim = 2
    field_names = ("displacement_m", "von_mises_pa")
    field_units = ("m", "Pa")
    field_scale = (1.0e-6, 1.0e6)
    characteristic_length_m = 1.0
    n_points = 96
    tolerance = 0.05
    geometry_params = {"leg_a_m": (0.40, 1.60), "leg_b_m": (0.30, 1.20), "leg_thickness_m": (0.05, 0.20)}
    operating_params = {"load_n": (1.0e3, 3.0e4), "angle_deg": (-60.0, 60.0), "e_gpa": (60.0, 210.0)}
    reference = Reference(
        name="immersed finite-element elasticity, plane stress-free 2-D, checked on equilibrium",
        citation=("pinneapple_simulation.immersed (Burman & Hansbo ghost-penalty finite cells; see "
                  "docs/guides/immersed_analysis.md), verified against conforming FEM in "
                  "immersed.verification and docs/guides/immersed_analysis.md"),
        assumptions=("linear, isotropic, small-strain elasticity in 2-D with Poisson ratio 0.33",
                     "the leg is a solid 2-D section: no out-of-plane bending, no plate or shell behaviour",
                     "the base edge is fully clamped, the load is a total force spread over the tip edge",
                     "the mesh is fixed at 1/50 of the longer leg, so a coarse-mesh error remains in every field"),
        implements="solver",
    )

    #: which edge carries the load, and which one is clamped, in the geometry the points describe
    @staticmethod
    def _outline(spec: DesignSpec, per_edge: int) -> tuple[np.ndarray, np.ndarray]:
        g = spec.geometry
        a, b, t = g["leg_a_m"], g["leg_b_m"], g["leg_thickness_m"]
        corners = np.array([[0.0, 0.0], [a, 0.0], [a, t], [t, t], [t, b], [0.0, b]])
        normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        points, out = [], []
        for i in range(6):
            start, end = corners[i], corners[(i + 1) % 6]
            for s in (np.arange(per_edge) + 0.5) / per_edge:
                points.append(start + s * (end - start))
                out.append(normals[i])
        return np.asarray(points, dtype=np.float64), np.asarray(out, dtype=np.float64)

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        return self._outline(spec, max(self.n_points // 6, 6))

    def _solve(self, spec: DesignSpec):
        from pinneapple_simulation import immersed as im

        g, op = spec.geometry, spec.operating
        a, b, t = g["leg_a_m"], g["leg_b_m"], g["leg_thickness_m"]
        geometry = im.geometry.union(im.geometry.box([0.0, 0.0], [a, t]), im.geometry.box([0.0, 0.0], [t, b]))
        step = max(a, b) / 50.0
        disc = im.Discretization(geometry, h=step, order=1)
        angle = math.radians(op["angle_deg"])
        load = float(op["load_n"])
        solution = im.solve_elasticity(
            disc, E=float(op["e_gpa"]) * 1e9, nu=0.33,
            bcs=[im.Displacement([0.0, 0.0], where=lambda X, n: np.isclose(X[:, 0], 0.0)),
                 im.Force([load * math.cos(angle), load * math.sin(angle)],
                          where=lambda X, n: np.isclose(X[:, 1], b))],
        )
        imbalance = float(abs(solution.balance["imbalance"]))
        if not math.isfinite(imbalance) or imbalance > 1e-8:
            raise RuntimeError(f"pp_bracket: the solver did not close its own equilibrium (imbalance {imbalance:g})")
        return solution, imbalance

    def field(self, spec: DesignSpec) -> np.ndarray:
        solution, _ = self._solve(spec)
        points, normals = self.points_and_normals(spec)
        # evaluate just inside the surface: the immersed solution is defined on the background grid
        probe = points - 1e-6 * normals * max(spec.geometry["leg_a_m"], spec.geometry["leg_b_m"])
        displacement = np.asarray(solution(probe), dtype=np.float64)
        von_mises = np.asarray(solution.von_mises(probe), dtype=np.float64)
        return np.stack([np.linalg.norm(displacement, axis=1), np.abs(von_mises)], axis=1)

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        op = {"load_n": 5.0e3, "angle_deg": -30.0, "e_gpa": 70.0}
        return [
            ("a 4.5 m leg, 2.8x the trained arm",
             DesignSpec({"leg_a_m": 4.5, "leg_b_m": 0.8, "leg_thickness_m": 0.12}, op, label="probe:leg")),
            ("a 300 kN load, 10x the trained load",
             DesignSpec({"leg_a_m": 1.0, "leg_b_m": 0.8, "leg_thickness_m": 0.12},
                        {**op, "load_n": 3.0e5}, label="probe:load")),
        ]


FIN = FinHeatSink()
AIRFOIL = JoukowskiAirfoil()
NOZZLE = ConicalNozzle()
EXCHANGER = ShellTubeExchanger()
BRACKET = LoadedBracket()

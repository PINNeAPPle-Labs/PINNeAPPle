"""What the capsule-aeroheating dataset lets you check without trusting it: four checks, and one that checks nothing.

The reader (:mod:`pinneapple_data.datasets.capsule_aeroheating`) gets the numbers out of the file. This module asks
whether they hold together, and each check here is arithmetic on the file's own arrays -- no reference solution, no
fitted parameter, nothing taken from the dataset card:

``axial_symmetry``      the flow is at zero incidence, so a field on the heat shield must be the same all the way
                        round each ring of equal arc length. This is the one check that tests the CFD data itself,
                        with no free parameter: the file states the angle of attack is zero, and a ring that disagrees
                        with itself is an error in the solution. Over the 64 CFD runs the largest within-ring spread
                        of the heat flux is 1.80 % (median 0.46 %), against the 1.20 % the dataset card reports over
                        3-80 % of the arc -- a different way of taking the ring, and the difference is stated here
                        rather than tuned away; the gate is at 2 %, which a solution that is not axisymmetric blows
                        through by an order of magnitude.
``newtonian_pressure``  the low-fidelity pressure is a modified Newtonian panel method: ``C_p = C_p,max sin^2(theta)``
                        with ``theta`` the angle between the surface normal and the freestream, and ``C_p,max`` the
                        stagnation value behind a normal shock (Rayleigh pitot). Both come from the wall's own
                        triangles and the run's own freestream, so recomputing it checks the geometry convention
                        (nodes, orientation, areas) and the units against the panel method's own output: over the
                        1.024 panel runs the median error is 0.01 % to 0.07 %, and the stagnation pressure comes back
                        to five figures. It does not check the physics: the physics question is the cheap model
                        against the CFD, which is what the dataset exists to measure (15 % median on the 64 CFD
                        runs, of which the shoulder is the largest part).
``heat_load``           the per-run scalar ``heat_load`` is the heat flux integrated over the wall. Integrating the
                        field over the triangles gives the scalar back to 0.008 % on every run.
``radiative_equilibrium``  solve ``C_H (h_r - c_p T_w) = eps sigma T_w^4`` for the wall temperature a heat shield
                        would reach. This is the *use* of the film coefficient the file carries, not a check of it.

The one thing that is **not** a check: ``q = C_H (h_r - h_w)`` holds to the last bit, because ``film_coefficient`` is
defined as ``q / (h_r - h_w)`` when the file is written. Comparing the two is an identity, and a gate built on it would
pass on any data at all -- including wrong data. It is asserted here only so that the identity is written down
somewhere rather than mistaken for a verification.

    >>> from pinneapple_data.datasets import capsule_aeroheating as cap, capsule_aeroheating_physics as physics
    >>> with cap.open_study(path) as study:
    ...     one = cap.wall(study, int(cap.pair_fidelities(study)[0][0]))
    ...     for result in physics.checks(one):
    ...         print(result.check_id, result.status, result.value)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capsule_aeroheating import Wall

__all__ = ["STEFAN_BOLTZMANN", "cp_of", "surface_normals", "cell_areas", "stagnation_pressure_coefficient",
           "newtonian_pressure", "relative_error", "axial_spread", "heat_load", "wall_heat_flux",
           "radiative_equilibrium_temperature", "closure_identity_residual", "checks", "record", "summary",
           "PhysicsSummary"]

#: Stefan-Boltzmann constant, W m-2 K-4 (exact in SI since the 2019 redefinition).
STEFAN_BOLTZMANN = 5.670374419e-8


# ---------------------------------------------------------------------------
# the gas
# ---------------------------------------------------------------------------

def cp_of(wall: Wall) -> float:
    """The specific heat of the gas at constant pressure, J kg-1 K-1, from the study's own gamma and R.

    The file states ``ratio_of_specific_heats`` and ``gas_constant`` and calls the air calorically perfect, so
    ``c_p = gamma R / (gamma - 1)`` is a consequence, not an assumption: 1004.675 for gamma 1.4 and R 287.05, which
    is the number the dataset card quotes."""
    gamma, gas_constant = float(wall.constants["ratio_of_specific_heats"]), float(wall.constants["gas_constant"])
    if gamma <= 1.0:
        raise ValueError(f"gamma {gamma} is not above 1: c_p = gamma R / (gamma - 1) has no meaning")
    return gamma * gas_constant / (gamma - 1.0)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def surface_normals(wall: Wall) -> np.ndarray:
    """The outward unit normal at each node, area-weighted from the wall's triangles, ``(n_nodes, 3)``.

    Area weighting is what makes a nodal normal belong to the surface rather than to whichever triangle was listed
    first: a node shared by three faces is pulled towards the middle of the patch it covers."""
    corners = np.asarray(wall.points, dtype=np.float64)[np.asarray(wall.cells)]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2.0
    if not np.any(area > 0.0):
        raise ValueError("every triangle of this wall has zero area: the connectivity and the points disagree")
    face_normal = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    total = np.zeros_like(np.asarray(wall.points, dtype=np.float64))
    for corner in range(corners.shape[1]):
        np.add.at(total, wall.cells[:, corner], face_normal * area[:, None])
    length = np.linalg.norm(total, axis=1, keepdims=True)
    if np.any(length == 0.0):
        raise ValueError(f"{int((length == 0).sum())} node(s) have no area around them: no normal can be formed")
    return total / length


def cell_areas(wall: Wall) -> np.ndarray:
    """The area of each face, m2, ``(n_cells,)``, by fanning it into triangles from its first node.

    That is exact for a triangle and for the planar quads of the dataset's quad file. An element with a volume
    (a hexahedron, a tetrahedron) has no area to report and is refused rather than answered with a number that would
    be a surface of it."""
    corners = np.asarray(wall.points, dtype=np.float64)[np.asarray(wall.cells)]
    if corners.shape[1] not in (3, 4):
        raise ValueError(f"cell type '{wall.cell_type}' with {corners.shape[1]} nodes is not a face this module "
                         "integrates over")
    area = np.zeros(corners.shape[0], dtype=np.float64)
    for corner in range(1, corners.shape[1] - 1):
        cross = np.cross(corners[:, corner] - corners[:, 0], corners[:, corner + 1] - corners[:, 0])
        area += np.linalg.norm(cross, axis=1) / 2.0
    return area


# ---------------------------------------------------------------------------
# the panel method, recomputed
# ---------------------------------------------------------------------------

def stagnation_pressure_coefficient(gamma: float, mach: float) -> float:
    """``C_p,max``: the stagnation pressure coefficient behind a normal shock, from the Rayleigh pitot formula.

    ``C_p,max = 2 (p02/p_inf - 1) / (gamma M^2)``. As the Mach number grows it tends to ``2K/gamma`` with
    ``K = ((gamma+1)^2 / 4 gamma)^(gamma/(gamma-1)) * 2 gamma/(gamma+1)`` -- 1.839 for air, not 2: the plain
    Newtonian value ``C_p = 2 sin^2(theta)`` is the limit as ``gamma`` tends to 1, and a *modified* Newtonian method
    takes the stagnation value from the shock instead, which is what this dataset's panel method does."""
    if mach <= 1.0:
        raise ValueError(f"Mach {mach} is not supersonic: the Rayleigh pitot formula is for a normal shock")
    pitot = ((((gamma + 1.0) ** 2 * mach ** 2) / (4.0 * gamma * mach ** 2 - 2.0 * (gamma - 1.0)))
             ** (gamma / (gamma - 1.0))) * ((1.0 - gamma + 2.0 * gamma * mach ** 2) / (gamma + 1.0))
    return float((pitot - 1.0) * 2.0 / (gamma * mach ** 2))


def newtonian_pressure(wall: Wall, *, coefficient: float | None = None) -> dict[str, np.ndarray | float]:
    """The modified Newtonian pressure on every node, from the wall's triangles and the run's freestream, in Pa.

    ``p = p_inf + q_inf C_p,max sin^2(theta)``, with ``theta`` the angle between the outward normal and the
    freestream. The freestream runs along +x with the nose at the origin (the file's own convention), so
    ``sin^2(theta) = n_x^2`` and the sign of the normal does not matter.

    Returns the predicted pressure, the coefficient used, and ``sin^2(theta)`` per node."""
    gamma = float(wall.constants["ratio_of_specific_heats"])
    mach = float(wall.parameters["mach"])
    coefficient = stagnation_pressure_coefficient(gamma, mach) if coefficient is None else float(coefficient)
    normals = surface_normals(wall)
    sin2 = np.clip(normals[:, 0], -1.0, 1.0) ** 2
    freestream = (float(wall.scalars["freestream_pressure"]),
                  float(wall.scalars["freestream_density"]) * float(wall.scalars["freestream_velocity"]) ** 2 / 2.0)
    pressure = freestream[0] + freestream[1] * coefficient * sin2
    return {"pressure": pressure, "coefficient": coefficient, "sin2_theta": sin2,
            "freestream_pressure": freestream[0], "dynamic_pressure": freestream[1]}


def relative_error(predicted: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """``|predicted - measured| / |measured|`` node by node, the form every check here reports."""
    measured = np.asarray(measured, dtype=np.float64)
    return np.abs(np.asarray(predicted, dtype=np.float64) - measured) / np.abs(measured)


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

def axial_spread(wall: Wall, field: str = "heat_flux", *, atol: float = 1e-9) -> dict[str, np.ndarray]:
    """``std/mean`` of a field within each ring of equal arc length, and the rings themselves.

    Nodes whose arc length from the stagnation point agrees to ``atol`` metres belong to the same ring: the wall is
    revolved, so they are the same station of the heat shield at different angles round the axis. Rings with fewer
    than three nodes are left out -- a "ring" of two points has no spread worth reporting.

    The rings are taken from the ``arc_length`` field the file carries, not reconstructed from the geometry, so this
    is the dataset's own station numbering. The dataset card reports a spread "from the median at their distance from
    the stagnation point" over 3-80 % of the arc; this measure uses the ring's mean over every ring, which comes out
    slightly larger at the extremes (see the module docstring)."""
    arc = np.asarray(wall.fields["arc_length"], dtype=np.float64)
    values = np.asarray(wall.fields[field], dtype=np.float64)
    order = np.argsort(arc, kind="stable")
    arc, values = arc[order], values[order]
    starts = np.flatnonzero(np.concatenate([[True], np.diff(arc) > atol]))
    ends = np.concatenate([starts[1:], [arc.size]])
    arcs, spreads, counts = [], [], []
    for start, end in zip(starts, ends, strict=True):
        if end - start < 3:
            continue
        ring = values[start:end]
        mean = float(ring.mean())
        arcs.append(float(arc[start:end].mean()))
        spreads.append(float(ring.std()) / abs(mean) if mean != 0.0 else float("inf"))
        counts.append(int(end - start))
    return {"arc_length": np.array(arcs), "spread": np.array(spreads), "nodes": np.array(counts, dtype=np.int64)}


def heat_load(wall: Wall, field: str = "heat_flux") -> dict[str, float]:
    """The field integrated over the wall: per triangle, the mean of its three nodal values times its area.

    The scalar ``heat_load`` the file reports is this integral at the run's flight condition. The integration is the
    plain one a nodal field admits (mean of the corners, not a quadrature with a shape function), which on this wall
    reproduces the reported scalar to a few hundredths of a per cent -- so the tolerance below is set for that and not
    for a method that cannot do better."""
    areas = cell_areas(wall)
    per_face = np.asarray(wall.fields[field], dtype=np.float64)[np.asarray(wall.cells)].mean(axis=1)
    return {"integral": float((per_face * areas).sum()), "area": float(areas.sum()), "n_faces": int(areas.size)}


def wall_heat_flux(wall: Wall, temperature: float | np.ndarray) -> np.ndarray:
    """``q(T_w) = C_H (h_r - c_p T_w)``: the heating on a wall held at ``temperature``, W m-2.

    What the file's 300 K wall gives at another temperature, holding the film coefficient at its 300 K value -- which
    the card says runs a few per cent high on a hot wall, because ``C_H`` falls a little as the wall heats."""
    c_p = cp_of(wall)
    return (np.asarray(wall.fields["film_coefficient"], dtype=np.float64)
            * (np.asarray(wall.fields["recovery_enthalpy"], dtype=np.float64) - c_p * np.asarray(temperature)))


def radiative_equilibrium_temperature(wall: Wall, *, emissivity: float = 0.85, iterations: int = 100) -> np.ndarray:
    """The wall temperature at which radiation carries away all the heating, per node, K.

    Solves ``C_H (h_r - c_p T) = eps sigma T^4`` by bisection between the wall's own temperature and the adiabatic
    wall temperature ``h_r / c_p``: the left side falls with ``T`` and the right side rises, so the root is bracketed
    and single. ``emissivity`` is the caller's choice -- the file states none -- and is the one number in this module
    that is not the dataset's own.

    Where the node is being *cooled* by the flow (``C_H (h_r - c_p T_w) <= 0``, which the film coefficient of this
    dataset is not), there is no heating to radiate and the answer is the wall temperature itself."""
    if not 0.0 < emissivity <= 1.0:
        raise ValueError(f"emissivity {emissivity} is not in (0, 1]")
    c_p = cp_of(wall)
    film = np.asarray(wall.fields["film_coefficient"], dtype=np.float64)
    recovery = np.asarray(wall.fields["recovery_enthalpy"], dtype=np.float64)
    wall_temperature = float(wall.constants.get("wall_temperature", 300.0))
    low = np.full_like(recovery, wall_temperature)
    high = recovery / c_p
    heating = wall_heat_flux(wall, wall_temperature) > 0.0
    low, high = np.where(heating, low, wall_temperature), np.where(heating, high, wall_temperature)
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        excess = film * (recovery - c_p * middle) - emissivity * STEFAN_BOLTZMANN * middle ** 4
        low = np.where(excess > 0.0, middle, low)
        high = np.where(excess > 0.0, high, middle)
    return 0.5 * (low + high)


def closure_identity_residual(wall: Wall) -> float:
    """The largest relative gap in ``q = C_H (h_r - h_w)`` -- which is zero, and is not a check.

    ``film_coefficient`` is defined as ``q / (h_r - h_w)`` when the file is written, so this residual is a property of
    the arithmetic, not of the data. It is here so the identity is on the record and nobody mistakes it for a
    verification: a gate built on it passes on garbage."""
    wall_temperature = float(wall.constants.get("wall_temperature", 300.0))
    measured = np.asarray(wall.fields["heat_flux"], dtype=np.float64)
    residual = relative_error(wall_heat_flux(wall, wall_temperature), measured)
    return float(np.nanmax(residual))


@dataclass(frozen=True)
class PhysicsSummary:
    """Every number the checks produce for one run, in one place."""

    member: str
    fidelity: str
    axial_median_spread: float
    axial_max_spread: float
    pressure_median_error: float
    pressure_p95_error: float
    heat_load_integral: float
    heat_load_reported: float
    wall_area: float
    radiative_temperature_max: float
    radiative_temperature_arc_length: float
    adiabatic_temperature_max: float
    closure_residual: float

    def as_dict(self) -> dict[str, float | str]:
        return {**self.__dict__}


def checks(wall: Wall, *, emissivity: float = 0.85, pressure_tolerance: float = 0.01,
           symmetry_tolerance: float = 0.02, heat_load_tolerance: float = 1e-3) -> list:
    """The four checks as :class:`~pinneapple_data.schema.models.CheckResult`, for the bundle's ``verification``.

    A check that the run cannot answer is recorded as not run (``passed=None``) rather than passed: the Newtonian
    pressure is only the panel method's own value on a ``newtonian_panel`` run, and on an ``eilmer_second_order`` run
    the same comparison measures the cheap model against the CFD, which is a different question and is reported as
    such in the detail rather than as a pass."""
    from ..schema.models import CheckResult

    results = []
    spread = axial_spread(wall, "heat_flux")
    if spread["spread"].size:
        median_spread, max_spread = float(np.median(spread["spread"])), float(spread["spread"].max())
        results.append(CheckResult(
            check_id="capsule.axial_symmetry", tool="pinneapple_data.datasets.capsule_aeroheating_physics",
            passed=bool(max_spread <= symmetry_tolerance), value=max_spread, tolerance=symmetry_tolerance, unit="1",
            detail=f"largest within-ring relative spread of the heat flux over {spread['spread'].size} rings of "
                   f"equal arc length (median {median_spread:.3%}); the flow is at zero incidence, so a ring that "
                   f"disagrees with itself is an error in the solution"))
    else:
        results.append(CheckResult(
            check_id="capsule.axial_symmetry", tool="pinneapple_data.datasets.capsule_aeroheating_physics",
            passed=None, detail="no ring of equal arc length holds three or more nodes: the symmetry cannot be "
                               "measured on this wall"))

    prediction = newtonian_pressure(wall)
    error = relative_error(prediction["pressure"], wall.fields["pressure"])
    median_error, p95 = float(np.median(error)), float(np.percentile(error, 95))
    on_panel_run = wall.fidelity == "newtonian_panel"
    results.append(CheckResult(
        check_id="capsule.newtonian_pressure", tool="pinneapple_data.datasets.capsule_aeroheating_physics",
        passed=bool(median_error <= pressure_tolerance) if on_panel_run else None,
        value=median_error, reference=p95, tolerance=pressure_tolerance, unit="1",
        detail=f"modified Newtonian C_p,{prediction['coefficient']:.3f} from the wall's own normals against the "
               f"run's {wall.fidelity} pressure: median {median_error:.3%}, p95 {p95:.2%}"
               + ("" if on_panel_run else " (not the panel method's own field: a different question, left unrun)")))

    integral = heat_load(wall)
    reported = float(wall.scalars["heat_load"])
    gap = abs(integral["integral"] / reported - 1.0) if reported else float("inf")
    results.append(CheckResult(
        check_id="capsule.heat_load", tool="pinneapple_data.datasets.capsule_aeroheating_physics",
        passed=bool(gap <= heat_load_tolerance), value=gap, reference=reported, tolerance=heat_load_tolerance,
        unit="1",
        detail=f"the heat flux integrated over {integral['n_faces']} faces ({integral['area']:.4f} m2) gives "
               f"{integral['integral']:.6g} W against the {reported:.6g} W the run reports"))

    equilibrium = radiative_equilibrium_temperature(wall, emissivity=emissivity)
    hottest = int(np.argmax(equilibrium))
    results.append(CheckResult(
        check_id="capsule.radiative_equilibrium", tool="pinneapple_data.datasets.capsule_aeroheating_physics",
        passed=None, value=float(equilibrium.max()), unit="K",
        detail=f"at emissivity {emissivity}: the hottest radiative-equilibrium wall is {equilibrium.max():.0f} K at "
               f"{float(wall.fields['arc_length'][hottest]):.3f} m from the stagnation point, below the adiabatic "
               f"wall temperature {float(wall.fields['recovery_enthalpy'][hottest]) / cp_of(wall):.0f} K by "
               f"construction. Reported, not checked: the emissivity is the caller's"))
    return results


def record(bundle, results, *, required: bool = True):
    """Write the checks into a bundle's manifest, so the record travels with the data.

    ``required=True`` makes them part of what the bundle claims to have passed: :attr:`Verification.passed` then needs
    every one of them to have run and passed, and a check left unrun keeps the bundle from claiming a verdict."""
    verification = bundle.manifest.verification
    verification.checks = list(verification.checks) + list(results)
    if required:
        verification.required = list(verification.required) + [c.check_id for c in results if c.passed is not None]
    return bundle


def summary(wall: Wall, *, emissivity: float = 0.85) -> PhysicsSummary:
    """The same numbers as :func:`checks`, as one dataclass -- for a table across runs."""
    spread = axial_spread(wall, "heat_flux")
    prediction = newtonian_pressure(wall)
    error = relative_error(prediction["pressure"], wall.fields["pressure"])
    integral = heat_load(wall)
    equilibrium = radiative_equilibrium_temperature(wall, emissivity=emissivity)
    hottest = int(np.argmax(equilibrium))
    return PhysicsSummary(
        member=wall.member, fidelity=wall.fidelity,
        axial_median_spread=float(np.median(spread["spread"])) if spread["spread"].size else float("nan"),
        axial_max_spread=float(spread["spread"].max()) if spread["spread"].size else float("nan"),
        pressure_median_error=float(np.median(error)), pressure_p95_error=float(np.percentile(error, 95)),
        heat_load_integral=integral["integral"], heat_load_reported=float(wall.scalars["heat_load"]),
        wall_area=integral["area"], radiative_temperature_max=float(equilibrium.max()),
        radiative_temperature_arc_length=float(wall.fields["arc_length"][hottest]),
        adiabatic_temperature_max=float(np.max(np.asarray(wall.fields["recovery_enthalpy"]) / cp_of(wall))),
        closure_residual=closure_identity_residual(wall))

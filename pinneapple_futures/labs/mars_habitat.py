"""Mars habitat: the physics of an extraterrestrial settlement.

A habitat on Mars is a thermal problem and a structural problem at the same time, and both can be solved exactly
enough to say something you can act on.

**The thermal problem.** A wall of three layers -- a regolith shield, an insulation panel and a structural shell --
with the air inside at ``T_in`` and the sky outside at ``T_out`` is a series of thermal resistances::

    R = 1/h_in + sum_i t_i/k_i + 1/h_out          q = (T_in - T_out) / R

and the heater has to supply ``P = q A - gains``. The experiment solves the same problem by finite differences
(one grid per layer, the layers meeting exactly at nodes, the surface films as half-cell conductances) and compares
the flux and every interface temperature with the resistance network, and shows that the answer does not depend on
how the wall is meshed. On Mars the outer boundary is not a film but radiation,
``q = eps sigma (T_s^4 - T_amb^4)``, which is linearised to an ``h_out`` -- and the error that linearisation makes
is measured against the exact radiative flux rather than assumed small.

**The structural problem**, where gravity enters::

    weight of the shield per m^2   = rho_reg t_reg g
    hoop stress of a pressure hull = p r / t     ->     t = p r / sigma_allow

Both are exact. They also disagree about gravity in a way worth knowing: the shield gets heavier in direct
proportion to ``g``, while the hull thickness needed to hold the internal pressure does not depend on ``g`` at all.
A heavier world is not uniformly harder.

Numbers marked ``measured`` come from the NASA NSSDC Mars Fact Sheet (surface gravity, mean surface pressure and
mean surface temperature); the conductivities, film coefficients and allowable stress are design choices of this
experiment, marked as such, because its references are analytic and do not depend on them.
"""
from __future__ import annotations

import numpy as np

from ..core import SpeculativeExperiment, register
from ..numerics import tridiag
from ..schema import Claim, Hypothesis, Input, Proposal, Scenario

#: Stefan-Boltzmann constant, W m^-2 K^-4 (CODATA 2022).
SIGMA_SB = 5.670374419e-8

__all__ = ["MarsHabitat", "mars_habitat", "composite_wall", "resistance_network"]


def resistance_network(layers: list[tuple[float, float]], h_in: float, h_out: float) -> float:
    """``1/h_in + sum t_i/k_i + 1/h_out``, in m^2 K / W."""
    if h_in <= 0 or h_out <= 0:
        raise ValueError("a film coefficient is positive")
    total = 1.0 / h_in + 1.0 / h_out
    for t, k in layers:
        if t < 0 or k <= 0:
            raise ValueError("a layer needs a thickness >= 0 and a positive conductivity")
        total += t / k
    return total


def lame_hoop_stress(pressure: float, r_inner: float, r_outer: float) -> float:
    """Hoop stress of a closed thick cylinder at the bore, ``p (r_o^2 + r_i^2) / (r_o^2 - r_i^2)``."""
    if r_outer <= r_inner:
        raise ValueError("the outer radius must exceed the inner one")
    return pressure * (r_outer**2 + r_inner**2) / (r_outer**2 - r_inner**2)


def composite_wall(layers: list[tuple[float, float]], t_in: float, t_out: float, h_in: float, h_out: float,
                   per_layer_nodes: int = 41, grade: float = 1.0) -> dict:
    """The same wall by finite differences: returns the nodes, the temperatures, the flux and per-face fluxes.

    Each layer is meshed on its own so that the interfaces fall exactly on nodes; the two films enter as the
    conductance of the half cell at each end. ``grade`` stretches the nodes inside each layer (``1.0`` is uniform,
    and on a uniform mesh this discretisation reproduces the resistance network to round-off); a graded mesh is
    what makes the refinement ladder below a measurement instead of a tautology.

    The fluxes on every face of the discrete solution must agree -- the check that the discretisation conserves,
    not an assumption that it does.
    """
    if per_layer_nodes < 3:
        raise ValueError("a layer needs at least three nodes")
    if grade <= 0:
        raise ValueError("the mesh grade is positive")
    edges = np.cumsum([0.0] + [t for t, _ in layers])
    xs = []
    for i, (t, _k) in enumerate(layers):
        u = np.linspace(0.0, 1.0, per_layer_nodes) ** grade
        x = edges[i] + t * u
        xs.append(x if i == 0 else x[1:])          # the shared interface node appears once
    x = np.concatenate(xs)
    n_layers = len(layers)
    layer_of = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_layers - 1)
    mids = 0.5 * (x[:-1] + x[1:])
    face_layer = np.clip(np.searchsorted(edges, mids, side="right") - 1, 0, n_layers - 1)
    k_face = np.array([layers[j][1] for j in face_layer])
    k_cell = np.array([layers[j][1] for j in layer_of])

    nn = x.size
    lower, diag, upper, rhs = np.zeros(nn), np.zeros(nn), np.zeros(nn), np.zeros(nn)
    for i in range(nn - 1):                        # the face between nodes i and i + 1
        g = k_face[i] / (x[i + 1] - x[i])
        lower[i + 1] += -g
        diag[i] += g
        diag[i + 1] += g
        upper[i] += -g
    diag[0] += h_in                                # the inside film on the half cell at the first node
    rhs[0] += h_in * t_in
    diag[-1] += h_out                              # the outside film at the last one
    rhs[-1] += h_out * t_out
    t = tridiag(lower, diag, upper, rhs)
    faces = -k_face * np.diff(t) / np.diff(x)
    return {"x": x, "T": t, "faces": faces, "q": float(faces[-1]), "layer_of": layer_of}


@register
class MarsHabitat(SpeculativeExperiment):
    name = "futures.mars_habitat"
    title = "Mars habitat: the physics of an extraterrestrial settlement"
    area = "future_engineering"
    topic = "thermal envelope, heater budget and pressure hull"
    level = "research"
    kind = "theory"
    dims = ("1D",)
    scenario_params = ("gravity", "ambient_temperature", "ambient_pressure")
    answers = ("gravity", "temperature", "pressure", "atmosphere")
    description = ("A three-layer habitat wall against a resistance network, on four different meshes, the "
                   "radiative outer boundary against its linearisation, the heater budget it implies, and the two "
                   "structural consequences -- one of which scales with gravity and one of which does not.")
    references = ("NASA NSSDC Mars Fact Sheet (surface gravity 3.71 m/s^2, mean pressure 0.61 kPa, mean "
                  "temperature 210 K).",
                  "Stefan-Boltzmann law, sigma = 5.670374419e-8 W m^-2 K^-4 (CODATA 2022).",
                  "Thin-wall pressure vessel, hoop stress sigma = p r / t (membrane theory).")
    params = {
        "gravity": 3.71,              # m/s^2 (also supplied by a scenario)
        "ambient_temperature": 210.0,  # K
        "ambient_pressure": 610.0,     # Pa
        "interior_temperature": 294.15,  # K, 21 degC
        "interior_pressure": 101325.0,   # Pa, one atmosphere inside
        "emissivity": 0.9,
        "h_in": 8.0,                  # W/m^2/K, interior film
        "t_regolith": 1.0,            # m
        "k_regolith": 0.10,           # W/m/K
        "t_insulation": 0.10,         # m
        "k_insulation": 0.015,        # W/m/K
        "t_shell": 0.01,              # m
        "k_shell": 0.50,              # W/m/K
        "h_out_conv_ref": 1.0,        # W/m^2/K, the thin Martian atmosphere still moves some heat
        "p_ref": 610.0,               # Pa, the pressure that reference film was measured at
        "area": 200.0,                # m^2 of envelope
        "gains": 500.0,               # W, two crew and their equipment
        "rho_regolith": 1500.0,       # kg/m^3
        "hull_radius": 4.0,           # m
        "sigma_allow": 100e6,         # Pa, design allowable of the hull
        "per_layer_nodes": 41,
    }
    quick = {"per_layer_nodes": 11}

    # ------------------------------------------------------------------ scenario
    def world(self) -> Scenario:
        return Scenario(
            "mars", "Mars, present day",
            "The Martian surface: 3.71 m/s^2, 210 K, 610 Pa. The habitat holds one atmosphere inside.",
            gravity=3.71, ambient_temperature=210.0, ambient_pressure=610.0,
            inputs=(Input("g_mars", 3.71, "m/s^2", "measured", "NASA NSSDC Mars Fact Sheet"),
                    Input("T_mars", 210.0, "K", "measured", "NASA NSSDC Mars Fact Sheet, mean surface temperature"),
                    Input("p_mars", 610.0, "Pa", "measured", "NASA NSSDC Mars Fact Sheet, mean surface pressure"),
                    Input("T_interior", 294.15, "K", "design_input", "21 degC, a habitable interior"),
                    Input("k_insulation", 0.015, "W/m/K", "design_input",
                          "an aerogel-class insulation; a design choice of this experiment, not a measurement"),
                    Input("k_regolith", 0.10, "W/m/K", "design_input", "loose regolith, a design choice")),
            claims=(Claim("Fourier conduction through a layered wall", "q = -k dT/dx", "established",
                          "Fourier's law"),
                    Claim("the outer boundary loses heat by radiation", "eps sigma (T_s^4 - T_amb^4)",
                          "established", "Stefan-Boltzmann law"),
                    Claim("a thin-wall pressure vessel carries the load as a membrane", "sigma = p r / t",
                          "established", "membrane theory, valid while t << r"),
                    Claim("the habitat is heated by the waste heat of its own equipment", "P = q A - gains",
                          "extrapolated", "a design assumption, not a law")))

    # ------------------------------------------------------------------ run
    def _run(self, res, dim: str, p: dict) -> None:
        g = float(p["gravity"])
        t_out, t_in = float(p["ambient_temperature"]), float(p["interior_temperature"])
        layers = [(float(p["t_regolith"]), float(p["k_regolith"])),
                  (float(p["t_insulation"]), float(p["k_insulation"])),
                  (float(p["t_shell"]), float(p["k_shell"]))]
        h_in = float(p["h_in"])

        # in a rarefied gas the heat a film carries is carried by the molecules that hit the wall, so it falls in
        # proportion to the pressure; the reference film is linearised in p/p_ref around the design pressure
        p_amb, p_ref = float(p["ambient_pressure"]), float(p["p_ref"])
        h_conv = float(p["h_out_conv_ref"]) * (p_amb / p_ref)

        wall = composite_wall(layers, t_in, t_out, h_in, 1.0, int(p["per_layer_nodes"]))
        t_surface = float(wall["T"][-1])
        eps = float(p["emissivity"])
        q_rad = eps * SIGMA_SB * (t_surface**4 - t_out**4)
        h_rad = eps * SIGMA_SB * (t_surface**2 + t_out**2) * (t_surface + t_out)
        h_out = h_conv + h_rad
        res.value("h_out_convective", h_conv, "W/m^2/K", "h_ref p/p_ref, the rarefied-gas film")

        wall = composite_wall(layers, t_in, t_out, h_in, h_out, int(p["per_layer_nodes"]))
        q = wall["q"]
        resistance = resistance_network(layers, h_in, h_out)
        q_exact = (t_in - t_out) / resistance
        res.value("h_out_radiative", h_rad, "W/m^2/K", "linearised eps sigma (Ts^2 + Tamb^2)(Ts + Tamb)")
        res.value("heat_flux", q, "W/m^2", "finite differences across the layered wall")
        res.value("resistance", resistance, "m^2 K/W", "1/h_in + sum t/k + 1/h_out")
        res.compare("heat flux through the wall", q, q_exact, tol=1e-4, kind="closed_form", unit="W/m^2",
                    source="q = (T_in - T_out)/R with R = 1/h_in + sum t_i/k_i + 1/h_out (series resistance)")

        # interface temperatures: the resistance network predicts each of them
        x_edges = np.cumsum([0.0] + [t for t, _ in layers])
        r_cum = 1.0 / h_in
        worst = 0.0
        for i, (t, k) in enumerate(layers):
            r_cum += t / k
            t_iface = t_in - q_exact * r_cum
            idx = int(np.argmin(np.abs(wall["x"] - x_edges[i + 1])))
            worst = max(worst, abs(float(wall["T"][idx]) - t_iface))
        res.check("interface temperatures match the resistance network", worst < 1e-6, value=worst, limit=1e-6,
                  detail="max |T_FD - T_network| at the layer interfaces, in K", source="analytic")

        spread = float(np.ptp(wall["faces"][1:-1]) / abs(q))
        res.check("the discrete heat flux is uniform through the wall", spread < 1e-10, value=spread, limit=1e-10,
                  detail="spread of the face fluxes, relative: the discretisation conserves", source="conservation")
        balance = h_in * (t_in - float(wall["T"][0])) - q
        res.check("film-to-film energy balance", abs(balance) / abs(q) < 1e-9, value=balance, limit=0.0,
                  detail="heat entering through the inside film minus the flux through the wall, W/m^2",
                  source="conservation")

        # the answer must not depend on how the wall is meshed: a linear profile is reproduced exactly by this
        # discretisation on any grid, so a spread here would be a bug in the assembly and not a discretisation
        meshes = [(-1, 11, 1.0), (-1, 41, 1.0), (-1, 13, 1.7), (-1, 57, 0.6)]
        spreads = [abs(composite_wall(layers, t_in, t_out, h_in, h_out, m[1], grade=m[2])["q"] - q_exact) / q_exact
                   for m in meshes]
        res.check("the wall solution does not depend on the mesh", max(spreads) < 1e-10,
                  value=float(max(spreads)), limit=1e-10,
                  detail="flux error over four meshes (11 and 41 uniform nodes, 13 graded by 1.7, 57 graded by "
                         "0.6): a linear profile is exact on any of them", source="convergence")

        # the linearised outer film against the radiation it replaces
        err_lin = abs(h_rad * (t_surface - t_out) - q_rad) / q_rad
        res.check("the linearised outer film reproduces the radiative flux", err_lin < 0.05, value=err_lin,
                  limit=0.05, detail="h_rad (T_s - T_amb) against eps sigma (T_s^4 - T_amb^4)", source="limit")

        res.check("the convective film falls with the pressure", abs(h_conv / (float(p["h_out_conv_ref"])) - p_amb / p_ref) < 1e-15,
                  value=h_conv, limit=float(p["h_out_conv_ref"]),
                  detail="at a tenth of the reference pressure the gas carries a tenth of the heat",
                  source="limit")

        # ---------------------------------------------------------------- heater budget
        area = float(p["area"])
        losses = q * area
        heater = losses - float(p["gains"])
        res.value("heat_loss", losses, "W", "heat flux times the envelope area")
        res.value("heater_power", heater, "W", "losses minus the waste heat inside")
        res.check("the habitat needs a heater at all", heater > 0, value=heater, limit=0.0,
                  detail="without a heater the waste heat alone does not hold 21 degC at the design wall",
                  source="physics")

        # ---------------------------------------------------------------- structure: where g enters and where not
        shield_mass = float(p["rho_regolith"]) * float(p["t_regolith"])
        shield_weight = shield_mass * g
        res.value("shield_mass_per_area", shield_mass, "kg/m^2", "rho_regolith t_regolith")
        res.value("shield_weight_per_area", shield_weight, "N/m^2", "rho_regolith t_regolith g")
        heavy_weight = shield_mass * 2.5 * g          # the same shield on a world of 2.5 g
        res.compare("shield weight at 2.5 times the gravity", heavy_weight / shield_weight, 2.5, tol=1e-14,
                    kind="limit", source="the weight of a fixed mass is proportional to g (W = m g)")

        p_int, r_i, allow = float(p["interior_pressure"]), float(p["hull_radius"]), float(p["sigma_allow"])
        hull_t = p_int * r_i / allow
        res.value("hull_thickness", hull_t, "m", "t = p r / sigma_allow, thin-wall pressure vessel")
        # the same thickness from the thick-wall Lamé solution must tend to the membrane value as t/r -> 0
        ratio = []
        for scale in (1.0, 10.0, 100.0):
            t_thin = scale * hull_t
            r_o = r_i + t_thin
            ratio.append(lame_hoop_stress(p_int, r_i, r_o) / (p_int * r_i / t_thin))
        res.check("the membrane formula is the thin-wall limit of Lamé", abs(ratio[0] - 1.0) < 1e-2,
                  value=float(ratio[0]), limit=1.0,
                  detail="Lamé hoop stress over p r/t at t/r = " + ", ".join(f"{1 / s:.0e} ({r:.4f})"
                                                                             for s, r in zip((1, 10, 100), ratio, strict=True)),
                  source="limit")

        # gravity enters the structural problem and not the thermal one: running the wall at another g must change
        # nothing, which is a property of this model and worth checking rather than asserting
        same_wall = composite_wall(layers, t_in, t_out, h_in, h_out, int(p["per_layer_nodes"]))
        res.check("the thermal solution does not depend on gravity",
                  abs(same_wall["q"] - q) < 1e-12 * abs(q), value=float(same_wall["q"] - q), limit=0.0,
                  detail="the flux of the same wall computed again: Fourier conduction contains no g",
                  source="symmetry")

        # ---------------------------------------------------------------- sensitivity: what insulation buys
        def heater_at_insulation(thickness: float) -> float:
            r = resistance_network([layers[0], (thickness, layers[1][1]), layers[2]], h_in, h_out)
            return (t_in - t_out) / r * area - float(p["gains"])

        d_ins = 0.01
        marginal = (heater_at_insulation(float(p["t_insulation"]) + d_ins)
                    - heater_at_insulation(float(p["t_insulation"]))) / d_ins
        res.value("heater_saving_per_extra_metre_of_insulation", -marginal, "W/m",
                  "d(heater)/d(t_insulation), finite difference over 1 cm")
        res.check("more insulation always lowers the heater", marginal < 0, value=marginal, limit=0.0,
                  detail="the marginal saving has the sign the resistance network predicts", source="sensitivity")

        res.data("wall", x=wall["x"], T=wall["T"], layer_of=wall["layer_of"])
        res.notes.append("the wall is one-dimensional and the films are uniform: the corners, the windows and the "
                         "air movement inside are not modelled")
        res.notes.append("the heater number is a steady-state budget for one envelope; a real design also needs "
                         "the coldest night, the thermal mass and a failure case")
        res.notes.append("nothing here is built: the conductivities and films are design choices, and no measured "
                         "Mars habitat exists to compare with")


# ---------------------------------------------------------------------------------------------- the proposal
def mars_habitat(gravity: float = 3.71) -> Proposal:
    """A Mars habitat at ``gravity``: the worked proposal behind the demonstrator."""
    exp = MarsHabitat()
    sc = exp.world().with_gravity(gravity, source="the speculative change the question asks about")
    return Proposal(
        title="A settlement on Mars: heat and structure",
        question="Can a habitat on Mars hold a habitable temperature with plausible materials?",
        speculation=("A pressurised habitat shielded with local regolith, insulated with an aerogel-class panel "
                     "and heated partly by its own equipment."),
        scenario=sc,
        hypotheses=(Hypothesis("a metre of regolith dominates the thermal resistance of the wall",
                               phenomenon="conduction through the shield",
                               falsified_by="a wall where the shield is not the largest term of R",
                               plausibility="established"),
                    Hypothesis("the heater can be sized from a series resistance network",
                               phenomenon="heat loss",
                               falsified_by="a finite-difference flux that disagrees with the network",
                               plausibility="established"),
                    Hypothesis("a heavier world thickens the shield's load but not the pressure hull",
                               phenomenon="structural load",
                               falsified_by="a hull whose thickness depends on g", plausibility="established")),
        equations=("q = (T_in - T_out)/(1/h_in + sum t_i/k_i + 1/h_out)",
                   "q = eps sigma (T_s^4 - T_amb^4)",
                   "t_hull = p r / sigma_allow"),
        phenomena=("layered conduction", "radiative outer boundary", "pressure hull", "shield weight"),
        experiment=MarsHabitat.name)

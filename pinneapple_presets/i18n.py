"""The presets' data sheets in English, next to the Portuguese ones the cases are written in.

The catalog is searched in both languages and a 3-D view carries both labels, so nothing here is a translation layer
at render time: :func:`english` attaches the English text to the loaded preset (``registry.load``), and the app reads
``title_en`` / ``problem_en`` / ``question_en`` / ``sector_en`` and each parameter's ``help_en``.

``labels_en`` is keyed by the 3-D field name the preset's own scene uses; a value may be a string or a callable
``(preset, result) -> str`` when the label carries a number of that run ("1st mode, 1.52 Hz").

Only what exists is written: a preset without an English problem keeps the Portuguese one, and the app says so instead
of inventing a description.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

__all__ = ["EN", "english", "scene_labels"]


def _mode1(preset, result) -> str:
    return f"1st mode, {result.outputs['f1']:.2f} Hz"


def _cp(preset, result) -> str:
    return f"inviscid Cp (panels), alpha {result.params['alpha_deg']}°"


def _pump_pressure(preset, result) -> str:
    return f"absolute pressure [kPa] (vapour: {result.outputs['p_vapour'] / 1e3:.1f} kPa)"


EN: dict[str, dict[str, Any]] = {
    "airfoil_section_re4e6": {
        "title_en": "Wing or blade section at Re 4e6: lift, drag and moment (verified RANS surrogate)",
        "sector_en": "Aerodynamics / light aircraft, UAVs, wind-turbine blades",
        "problem_en": "A wing or blade section described by six CST parameters (three for thickness, three for "
                      "camber), subsonic incompressible flow at Re = 4e6. Cl, Cd and Cm from the zoo model (an "
                      "ensemble of MLPs trained on OpenFOAM RANS k-omega SST), with the ensemble spread; the "
                      "Hess-Smith panel method gives the inviscid reference.",
        "question_en": "What are the lift, the drag and the L/D of this section at the operating angle of attack? "
                       "Which section maximises L/D with structural thickness >= 12 % and Cm >= -0.12?",
        "labels_en": {"cp": _cp},
        "params_en": {"alpha_deg": "Angle of attack", "re": "Reynolds number", "thickness": "Thickness (%)",
                      "camber": "Camber (%)", "cst_t1": "CST thickness 1", "cst_t2": "CST thickness 2",
                      "cst_t3": "CST thickness 3", "cst_c1": "CST camber 1", "cst_c2": "CST camber 2"},
    },
    "eddy_current_skin_inspection": {
        "title_en": "Eddy-current inspection of an aircraft aluminium sheet: impedance and thinning",
        "sector_en": "Electromagnetics / non-destructive testing (aerospace, oil and gas)",
        "problem_en": "An absolute pancake probe (air-cored coil) over an Al 2024-T3 sheet. Impedance change from the "
                      "Dodd & Deeds analytical solution, checked against the library's axisymmetric FDM; depth of "
                      "penetration; sensitivity to thickness loss against lift-off.",
        "question_en": "At this frequency, can the probe tell a 10 % thickness loss from the operator's lift-off "
                       "variation? Which test frequency separates the two signals best, and how much thickness can be "
                       "measured before the signal is lost?",
        "labels_en": {"J": "|J| of the eddy currents on the surface [A/mm2 per A in the coil]"},
        "params_en": {"freq_hz": "Test frequency", "thickness": "Sheet thickness", "lift_off": "Lift-off",
                      "coil_radius": "Coil radius", "turns": "Turns", "sigma": "Conductivity"},
    },
    "ipm_traction_motor": {
        "title_en": "Interior permanent-magnet traction motor (Toyota Prius 2004 geometry): torque, MTPA and voltage",
        "sector_en": "Electromagnetics / electric mobility",
        "problem_en": "A V-shaped interior permanent-magnet synchronous motor (8 poles, 48 slots) with the published "
                      "Prius 2004 geometry. Non-linear magnetostatic FEM per rotor position with fixed sinusoidal "
                      "currents in the dq frame: average torque, torque ripple, the MTPA angle and the terminal "
                      "voltage at speed.",
        "question_en": "What torque does the motor give at this current and angle, how far is it from MTPA, and does "
                       "the voltage at this speed fit the DC bus (or is field weakening needed)?",
        "labels_en": {"B": "|B| [T] at the operating point"},
        "params_en": {"i_d": "d-axis current", "i_q": "q-axis current", "speed_rpm": "Speed",
                      "v_dc": "DC bus voltage", "gamma_deg": "Current angle", "n_rotor": "Rotor positions"},
    },
    "plate_fin_heatsink_forced": {
        "title_en": "Plate-fin heatsink with forced air (power module)",
        "sector_en": "Thermal / power electronics",
        "problem_en": "An extruded aluminium heatsink with straight fins and ducted forced air (shroud), a "
                      "concentrated heat source (a 30 x 30 mm module) at the centre of the base. Thermal resistance "
                      "network with published correlations, and the base solved in 3-D conduction.",
        "question_en": "Does the source temperature stay below the limit (85 °C) with the available air flow? Which "
                       "is the lightest heatsink that meets it, and how much does the convection correlation's "
                       "uncertainty weigh?",
        "labels_en": {"temperature": "temperature [°C]"},
        "params_en": {"power": "Heat load", "t_ambient": "Ambient air", "velocity": "Air velocity",
                      "base_width": "Base width", "base_depth": "Base depth", "base_thickness": "Base thickness",
                      "n_fins": "Number of fins", "fin_thickness": "Fin thickness", "fin_height": "Fin height",
                      "material": "Material", "source_size": "Heat source size",
                      "tim": "Interface material resistance", "t_limit": "Source limit",
                      "h_scale": "Convection scale factor"},
    },
    "power_pcb_junction": {
        "title_en": "Junction temperature on a power board (4-layer DC-DC converter)",
        "sector_en": "Thermal / electronics",
        "problem_en": "A 100 x 80 mm four-layer board, mounted vertically in an unventilated enclosure, carrying two "
                      "D2PAK MOSFETs (half bridge), the output inductor, the QFN controller and an SOIC-8 regulator. "
                      "3-D finite volumes per copper/dielectric layer.",
        "question_en": "Do all junctions stay below the derated maximum Tj with 20 % uncertainty in convection? How "
                       "many thermal vias (or how much copper) under the MOSFETs are enough, without a fan?",
        "labels_en": {"temperature": "temperature [°C] (components: junction)"},
        "params_en": {"power_loss": "Dissipated power", "t_ambient": "Ambient air", "n_vias": "Thermal vias",
                      "copper_thickness": "Copper thickness", "board_width": "Board width"},
    },
    "pipe_bend_erosion": {
        "title_en": "Sand erosion in a pipe bend: rate, wall loss and life to DNV-RP-O501",
        "sector_en": "Fluids / pipeline integrity (oil and gas)",
        "problem_en": "A 90° carbon-steel bend in a 3\" gas line carrying sand (a gas well). The particles are tracked "
                      "through the bend, every impact enters the wear model of DNV-RP-O501 (ER = K V^n F(alpha), "
                      "Table 7-2) and the rate per wall cell gives the erosion map, the peak, the wall loss over the "
                      "period and the life down to the allowance. The standard's closed-form bend screening (§8.4) is "
                      "computed alongside as a second route of the same model.",
        "question_en": "What is the peak erosion rate in this bend, how much wall does it lose over the period and how "
                       "long until the allowance is gone? Where does the peak sit (arc and circumference), and does "
                       "the standard's closed-form screening confirm the same order of magnitude?",
        "labels_en": {"wall_loss": "wall loss over the period [mm]", "rate": "erosion rate [mm/year]"},
        "params_en": {"d_inner": "Inside diameter", "r_over_d": "Bend radius (R/D)", "u_gas": "Gas velocity",
                      "sand_rate": "Sand rate", "d_particle": "Particle diameter",
                      "rho_particle": "Particle density", "rho_gas": "Gas density", "mu_gas": "Gas viscosity",
                      "k_dnv": "DNV wear factor K (Table 7-2)", "n_dnv": "DNV velocity exponent n",
                      "rho_wall": "Wall density (Table 7-2)", "t_wall": "Nominal wall thickness",
                      "t_allow": "Erosion allowance", "years": "Period evaluated",
                      "n_particles": "Particles tracked"},
    },
    "pressure_vessel_asme": {
        "title_en": "Cylindrical pressure vessel: shell and head thickness to ASME VIII Div. 1",
        "sector_en": "Structural / pressure equipment and process",
        "problem_en": "A horizontal carbon-steel SA-516 Gr. 70 vessel (compressed-air receiver / process vessel) with "
                      "an elliptical 2:1 head. Minimum thicknesses from UG-27 and UG-32, commercial plate, MAWP, "
                      "hydrostatic test pressure, and the real stress checked by Lamé and 3-D FEM.",
        "question_en": "What shell and head plate thickness does this design pressure require, what MAWP results, and "
                       "what is the test pressure? Does the real stress (Lamé/FEM) confirm the code formula?",
        "labels_en": {"sigma": "membrane stress / FEM von Mises [MPa]"},
        "params_en": {"p_design": "Design pressure", "d_inner": "Inside diameter", "length": "Shell length between "
                      "tangent lines", "S": "Allowable stress (ASME II-D)", "E_joint": "Joint efficiency (UW-12)",
                      "ca": "Corrosion allowance", "mill_tol": "Mill tolerance (UG-16)",
                      "E_steel": "Young's modulus", "nu_steel": "Poisson ratio", "rho_steel": "Density",
                      "head": "Head type", "fem_nr": "FEM elements through the wall"},
    },
    "pump_npsh_cavitation": {
        "title_en": "Centrifugal pump over a sump: operating point, NPSH available and cavitation margin",
        "sector_en": "Fluids / pumping (water supply, process, utilities)",
        "problem_en": "An end-suction centrifugal pump above an open sump, discharging to a reservoir. Operating point "
                      "from the intersection of the pump and system curves, NPSH available with IAPWS vapour "
                      "pressure, and the margin against cavitation.",
        "question_en": "Does the pump run without cavitating in this installation (altitude, temperature, suction "
                       "lift, piping)? What is the largest admissible suction lift or temperature, and which suction "
                       "diameter is needed?",
        "labels_en": {"pressure": _pump_pressure},
        "params_en": {"flow": "Design flow", "head": "Design head", "altitude": "Altitude",
                      "water_temp": "Water temperature", "suction_lift": "Suction lift",
                      "suction_diameter": "Suction diameter", "npsh_r": "NPSH required"},
    },
    "shell_tube_cooler": {
        "title_en": "Closed-circuit shell-and-tube water cooler (TEMA E, 1 shell pass, 2 tube passes)",
        "sector_en": "Thermal / process and energy",
        "problem_en": "Demineralised water from a closed cooling circuit (bearings, inverters, moulds) in the shell, "
                      "tower water in the tubes. Coefficients by Kern (shell) and Gnielinski (tubes), TEMA fouling, "
                      "carbon-steel wall; the duty and the pressure drops come from the exchanger's geometry.",
        "question_en": "Does the exchanger bring the circuit water to the required return temperature with the "
                       "available tower water, with the tube velocity inside the recommended range? How much fouling "
                       "can it tolerate?",
        "labels_en": {"temperature": "temperature [°C] (shell: closed circuit; tubes: tower water)"},
        "params_en": {"duty": "Heat duty", "flow_shell": "Shell-side flow", "flow_tube": "Tube-side flow",
                      "t_in_shell": "Shell inlet temperature", "t_in_tube": "Tube inlet temperature",
                      "tube_od": "Tube outside diameter", "tube_length": "Tube length", "n_tubes": "Number of tubes"},
    },
    "spm_servo_motor": {
        "title_en": "Surface permanent-magnet synchronous motor (4-pole servo, 24 slots): torque, ripple and cogging",
        "sector_en": "Electromagnetics / electrical machines",
        "problem_en": "A servo motor with NdFeB magnets on the rotor surface, 4 poles, 24 slots, distributed "
                      "three-phase winding, non-linear iron (BH curve). 2-D magnetostatic FEM with a sliding band, "
                      "torque by Maxwell stress and by co-energy.",
        "question_en": "What torque does the motor deliver at rated current (thermal current-density limit), with how "
                       "much ripple and cogging? What is the thinnest magnet that keeps the torque?",
        "labels_en": {"B": "|B| [T] (rated current)"},
        "params_en": {"current": "Rated current", "turns": "Turns per phase", "magnet_thickness": "Magnet thickness",
                      "airgap": "Air gap", "stack_length": "Stack length", "rotor_diameter": "Rotor diameter",
                      "Br": "Remanence", "mu_r": "Relative permeability", "n_rotor": "Rotor positions"},
    },
    "steam_pipe_insulation": {
        "title_en": "Steam-line insulation: heat loss, surface temperature and economic thickness",
        "sector_en": "Process and energy / utilities",
        "problem_en": "A 10 bar(a) saturated-steam line, DN150, outdoors, insulated with rock wool and clad in "
                      "aluminium. Heat loss per metre, surface temperature, and the thickness that minimises annual "
                      "cost (energy lost against insulation cost).",
        "question_en": "Which insulation thickness minimises the annual cost per metre, and which is the smallest "
                       "that keeps the surface below 60 °C (personal protection)? How much is lost uninsulated?",
        "labels_en": {"temperature": "temperature [°C]"},
        "params_en": {"diameter": "Pipe outside diameter", "steam_pressure": "Steam pressure",
                      "insulation_thickness": "Insulation thickness", "k_insulation": "Insulation conductivity",
                      "t_ambient": "Ambient air", "wind_speed": "Wind speed", "t_surface_limit": "Surface limit"},
    },
    "steel_footbridge_pedestrian": {
        "title_en": "30 m steel footbridge: strength, deflection, lateral buckling and pedestrian vibration",
        "sector_en": "Structural / construction",
        "problem_en": "A 30 m simply supported footbridge with two welded I-girders and a 2.5 m deck (precast "
                      "planks). Ultimate limit state in bending under a crowd (EN 1991-2), deflection L/360, "
                      "lateral-torsional buckling (EN 1993-1-1) and comfort under pedestrian loading.",
        "question_en": "Does the girder section meet all four failure modes? Which is the lightest welded section "
                       "that does? Does the first vertical mode fall in the walking-pace range (1.6-2.4 Hz)?",
        "labels_en": {"sigma": "design bending stress [MPa]", "mode1": _mode1},
        "params_en": {"span": "Span", "deck_width": "Deck width", "deck_dead_load": "Deck dead load",
                      "girder_h": "Girder depth", "girder_b": "Flange width", "girder_tf": "Flange thickness",
                      "girder_tw": "Web thickness", "fy": "Yield strength", "lateral_braces": "Lateral braces",
                      "damping_ratio": "Damping ratio", "pedestrian_weight": "Pedestrian weight",
                      "acc_max": "Comfort limit", "deflection_ratio": "Deflection ratio (L/n)"},
    },
    "water_distribution_network": {
        "title_en": "Water distribution network: pressures, flows and least-cost pipe sizing",
        "sector_en": "Fluids / water supply",
        "problem_en": "A looped distribution network (two loops, one reservoir, six demand nodes, eight 1 km pipes; "
                      "the Alperovits & Shamir 1977 reference network). Steady-state pressures and flows "
                      "(Hazen-Williams or Darcy-Weisbach).",
        "question_en": "Do all nodes have the minimum service pressure (30 m) at the design demand? Which set of "
                       "commercial diameters meets it at the lowest cost?",
        "labels_en": {"pressure": "pressure [m water column]", "velocity": "velocity [m/s]"},
        "params_en": {"demand": "Node demands", "head": "Reservoir head", "roughness": "Roughness (C or ks)",
                      "p_min": "Minimum service pressure"},
    },
    "water_hammer_valve_closure": {
        "title_en": "Water hammer on valve closure in a transmission main: surge pressure and protection",
        "sector_en": "Fluids / water supply and oil and gas",
        "problem_en": "A transmission main from a reservoir to a downstream valve that closes in a given time. The "
                      "transient is solved by the method of characteristics; wave speed from the pipe and fluid, "
                      "maximum and minimum pressure along the line, and the protection that keeps the surge within "
                      "the pipe rating.",
        "question_en": "What is the maximum surge pressure if the valve closes in this time, and where along the line "
                       "does it happen? Does a slow closure or an air vessel keep the line within its rating?",
        "labels_en": {"p_max": "maximum pressure [MPa]", "p_min": "minimum pressure [MPa]"},
        "params_en": {"length": "Main length", "diameter": "Inside diameter", "flow": "Steady flow",
                      "wave_speed": "Wave speed", "t_close": "Closure time", "p_rating": "Pipe rating"},
    },
}


def english(preset) -> Any:
    """The preset with its English data sheet attached (``title_en``, ``problem_en``, ``question_en``, ``sector_en``
    and each parameter's ``help_en``); a preset without an English entry comes back unchanged."""
    entry = EN.get(getattr(preset, "name", "") or "")
    if not entry:
        return preset
    preset.title_en = preset.title_en or entry.get("title_en", "")
    preset.sector_en = preset.sector_en or entry.get("sector_en", "")
    preset.problem_en = preset.problem_en or entry.get("problem_en", "")
    preset.question_en = preset.question_en or entry.get("question_en", "")
    labels = entry.get("params_en") or {}
    if labels:
        preset.params = [replace(q, help_en=q.help_en or labels.get(q.name, "")) for q in preset.params]
    return preset


def scene_labels(preset, result) -> dict[str, str]:
    """The English labels of the preset's 3-D fields for this result (the callable ones read its numbers)."""
    out: dict[str, str] = {}
    for key, value in ((EN.get(getattr(preset, "name", "") or "") or {}).get("labels_en") or {}).items():
        try:
            out[key] = value(preset, result) if isinstance(value, Callable) else str(value)
        except Exception:                        # noqa: BLE001 - a label is never a reason to fail a run
            continue
    return out

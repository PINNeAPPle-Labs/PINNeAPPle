"""Derivative-tree nodes (``pinneapple_physics.potential``): the tree of an equation of state, a hyperelastic stress
and tangent, a Lagrangian, and the torque of a surface-PM machine from its co-energy, each against its closed form or
an independent computation."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("potential.eos_tree", "Equation of state: properties from one Helmholtz energy",
      category="Analysis",
      outputs=[Port("metrics", "metrics")],
      params=[Param("eos", "enum", "van_der_waals", choices=["ideal_gas", "van_der_waals", "peng_robinson"],
                    help="Helmholtz energy (CO2 constants for the cubic equations)"),
              Param("T", "float", 320.0, minimum=50.0, maximum=2000.0, help="temperature [K]"),
              Param("V", "float", 2e-4, minimum=5e-5, maximum=1.0, help="molar volume [m3/mol]")],
      calls="pinneapple_physics.potential.tree + thermo.ideal_gas / van_der_waals / peng_robinson + "
            "potential.consistency_check + potential.convexity")
def potential_eos_tree(ctx, inputs, params):
    """Depth-2 derivative tree of a Helmholtz energy A(T, V): pressure, entropy, c_v, c_p, compressibility,
    expansion and speed of sound at one state; the Maxwell relation checked by finite differences, and the share of
    a (T, V) grid where A is convex in V (outside it the homogeneous fluid is unstable)."""
    import numpy as np

    from pinneapple_physics.potential import consistency_check, convexity, thermo, tree
    T, V, eos = float(params["T"]), float(params["V"]), params["eos"]
    M = 0.0440098  # CO2, kg/mol
    if eos == "ideal_gas":
        A = thermo.ideal_gas(c=2.5, molar_mass=M)
    elif eos == "van_der_waals":
        A = thermo.van_der_waals(0.3640, 4.267e-5, c=2.5, molar_mass=M)
    else:
        A = thermo.peng_robinson(304.1282, 7.3773e6, 0.22394, c=2.5, molar_mass=M)
    t = tree(A, 2, at={"T": T, "V": V})
    for q, how in [("p", "Pa"), ("S", "J/(mol K)"), ("cv", "J/(mol K)"), ("cp", "J/(mol K)"),
                   ("kappa_T", "1/Pa"), ("alpha_p", "1/K"), ("w", "m/s")]:
        val = float(t[q])
        if np.isfinite(val):
            ctx.metric(q, val, how)
    rep = consistency_check(A, A, at={"T": T * np.linspace(0.95, 1.05, 5), "V": V * np.linspace(0.9, 1.1, 5)})
    ctx.metric("maxwell_violation_p95", rep.rows[0]["rel_p95"], "relative, finite differences of the tree")
    conv = convexity(A, "V", {"T": (0.8 * T, 1.2 * T), "V": (0.3 * V, 3 * V)}, n_grid=41)
    ctx.metric("convex_fraction", conv.fraction_stable, "share of the (T, V) grid where A is convex in V")
    ctx.summary(f"{eos}: p = {float(t['p']):.4g} Pa, c_p = {float(t['cp']):.4g} J/(mol K); convex on "
                f"{100 * conv.fraction_stable:.0f} % of the grid")
    return {"metrics": ctx.report["metrics"]}


@node("potential.hyperelastic", "Hyperelastic stress and tangent from a strain energy",
      category="Analysis",
      outputs=[Port("metrics", "metrics")],
      params=[Param("stretch", "float", 1.3, minimum=0.3, maximum=3.0, help="uniaxial stretch"),
              Param("shear", "float", 0.2, minimum=-1.0, maximum=1.0, help="simple shear added to F"),
              Param("mu", "float", 0.8e6, minimum=1.0, help="shear modulus [Pa]"),
              Param("lam", "float", 2.4e6, minimum=0.0, help="Lame lambda [Pa]")],
      calls="pinneapple_physics.potential.hyperelastic.neo_hookean + neo_hookean_stress + neo_hookean_tangent + "
            "major_symmetry")
def potential_hyperelastic(ctx, inputs, params):
    """Neo-Hookean W(F): the first Piola-Kirchhoff stress and the 3x3x3x3 tangent by automatic differentiation,
    against their closed forms, and the major symmetry of the tangent."""
    import numpy as np

    from pinneapple_physics.potential import hyperelastic as he
    from pinneapple_physics.potential import tree
    mu, lam, a, g = float(params["mu"]), float(params["lam"]), float(params["stretch"]), float(params["shear"])
    F = np.diag([a, 1 / np.sqrt(a), 1 / np.sqrt(a)])
    F[0, 1] = g
    t = tree(he.neo_hookean(mu, lam), 2, at={"F": F})
    P, A = he.neo_hookean_stress(F, mu, lam), he.neo_hookean_tangent(F, mu, lam)
    ctx.metric("P11", float(t["P"][0, 0]), "Pa")
    ctx.metric("stress_rel_error", float(np.abs(t["P"] - P).max() / np.abs(P).max()), "against the closed form")
    ctx.metric("tangent_rel_error", float(np.abs(t["A"] - A).max() / np.abs(A).max()), "against the closed form")
    ctx.metric("major_symmetry", he.major_symmetry(t["A"]), "max |A_iJkL - A_kLiJ| / max |A|")
    ctx.summary(f"P11 = {float(t['P'][0, 0]):.4g} Pa; stress and tangent match the closed forms")
    return {"metrics": ctx.report["metrics"]}


@node("potential.double_pendulum", "Double pendulum from its Lagrangian",
      category="Simulation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("q1", "float", 1.2, minimum=-3.1, maximum=3.1, help="initial angle of the upper rod [rad]"),
              Param("q2", "float", -0.4, minimum=-3.1, maximum=3.1, help="initial angle of the lower rod [rad]"),
              Param("t_end", "float", 5.0, minimum=0.1, maximum=60.0, help="duration [s]"),
              Param("dt", "float", 0.005, minimum=1e-4, maximum=0.05, help="RK4 step [s]")],
      calls="pinneapple_physics.potential.mechanics.double_pendulum + mechanics.simulate + mechanics.hamiltonian")
def potential_double_pendulum(ctx, inputs, params):
    """Equations of motion from the derivative tree of L(q, qd) (mass matrix, coupling, forces), integrated with
    RK4; energy drift, and the Hamiltonian from the Legendre transform checked against the energy."""
    import numpy as np

    from pinneapple_physics.potential import mechanics, tree
    L = mechanics.double_pendulum()
    run = mechanics.simulate(L, [float(params["q1"]), float(params["q2"])], [0.0, 0.0], float(params["t_end"]),
                             float(params["dt"]))
    drift = float(np.abs(run["E"] - run["E"][0]).max() / abs(run["E"][0]))
    ctx.metric("energy_drift", drift, "max |E - E0| / |E0| over the run")
    pt = {"q1": run["q"][-1, 0], "q2": run["q"][-1, 1], "qd1": run["qd"][-1, 0], "qd2": run["qd"][-1, 1]}
    t = tree(L, 1, at=pt)
    H = mechanics.hamiltonian(L).potential
    h = float(H(q1=pt["q1"], q2=pt["q2"], p1=t["p1"], p2=t["p2"]))
    ctx.metric("hamiltonian_vs_energy", abs(h - float(t["E"])) / abs(float(t["E"])), "relative, final state")
    ctx.summary(f"energy drift {drift:.2e} over {params['t_end']} s")
    return {"metrics": ctx.report["metrics"]}


@node("potential.machine_coenergy", "Machine torque, flux linkage and inductance from the co-energy",
      category="Electromagnetics",
      outputs=[Port("metrics", "metrics")],
      params=[Param("theta_deg", "float", 1.3, minimum=0.0, maximum=180.0, help="rotor angle [deg]"),
              Param("I_peak", "float", 30.0, minimum=0.0, maximum=200.0, help="phase current amplitude [A]"),
              Param("gamma_deg", "float", 0.0, minimum=-90.0, maximum=90.0, help="current angle [deg]")],
      calls="pinneapple_physics.potential.machines.machine_coenergy + "
            "pinneapple_simulation.electromagnetics.machine.SPMMachine")
def potential_machine_coenergy(ctx, inputs, params):
    """A local co-energy potential W'(i, theta) of the surface-PM machine (linear iron) fitted to FEA co-energies:
    torque dW'/dtheta against the Maxwell-stress torque, flux linkages dW'/di against the winding integral, and
    the inductance matrix."""
    import math

    import numpy as np

    from pinneapple_physics.potential.machines import machine_coenergy
    from pinneapple_simulation.electromagnetics.machine import SPMMachine
    from pinneapple_simulation.electromagnetics.materials import LinearMaterial
    m = SPMMachine(n_theta=360, iron=LinearMaterial(2000.0), slot_opening=2 / 15, slot_width=0.4)
    th = math.radians(float(params["theta_deg"]))
    cur = m.phase_currents(th, float(params["I_peak"]), math.radians(float(params["gamma_deg"])))
    r = machine_coenergy(m, th, cur).at_operating_point()
    ms = m.solve(th, cur)
    T_mst = m.torque(ms)
    ctx.metric("torque_coenergy", r["torque"], "N m, dW'/dtheta of the fitted potential")
    ctx.metric("torque_maxwell_stress", T_mst, "N m, Arkkio")
    ctx.metric("psi_max_abs_diff", float(np.abs(r["psi"] - m.flux_linkage(ms)).max()), "Wb, against the FEA")
    ctx.metric("L_self", float(r["L"][0, 0]), "H, d2W'/di0^2")
    ctx.summary(f"torque {r['torque']:.4f} N m from the co-energy vs {T_mst:.4f} N m by Maxwell stress")
    return {"metrics": ctx.report["metrics"]}

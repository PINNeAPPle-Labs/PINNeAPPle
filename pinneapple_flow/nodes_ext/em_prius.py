"""The longer Prius-motor studies of ``examples/electrodynamics`` (12-14, 16) as nodes on
:mod:`pinneapple_simulation.electromagnetics.machines`: rotor design (GP-surrogate NSGA-II against FEA-only), the
thermal network over WLTC cycles with demagnetisation margins, the cross-check against gmsh + GetDP, and the end
effects (end-winding inductance, 3-D magnet end leakage). Metric names follow the examples' results JSON."""
from __future__ import annotations

import dataclasses
import math
import shutil

import numpy as np

from ..registry import Param, Port, node

RPM = 2 * math.pi / 60
TWO_PI = 2 * math.pi
_VEH = {"mass": 1450.0, "c_rr": 0.009, "cd_a": 0.58, "wheel_radius": 0.30, "gear_ratio": 4.113, "gear_efficiency": 0.97}


def _getdp_available():
    if shutil.which("getdp") is None or shutil.which("gmsh") is None:
        return False, "needs gmsh and GetDP on the PATH (apt install gmsh getdp)"
    return True, ""


def _fm(v):
    from pinneapple_simulation.electromagnetics.machines.drive import FluxMap
    return FluxMap.from_npz(v["$artifact"])


@node("em.prius_rotor_design", "Prius rotor design: GP-surrogate NSGA-II against FEA-only", category="Electromagnetics",
      outputs=[Port("metrics", "metrics")],
      params=[Param("seeds", "list", [0, 1, 2]), Param("n_initial", "int", 24, minimum=4),
              Param("rounds", "int", 4, minimum=1), Param("n_verify", "int", 8, minimum=1),
              Param("ga_budget", "int", 2000, minimum=10), Param("surrogate_budget", "int", 2000, minimum=10)],
      calls="pinneapple_simulation.electromagnetics.machines.design.rotor_evaluator + rotor_design_problem + "
            "hybrid_vs_fea_only")
def rotor_design(ctx, inputs, params):
    """examples/electrodynamics/12_prius_rotor_design.py: hypervolume and FEA calls of the hybrid search against
    FEA-only, the hybrid front re-evaluated on a finer FEA, and the Prius rotor through the same evaluators."""
    import warnings

    from pinneapple_simulation.electromagnetics.machines import design as D
    warnings.filterwarnings("ignore")
    for seed in map(int, params["seeds"]):
        ctx.check_stop()
        fea = D.rotor_evaluator(n_gap=288, n_positions=6)
        fine = D.rotor_evaluator(n_gap=480, n_positions=12, name="ipm-fea-fine")
        r = D.hybrid_vs_fea_only(D.rotor_design_problem(fea), n_initial=int(params["n_initial"]),
                                 rounds=int(params["rounds"]), n_verify=int(params["n_verify"]),
                                 ga_budget=int(params["ga_budget"]), surrogate_budget=int(params["surrogate_budget"]),
                                 seed=seed, verifier=fine, progress=ctx.log)
        for k in ("hypervolume", "fea_calls"):
            ctx.metric(f"seed{seed}.hybrid.{k}", float(r["hybrid"][k]))
            ctx.metric(f"seed{seed}.fea_only.{k}", float(r["fea_only"][k]))
        if isinstance(r["fea_only"].get("calls_to_reach_hybrid"), (int, float)):
            ctx.metric(f"seed{seed}.fea_only.calls_to_reach_hybrid", float(r["fea_only"]["calls_to_reach_hybrid"]))
    fea, fine = D.rotor_evaluator(n_gap=288, n_positions=6), D.rotor_evaluator(n_gap=480, n_positions=12)
    x0 = np.array([[getattr(fea.base.rotor, n) for n in fea.names]])
    y, yf = fea.evaluate(x0), fine.evaluate(x0)
    ctx.metric("baseline.torque", float(y["torque"][0]), "the Prius rotor, coarse FEA")
    ctx.metric("baseline.ripple", float(y["ripple"][0]))
    ctx.metric("baseline.torque_fine", float(yf["torque"][0]), "the Prius rotor, fine FEA")
    return {"metrics": ctx.report["metrics"]}


@node("em.prius_thermal", "Prius thermal network over WLTC, demagnetisation", category="Electromagnetics",
      inputs=[Port("flux_map", "flux_map")], outputs=[Port("metrics", "metrics")],
      params=[Param("n_gap", "int", 960, minimum=96), Param("magnet_temperatures", "list", [20.0, 60.0, 100.0, 150.0]),
              Param("cycles", "int", 4, minimum=1, maximum=50), Param("beta_hcj", "float", -0.0055),
              Param("refinement", "bool", True, help="the demagnetisation mesh-refinement study")],
      calls="pinneapple_simulation.electromagnetics.machines.thermal: magnet_temperature_sweep, machine_network "
            "(transient), demagnetisation_margin / demagnetisation_refinement, intrinsic_coercivity; drive: Drive, "
            "cycle_energy, iron_split")
def prius_thermal(ctx, inputs, params):
    """examples/electrodynamics/13_prius_thermal.py: magnet temperature sweep, a lumped thermal network driven by
    the WLTC losses (peak node temperatures), the time to 180 C at peak load, and demagnetisation margins per NdFeB
    grade at the hottest magnet temperature."""
    from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004
    from pinneapple_simulation.electromagnetics.machines.drive import (
        Drive,
        Vehicle,
        cycle_energy,
        prius_phase_resistance,
        v_max_svpwm,
        wltc_class3b,
    )
    from pinneapple_simulation.electromagnetics.machines.thermal import (
        NDFEB_HCJ_MIN_KAM,
        demagnetisation_margin,
        demagnetisation_refinement,
        intrinsic_coercivity,
        iron_split,
        machine_network,
        magnet_temperature_sweep,
    )
    g, N = prius_2004(), int(params["n_gap"])
    for r in magnet_temperature_sweep(g, tuple(map(float, params["magnet_temperatures"])), n_gap=N):
        ctx.metric(f"sweep_{r['magnet_temperature_C']:.0f}C.torque_Nm", r["torque_Nm"])
        ctx.metric(f"sweep_{r['magnet_temperature_C']:.0f}C.Br_T", r["Br_T"])
    R, wdata = prius_phase_resistance(100.0)
    drive = Drive(_fm(inputs["flux_map"]), R=R, I_max=250.0, V_max=v_max_svpwm(500.0))
    t, v = wltc_class3b()
    E = cycle_energy(drive, Vehicle(**_VEH), t, v)
    split = iron_split(IPMMachine(g, n_gap=N))
    end = wdata["end_turn_per_side_m"]
    f_slot = g.length / (g.length + end)
    net = machine_network(g, end_length=end)

    def node_losses(cu, fe, mg):
        return {"yoke": fe * split["yoke"], "teeth": fe * split["teeth"], "winding": cu * f_slot,
                "end_winding": cu * (1 - f_slot), "rotor": fe * split["rotor"], "magnets": mg}
    n = len(E["t"])
    P = np.array([[node_losses(E["loss_copper"][k], E["loss_iron"][k], E["loss_magnet"][k])[nm] for nm in net.names]
                  for k in range(n)])
    cyc = int(params["cycles"])
    _, T = net.transient(lambda s: P[min(int(s), cyc * n - 1) % n], cyc * n - 1, 1.0, net.boundaries[0][2])
    hot = {nm: float(T[:, i].max()) for i, nm in enumerate(net.names)}
    for nm, val in hot.items():
        ctx.metric(f"peak_temperatures_C.{nm}", val, f"WLTC class 3b x {cyc}")
    w = 1500 * RPM
    op = drive.operating_point(0.999 * drive.max_torque(w)["T"], w)
    Pk = node_losses(op["copper"], op["iron"], op["magnet"])
    ts, Tp = net.transient([Pk[nm] for nm in net.names], 1800.0, 1.0, T[-1])
    hot_cu = np.maximum(Tp[:, net.index("winding")], Tp[:, net.index("end_winding")])
    reach = np.nonzero(hot_cu >= 180.0)[0]
    ctx.metric("peak_load.torque_Nm", float(op["T"]))
    if len(reach):
        ctx.metric("peak_load.time_to_180C_winding_s", float(ts[reach[0]]))
    T_mag = float(max(hot["magnets"], Tp[:, net.index("magnets")].max()))
    ctx.metric("demagnetisation.magnet_temperature_C", T_mag)
    m = IPMMachine(g, n_gap=N, magnet_temperature=T_mag)
    ind = m.inductances(0.0, 0.0)
    i_sc = 2.0 * ind["psi_pm_d"] / ind["L_dd"]
    ctx.metric("demagnetisation.short_circuit_peak_estimate_A", float(i_sc))
    beta = float(params["beta_hcj"])
    hcj = {gr: intrinsic_coercivity(NDFEB_HCJ_MIN_KAM[gr] * 1e3, T_mag, beta) for gr in ("N", "M", "H", "SH", "UH")}
    rows = []
    for label, i_d in (("i_d = -250 A", -250.0), ("short circuit", -i_sc)):
        ms = m.solve(0.0, m.currents_dq(0.0, i_d, 0.0))
        for gr, H in hcj.items():
            d = demagnetisation_margin(m, ms, H)
            ctx.metric(f"demag.{'id250' if i_d == -250.0 else 'short_circuit'}.{gr}.demagnetised_fraction",
                       float(d["demagnetised_fraction"]))
            rows.append([label, gr, H / 1e3, d["H_rev_max"] / 1e3, d["demagnetised_fraction"]])
    ctx.table("demagnetisation at the hottest magnet temperature", ["case", "grade", "Hcj [kA/m]",
                                                                    "worst reverse field [kA/m]",
                                                                    "demagnetised fraction"], rows)
    if params["refinement"]:
        cases = {"i_d = -250 A": -250.0, f"short-circuit peak 2 psi/L_d = -{i_sc:.0f} A": -i_sc}
        study = demagnetisation_refinement(g, T_mag, cases, hcj, n_gap=N)
        ctx.metric("refinement_levels", len(study))
    return {"metrics": ctx.report["metrics"]}


@node("em.prius_getdp_crosscheck", "Prius FEA against gmsh + GetDP", category="Electromagnetics",
      outputs=[Port("metrics", "metrics")],
      params=[Param("n_gap", "int", 960, minimum=96), Param("positions", "int", 12, minimum=2),
              Param("angles_every", "int", 1, minimum=1, help="every k-th angle of Yang 2013 (3 in the quick run)"),
              Param("emf_positions", "int", 60, minimum=6)],
      calls="pinneapple_simulation.electromagnetics.machines.getdp.solve (gmsh + GetDP) against IPMMachine.solve; "
            "machine.belt_winding; verification.YANG_2013_*", available=_getdp_available)
def getdp_crosscheck(ctx, inputs, params):
    """examples/electrodynamics/14_prius_getdp_crosscheck.py: torque against current angle, cogging, back-EMF and
    apparent inductances, PINNeAPPle's FEA against an independent gmsh + GetDP model of the same machine."""
    from pinneapple_simulation.electromagnetics.machine import belt_winding
    from pinneapple_simulation.electromagnetics.machines import M400_50A, IPMMachine, prius_2004
    from pinneapple_simulation.electromagnetics.machines import getdp as G
    from pinneapple_simulation.electromagnetics.machines import verification as mv
    g, N = prius_2004(), int(params["n_gap"])
    m = IPMMachine(g, n_gap=N)
    lay = belt_winding(48, 4)

    def gd(theta, cur):
        return G.solve(g, theta, cur, M400_50A, m.Br, m.magnet.mu_r, lay, 9.0, n_gap=N)
    npos = int(params["positions"])
    T_o, T_g, T_y = [], [], []
    for p in mv.YANG_2013_PHI0_DEG[::int(params["angles_every"])]:
        ctx.check_stop()
        gm = math.radians(p - 90.0)
        to, tg = [], []
        for th in m.positions(npos):
            cur = m.phase_currents(th, 250.0, gm)
            to.append(m.torque(m.solve(th, cur)))
            tg.append(gd(th, cur)["torque"])
        T_o.append(float(np.mean(to)))
        T_g.append(float(np.mean(tg)))
        T_y.append(float(mv.YANG_2013_TORQUE_NM[list(mv.YANG_2013_PHI0_DEG).index(p)]))
        ctx.metric(f"torque_phi{p:.0f}.pinneapple_Nm", T_o[-1])
        ctx.metric(f"torque_phi{p:.0f}.getdp_Nm", T_g[-1])
    T_o, T_g, T_y = map(np.asarray, (T_o, T_g, T_y))
    ctx.metric("torque_difference_getdp_minus_pinneapple.max_abs_Nm", float(np.abs(T_g - T_o).max()))
    ctx.metric("torque_difference_getdp_minus_pinneapple.rms_Nm", float(np.sqrt(np.mean((T_g - T_o) ** 2))))
    ctx.metric("vs_yang.pinneapple_rms_Nm", float(np.sqrt(np.mean((T_o - T_y) ** 2))))
    ctx.metric("vs_yang.getdp_rms_Nm", float(np.sqrt(np.mean((T_g - T_y) ** 2))))
    th_c = m.positions(npos, period=m.cogging_period)
    ctx.metric("cogging.pinneapple_pkpk_Nm", float(np.ptp([m.torque(m.solve(t)) for t in th_c])))
    ctx.metric("cogging.getdp_pkpk_Nm", float(np.ptp([gd(t, (0.0, 0.0, 0.0))["torque"] for t in th_c])))
    nf = int(params["emf_positions"])
    th_e = m.positions(nf, period=TWO_PI / m.pole_pairs)
    e_o = m.back_emf(np.array([m.flux_linkage(m.solve(t)) for t in th_e]), th_e, 1000.0)
    e_g = m.back_emf(np.array([gd(t, (0.0, 0.0, 0.0))["psi"] for t in th_e]), th_e, 1000.0)
    ctx.metric("back_emf_1000rpm.pinneapple_fundamental_peak_V", e_o["fundamental_peak"])
    ctx.metric("back_emf_1000rpm.getdp_fundamental_peak_V", e_g["fundamental_peak"])

    def dq_mean(code, i_d, i_q):
        out = []
        for th in m.positions(npos):
            cur = m.currents_dq(th, i_d, i_q)
            psi = m.flux_linkage(m.solve(th, cur)) if code == "o" else gd(th, cur)["psi"]
            out.append(m.park(psi, th))
        return np.mean(out, axis=0)
    i_d, i_q = -250 * math.sin(math.radians(50)), 250 * math.cos(math.radians(50))
    for code, name in (("o", "pinneapple"), ("g", "getdp")):
        d0, _ = dq_mean(code, 0.0, 0.0)
        dd, _ = dq_mean(code, i_d, 0.0)
        _, qq = dq_mean(code, 0.0, i_q)
        ctx.metric(f"inductances.{name}.L_d_mH", 1e3 * float((dd - d0) / i_d))
        ctx.metric(f"inductances.{name}.L_q_mH", 1e3 * float(qq / i_q))
    return {"metrics": ctx.report["metrics"]}


@node("em.prius_end_effects", "Prius end effects: end-winding inductance, 3-D magnet end leakage",
      category="Electromagnetics", inputs=[Port("flux_map", "flux_map")], outputs=[Port("metrics", "metrics")],
      params=[Param("leakage_levels", "list", [[144, 4, 4, 0.04], [240, 6, 6, 0.04], [336, 8, 8, 0.04],
                                               [240, 6, 9, 0.06]],
                    help="3-D levels: n_gap, layers in the stack, layers in the end air, end air [m]"),
              Param("finest_level", "int", 2, minimum=0, help="the level whose flux ratio corrects the 2-D model")],
      calls="pinneapple_simulation.electromagnetics.machines.endwinding.end_winding_inductance + getdp.end_leakage_3d "
            "(gmsh + GetDP) + drive.Drive(L_end) + IPMMachine.mtpa / no_load", available=_getdp_available)
def end_effects(ctx, inputs, params):
    """examples/electrodynamics/16_prius_end_effects.py."""
    from pinneapple_simulation.electromagnetics.machine import belt_winding
    from pinneapple_simulation.electromagnetics.machines import (
        M400_50A,
        MAGNET_PRIUS,
        IPMMachine,
        prius_2004,
    )
    from pinneapple_simulation.electromagnetics.machines import getdp as G
    from pinneapple_simulation.electromagnetics.machines.drive import (
        Drive,
        prius_phase_resistance,
        v_max_svpwm,
    )
    from pinneapple_simulation.electromagnetics.machines.endwinding import end_winding_inductance
    g = prius_2004()
    e = end_winding_inductance(g, belt_winding(48, 4), 9.0, 6, 0.019366, pieces_per_metre=300)
    ctx.metric("end_winding.air.L_end_dq_uH", 1e6 * e["air"]["L_end_dq"])
    ctx.metric("end_winding.iron_face.L_end_dq_uH", 1e6 * e["iron_face"]["L_end_dq"])
    fm = _fm(inputs["flux_map"])
    R, _ = prius_phase_resistance(100.0)
    speeds = np.array([1000, 2000, 3000, 4000, 5000, 6000]) * RPM
    for label, L in (("2d", 0.0), ("air", e["air"]["L_end_dq"]), ("iron_face", e["iron_face"]["L_end_dq"])):
        d = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0), L_end=L)
        for w in speeds:
            ctx.metric(f"envelope.{label}.{w / RPM:.0f}rpm_Nm", float(d.max_torque(w)["T"]))
    d0 = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0))
    d1 = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0), L_end=e["air"]["L_end_dq"])
    for rpm, T in ((4000, 100.0), (6000, 60.0)):
        ctx.metric(f"efficiency.{rpm}rpm_{T:.0f}Nm.2d", float(d0.operating_point(T, rpm * RPM)["efficiency"]))
        ctx.metric(f"efficiency.{rpm}rpm_{T:.0f}Nm.with_end_winding",
                   float(d1.operating_point(T, rpm * RPM)["efficiency"]))
    ratios = []
    for i, (n_gap, ls, la, ea) in enumerate(params["leakage_levels"]):
        ctx.check_stop()
        r = G.end_leakage_3d(g, M400_50A, 1.24, 1.05, n_gap=int(n_gap), layers_stack=int(ls), end_air=float(ea),
                             layers_air=int(la))
        ratios.append(float(r["ratio"]))
        ctx.metric(f"leakage_level{i}.ratio", float(r["ratio"]), f"n_gap {n_gap}, {ls} + {la} layers, {ea} m air")
    k = ratios[min(int(params["finest_level"]), len(ratios) - 1)]
    ctx.metric("flux_correction.ratio_finest", k)
    for label, scale in (("2d", 1.0), ("end_leakage", k)):
        m = IPMMachine(g, magnet=dataclasses.replace(MAGNET_PRIUS, Br20=MAGNET_PRIUS.Br20 * scale), n_gap=960)
        _, T = m.mtpa(250.0, n_positions=12)
        ctx.metric(f"corrected_2d.{label}.torque_Nm", float(T))
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    ref = ("brenner2008fem",)
    register_capability(Capability(
        id="em.prius_rotor_design", title="PM rotor design: surrogate-assisted NSGA-II against FEA-only",
        domain="electromagnetics", example="electrodynamics",
        keywords={"rotor design": 3, "nsga": 2, "multi-objective": 2, "pareto": 2, "hypervolume": 2},
        requires_all=(("rotor", "motor", "prius", "ipm"),), min_score=3,
        nodes=("em.prius_rotor_design",), references=ref + ("deb2002nsga2", "rasmussen2006gp"),
        template="electromagnetics/12_prius_rotor_design", quick=True,
        description="examples/electrodynamics/12_prius_rotor_design.py"))
    register_capability(Capability(
        id="em.prius_thermal", title="Motor thermal network over a drive cycle, demagnetisation", domain="electromagnetics",
        example="electrodynamics", keywords={"thermal": 2, "demagnetisation": 3, "demagnetization": 3, "hot spot": 1.5,
                                             "winding temperature": 2, "magnet temperature": 2},
        requires_all=(_MOTOR := ("motor", "prius", "ipm", "magnet", "machine"),), min_score=3,
        nodes=("em.prius_flux_map", "em.prius_thermal"), references=ref + ("incropera2007",),
        template="electromagnetics/13_prius_thermal", quick=True,
        description="examples/electrodynamics/13_prius_thermal.py"))
    register_capability(Capability(
        id="em.prius_getdp", title="Motor FEA cross-checked against gmsh + GetDP", domain="electromagnetics",
        example="electrodynamics", keywords={"getdp": 3, "cross-check": 2, "crosscheck": 2, "independent solver": 2},
        requires_all=(_MOTOR,), min_score=3, nodes=("em.prius_getdp_crosscheck",), references=ref,
        template="electromagnetics/14_prius_getdp_crosscheck", quick=True,
        description="examples/electrodynamics/14_prius_getdp_crosscheck.py"))
    register_capability(Capability(
        id="em.prius_end_effects", title="Motor end effects: end-winding inductance, 3-D end leakage",
        domain="electromagnetics", example="electrodynamics",
        keywords={"end winding": 3, "end-winding": 3, "end effects": 3, "end leakage": 3, "3-d leakage": 2},
        requires_all=(_MOTOR,), min_score=3, nodes=("em.prius_flux_map", "em.prius_end_effects"), references=ref,
        template="electromagnetics/16_prius_end_effects", quick=True,
        description="examples/electrodynamics/16_prius_end_effects.py"))


_register_capabilities()

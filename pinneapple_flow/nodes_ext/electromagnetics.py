"""Electromagnetics nodes: the studies of ``examples/motor_magnetics`` and ``examples/electrodynamics`` on
:mod:`pinneapple_simulation.electromagnetics` (the verification table, the surface-PM machine, the 2004 Prius
interior-PM motor: no-load, torque against current angle and MTPA, flux map and drive with the WLTC cycle, step skew,
manufacturing tolerances). Metric names follow the keys of the examples' results JSON."""
from __future__ import annotations

import math
import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact
from ..types import register_port_type

register_port_type("flux_map", "an interior-PM machine flux map (psi_d, psi_q, torque, loss data on an i_d x i_q grid; "
                               "machines.drive.FluxMap .npz)")
RPM = 2 * math.pi / 60


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _field_figure(ctx, sol, name, title, scale=1.0, xlim=None, ylim=None, clim=None):
    plt = _plt()
    from matplotlib.collections import PolyCollection
    mesh = sol.mesh
    fig, ax = plt.subplots(figsize=(6.5, 6))
    pc = PolyCollection(mesh.points[mesh.tris] * scale, array=np.hypot(sol.B[:, 0], sol.B[:, 1]), cmap="viridis",
                        edgecolors="none")
    if clim:
        pc.set_clim(*clim)
    ax.add_collection(pc)
    ax.tricontour(mesh.points[:, 0] * scale, mesh.points[:, 1] * scale, mesh.tris, sol.A, 20, colors="w",
                  linewidths=0.4)
    fig.colorbar(pc, ax=ax, label="|B| (T)")
    ax.set_aspect("equal")
    ax.autoscale()
    if xlim:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    ax.set_title(title)
    return ctx.figure(fig, name, title)


@node("em.verification", "Electromagnetics verification table", category="Electromagnetics",
      outputs=[Port("metrics", "metrics")],
      params=[Param("quick", "bool", False, help="coarser refinement ladders")],
      calls="pinneapple_simulation.electromagnetics.verification.run + to_markdown")
def em_verification(ctx, inputs, params):
    """Every electromagnetics solver and reference against an independent result: errors and observed orders
    (examples/electrodynamics/07_em_verification.py)."""
    from pinneapple_simulation.electromagnetics import verification
    res = verification.run(quick=bool(params["quick"]))
    rows = []
    for i, r in enumerate(res):
        ctx.metric(f"case{i}.error", r.error, f"{r.name}: {r.what} against {r.reference}")
        ctx.metric(f"case{i}.passed", int(bool(r.passed)))
        rows.append([r.name, r.what, r.reference, r.error, ", ".join(f"{o:.2f}" for o in r.orders), r.passed])
    ctx.metric("cases", len(res))
    ctx.metric("passed", sum(bool(r.passed) for r in res))
    ctx.table("verification", ["case", "quantity", "reference", "error", "orders", "passed"], rows)
    ctx.summary(f"{sum(bool(r.passed) for r in res)} of {len(res)} cases pass")
    return {"metrics": ctx.report["metrics"]}


@node("em.spm_machine", "Surface-PM machine: cogging, torque, back-EMF, losses", category="Electromagnetics",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("n_theta", "int", 360, minimum=48, maximum=1440, help="air-gap nodes (1-degree grid at 360)"),
              Param("slot_opening", "float", 2 / 15, minimum=0.01, maximum=0.9),
              Param("slot_width", "float", 0.4, minimum=0.05, maximum=0.9),
              Param("I_rated", "float", 30.0, minimum=0.0, help="A peak, 20 turns per slot"),
              Param("rpm", "float", 1500.0, minimum=1.0),
              Param("cogging_points", "int", 16, minimum=4, maximum=128),
              Param("emf_steps", "int", 72, minimum=12, maximum=720),
              Param("loss_steps", "int", 24, minimum=6, maximum=360),
              Param("iron_loss", "dict", {"k_h": 0.02, "alpha": 1.0, "beta": 2.0, "k_c": 5e-5, "k_e": 1e-4,
                                          "density": 7650.0}, help="IronLossModel coefficients (illustrative)")],
      calls="pinneapple_simulation.electromagnetics.machine.SPMMachine (cogging, torque, torque_coenergy, "
            "torque_vs_current_angle, no_load_flux, losses) + postprocess.back_emf / IronLossModel")
def spm_machine(ctx, inputs, params):
    """examples/motor_magnetics/01_spm_cogging_and_torque.py: 4 poles, 24 slots, nonlinear iron; torque by Maxwell
    stress (Arkkio) and by virtual work, side by side."""
    from pinneapple_simulation.electromagnetics.machine import SPMMachine
    from pinneapple_simulation.electromagnetics.postprocess import IronLossModel, back_emf
    t0 = time.perf_counter()
    m = SPMMachine(n_theta=int(params["n_theta"]), slot_opening=float(params["slot_opening"]),
                   slot_width=float(params["slot_width"]))
    i_rated = float(params["I_rated"])
    period = m.cogging_period()
    th_cog = np.linspace(0.0, period, int(params["cogging_points"]), endpoint=False)
    T_cog = m.cogging(th_cog)
    ctx.metric("cogging.period_deg", math.degrees(period))
    ctx.metric("cogging.peak_to_peak_Nm", float(np.ptp(T_cog)), "Maxwell stress over one cogging period")
    pairs = []
    for d in (1.3, 4.1, 9.7):
        t = math.radians(d)
        a, b = m.torque(m.solve(t)), m.torque_coenergy(t, (0.0, 0.0, 0.0))
        pairs.append((d, a, b))
        ctx.metric(f"cogging.maxwell_{d}deg_Nm", a)
        ctx.metric(f"cogging.coenergy_{d}deg_Nm", b)
    theta0 = math.radians(1.3)
    gammas = np.radians(np.arange(-90, 91, 15))
    T_gam = m.torque_vs_current_angle(i_rated, gammas, theta=theta0)
    for g, T in zip(np.degrees(gammas), T_gam, strict=True):
        ctx.metric(f"torque_vs_gamma.gamma{g:.0f}_Nm", float(T))
    load = []
    for g in (-30.0, 0.0, 30.0):
        cur = m.phase_currents(theta0, i_rated, math.radians(g))
        load.append((g, m.torque(m.solve(theta0, cur)), m.torque_coenergy(theta0, cur)))
        ctx.metric(f"torque_vs_gamma.coenergy_gamma{g:.0f}_Nm", load[-1][2])
    rpm = float(params["rpm"])
    steps = int(params["emf_steps"])
    th_e = (2 * math.pi / m.pole_pairs) * np.arange(steps) / steps
    psi = m.no_load_flux(th_e)
    emf = back_emf(psi, th_e, rpm * RPM)
    ctx.metric("back_emf.psi_peak_Wb", float(np.abs(psi).max()))
    ctx.metric("back_emf.emf_peak_V", float(np.abs(emf).max()), f"at {rpm:g} rpm")
    ctx.metric("back_emf.emf_rms_V", float(np.sqrt(np.mean(emf[:, 0] ** 2))))
    loss = m.losses(rpm, i_rated, 0.0, IronLossModel(**params["iron_loss"]), steps=int(params["loss_steps"]))
    for k, v in loss.items():
        if isinstance(v, (int, float)):
            ctx.metric(f"losses.{k}", float(v), f"{rpm:g} rpm, {i_rated:g} A peak")
    ctx.metric("seconds", round(time.perf_counter() - t0, 1))
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(np.degrees(th_cog), T_cog, "o-", label="Maxwell stress")
    ax[0].plot([p[0] for p in pairs], [p[2] for p in pairs], "s", label="co-energy")
    ax[0].set(xlabel="rotor angle (deg)", ylabel="cogging torque (N m)", title="Cogging")
    ax[0].legend()
    ax[1].plot(np.degrees(gammas), T_gam, "o-", label="Maxwell stress")
    ax[1].plot([p[0] for p in load], [p[2] for p in load], "s", label="co-energy")
    ax[1].set(xlabel="current angle from q-axis (deg)", ylabel="torque (N m)", title=f"I = {i_rated:g} A peak")
    ax[1].legend()
    for k, name in enumerate("ABC"):
        ax[2].plot(np.degrees(m.pole_pairs * th_e), emf[:, k], label=name)
    ax[2].set(xlabel="electrical angle (deg)", ylabel="back-EMF (V)", title=f"{rpm:g} rpm")
    ax[2].legend()
    fig.tight_layout()
    f = ctx.figure(fig, "spm_torque_cogging_emf", "cogging, torque against current angle, back-EMF")
    _field_figure(ctx, m.solve(theta0, m.phase_currents(theta0, i_rated, 0.0)), "spm_flux_density",
                  "rated load, flux lines and |B|")
    ctx.table("torque: Maxwell stress vs co-energy", ["case", "Maxwell [N m]", "co-energy [N m]"],
              [[f"cogging {d:g} deg", a, b] for d, a, b in pairs] + [[f"gamma {g:g} deg", a, b] for g, a, b in load])
    ctx.summary(f"cogging {np.ptp(T_cog):.3f} N m pk-pk, peak torque {T_gam.max():.2f} N m, "
                f"EMF {np.abs(emf).max():.1f} V peak at {rpm:g} rpm")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("em.prius_machine", "Prius IPM motor: no-load, torque vs angle, MTPA, losses", category="Electromagnetics",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("n_gap", "int", 960, minimum=96, maximum=3840, help="air-gap nodes (480 in the quick run)"),
              Param("cogging_positions", "int", 24, minimum=4, maximum=96),
              Param("flux_positions", "int", 120, minimum=12, maximum=480),
              Param("mtpa_currents", "list", [50.0, 100.0, 150.0, 200.0, 250.0], help="A peak"),
              Param("operating_positions", "int", 12, minimum=2, maximum=96),
              Param("loss_steps", "int", 48, minimum=6, maximum=360),
              Param("verification", "bool", True, help="also the machine verification cases"),
              Param("quick", "bool", False, help="machine verification: quick ladders")],
      calls="pinneapple_simulation.electromagnetics.machines.IPMMachine(prius_2004()): no_load, back_emf, mtpa, "
            "operating_point, inductances, losses; machines.verification.prius_torque_curve / yang_comparison / run; "
            "fit_iron_loss(M400_50A)")
def prius_machine(ctx, inputs, params):
    """examples/electrodynamics/10_pmsm_prius.py: the 2004 Prius motor (48 slots, 8 poles, V-shaped NdFeB, M400-50A),
    torque against current angle compared with Yang et al. (IEMDC 2013)."""
    from pinneapple_simulation.electromagnetics.machines import (
        M400_50A,
        IPMMachine,
        fit_iron_loss,
        prius_2004,
    )
    from pinneapple_simulation.electromagnetics.machines import verification as mv
    t0 = time.perf_counter()
    n_gap = int(params["n_gap"])
    m = IPMMachine(prius_2004(), n_gap=n_gap)
    ctx.metric("elements", int(m.mm.n_elements))
    nl = m.no_load(int(params["cogging_positions"]), n_flux=int(params["flux_positions"]))
    emf = m.back_emf(nl["psi"], nl["theta"], rpm=1000.0)
    ctx.metric("no_load.cogging_peak_to_peak_Nm", float(np.ptp(nl["cogging"])))
    ctx.metric("no_load.psi_pm_peak_Wb", float(np.max(np.abs(nl["psi"][:, 0]))))
    ctx.metric("no_load.emf_phase_fundamental_peak_V_at_1000rpm", emf["fundamental_peak"])
    ctx.metric("no_load.emf_line_rms_V_at_1000rpm", emf["line_rms"])
    ctx.metric("no_load.emf_thd_phase", emf["thd"])
    ctx.metric("no_load.emf_thd_line", emf["thd_line"])
    ctx.check_stop()
    curve = mv.prius_torque_curve(n_gap, n_positions=12)
    cmp_ = mv.yang_comparison(curve)
    for p, t in zip(curve["phi0_deg"], curve["torque"], strict=True):
        ctx.metric(f"torque_vs_angle.phi{p:.0f}_Nm", float(t), "250 A peak, Maxwell stress")
    for k, v in cmp_.items():
        if isinstance(v, (int, float)):
            ctx.metric(f"torque_vs_angle.comparison.{k}", float(v), "against Yang et al. 2013")
    ctx.table("torque against current angle, 250 A", ["Phi0 [deg]", "this FEA [N m]", "Yang 2013 [N m]"],
              [[float(p), float(t), float(r)] for p, t, r in zip(curve["phi0_deg"], curve["torque"],
                                                                  mv.YANG_2013_TORQUE_NM, strict=True)])
    mtpa = []
    for amp in params["mtpa_currents"]:
        ctx.check_stop()
        g, _ = m.mtpa(float(amp), n_positions=4, tol=math.radians(1.0))
        op = m.operating_point(I_peak=float(amp), gamma=g, n_positions=int(params["operating_positions"]))
        ind = m.inductances(op["i_d"], op["i_q"])
        mtpa.append({"I": float(amp), "gamma_deg": math.degrees(g), "T": op["torque_mean"], "ripple": op["ripple"],
                     "i_d": op["i_d"], "i_q": op["i_q"], "L_d_mH": 1e3 * ind["L_dd"], "L_q_mH": 1e3 * ind["L_qq"]})
        for k in ("gamma_deg", "T", "ripple", "L_d_mH", "L_q_mH"):
            ctx.metric(f"mtpa_{amp:.0f}A.{'torque_Nm' if k == 'T' else k}", mtpa[-1][k])
    ctx.table("MTPA", ["I peak [A]", "gamma [deg]", "torque [N m]", "ripple", "L_d [mH]", "L_q [mH]"],
              [[r["I"], r["gamma_deg"], r["T"], r["ripple"], r["L_d_mH"], r["L_q_mH"]] for r in mtpa])
    fit = fit_iron_loss(M400_50A, f_max=1000.0)
    ctx.metric("bertotti_fit.rms_rel_error", fit.rms_rel_error)
    top = mtpa[-1]
    tab = m.losses(1500.0, top["i_d"], top["i_q"], steps=int(params["loss_steps"]))
    bert = m.losses(1500.0, top["i_d"], top["i_q"], steps=int(params["loss_steps"]), loss_model=fit.model)
    for k in ("stator_iron_W", "rotor_iron_W", "magnet_W"):
        ctx.metric(f"losses_1500rpm.{k}_table", tab[k])
    ctx.metric("losses_1500rpm.stator_iron_W_bertotti", bert["stator_iron_W"])
    if params["verification"]:
        res = mv.run(quick=bool(params["quick"]))
        ctx.table("machine verification", ["case", "error", "passed", "detail"],
                  [[r.name, r.error, r.passed, r.detail] for r in res])
        ctx.metric("verification_passed", sum(bool(r.passed) for r in res), f"of {len(res)} cases")
    sol = m.solve(0.0, m.currents_dq(0.0, top["i_d"], top["i_q"]))
    f = _field_figure(ctx, sol, "pmsm_field", f"MTPA at {top['I']:.0f} A: |B| and flux lines", scale=1e3,
                      xlim=(0, 140), ylim=(-70, 70), clim=(0, 2.2))
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(curve["phi0_deg"], curve["torque"], "o-", label="this FEA (Maxwell stress)")
    ax[0].plot(curve["phi0_deg"], mv.YANG_2013_TORQUE_NM, "s--", label="Yang et al. 2013")
    ax[0].set(xlabel="current angle Phi0 (deg)", ylabel="mean torque (N m)", title="250 A peak")
    ax[0].legend()
    ax[1].plot(np.degrees(nl["theta_cogging"]), nl["cogging"], "o-")
    ax[1].set(xlabel="rotor angle (mech. deg)", ylabel="cogging torque (N m)", title="one cogging period")
    fig.tight_layout()
    ctx.figure(fig, "pmsm_curves", "torque against current angle and cogging")
    ctx.metric("seconds", round(time.perf_counter() - t0, 1))
    ctx.summary(f"n_gap {n_gap}: cogging {np.ptp(nl['cogging']):.2f} N m pk-pk, MTPA {top['T']:.1f} N m at "
                f"{top['I']:.0f} A")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("em.prius_flux_map", "Prius flux map (FEA)", category="Electromagnetics",
      outputs=[Port("flux_map", "flux_map"), Port("metrics", "metrics")],
      params=[Param("source", "enum", "fea", choices=["fea", "example"],
                    help="fea: sample the machine; example: the flux map committed with examples/electrodynamics "
                         "(results/11_prius_fluxmap.npz, the cache 11_prius_drive.py reads)"),
              Param("n_gap", "int", 480, minimum=96, maximum=3840), Param("I_max", "float", 250.0, minimum=1.0),
              Param("n_d", "int", 9, minimum=3, maximum=41), Param("n_q", "int", 9, minimum=3, maximum=41),
              Param("n_positions", "int", 12, minimum=2, maximum=96),
              Param("loss_n", "int", 5, minimum=2, maximum=21), Param("loss_steps", "int", 24, minimum=6, maximum=360)],
      calls="pinneapple_simulation.electromagnetics.machines.drive.FluxMap.from_fea (IPMMachine(prius_2004())) / "
            "from_npz + to_npz")
def prius_flux_map(ctx, inputs, params):
    """psi_d, psi_q and torque on an i_d x i_q grid (and loss data on a coarser grid) of the Prius motor."""
    from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004
    from pinneapple_simulation.electromagnetics.machines.drive import FluxMap

    from .examples import example_dir
    t0 = time.perf_counter()
    if params["source"] == "example":
        fm = FluxMap.from_npz(example_dir("electrodynamics") / "results" / "11_prius_fluxmap.npz")
        ctx.warn("flux map read from the example's committed cache, not recomputed")
    else:
        m = IPMMachine(prius_2004(), n_gap=int(params["n_gap"]))
        fm = FluxMap.from_fea(m, float(params["I_max"]), n_d=int(params["n_d"]), n_q=int(params["n_q"]),
                              n_positions=int(params["n_positions"]), loss_n=int(params["loss_n"]),
                              loss_steps=int(params["loss_steps"]), progress=ctx.log)
    p = ctx.path("flux_map.npz")
    fm.to_npz(p)
    ctx.metric("grid", f"{len(fm.i_d)} x {len(fm.i_q)}")
    ctx.metric("T_max_Nm", float(np.nanmax(fm.T)))
    ctx.metric("seconds", round(time.perf_counter() - t0, 1))
    return {"flux_map": artifact(p, "npz"), "metrics": ctx.report["metrics"]}


@node("em.prius_drive", "Prius drive: envelope, efficiency map, WLTC energy", category="Electromagnetics",
      inputs=[Port("flux_map", "flux_map")],
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("I_max", "float", 250.0, minimum=1.0), Param("V_dc", "float", 500.0, minimum=1.0),
              Param("n_d", "int", 241, minimum=11, maximum=2001, help="d-axis search points of the drive"),
              Param("envelope_speeds", "int", 25, minimum=3, maximum=201, help="0..6000 rpm"),
              Param("map_speeds", "int", 23, minimum=3, maximum=201, help="500..6000 rpm (plus 100, 250)"),
              Param("map_fractions", "int", 25, minimum=2, maximum=201, help="per sign, of the envelope"),
              Param("vehicle", "dict", {"mass": 1450.0, "c_rr": 0.009, "cd_a": 0.58, "wheel_radius": 0.30,
                                        "gear_ratio": 4.113, "gear_efficiency": 0.97},
                    help="illustrative compact car (machines.drive.Vehicle)")],
      calls="pinneapple_simulation.electromagnetics.machines.drive: Drive(FluxMap, prius_phase_resistance, "
            "v_max_svpwm).max_torque / efficiency_map, Vehicle, cycle_energy(wltc_class3b)")
def prius_drive(ctx, inputs, params):
    """examples/electrodynamics/11_prius_drive.py: MTPA + flux weakening envelope, the efficiency map and the
    battery energy over the WLTC class 3b cycle."""
    from pinneapple_simulation.electromagnetics.machines.drive import (
        Drive,
        FluxMap,
        Vehicle,
        cycle_energy,
        prius_phase_resistance,
        v_max_svpwm,
        wltc_class3b,
    )
    fm = FluxMap.from_npz(inputs["flux_map"]["$artifact"])
    R, wdata = prius_phase_resistance()
    drive = Drive(fm, R=R, I_max=float(params["I_max"]), V_max=v_max_svpwm(float(params["V_dc"])),
                  n_d=int(params["n_d"]))
    ctx.metric("assumptions.R_phase_ohm", float(R))
    speeds = np.linspace(0.0, 6000.0, int(params["envelope_speeds"])) * RPM
    env = [drive.max_torque(w) for w in speeds]
    for w, e in zip(speeds, env, strict=True):
        ctx.metric(f"envelope.torque_{w / RPM:.0f}rpm_Nm", e["T"])
    map_rpm = np.concatenate([[100.0, 250.0], np.linspace(500.0, 6000.0, int(params["map_speeds"]))])
    nf = int(params["map_fractions"])
    frac = np.concatenate([-np.linspace(1.0, 0.02, nf), np.linspace(0.02, 1.0, nf)])
    em = drive.efficiency_map(map_rpm * RPM, frac, relative=True)
    eff = em["efficiency"]
    ctx.metric("efficiency_map.peak_efficiency", float(np.nanmax(eff)))
    t, v = wltc_class3b()
    E = cycle_energy(drive, Vehicle(**params["vehicle"]), t, v, envelope_speeds=int(params["envelope_speeds"]))
    for k in ("battery_J", "motoring_J", "regenerated_J", "drive_losses_J", "distance_m", "battery_Wh_per_km"):
        ctx.metric(f"wltc_class3b.{k}", float(E[k]))
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(speeds / RPM, [e["T"] for e in env], "k-", lw=2)
    ax[0].set(xlabel="speed (rpm)", ylabel="max torque (N m)", title="torque envelope (MTPA + flux weakening)")
    R_, T_ = np.meshgrid(map_rpm, np.arange(eff.shape[1]), indexing="ij")
    cs = ax[1].contourf(np.repeat(map_rpm[:, None], eff.shape[1], 1), em["T"], 100 * eff,
                        levels=[50, 70, 80, 85, 88, 90, 92, 93, 94, 95, 96, 97, 98, 99], cmap="viridis")
    fig.colorbar(cs, ax=ax[1], label="efficiency (%)")
    ax[1].set(xlabel="speed (rpm)", ylabel="torque (N m)", title="efficiency map")
    fig.tight_layout()
    f = ctx.figure(fig, "prius_drive", "envelope and efficiency map")
    ctx.summary(f"peak efficiency {100 * np.nanmax(eff):.1f} %, WLTC {E['battery_Wh_per_km']:.1f} Wh/km")
    return {"metrics": ctx.report["metrics"], "figure": f}


@node("em.prius_skew", "Prius step skew (multi-slice FEA)", category="Electromagnetics",
      outputs=[Port("metrics", "metrics")],
      params=[Param("n_gap", "int", 960, minimum=96, maximum=3840),
              Param("positions", "int", 12, minimum=2, maximum=96, help="rotor positions (load and cogging)"),
              Param("fractions", "list", [0.0, 0.25, 0.5, 0.75, 1.0], help="total skew in slot pitches"),
              Param("steps", "list", [2, 3], help="number of axial slices"),
              Param("emf_positions", "int", 120, minimum=12, maximum=720)],
      calls="pinneapple_simulation.electromagnetics.machines.robustness.skewed_no_load / skewed_torque / "
            "step_skew_factor (IPMMachine(prius_2004()))")
def prius_skew(ctx, inputs, params):
    """examples/electrodynamics/15_prius_skew.py: cogging, torque and ripple against the step skew, and the skewed
    back-EMF harmonics against the closed-form skew factor (the 3-D VTU export is left to the script)."""
    from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004
    from pinneapple_simulation.electromagnetics.machines.robustness import (
        skewed_no_load,
        skewed_torque,
        step_skew_factor,
    )
    two_pi = 2 * math.pi
    m = IPMMachine(prius_2004(), n_gap=int(params["n_gap"]))
    slot = two_pi / m.geometry.stator.slots
    gm = math.radians(50.3)
    i_d, i_q = -250 * math.sin(gm), 250 * math.cos(gm)
    th_load = m.positions(int(params["positions"]))
    th_cog = m.positions(int(params["positions"]), period=m.cogging_period)
    rows = []
    for K in params["steps"]:
        for frac in params["fractions"]:
            ctx.check_stop()
            total = float(frac) * slot
            s = total / (int(K) - 1)
            cog = skewed_no_load(m, th_cog, int(K), s)["cogging"]
            T = skewed_torque(m, i_d, i_q, th_load, int(K), s)
            r = [int(K), float(frac), float(np.ptp(cog)), float(T.mean()), float(np.ptp(T) / T.mean())]
            rows.append(r)
            tag = f"K{K}_skew{frac:g}"
            ctx.metric(f"sweep.{tag}.cogging_pkpk_Nm", r[2])
            ctx.metric(f"sweep.{tag}.torque_mean_Nm", r[3])
            ctx.metric(f"sweep.{tag}.ripple_pkpk", r[4])
    ctx.table("step skew at 250 A, 50.3 deg", ["slices", "skew [slot pitches]", "cogging pk-pk [N m]",
                                                "mean torque [N m]", "ripple pk-pk"], rows)
    nf = int(params["emf_positions"])
    th_e = m.positions(nf, period=two_pi / m.pole_pairs)
    psi0 = np.array([m.flux_linkage(m.solve(t)) for t in th_e])
    psis = skewed_no_load(m, th_e, 2, slot)["psi"]
    e0, es = m.back_emf(psi0, th_e, 1000.0), m.back_emf(psis, th_e, 1000.0)
    nh = min(25, len(e0["harmonics_phase"]) - 1)
    H0, Hs = np.array(e0["harmonics_phase"][:nh]), np.array(es["harmonics_phase"][:nh])
    fac = step_skew_factor(np.arange(1, nh + 1), 2, slot, two_pi / m.pole_pairs)
    sig = H0 > 1e-3 * H0[0]
    ctx.metric("back_emf.fundamental_ratio", float(Hs[0] / H0[0]), "two-step skew by one slot pitch")
    ctx.metric("back_emf.factor_fundamental", float(fac[0]), "closed-form step-skew factor")
    ctx.metric("back_emf.max_deviation_from_factor_rel_fundamental",
               float((np.abs(Hs[sig] - fac[sig] * H0[sig]) / H0[0]).max()))
    ctx.metric("back_emf.thd_straight", e0["thd"])
    ctx.metric("back_emf.thd_skewed", es["thd"])
    return {"metrics": ctx.report["metrics"]}


@node("em.prius_tolerances", "Prius manufacturing tolerances (Monte Carlo FEA)", category="Electromagnetics",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("samples", "int", 200, minimum=4, maximum=5000), Param("seed", "int", 0, minimum=0),
              Param("n_gap", "int", 480, minimum=96, maximum=3840),
              Param("positions", "int", 12, minimum=2, maximum=96),
              Param("gamma_deg", "float", 50.3),
              Param("nominal", "list", [1.0, 0.0065, 0.0189, 0.00075],
                    help="Br scale [-], magnet thickness, magnet width, air gap [m]"),
              Param("sigma", "list", [0.01, 0.025e-3, 0.05e-3, 0.025e-3])],
      calls="pinneapple_analysis.uncertainty.tolerance.propagate + src over "
            "pinneapple_simulation.electromagnetics.machines.robustness.tolerance_model(prius_2004())")
def prius_tolerances(ctx, inputs, params):
    """examples/electrodynamics/17_prius_tolerances.py: torque and ripple spread from the tolerances of the magnet
    remanence, thickness, width and the air gap; standardised regression coefficients; probability within spec."""
    from pinneapple_analysis.uncertainty.tolerance import propagate, src
    from pinneapple_simulation.electromagnetics.machines import prius_2004
    from pinneapple_simulation.electromagnetics.machines.robustness import (
        TOLERANCE_VARIABLES,
        tolerance_model,
    )
    NOM, SIG = np.array(params["nominal"], float), np.array(params["sigma"], float)
    if NOM.shape != (len(TOLERANCE_VARIABLES),) or SIG.shape != NOM.shape:
        raise StepFailed(f"nominal and sigma: {len(TOLERANCE_VARIABLES)} values ({', '.join(TOLERANCE_VARIABLES)})")
    fn = tolerance_model(prius_2004(), gamma_deg=float(params["gamma_deg"]), n_gap=int(params["n_gap"]),
                         n_positions=int(params["positions"]))
    T0, r0 = fn(NOM[None, :])[0]
    ctx.metric("nominal_outputs.torque_Nm", float(T0))
    ctx.metric("nominal_outputs.ripple", float(r0))
    spec = (np.array([0.98 * T0, -np.inf]), np.array([np.inf, 0.25]))
    res = propagate(fn, NOM, SIG, n=int(params["samples"]), seed=int(params["seed"]), limits=spec)
    ok = np.all(np.isfinite(res.outputs), axis=1)
    X, Y = res.inputs[ok], res.outputs[ok]
    ctx.metric("valid_samples", int(ok.sum()))
    for j, name in enumerate(("torque_Nm", "ripple")):
        y = Y[:, j]
        coef, r2 = src(X, y)
        ctx.metric(f"outputs.{name}.mean", float(y.mean()))
        ctx.metric(f"outputs.{name}.std", float(y.std(ddof=1)))
        ctx.metric(f"outputs.{name}.p_within_spec", float(res.p_within_each[j]))
        ctx.metric(f"outputs.{name}.src_r2", float(r2))
        for v, c in zip(TOLERANCE_VARIABLES, coef, strict=True):
            ctx.metric(f"outputs.{name}.src.{v}", float(c))
    ctx.metric("p_within_spec_joint", float(res.p_within))
    ctx.metric("p_within_spec_joint_wilson95_low", float(res.p_within_ci[0]))
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(Y[:, 0], 20)
    ax[0].axvline(spec[0][0], color="r", ls="--", label="spec: 98 % of nominal")
    ax[0].set(xlabel="mean torque (N m)", title=f"{len(Y)} samples")
    ax[0].legend()
    ax[1].hist(100 * Y[:, 1], 20)
    ax[1].axvline(25, color="r", ls="--")
    ax[1].set(xlabel="ripple pk-pk (%)")
    f = ctx.figure(fig, "prius_tolerances", "torque and ripple under manufacturing tolerances")
    return {"metrics": ctx.report["metrics"], "figure": f}


# ------------------------------------------------------------------------------------------- planner capabilities
_MACHINE = ("motor", "machine", "máquina", "motor elétrico", "ipm", "pmsm", "prius", "rotor", "stator")


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    em = dict(domain="electromagnetics")
    register_capability(Capability(
        id="em.spm_machine", title="Surface-PM machine: cogging, torque vs current angle, back-EMF, losses",
        keywords={"surface-pm": 3, "surface pm": 3, "surface permanent": 3, "spm": 3, "cogging": 2,
                  "back-emf": 1.5, "back emf": 1.5, "permanent magnet": 1.5, "24 slots": 1, "24-slot": 1,
                  "ímã": 1, "imã": 1}, requires_all=(_MACHINE + ("cogging", "torque"),), min_score=3,
        nodes=("em.spm_machine", "examples.compare"), references=("brenner2008fem",),
        inputs=("machine dimensions (illustrative defaults), rated current, speed",),
        template="electromagnetics/motor_spm_cogging_and_torque", example="motor_magnetics",
        description="2-D A_z FEM with Maxwell stress and co-energy torques "
                    "(examples/motor_magnetics/01_spm_cogging_and_torque.py)", **em))
    register_capability(Capability(
        id="em.prius_machine", title="Interior-PM traction motor (2004 Prius): FEA, MTPA, losses",
        keywords={"prius": 3, "interior pm": 2.5, "interior permanent": 2.5, "ipm": 2.5, "mtpa": 2,
                  "traction motor": 2, "l_d": 1, "inductance": 1}, min_score=3,
        nodes=("em.prius_machine", "examples.compare"), references=("brenner2008fem",),
        template="electromagnetics/10_pmsm_prius", quick=True, example="electrodynamics",
        description="nonlinear 2-D FEA against Yang et al. (IEMDC 2013) (examples/electrodynamics/10_pmsm_prius.py)",
        **em))
    register_capability(Capability(
        id="em.prius_drive", title="Traction drive: torque-speed envelope, efficiency map, WLTC energy",
        keywords={"wltc": 3, "efficiency map": 3, "flux weakening": 2.5, "drive cycle": 2, "torque-speed": 2,
                  "envelope": 1, "wh/km": 2}, requires_all=(_MACHINE + ("drive", "inverter", "vehicle", "car"),),
        min_score=3, nodes=("em.prius_flux_map", "em.prius_drive", "examples.compare"),
        references=("brenner2008fem",), template="electromagnetics/11_prius_drive", quick=True,
        example="electrodynamics", description="flux map, MTPA + flux weakening, WLTC class 3b "
                                               "(examples/electrodynamics/11_prius_drive.py)", **em))
    register_capability(Capability(
        id="em.prius_skew", title="Rotor step skew: cogging, ripple, back-EMF harmonics",
        keywords={"skew": 3, "step skew": 3, "multi-slice": 2}, requires_all=(_MACHINE,), min_score=3,
        nodes=("em.prius_skew", "examples.compare"), references=("brenner2008fem",),
        template="electromagnetics/15_prius_skew", quick=True, example="electrodynamics",
        description="multi-slice 2-D FEA (examples/electrodynamics/15_prius_skew.py)", **em))
    register_capability(Capability(
        id="em.prius_tolerances", title="Manufacturing tolerances of a PM motor (Monte Carlo FEA)",
        keywords={"tolerance": 2.5, "tolerances": 2.5, "manufacturing": 1.5, "monte carlo": 1.5, "tolerância": 2.5},
        requires_all=(_MACHINE,), min_score=3, nodes=("em.prius_tolerances", "examples.compare"),
        references=("mckay1979lhs",), template="electromagnetics/17_prius_tolerances", quick=True,
        example="electrodynamics", description="examples/electrodynamics/17_prius_tolerances.py", **em))
    register_capability(Capability(
        id="em.verification", title="Electromagnetics solvers against independent references",
        keywords={"electromagnetic": 1.5, "electromagnetics": 1.5, "maxwell": 1, "mie": 1.5, "biot-savart": 2,
                  "fresnel": 1.5, "verification": 2, "verify": 1.5}, min_score=3.5,
        requires_all=(("electromagnetic", "electromagnetics", "maxwell", "mie", "biot", "fresnel"),),
        nodes=("em.verification", "examples.compare"), references=("brenner2008fem",),
        template="electromagnetics/07_em_verification", example="electrodynamics",
        description="examples/electrodynamics/07_em_verification.py", **em))


_register_capabilities()

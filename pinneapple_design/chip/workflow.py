"""The standard chip-design workflow (#1206): floorplan, die and package thermal with the leakage-temperature loop,
power-grid IR drop with electromigration, Elmore delay of the critical nets, a placement and power-grid design, and
a finer thermal model that re-verifies it.

1. **Floorplan** (:mod:`.floorplan`): blocks with dynamic and leakage power in placement slots, critical nets.
2. **Thermal** (:mod:`.thermal`): a 3-D conduction network of the die stack (board, C4, die, TIM, lid, heat sink)
   with the leakage of every cell at its own temperature, iterated to the fixed point; the loop gain is measured
   and a gain >= 1 is thermal runaway. Checks: the energy balance (sink + board heat = power) and the 1-D stack
   against its closed-form resistance network.
3. **Placement**: every assignment of blocks to slots screened by superposition of per-slot responses (exact for
   the linear model at fixed power), the critical-net delays as constraints; the best re-solved with the leakage
   loop.
4. **Power grid** (:mod:`.power_grid`): IR drop on the strap mesh with the block powers at their temperatures and the
   strap resistivity at the local temperature; electromigration against Black's equation; the least strap width that
   meets the IR-drop and lifetime limits, within the metal-area budget. Checked: the mesh solver against the exact
   discrete-sine solution of a uniform grid, and that grid's continuum limit.
5. **Interconnect** (:mod:`.interconnect`): Elmore delay of each critical net with the wire at its temperature;
   checked against a SPICE-like trapezoidal transient (first moment equal to Elmore, 50 % delay below it).
6. **Re-verification**: the chosen design with a finer thermal mesh (lateral and vertical refinement).
7. **Lithography**: printability of the critical layers is the ``litho.opc`` node of #1180 (bundled in the flow).
"""
from __future__ import annotations

import itertools
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .floorplan import Block, Floorplan, default_floorplan
from .interconnect import WireSpec, elmore_delay, net_delay, rc_ladder, transient_step, wire_rc
from .power_grid import (
    POISSON_SQUARE_MAX,
    GridSpec,
    mesh_drop,
    solve_grid,
    uniform_grid_closed_form,
)
from .thermal import (
    DieStack,
    _Operator,
    default_stack,
    slot_responses,
    solve_thermal,
    stack_closed_form,
)

__all__ = ["chip_design_workflow", "floorplan_from_dict"]


def floorplan_from_dict(d: Mapping[str, Any]) -> Floorplan:
    """``{"die_mm": [w, h], "slots": [nx, ny], "blocks": [{"name", "dynamic_w", "leakage_w", ...}],
    "placement": [...], "nets": [[a, b], ...]}``."""
    blocks = [Block(str(b["name"]), float(b["dynamic_w"]), float(b.get("leakage_w", 0.0)),
                    float(b.get("leak_beta", 0.02)), float(b.get("t_ref_c", 85.0))) for b in d["blocks"]]
    w, h = d.get("die_mm", (10.0, 10.0))
    return Floorplan(float(w), float(h), blocks, tuple(d.get("slots", (3, 2))),
                     list(d.get("placement", range(len(blocks)))), [tuple(n) for n in d.get("nets", [])])


def _net_delays(fp: Floorplan, placement, wire: WireSpec, tj: np.ndarray | None) -> dict[str, dict[str, float]]:
    out = {}
    pos = {b.name: fp.slot_centre_mm(s) for b, s in zip(fp.blocks, placement, strict=True)}
    for a, b in fp.nets:
        L = (abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1])) * 1e-3
        if tj is None:
            t_w = 25.0
        else:                                       # mean junction temperature under the L-shaped route
            nx, ny = tj.shape
            xs = np.linspace(pos[a][0], pos[b][0], 20) / fp.die_w_mm
            ys = np.linspace(pos[a][1], pos[b][1], 20) / fp.die_h_mm
            pts = np.concatenate([np.stack([xs, np.full_like(xs, ys[0])], 1), np.stack([np.full_like(ys, xs[-1]),
                                                                                         ys], 1)])
            ix = np.clip((pts[:, 0] * nx).astype(int), 0, nx - 1)
            iy = np.clip((pts[:, 1] * ny).astype(int), 0, ny - 1)
            t_w = float(tj[ix, iy].mean())
        out[f"{a}-{b}"] = {**net_delay(L, wire, t_w), "t_wire_c": t_w}
    return out


def chip_design_workflow(*, floorplan: Floorplan | Mapping[str, Any] | None = None, stack: DieStack | None = None,
                         grid: GridSpec | None = None, wire: WireSpec | None = None, nx: int = 24,
                         t_max_c: float = 105.0, ir_drop_max: float = 0.05, delay_budget_ps: float = 200.0,
                         metal_fraction_max: float = 0.30, mttf_target_years: float = 10.0,
                         width_range_um: Sequence[float] = (0.5, 30.0), optimize_placement: bool = True,
                         refine: int = 2, refine_tolerance_k: float = 1.0, seed: int = 0) -> dict[str, Any]:
    """Run the workflow; see the module docstring. ``floorplan``: a :class:`Floorplan` or its dict form (default
    :func:`default_floorplan`)."""
    t_start = time.perf_counter()
    fp = default_floorplan() if floorplan is None else (floorplan if isinstance(floorplan, Floorplan) else
                                                       floorplan_from_dict(floorplan))
    stack = stack or default_stack()
    grid = grid or GridSpec()
    wire = wire or WireSpec()
    W, H = fp.die_w_mm * 1e-3, fp.die_h_mm * 1e-3
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        status = "skip" if ok == "skip" else ("inconclusive" if ok is None else ("pass" if ok else "fail"))
        checks.append({"code": code, "status": status, "detail": detail, "value": value})

    # ---- verification of the thermal network: 1-D stack against its closed form
    one = Floorplan(fp.die_w_mm, fp.die_h_mm, [Block("uniform", 50.0)], (1, 1), [0])
    r1 = solve_thermal(one, stack, nx=6, leakage=False)
    cf = stack_closed_form(stack, W * H, 50.0)
    e1 = float(np.max(np.abs(r1["T_junction"] - cf["T_j_c"])) / (cf["T_j_c"] - stack.t_ambient_c))
    chk("verify.stack_1d", e1 < 1e-9, f"uniform 50 W: FV junction {r1['T_max_c']:.6f} C against the series-parallel "
                                      f"network {cf['T_j_c']:.6f} C (R_ja {cf['R_ja_k_w']:.5f} K/W, "
                                      f"{cf['fraction_up']:.1%} of the heat to the sink): {e1:.2g} of the rise", e1)

    # ---- thermal of the given placement, with the leakage loop
    op = _Operator(stack, W, H, nx, nx)
    th0 = solve_thermal(fp, stack, nx=nx, operator=op)
    T = th0["T"]
    g_top, g_bot = op.sys.g_top, op.sys.g_bottom
    q_out = float((g_top * (T[:, :, -1] - stack.t_ambient_c)).sum() + (g_bot * (T[:, :, 0] - stack.t_ambient_c)).sum())
    e_bal = abs(q_out - th0["power_w"]) / th0["power_w"]
    chk("thermal.energy_balance", e_bal < 1e-9, f"heat to the sink and the board {q_out:.6g} W = power "
                                                f"{th0['power_w']:.6g} W (incl. leakage {th0['leakage_w']:.4g} W): "
                                                f"{e_bal:.2g}", e_bal)
    chk("thermal.no_runaway", th0["converged"] and not th0["runaway"],
        f"leakage-temperature loop converged in {th0['iterations']} iterations, loop gain {th0['loop_gain']:.3g} "
        f"(< 1: stable)")

    # ---- placement search
    resp = slot_responses(fp, op)
    p_block = np.array([b.dynamic_w + b.leakage_w * math.exp(b.leak_beta * (th0["block_t_max_c"][b.name]
                                                                             - b.t_ref_c)) for b in fp.blocks])
    budget = delay_budget_ps * 1e-12
    cand = []
    perms = itertools.permutations(range(fp.n_slots), len(fp.blocks)) if optimize_placement else [fp.placement]
    for pl in perms:
        tmap = stack.t_ambient_c + np.tensordot(p_block, resp[list(pl)], axes=1)
        tmax = float(tmap.max())
        d = _net_delays(fp, pl, wire, tmap)                 # wires at the screened temperature
        worst = max((v["elmore_s"] for v in d.values()), default=0.0)
        cand.append((tmax, worst, list(pl)))
    n_cand = len(cand)
    feas = [c for c in cand if c[1] <= budget]
    # the coolest placement that meets the budget; when none does, the one with the least worst delay
    best = min(feas, key=lambda c: (c[0], c[1])) if feas else min(cand, key=lambda c: (c[1], c[0]))
    placement = best[2]
    th = solve_thermal(fp, stack, nx=nx, placement=placement, operator=op)
    screen_err = abs(best[0] - th["T_max_c"])
    chk("placement.feasible", bool(feas), f"{len(feas)} of {n_cand} placements meet the {delay_budget_ps:g} ps net "
                                          f"budget with the wires at their temperature; the best has peak {th['T_max_c']:.2f} C with the leakage "
                                          f"loop (initial placement {th0['T_max_c']:.2f} C; superposition screen "
                                          f"{best[0]:.2f} C)", len(feas))
    chk("thermal.t_max", th["T_max_c"] <= t_max_c, f"peak junction {th['T_max_c']:.2f} C in {th['hotspot_block']} "
                                                   f"(limit {t_max_c:g} C), theta_ja {th['theta_ja_k_w']:.4f} K/W",
        th["T_max_c"])

    # ---- power grid at the chosen placement: block powers at their temperatures
    Tj = th["T_junction"]
    qmap = np.zeros_like(Tj)
    masks = [fp.slot_mask(s, nx, nx) for s in placement]
    for b, m in zip(fp.blocks, masks, strict=True):
        qmap[m] = b.dynamic_w / m.sum() + b.leakage_w / m.sum() * np.exp(b.leak_beta * (Tj[m] - b.t_ref_c))

    def pg(width):
        return solve_grid(qmap, W, H, grid, width_m=width, temperature_c=Tj)
    lo, hi = width_range_um[0] * 1e-6, width_range_um[1] * 1e-6
    ok = lambda r: r["max_drop_fraction"] <= ir_drop_max and r["mttf_min_years"] >= mttf_target_years  # noqa: E731
    r_hi = pg(hi)
    if not ok(r_hi):
        width, r_pg, feasible_w = hi, r_hi, False
    else:
        for _ in range(30):
            mid = math.sqrt(lo * hi)
            lo, hi = (lo, mid) if ok(pg(mid)) else (mid, hi)
            if hi / lo < 1.002:
                break
        width, r_pg, feasible_w = hi, pg(hi), True
    chk("power_grid.ir_drop", r_pg["max_drop_fraction"] <= ir_drop_max,
        f"strap width {width * 1e6:.3g} um: worst IR drop {r_pg['max_drop_v'] * 1e3:.2f} mV = "
        f"{r_pg['max_drop_fraction']:.2%} of {grid.vdd:g} V (limit {ir_drop_max:.0%})", r_pg["max_drop_fraction"])
    chk("power_grid.electromigration", r_pg["mttf_min_years"] >= mttf_target_years,
        f"worst segment J {r_pg['j_max_a_m2'] / 1e10:.3g} MA/cm^2, Black MTTF {r_pg['mttf_min_years']:.3g} years at "
        f"its temperature (target {mttf_target_years:g})", r_pg["mttf_min_years"])
    chk("power_grid.area", r_pg["metal_fraction"] <= metal_fraction_max and feasible_w,
        f"top-metal area used by the straps {r_pg['metal_fraction']:.1%} (budget {metal_fraction_max:.0%})",
        r_pg["metal_fraction"])
    chk("power_grid.current_balance", r_pg["current_balance"] < 1e-9,
        f"bumps deliver {r_pg['bump_current_a']:.6g} A = load {r_pg['current_a']:.6g} A: {r_pg['current_balance']:.2g}",
        r_pg["current_balance"])
    # verification of the mesh solver: uniform grid, ideal boundary, against the discrete sine solution
    n = 39
    g_seg, i0 = 2.0, 1e-3
    fixed = np.zeros((n + 2, n + 2), bool)
    fixed[[0, -1], :] = True
    fixed[:, [0, -1]] = True
    load = np.zeros((n + 2, n + 2))
    load[1:-1, 1:-1] = i0
    dm = mesh_drop(load, np.full((n + 1, n + 2), g_seg), np.full((n + 2, n + 1), g_seg), fixed=fixed)
    dc = uniform_grid_closed_form(n, g_seg, i0)
    e_mesh = float(np.max(np.abs(dm[1:-1, 1:-1] - dc)) / dc.max())
    cont = POISSON_SQUARE_MAX * i0 / g_seg * (n + 1) ** 2
    chk("verify.ir_uniform_grid", e_mesh < 1e-10,
        f"{n} x {n} uniform mesh with an ideal pad ring: sparse solve against the exact discrete-sine solution "
        f"{e_mesh:.2g}; the continuum limit 0.07367 j R_s L^2 gives {cont * 1e3:.4f} mV vs {dc.max() * 1e3:.4f} mV "
        f"({abs(dc.max() / cont - 1):.2%}, O(h^2))", e_mesh)

    # ---- interconnect at the chosen placement, wires at their temperature
    delays = _net_delays(fp, placement, wire, Tj)
    worst_net = max(delays, key=lambda k: delays[k]["elmore_s"]) if delays else None
    if worst_net:
        dw = delays[worst_net]
        chk("interconnect.delay", dw["elmore_s"] <= budget,
            f"worst critical net {worst_net}: {dw['length_mm']:.3g} mm at {dw['t_wire_c']:.1f} C, Elmore "
            f"{dw['elmore_s'] * 1e12:.2f} ps (budget {delay_budget_ps:g} ps)", dw["elmore_s"])
        r_w, c_w = wire_rc(dw["length_mm"] * 1e-3, wire, dw["t_wire_c"])
        R, C = rc_ladder(r_w, c_w, r_driver=wire.r_driver_ohm, c_load=wire.c_load_f, n=wire.segments)
        tau = elmore_delay(R, C)
        tr = transient_step(R, C, t_end=12 * tau, dt=tau / 4000)
        e_m1 = abs(tr["first_moment_s"] - tau) / tau
        chk("verify.elmore_transient", e_m1 < 1e-3 and tr["t50_s"] <= tau,
            f"{worst_net} as a {wire.segments}-segment RC ladder, trapezoidal MNA transient: first moment "
            f"{tr['first_moment_s'] * 1e12:.3f} ps vs Elmore {tau * 1e12:.3f} ps ({e_m1:.2g}); 50 % delay "
            f"{tr['t50_s'] * 1e12:.3f} ps <= Elmore (ratio {tr['t50_s'] / tau:.3f})", e_m1)
        transient = {"net": worst_net, "elmore_s": tau, "t50_s": tr["t50_s"], "first_moment_s": tr["first_moment_s"],
                     "t50_over_elmore": tr["t50_s"] / tau}
    else:
        transient = {}

    # ---- re-verification on a finer thermal model
    fine = solve_thermal(fp, stack.refined(refine), nx=nx * refine, placement=placement)
    dT = abs(fine["T_max_c"] - th["T_max_c"])
    chk("reverify.thermal_refined", dT <= refine_tolerance_k and fine["T_max_c"] <= t_max_c,
        f"{fine['grid']} cells ({fine['cells']}): peak {fine['T_max_c']:.2f} C vs {th['T_max_c']:.2f} C on "
        f"{th['grid']} ({dT:.3f} K, tolerance {refine_tolerance_k:g} K)", dT)
    chk("lithography.printability", "skip", "critical-layer printability: the litho.opc node (#1180) in the bundled "
                                            "flow corrects and verifies the clip over the process window")

    failed = [c["code"] for c in checks if c["status"] == "fail"]
    names = [b.name for b in fp.blocks]
    summary = (f"{len(fp.blocks)} blocks, {th['power_w']:.1f} W (leakage {th['leakage_w']:.2f} W): placement "
               f"{dict(zip(names, placement, strict=True))} peaks at {th['T_max_c']:.1f} C (initial "
               f"{th0['T_max_c']:.1f} C); straps {width * 1e6:.3g} um, IR drop {r_pg['max_drop_fraction']:.2%}, "
               f"EM {r_pg['mttf_min_years']:.3g} years"
               + (f"; worst net {worst_net} {delays[worst_net]['elmore_s'] * 1e12:.1f} ps" if worst_net else "")
               + f"; refined thermal {fine['T_max_c']:.2f} C; {sum(c['status'] == 'pass' for c in checks)} checks "
                 f"pass, {len(failed)} fail" + (f" ({', '.join(failed)})" if failed else ""))
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_") and k not in ("T", "T_junction")}  # noqa: E731
    return {"floorplan": fp.with_placement(placement).to_dict(), "initial_placement": list(fp.placement),
            "placement": placement, "screen_error_k": screen_err, "placements_screened": n_cand, "placements_feasible": len(feas),
            "thermal_initial": strip(th0), "thermal": strip(th), "thermal_refined": strip(fine),
            "stack_closed_form": cf, "junction_map_c": Tj.tolist(),
            "power_grid": {k: v for k, v in r_pg.items() if k != "drop"}, "strap_width_um": width * 1e6,
            "net_delays": delays, "transient": transient, "checks": checks, "failed": failed,
            "verdict": "pass" if not failed else "fail", "summary": summary,
            "assumptions": list(stack.assumptions) + [
                "power-grid straps on one top metal; bumps an ideal supply behind r_bump",
                "Black's constant from the stated reference condition (not a foundry rule)",
                "nets routed centre to centre, Manhattan, one wire class, no repeaters"],
            "seconds": time.perf_counter() - t_start, "seed": seed}


"""A heat-sink work cell with every stack: the part is designed, served, handled by a robot, seen by a camera and put
in one OpenUSD scene, and PINNeAPPle checks each hand-off.

1. B (NVIDIA USD BRep): exact solids of sampled heat-sink designs (fins, height, thickness, root fillets): exact mass.
2. P (PhysicsNeMo): a ``FullyConnected`` surrogate ``(n, H, t) -> (T rise, dp, mass)`` trained on the B-rep masses
   and PINNeAPPle's closed-form thermal rating.
3. N (PINNeAPPle Model Services): the surrogate served behind the NIM-shaped ``/v1/infer`` with a trust report per
   design; the client asks it for the lightest design meeting T <= 75 C and dp <= 30 Pa, and PINNeAPPle verifies the
   proposal with the exact solid and the closed form.
4. I (PINNeAPPle Robot Sim on NVIDIA Newton): the two-link arm carries the verified part (plus a 2 kg gripper);
   PINNeAPPle finds the shortest move that passes the motor, stress, tracking and energy checks.
5. C (PINNeAPPle World Model): synthetic side-camera frames of that move, with the check that the part in the pixels
   is where the kinematics put it.
6. O (Omniverse / OpenUSD): the move with the exact part mesh, the camera and every decision in one validated stage.

    python examples/nvidia/18_all_stacks_heatsink_cell/run.py      # ~15 min on 4 CPU cores
"""
import argparse
import json
import os

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_lab import experiment
from pinneapple_simulation.robot_sim import TwoLinkArm
from pinneapple_simulation.robot_sim.checks import arm_pick_and_place
from pinneapple_tools.model_services.client import ModelServiceClient
from pinneapple_tools.model_services.registry import Registry
from pinneapple_tools.model_services.server import serve_in_thread
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.heatsink import (
    Duty,
    design_table,
    sample_designs,
    surrogate_service,
    verify,
)
from pinneapple_tools.nvidia.omniverse import kit_render_script, render_stage_png, validate_stage
from pinneapple_tools.nvidia.physicsnemo import MLPSurrogate, relative_errors
from pinneapple_tools.nvidia.runinfo import RunInfo
from pinneapple_worldmodel.robot_perception import (
    CameraVariation,
    arm_motions,
    camera_frames,
    payload_track_check,
    sample_camera,
    visibility,
    write_motion_stage,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DUTY = Duty()
T_MAX, DP_MAX = 75.0, 30.0
GRIPPER_KG = 2.0
Q_START, Q_END = (-0.6, 1.2), (0.9, -0.4)
DURATIONS = (0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25)
PART_DROP = 0.07          # m: part centre below the tool point (gripper)


def _xy(rows):
    X = np.array([[r["n_fins"], r["fin_height"], r["fin_t"]] for r in rows], float)
    Y = np.array([[r["t_source"] - DUTY.t_ambient_c, r["dp"], r["mass"]] for r in rows], float)
    return X, Y


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("18_all_stacks_heatsink_cell")
    architecture_diagram(
        "18 Heat-sink work cell with every stack: PINNeAPPle + B + P + N + I + C + O",
        [Block("brep", "USD BRep: exact solids,\nexact mass", "brep", 0, 0),
         Block("cf", "Closed-form thermal\nrating (PINNeAPPle)", "pinneapple", 0, 1),
         Block("pn", "PhysicsNeMo surrogate:\n(n, H, t) -> T, dp, mass", "physicsnemo", 1, 0),
         Block("svc", "Model Services /v1/infer:\ndesigns + trust report", "services", 2, 0),
         Block("ver", "Verify the proposal:\nB-rep + closed form", "pinneapple", 2, 1),
         Block("rob", "Robot Sim (Newton): arm\ncarries part + gripper", "robotsim", 3, 0),
         Block("rchk", "Fastest move passing motor,\nstress, energy checks", "pinneapple", 3, 1),
         Block("cam", "Synthetic camera frames\nof the move", "worldmodel", 4, 0),
         Block("pix", "Part in the pixels where\nthe kinematics put it?", "pinneapple", 4, 1),
         Block("usd", "One USD stage: move, exact\npart mesh, camera, decisions", "omniverse", 5, 0),
         Block("kit", "Kit RTX / Isaac Sim /\nCosmos / NVIDIA NIM", "omniverse", 5, 1, ran=False)],
        [("brep", "pn", "mass"), ("cf", "pn", "T, dp"), ("pn", "svc", "model"), ("svc", "ver", "proposal"),
         ("ver", "rob", "mass"), ("rob", "rchk", "tau"), ("rob", "cam", "poses"), ("cam", "pix", "frames"),
         ("rchk", "usd", "move"), ("ver", "usd", "mesh"), ("pix", "usd", "decision"), ("usd", "kit", ".usda")],
        os.path.join(OUT, "architecture.png"),
        subtitle="Each stack hands its result to the next; PINNeAPPle checks every hand-off. Hatched: not run here")
    n_train, n_test, steps = (14, 5, 300) if smoke else (90, 20, 12000)
    with experiment("examples/nvidia/18_all_stacks_heatsink_cell", title="Heat-sink work cell: B-rep design, "
                    "PhysicsNeMo surrogate served as a model service, Newton robot move, synthetic camera, USD scene",
                    domain="engineering design", methods=["B-rep (SMLib)", "FullyConnected (PhysicsNeMo)",
                                                          "model service", "Newton Featherstone", "synthetic camera",
                                                          "OpenUSD"],
                    out_dir=base, register=False, modules=["pinneapple_tools.nvidia.heatsink",
                                                           "pinneapple_tools.model_services",
                                                           "pinneapple_simulation.robot_sim",
                                                           "pinneapple_worldmodel.robot_perception"],
                    params={"T_max_C": T_MAX, "dp_max_Pa": DP_MAX, "gripper_kg": GRIPPER_KG}) as ex:
        # B + closed form
        with info.step("designs: USD BRep + closed form"):
            train = design_table(sample_designs(n_train, seed=0), DUTY)
            test = design_table(sample_designs(n_test, seed=1), DUTY)
        X, Y = _xy(train)
        Xt, Yt = _xy(test)
        # P
        net = MLPSurrogate(3, 3, layer_size=64, num_layers=3)
        with info.step("surrogate (PhysicsNeMo FullyConnected)"):
            fit = net.fit(X, Y, ComputeBudget(steps=steps), batch_size=64, lr=3e-3)
        held = relative_errors(net.predict(Xt), Yt)
        # N
        nn_, hh, tt = np.meshgrid(np.arange(6, 21), np.linspace(0.015, 0.045, 31), np.linspace(0.001, 0.003, 21),
                                  indexing="ij")
        G = np.c_[nn_.ravel(), hh.ravel(), tt.ravel()]
        G = G[(DUTY.width - G[:, 0] * G[:, 2]) / (G[:, 0] - 1) >= 0.002]
        srv, url = serve_in_thread(Registry([surrogate_service(net, X, t_ambient=DUTY.t_ambient_c)]))
        try:
            with info.step("model service query (HTTP /v1/infer)"):
                ans = ModelServiceClient(url).infer("pinneapple-heatsink-surrogate", {"designs": G.tolist()})
        finally:
            srv.shutdown()
        o = ans["outputs"]
        T, dp, mass = (np.array(o[k]) for k in ("t_source", "dp", "mass"))
        trusted = np.array(ans["trust"]["per_design"])
        ok = trusted & (T <= T_MAX) & (dp <= DP_MAX)
        order = np.argsort(np.where(ok, mass, np.inf))[:int(ok.sum())]
        attempts = []
        with info.step("verify proposals (USD BRep + closed form)"):
            for i in order[:8]:
                v = verify(int(G[i, 0]), float(G[i, 1]), float(G[i, 2]), DUTY, chord_tolerances=[2e-4])
                good = v["t_source"] <= T_MAX and v["dp"] <= DP_MAX
                attempts.append({"n_fins": int(G[i, 0]), "fin_height_mm": G[i, 1] * 1e3, "fin_t_mm": G[i, 2] * 1e3,
                                 "served_t": float(T[i]), "served_mass": float(mass[i]),
                                 "verified_t": v["t_source"], "verified_dp": v["dp"], "verified_mass": v["mass"],
                                 "accepted": bool(good)})
                picked = i
                if good:
                    break
        part = v
        rep = part.pop("_rep")
        # I
        payload = GRIPPER_KG + part["mass"]
        moves = []
        with info.step("robot sim: shortest passing move (Newton)"):
            for d in DURATIONS[:3] if smoke else DURATIONS:
                r = arm_pick_and_place(payload=payload, duration=d, q_start=Q_START, q_end=Q_END,
                                       dt=1e-3 if smoke else 5e-4)
                moves.append({"duration": d, "decision": r["decision"], "peak_torque_Nm": r["peak_torque_Nm"],
                              "failed": [c["name"] for c in r["checks"] if not c["passed"]]})
        passing = [m for m in moves if m["decision"] == "APPROVED"]
        best_d = min(m["duration"] for m in passing) if passing else DURATIONS[0]
        # C
        arm = TwoLinkArm(payload=payload)
        Vp, Fp = rep.mesh(0)
        Vp = np.asarray(Vp) - np.asarray(Vp).mean(0)
        Vp = Vp[:, [0, 2, 1]] * [1, -1, 1] - [0.0, 0.0, PART_DROP]    # base facing the camera, below the gripper
        with info.step("synthetic camera frames of the move"):
            motion = arm_motions(arm, 1, 0, duration=best_d, moves=[(Q_START, Q_END)])[0]
            rng = np.random.default_rng(3)
            cams = [CameraVariation(albedo=(0.5, 0.5), background=0.08)] + [
                sample_camera(rng, occluder_p=0.0) for _ in range(1 if smoke else 3)]
            stage = os.path.join(OUT, "cell.usda")
            tracks, frames0 = [], None
            for k, cam in enumerate(cams):
                cam.albedo = (min(cam.albedo[0], 0.5), min(cam.albedo[1], 0.5))
                cam.gain = min(cam.gain, 1.0)
                write_motion_stage(stage, arm, motion, cam, payload=(Vp, Fp, 1.0))
                fr = camera_frames(stage, arm, motion, cam, 128, 128, seed=k)
                frames0 = fr if k == 0 else frames0
                keep = visibility(arm, motion, cam)
                tracks.append(payload_track_check(fr[keep], arm, {**motion, "q": motion["q"][keep]}, cam,
                                                  offset=(0.0, -PART_DROP)))
        # O: the nominal camera's stage is the cell scene
        write_motion_stage(stage, arm, motion, cams[0], payload=(Vp, Fp, 1.0))
        from pxr import Sdf, Usd
        st = Usd.Stage.Open(stage)
        w = st.GetPrimAtPath("/World")
        w.CreateAttribute("pinneapple:design:verified_t_source_C", Sdf.ValueTypeNames.Float).Set(float(part["t_source"]))
        w.CreateAttribute("pinneapple:design:mass_kg", Sdf.ValueTypeNames.Float).Set(float(part["mass"]))
        w.CreateAttribute("pinneapple:robot:move_s", Sdf.ValueTypeNames.Float).Set(float(best_d))
        st.GetRootLayer().Save()
        vs = validate_stage(stage)
        with open(os.path.join(OUT, "kit_render.py"), "w") as f:
            f.write(kit_render_script(stage, os.path.join(OUT, "cell_rtx.png")))
        pix = max(t["max_px"] for t in tracks)
        checks = [
            {"name": "B-rep solids manifold (all designs)", "value": None, "limit": None,
             "passed": all(r["manifold"] for r in train + test)},
            *[{"name": f"surrogate held-out error: {k}", "value": float(e), "limit": 0.03, "passed": bool(e <= 0.03)}
              for k, e in zip(("T rise", "dp", "mass"), held, strict=True)],
            {"name": "served proposal inside the service's trust envelope", "value": None, "limit": None,
             "passed": bool(trusted[picked]),
             "detail": f"the service flagged {int((~trusted).sum())} of {len(G)} queried designs (outside the "
                       "training box or unphysical); they were not considered"},
            {"name": "served proposal verified: T_source (C)", "value": part["t_source"], "limit": T_MAX,
             "passed": part["t_source"] <= T_MAX, "detail": f"{len(attempts)} proposal(s) verified"},
            {"name": "served proposal verified: dp (Pa)", "value": part["dp"], "limit": DP_MAX,
             "passed": part["dp"] <= DP_MAX},
            {"name": "a move passes every robot check", "value": float(best_d), "limit": DURATIONS[0],
             "passed": bool(passing), "detail": f"shortest passing move {best_d} s"},
            {"name": "part in the pixels at the kinematic tool point (max, px)", "value": pix, "limit": 2.0,
             "passed": pix <= 2.0, "detail": f"{len(cams)} camera variations"},
            {"name": "cell USD stage valid", "value": None, "limit": None, "passed": vs["ok"],
             "detail": "; ".join(vs["problems"])[:300] or f"{vs['time_samples']} time samples"}]
        decision = "APPROVED" if all(c["passed"] for c in checks) else "REVIEW"
        for c in checks:
            lim = {"min": c["limit"]} if c.get("kind") == "min" else {"max": c["limit"]}
            ex.check(c["name"], c["passed"], value=c["value"], detail=c.get("detail", ""), **lim)
        ex.log_metric("part_mass", part["mass"], "kg", "exact B-rep mass of the accepted design")
        ex.log_metric("move_time", best_d, "s", "shortest move passing every robot check")
        for comp, ran, why in (("NVIDIA USD BRep", True, ""), ("PhysicsNeMo FullyConnected", True, ""),
                               ("PINNeAPPle Model Services (local HTTP)", True, ""),
                               ("PINNeAPPle Robot Sim on NVIDIA Newton", True, ""),
                               ("PINNeAPPle World Model (synthetic camera)", True, ""),
                               ("OpenUSD stage (usd-core)", True, ""),
                               ("Omniverse Kit RTX / Isaac Sim", False, "needs an RTX GPU"),
                               ("NVIDIA NIM / Cosmos", False, "need NVIDIA_API_KEY or a GPU")):
            info.mark(comp, ran=ran, why=why)
        _figures(OUT, stage, motion, frames0, cams[0], arm, attempts, moves, checks, decision, G, mass, T, ok)
        summary = {"decision": decision, "held_out_rel_rms": held.tolist(), "train_steps": fit.steps,
                   "service_timing_ms": ans["timing_ms"], "queried_designs": int(len(G)),
                   "flagged_designs": int((~trusted).sum()), "attempts": attempts,
                   "part": {k: part[k] for k in ("n_fins", "fin_height", "fin_t", "t_source", "dp", "mass")},
                   "payload_kg": payload, "moves": moves, "move_time_s": best_d, "pixel_track_max_px": pix,
                   "usd": {k: vs.get(k) for k in ("ok", "meshes", "time_samples")}, "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "part", "move_time_s", "pixel_track_max_px",
                                              "held_out_rel_rms")}, indent=1, default=float))
    return summary


def _figures(OUT, stage, motion, frames, cam, arm, attempts, moves, checks, decision, G, mass, T, ok):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    sc = axs[0].scatter(mass[ok] * 1e3, T[ok], c=G[ok, 0], s=6, cmap="viridis")
    axs[0].scatter(mass[~ok] * 1e3, T[~ok], c="#cccccc", s=4, label="infeasible or untrusted")
    for a in attempts:
        axs[0].plot(a["verified_mass"] * 1e3, a["verified_t"], "r*" if a["accepted"] else "kx", ms=11)
    axs[0].axhline(T_MAX, color="r", ls="--", lw=1)
    axs[0].set_xlabel("mass [g] (served)")
    axs[0].set_ylabel("T_source [C] (served)")
    axs[0].set_title("Designs from the model service; * = verified, accepted", fontsize=9)
    fig.colorbar(sc, ax=axs[0], label="fins")
    axs[0].legend(fontsize=7)
    d = [m["duration"] for m in moves]
    pk = np.array([m["peak_torque_Nm"] for m in moves])
    axs[1].plot(d, pk[:, 0], "o-", label="shoulder")
    axs[1].plot(d, pk[:, 1], "s-", label="elbow")
    axs[1].axhline(arm.effort_limits[0], color="C0", ls="--", lw=1)
    axs[1].axhline(arm.effort_limits[1], color="C1", ls="--", lw=1)
    for m in moves:
        axs[1].annotate("ok" if m["decision"] == "APPROVED" else "x", (m["duration"], max(m["peak_torque_Nm"])),
                        fontsize=8, ha="center", va="bottom")
    axs[1].set_xlabel("move duration [s]")
    axs[1].set_ylabel("peak torque [N m]")
    axs[1].set_title("Robot sim (Newton): arm with part + gripper", fontsize=9)
    axs[1].legend(fontsize=7)
    axs[1].grid(alpha=0.3)
    k = len(frames) // 2
    axs[2].imshow(frames[k], cmap="gray", vmin=0, vmax=1)
    axs[2].set_title(f"Synthetic camera, frame {k} (part in white)", fontsize=9)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cell.png"), dpi=130)
    plt.close(fig)
    fps = 1.0 / motion["dt"]
    render_stage_png(stage, os.path.join(OUT, "usd_cell.png"), time=float(motion["t"][len(motion["t"]) // 2] * fps),
                     elev=10, azim=-70, title="cell.usda read back: arm, exact heat-sink mesh, mid-move")
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Work-cell checks", decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

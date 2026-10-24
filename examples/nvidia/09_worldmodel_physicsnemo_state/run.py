"""A world model that keeps the physics: PINNeAPPle World Model reads the state from the pixels, PhysicsNeMo learns
the state dynamics from the simulation's exact labels, and the same video physics checks as example 04 judge it.

Example 04's pixel-space world model fails the physics checks. Here the synthetic videos' exact labels are used for
what they are worth: a small CNN (``StateEncoder``) reads the ball's position from each frame, a PhysicsNeMo
``FullyConnected`` network learns the one-step state map ``(x, y, vx, vy) -> next`` (flight and bounces) from 9 400
labelled transitions, and the rollout is rendered with each test video's own camera and appearance. The checks read
only the rendered pixels, exactly as for any other video.

    python examples/nvidia/09_worldmodel_physicsnemo_state/run.py        # ~3 min on 4 CPU cores
"""
import argparse
import json
import os

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_lab import experiment
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.physicsnemo import MLPSurrogate, relative_errors
from pinneapple_tools.nvidia.runinfo import RunInfo
from pinneapple_worldmodel.physical_world import StateEncoder, dataset, physics_checks, render_like, state_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT, HORIZON = 8, 40


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("09_worldmodel_physicsnemo_state")
    architecture_diagram(
        "09 A world model that keeps the physics: PINNeAPPle + World Model + PhysicsNeMo",
        [Block("sim", "Exact ballistics + bounces,\nrandomized rendering", "pinneapple", 0, 0),
         Block("enc", "StateEncoder (CNN):\nframe -> ball position", "worldmodel", 1, 0),
         Block("dyn", "physicsnemo FullyConnected:\n(x, y, vx, vy) -> next state", "physicsnemo", 2, 0),
         Block("ren", "Render the rollout with the\nvideo's own camera", "worldmodel", 3, 0),
         Block("lab", "Exact labels\n(positions, velocities)", "data", 1, 1),
         Block("chk", "Video physics checks (pixels):\ng, area, speed, apex energy", "pinneapple", 3, 1)],
        [("sim", "enc", "frames"), ("sim", "lab", ""), ("lab", "enc", "targets"), ("lab", "dyn", "transitions"),
         ("enc", "dyn", "initial state"), ("dyn", "ren", "states"), ("ren", "chk", "frames")],
        os.path.join(OUT, "architecture.png"),
        subtitle="Exact simulation labels make the latent a physical state; the checks still read only the pixels")
    n_train, n_test = (20, 4) if smoke else (200, 20)
    with experiment("examples/nvidia/09_worldmodel_physicsnemo_state", title="State-space world model: CNN encoder, "
                    "PhysicsNeMo dynamics, video physics checks", domain="physical AI",
                    methods=["synthetic video", "FullyConnected (PhysicsNeMo)", "video physics checks"],
                    out_dir=base, register=False, modules=["pinneapple_worldmodel.physical_world",
                                                           "pinneapple_tools.nvidia.physicsnemo"]) as ex:
        train = dataset(n_train, seed=100)
        test = dataset(n_test, seed=5000)
        with info.step("state encoder (CNN, exact labels)"):
            enc = StateEncoder().fit(train, steps=100 if smoke else 1500)
        S, S1 = state_pairs(train)
        dyn = MLPSurrogate(4, 4, layer_size=128, num_layers=4)
        with info.step("state dynamics (PhysicsNeMo FullyConnected)"):
            run = dyn.fit(S, S1 - S, ComputeBudget(steps=200 if smoke else 20000), batch_size=256, lr=2e-3)
        St, St1 = state_pairs(test)
        step_err = relative_errors(St + dyn.predict(St), St1)
        res, enc_err, traj = [], [], []
        for v in test:
            c = enc.predict(v.frames[:CONTEXT])
            enc_err.append(float(np.max(np.hypot(c[:, 0] - v.labels["cx_px"][:CONTEXT],
                                                 c[:, 1] - v.labels["cy_px"][:CONTEXT]))))
            H = v.frames.shape[1]
            xm = (c[:, 0] - v.params["offset_px"][0]) / v.px_per_m
            ym = (H - 4 - c[:, 1] + v.params["offset_px"][1]) / v.px_per_m
            vx = np.polyfit(np.arange(CONTEXT) * v.dt, xm, 1)[0]
            k = CONTEXT - 1
            s = np.array([xm[k], ym[k], vx, (ym[k] - ym[k - 2]) / (2 * v.dt)])
            states = [s]
            for _ in range(HORIZON - 1):
                states.append(states[-1] + dyn.predict(states[-1][None])[0])
            states = np.asarray(states)
            frames = render_like(v, states[:, 0], states[:, 1])
            r = physics_checks(frames, dt=v.dt, px_per_m=v.px_per_m)
            res.append(r)
            ref_y = v.labels["y_m"][k:k + HORIZON] if k + HORIZON <= v.n else None
            traj.append((v, states, ref_y))
        rate = float(np.mean([r["passed"] for r in res]))
        fails = {c["name"]: int(sum(not r["checks"][i]["passed"] for r in res)) for i, c in enumerate(res[0]["checks"])}
        g_fit = [r["g_fit"] for r in res]
        checks = [{"name": "state encoder error on test frames (max, px)", "value": max(enc_err), "limit": 1.5,
                   "passed": max(enc_err) <= 1.5},
                  {"name": "one-step state error (worst component, rel. RMS)", "value": float(step_err.max()),
                   "limit": 0.05, "passed": step_err.max() <= 0.05},
                  {"name": "rendered rollouts pass every video physics check", "value": rate, "limit": 0.9,
                   "passed": rate >= 0.9, "detail": f"failures per check: {fails}"},
                  {"name": "g fitted from the rollout pixels (median)", "value": float(abs(np.nanmedian(g_fit) - 9.81) / 9.81),
                   "limit": 0.05, "passed": abs(np.nanmedian(g_fit) - 9.81) / 9.81 <= 0.05,
                   "detail": f"median g {np.nanmedian(g_fit):.2f} m/s2"}]
        decision = "APPROVED" if all(c["passed"] for c in checks) else "REVIEW"
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("rollout_pass_rate", rate, "", f"fraction of {len(res)} rendered rollouts passing every check")
        ex.log_metric("rollout_g_median", float(np.nanmedian(g_fit)), "m/s^2", "fitted from the rendered rollout pixels")
        info.mark("PINNeAPPle World Model (state encoder, renderer)", ran=True)
        info.mark("PhysicsNeMo FullyConnected", ran=True, how=f"{run.steps} steps on CPU")
        _figures(OUT, traj, res, checks, decision)
        summary = {"decision": decision, "rollout_pass": rate, "failures": fails, "g_fit": g_fit,
                   "encoder_max_err_px": max(enc_err), "one_step_rel_rms": step_err.tolist(),
                   "train_transitions": int(len(S)), "train_steps": run.steps, "train_wall_s": run.wall_s,
                   "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "rollout_pass", "failures", "encoder_max_err_px",
                                              "one_step_rel_rms")}, indent=1))
    return summary


def _figures(OUT, traj, res, checks, decision):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for j, (v, states, ref) in enumerate(traj[:3]):
        t = (np.arange(len(states)) + CONTEXT - 1) * v.dt
        line, = axs[0].plot(t, states[:, 1], "-", lw=1.5, label=f"rollout {j}")
        axs[0].plot(v.labels["t"], v.labels["y_m"], "--", lw=1, color=line.get_color(), label=f"exact {j}")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("height [m]")
    axs[0].set_title("State rollouts (PhysicsNeMo dynamics) vs exact", fontsize=9)
    axs[0].legend(fontsize=6, ncol=2)
    axs[0].grid(alpha=0.3)
    g = [r["g_fit"] for r in res]
    axs[1].hist(g, bins=12, color="#4f7d00")
    axs[1].axvline(9.81, color="r", ls="--", label="g = 9.81")
    axs[1].set_xlabel("g fitted from the rendered rollout pixels [m/s2]")
    axs[1].set_ylabel("rollouts")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rollouts.png"), dpi=130)
    plt.close(fig)
    v, states, _ = traj[0]
    frames = render_like(v, states[:, 0], states[:, 1])
    fig, axs = plt.subplots(1, 5, figsize=(13, 2.4))
    for a, k in zip(axs, np.linspace(0, len(frames) - 1, 5).astype(int), strict=True):
        a.imshow(frames[k], cmap="gray", vmin=0, vmax=1)
        a.set_title(f"rendered rollout, step {k}", fontsize=8)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rollout_frames.png"), dpi=120)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="State-space world model checks", decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

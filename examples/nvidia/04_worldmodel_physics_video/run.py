"""PINNeAPPle World Model (the CPU alternative to NVIDIA Cosmos): physics-labelled synthetic video, a world model
trained on it, and the physics checks that judge any video, real, predicted or generated.

A thrown ball bounces on the floor (exact event-driven trajectory, restitution 0.6 to 0.9). Each video is rendered with
domain randomization (background, texture, lighting, camera offset and zoom, blur, noise, an occluder in one video in
five) and carries exact labels. The physics checks read only the pixels: the object stays visible and keeps its size,
its free-flight acceleration equals g, no frame-to-frame jump exceeds what the fall height allows, and the apex heights
never grow (restitution <= 1). They are run on the ground truth, on four kinds of non-physical edits (time warp,
teleport, fade-out, time reversal), on the world model's reconstructions and on its rollouts.

    python examples/nvidia/04_worldmodel_physics_video/run.py         # ~12 min on 4 CPU cores
"""
import argparse
import json
import os

import numpy as np

from pinneapple_lab import experiment
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.runinfo import RunInfo
from pinneapple_worldmodel.physical_world import WorldModel, corrupt, dataset, physics_checks

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("time_warp", "teleport", "fade", "gravity_flip")
CONTEXT, HORIZON = 8, 24


def rate(results):
    return float(np.mean([r["passed"] for r in results])) if results else float("nan")


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("04_worldmodel_physics_video")
    architecture_diagram(
        "04 Physics-consistent synthetic video: PINNeAPPle + PINNeAPPle World Model",
        [Block("phys", "Exact ballistics with bounces\n(event-driven, restitution)", "pinneapple", 0, 0),
         Block("gen", "Synthetic video + domain\nrandomization, exact labels", "worldmodel", 1, 0),
         Block("wm", "World model (conv encoder,\nGRU latent dynamics, decoder)", "worldmodel", 2, 0),
         Block("chk", "Physics checks from pixels: g,\narea, speed bound, apex energy", "pinneapple", 2, 1),
         Block("judge", "Verdicts: ground truth, edits,\nreconstructions, rollouts", "pinneapple", 3, 1),
         Block("cosmos", "NVIDIA Cosmos WFM\n(CosmosAdapter, /v1/infer)", "cosmos", 3, 0, ran=False)],
        [("phys", "gen", "trajectory"), ("gen", "wm", "frames"), ("gen", "chk", "videos"), ("wm", "chk", "rollouts"),
         ("chk", "judge", "checks"), ("cosmos", "chk", "same checks")],
        os.path.join(OUT, "architecture.png"),
        subtitle="The checks do not care who made the video: simulator, world model or foundation model")
    n_train, n_test, steps = (12, 3, 40) if smoke else (200, 20, 3000)
    with experiment("examples/nvidia/04_worldmodel_physics_video", title="World model on physics-labelled video, "
                    "judged by physics checks", domain="physical AI", methods=["synthetic video", "world model",
                                                                                "video physics checks"],
                    out_dir=base, register=False, modules=["pinneapple_worldmodel.physical_world"],
                    params={"n_train": n_train, "n_test": n_test, "steps": steps, "context": CONTEXT,
                            "horizon": HORIZON}) as ex:
        with info.step("synthetic videos (exact physics + randomization)"):
            train = dataset(n_train, seed=100, defect=False)
            test = [v for v in dataset(n_test, seed=5000)]
            test_def = dataset(max(2, n_test // 5), seed=7000, defect=True)
        with info.step("checks on ground truth and edits"):
            gt = [physics_checks(v) for v in test + test_def]
            bad = {k: [physics_checks(corrupt(v, k)) for v in test] for k in KINDS}
            g_err = [abs(r["g_fit"] - 9.81) / 9.81 for r in gt]
            track_err = [float(np.nanmax(np.hypot(r["track"]["cx"] - v.labels["cx_px"], r["track"]["cy"] -
                                                  v.labels["cy_px"]))) for r, v in zip(gt, test + test_def, strict=True)]
        wm = WorldModel()
        with info.step("train the world model (CPU)"):
            wm.fit(train, steps=steps, batch=8, window=24 if not smoke else 12, free_run=CONTEXT if not smoke else 4)
        with info.step("reconstructions and rollouts, judged"):
            import torch
            recon, roll, pix = [], [], []
            for v in test:
                with torch.no_grad():
                    x = torch.tensor(v.frames[None])
                    rec = wm.net.decode(wm.net.encode(x))[0].numpy()
                recon.append(physics_checks(rec, dt=v.dt, px_per_m=v.px_per_m))
                r = wm.rollout(v.frames[:CONTEXT], HORIZON)
                pix.append(float(np.abs(r - v.frames[CONTEXT:CONTEXT + HORIZON]).mean()))
                roll.append(physics_checks(r, dt=v.dt, px_per_m=v.px_per_m))
        detected = {k: 1.0 - rate(v) for k, v in bad.items()}
        fail_by = {}
        for grp, rs in (("reconstructions", recon), ("rollouts", roll)):
            fail_by[grp] = {c["name"]: int(sum(not r["checks"][i]["passed"] for r in rs))
                            for i, c in enumerate(rs[0]["checks"])}
        checks = [{"name": "ground-truth videos pass every check", "value": rate(gt), "limit": 1.0, "passed": rate(gt) == 1.0,
                   "detail": f"{len(gt)} videos, {len(test_def)} with an occluder"},
                  {"name": "tracker vs exact labels (max error, px)", "value": max(track_err), "limit": 1.0,
                   "passed": max(track_err) <= 1.0},
                  {"name": "g from the pixels of ground truth (worst)", "value": max(g_err), "limit": 0.03,
                   "passed": max(g_err) <= 0.03},
                  *[{"name": f"edit detected: {k}", "value": detected[k], "limit": 0.9, "passed": detected[k] >= 0.9}
                    for k in KINDS],
                  {"name": "world-model reconstructions pass", "value": rate(recon), "limit": 0.9,
                   "passed": rate(recon) >= 0.9},
                  {"name": "world-model rollouts pass", "value": rate(roll), "limit": 0.9, "passed": rate(roll) >= 0.9}]
        decision = "APPROVED" if all(c["passed"] for c in checks) else (
            "REVIEW" if all(c["passed"] for c in checks[:3]) else "REJECT")
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("rollout_pass_rate", rate(roll), "", f"fraction of {len(roll)} rollouts passing every check")
        ex.log_metric("rollout_pixel_mae", float(np.mean(pix)), "", "mean |frame - truth| over the rollout horizon")
        info.mark("PINNeAPPle World Model", ran=True, how=f"{wm.n_parameters()} parameters, {steps} steps on CPU")
        info.mark("NVIDIA Cosmos", ran=False, why="needs a Cosmos NIM on a GPU or a hosted endpoint and key")
        _figures(OUT, test[0], wm, recon, roll, gt, bad, checks, decision)
        summary = {"decision": decision, "ground_truth_pass": rate(gt), "edits_detected": detected,
                   "reconstruction_pass": rate(recon), "rollout_pass": rate(roll), "failures": fail_by,
                   "rollout_g_fit": [r["g_fit"] for r in roll], "rollout_pixel_mae": float(np.mean(pix)),
                   "tracker_max_err_px": max(track_err), "g_err_gt_max": max(g_err),
                   "train": {"videos": n_train, "steps": steps, "parameters": wm.n_parameters(),
                             "wall_s": wm.history[-1]["wall_s"], "final_loss": wm.history[-1]}, "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "ground_truth_pass", "edits_detected", "reconstruction_pass",
                                              "rollout_pass", "failures")}, indent=1))
    return summary


def _figures(OUT, v, wm, recon, roll, gt, bad, checks, decision):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = wm.rollout(v.frames[:CONTEXT], HORIZON)
    ks = [CONTEXT, CONTEXT + 6, CONTEXT + 12, CONTEXT + 18]
    fig, axs = plt.subplots(2, 5, figsize=(13, 4.4))
    axs[0, 0].imshow(v.frames[CONTEXT - 1], cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title(f"context, frame {CONTEXT - 1}", fontsize=8)
    axs[1, 0].axis("off")
    axs[1, 0].text(0, 0.5, "top: ground truth\nbottom: world model\nrollout", fontsize=9)
    for j, k in enumerate(ks):
        axs[0, j + 1].imshow(v.frames[k], cmap="gray", vmin=0, vmax=1)
        axs[0, j + 1].set_title(f"truth, frame {k}", fontsize=8)
        axs[1, j + 1].imshow(r[k - CONTEXT], cmap="gray", vmin=0, vmax=1)
        axs[1, j + 1].set_title(f"rollout, frame {k}", fontsize=8)
    for a in axs.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rollout.png"), dpi=120)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8))
    tr = physics_checks(v)["track"]
    axs[0].plot(v.labels["t"], v.labels["y_m"], "k-", lw=1, label="exact label")
    axs[0].plot(v.labels["t"], (v.frames.shape[1] - 4 - tr["cy"] + v.params["offset_px"][1]) / v.px_per_m, "o", ms=3,
                label="tracked from pixels")
    tr_r = roll[0]["track"]
    t_r = v.labels["t"][CONTEXT:CONTEXT + HORIZON]
    axs[0].plot(t_r, (v.frames.shape[1] - 4 - tr_r["cy"] + v.params["offset_px"][1]) / v.px_per_m, "s", ms=3,
                label="tracked from the rollout")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("height [m]")
    axs[0].set_title("Ball height: label, ground-truth pixels, world-model pixels", fontsize=9)
    axs[0].legend(fontsize=7)
    axs[0].grid(alpha=0.3)
    groups = ["ground truth", *KINDS, "reconstruction", "rollout"]
    rates = [np.mean([x["passed"] for x in gt]), *[np.mean([x["passed"] for x in bad[k]]) for k in KINDS],
             np.mean([x["passed"] for x in recon]), np.mean([x["passed"] for x in roll])]
    axs[1].bar(range(len(groups)), rates, color=["#2e9d4f"] + ["#c0392b"] * 4 + ["#1f6f99"] * 2)
    axs[1].set_xticks(range(len(groups)))
    axs[1].set_xticklabels(groups, rotation=25, fontsize=8)
    axs[1].set_ylabel("fraction passing every physics check")
    axs[1].set_ylim(0, 1.05)
    axs[1].set_title("Who passes the physics checks", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "physics_verdicts.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Video physics checks", decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

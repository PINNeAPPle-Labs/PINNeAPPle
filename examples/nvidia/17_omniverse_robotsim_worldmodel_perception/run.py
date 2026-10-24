"""Robot perception from synthetic camera data: PINNeAPPle Robot Sim (NVIDIA Newton) moves the arm, OpenUSD holds
each motion as a timeline with a camera, PINNeAPPle World Model renders randomized camera frames from it with the
simulator's exact joint angles as labels, and PINNeAPPle checks what the trained perception model claims.

The two-link arm of examples 03 and 12 makes random point-to-point moves in Newton (PD + model feedforward). Each
motion becomes a USD stage (animated links, a side ``UsdGeom.Camera``, sometimes an occluder in front, the joint
angles time-sampled as ``pinneapple:label:q``). The synthetic camera rasterizes the stage at every frame with a
random zoom, framing, gain, noise, blur, albedo and background. The label check keeps a frame only if the elbow and
the tool point are visible. A small CNN reads both joint angles from a frame; the checks go beyond the angle error:
the tool point through the forward kinematics, the joint speed the predicted sequence implies against Newton's, and
frame-to-frame jumps the motors cannot make.

    python examples/nvidia/17_omniverse_robotsim_worldmodel_perception/run.py      # ~80 min on 4 CPU cores shared with other runs
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np

from pinneapple_lab import experiment
from pinneapple_simulation.robot_sim import TwoLinkArm
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.omniverse import render_stage_png, validate_stage
from pinneapple_tools.nvidia.runinfo import RunInfo
from pinneapple_worldmodel.robot_perception import (
    ImageRegressor,
    angle_features,
    arm_motions,
    camera_frames,
    feature_angles,
    forward_points,
    perception_checks,
    sample_camera,
    visibility,
    write_motion_stage,
)

HERE = os.path.dirname(os.path.abspath(__file__))
H = W = 72


def _render(motions, arm, seed, work, keep=0):
    rng = np.random.default_rng(seed)
    frames, vis, cams, stages = [], [], [], []
    for i, m in enumerate(motions):
        var = sample_camera(rng)
        path = write_motion_stage(os.path.join(work, f"motion_{seed}_{i:03d}.usda"), arm, m, var)
        frames.append(camera_frames(path, arm, m, var, H, W, seed=seed * 1000 + i))
        vis.append(visibility(arm, m, var))
        cams.append(var)
        if i < keep:
            stages.append(path)
        else:
            os.remove(path)
    return frames, vis, cams, stages


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("17_omniverse_robotsim_worldmodel_perception")
    architecture_diagram(
        "17 Robot perception from synthetic camera data: PINNeAPPle + Omniverse (USD) + Robot Sim + World Model",
        [Block("sim", "Robot Sim on NVIDIA Newton:\nrandom arm moves", "robotsim", 0, 0),
         Block("lab", "Exact joint angles,\nforward kinematics", "pinneapple", 0, 1),
         Block("usd", "USD timeline per motion:\nlinks, occluder, Camera, q", "omniverse", 1, 0),
         Block("cam", "Synthetic camera: zoom, gain,\nnoise, blur, albedo", "worldmodel", 2, 0),
         Block("vis", "Label check: elbow and tool\npoint visible?", "pinneapple", 2, 1),
         Block("cnn", "CNN: frame -> joint angles", "worldmodel", 3, 0),
         Block("chk", "Angle, tool point, speed vs\nNewton, rate-limit jumps", "pinneapple", 3, 1),
         Block("isaac", "Isaac Sim camera /\nReplicator on the same USD", "isaac", 1, 1, ran=False),
         Block("cos", "Cosmos Transfer\n(photoreal frames)", "cosmos", 4, 0, ran=False)],
        [("sim", "usd", "poses"), ("sim", "lab", "q, qd"), ("usd", "cam", "stage"), ("lab", "vis", "q"),
         ("cam", "vis", "frames"), ("vis", "cnn", "kept frames"), ("cnn", "chk", "q pred"), ("lab", "chk", "q, qd"),
         ("usd", "isaac", ""), ("cam", "cos", "")],
        os.path.join(OUT, "architecture.png"),
        subtitle="The simulator gives exact labels; PINNeAPPle checks the perception against kinematics and dynamics. "
                 "Hatched: not run here")
    arm = TwoLinkArm()
    n_train, n_test, steps = (6, 2, 80) if smoke else (160, 20, 4000)
    work = tempfile.mkdtemp(prefix="perc17_")
    try:
        with experiment("examples/nvidia/17_omniverse_robotsim_worldmodel_perception", title="Robot perception "
                        "from synthetic camera data: Newton motions, USD timelines, checked joint-angle CNN",
                        domain="robotics", methods=["Newton Featherstone", "OpenUSD", "synthetic camera", "CNN"],
                        out_dir=base, register=False, modules=["pinneapple_worldmodel.robot_perception",
                                                               "pinneapple_simulation.robot_sim"]) as ex:
            with info.step("Newton motions"):
                train_m = arm_motions(arm, n_train, seed=1)
                test_m = arm_motions(arm, n_test, seed=2)
            label_err = max(float(np.abs(forward_points(arm, m["q"])["elbow"] - m["body_pos"][:, 2][:, [0, 2]]).max())
                            for m in train_m + test_m)
            with info.step("USD stages and camera frames"):
                tr_f, tr_v, _, keep = _render(train_m, arm, 1, work, keep=2)
                te_f, te_v, te_c, _ = _render(test_m, arm, 2, work)
            val = [validate_stage(p) for p in keep]
            X = np.concatenate(tr_f)
            vis = np.concatenate(tr_v)
            Y = angle_features(np.concatenate([m["q"] for m in train_m]))
            with info.step("joint-angle CNN"):
                cnn = ImageRegressor(H, W, 4).fit(X[vis], Y[vis], steps=steps)
            res, preds = [], []
            for f, v, m in zip(te_f, te_v, test_m, strict=True):
                qp = feature_angles(cnn.predict(f))
                preds.append(qp)
                if v.all():
                    res.append(perception_checks(qp, m, arm))
            qt = np.concatenate([m["q"] for m in test_m])
            qp_all = np.concatenate(preds)
            vt = np.concatenate(te_v)
            err = np.abs(np.angle(np.exp(1j * (qp_all - qt)))).max(1)
            rms_vis = float(np.sqrt(np.mean(err[vt] ** 2)))
            rms_hidden = float(np.sqrt(np.mean(err[~vt] ** 2))) if (~vt).any() else float("nan")
            tool = np.hypot(*(forward_points(arm, qp_all)["tool"] - forward_points(arm, qt)["tool"]).T)
            tool_rms = float(np.sqrt(np.mean(tool[vt] ** 2)))
            vel = float(np.median([r["velocity_rel_rms"] for r in res])) if res else float("nan")
            jumps = int(sum(r["jumps_over_limit"] for r in res))
            checks = [
                {"name": "labels: forward kinematics = Newton link poses (m)", "value": label_err, "limit": 1e-6,
                 "passed": label_err <= 1e-6},
                {"name": "USD timelines valid (time samples, camera, UsdValidation)",
                 "value": float(sum(not v["ok"] for v in val)), "limit": 0.0, "passed": all(v["ok"] for v in val),
                 "detail": "; ".join(sum((v["problems"] for v in val), []))[:300]},
                {"name": "joint angle error on visible test frames (RMS, rad)", "value": rms_vis, "limit": 0.035,
                 "passed": rms_vis <= 0.035, "detail": f"{int(vt.sum())} frames; 0.035 rad = 2 deg"},
                {"name": "tool point error through the kinematics (RMS, m)", "value": tool_rms, "limit": 0.03,
                 "passed": tool_rms <= 0.03},
                {"name": "joint speed implied by the predictions vs Newton (median rel. RMS)", "value": vel,
                 "limit": 0.25, "passed": vel <= 0.25, "detail": f"{len(res)} fully visible test motions"},
                {"name": "frame-to-frame jumps beyond the joint rate limit", "value": float(jumps), "limit": 0.0,
                 "passed": jumps == 0, "detail": "limit 12 rad/s"}]
            decision = "APPROVED" if all(c["passed"] for c in checks) else "REVIEW"
            for c in checks:
                ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
            ex.log_metric("angle_rms_visible", rms_vis, "rad", "test frames where the label check keeps the frame")
            if np.isfinite(rms_hidden):
                ex.log_metric("angle_rms_hidden", rms_hidden, "rad", "test frames the label check drops")
            info.mark("PINNeAPPle Robot Sim on NVIDIA Newton", ran=True)
            info.mark("OpenUSD timelines (usd-core)", ran=True)
            info.mark("PINNeAPPle World Model (synthetic camera, CNN)", ran=True)
            info.mark("Isaac Sim camera / Omniverse Replicator", ran=False, why="needs Isaac Sim and an RTX GPU")
            info.mark("NVIDIA Cosmos Transfer", ran=False, why="needs a GPU or NVIDIA_API_KEY")
            _figures(OUT, arm, keep, train_m, test_m, te_f, te_v, te_c, preds, err, vt, checks, decision)
            summary = {"decision": decision, "train_frames": int(len(X)), "train_frames_kept": int(vis.sum()),
                       "test_frames": int(len(vt)), "test_frames_visible": int(vt.sum()),
                       "angle_rms_visible": rms_vis, "angle_rms_hidden": rms_hidden, "tool_rms": tool_rms,
                       "velocity_rel_rms_median": vel, "jumps": jumps, "label_error": label_err,
                       "motions": res, "checks": checks}
            with open(os.path.join(OUT, "summary.json"), "w") as f:
                json.dump(summary, f, indent=1, default=float)
            info.save(os.path.join(OUT, "run_info.json"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: summary[k] for k in ("decision", "train_frames_kept", "angle_rms_visible", "angle_rms_hidden",
                                              "tool_rms", "velocity_rel_rms_median", "jumps")}, indent=1))
    return summary


def _figures(OUT, arm, keep, train_m, test_m, te_f, te_v, te_c, preds, err, vt, checks, decision):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fps = 1.0 / train_m[0]["dt"]
    render_stage_png(keep[0], os.path.join(OUT, "usd_preview.png"), time=float(train_m[0]["t"][12] * fps), elev=0,
                     azim=-90, title="Motion stage read back from USD (side camera view)")
    fig, axs = plt.subplots(2, 5, figsize=(13, 5.8))
    picks = [(i, k) for i in range(len(te_f)) for k in (0, 12, 24) if k < len(te_f[i])][:10]
    for a, (i, k) in zip(axs.ravel(), picks, strict=False):
        a.imshow(te_f[i][k], cmap="gray", vmin=0, vmax=1)
        for q, style in ((test_m[i]["q"][k], "c-"), (preds[i][k], "m--")):
            P = forward_points(arm, q[None])
            pts = np.r_[[[0.0, arm.shoulder_z]], P["elbow"], P["tool"]]
            a.plot(*_to_px(pts, te_c[i].bounds(arm)), style, lw=1.5)
        a.set_title(f"{'kept' if te_v[i][k] else 'DROPPED (hidden)'}  err {np.degrees(err[_flat(te_f, i, k)]):.1f} deg",
                    fontsize=8, color="k" if te_v[i][k] else "#b00020")
        a.set_xticks([])
        a.set_yticks([])
    for a in axs.ravel()[len(picks):]:
        a.axis("off")
    fig.suptitle("Test frames from USD timelines: Newton angles (cyan) vs CNN (magenta dashed)", fontsize=10)
    fig.tight_layout(h_pad=1.6)
    fig.savefig(os.path.join(OUT, "frames.png"), dpi=120)
    plt.close(fig)
    full = [i for i, v in enumerate(te_v) if v.all()] or [0]
    i = full[0]
    m = test_m[i]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, name in enumerate(["shoulder", "elbow"]):
        axs[j].plot(m["t"], m["q"][:, j], "c-", lw=2, label="Newton")
        axs[j].plot(m["t"], preds[i][:, j], "m.--", lw=1, label="CNN from frames")
        axs[j].set_title(f"{name} angle, test motion {i}", fontsize=9)
        axs[j].set_xlabel("t [s]")
        axs[j].set_ylabel("rad")
        axs[j].legend(fontsize=7)
        axs[j].grid(alpha=0.3)
    bins = np.linspace(0, 10, 41)
    axs[2].hist(np.degrees(np.clip(err[vt], 0, np.radians(10))), bins=bins, alpha=0.8, label="kept frames")
    if (~vt).any():
        axs[2].hist(np.degrees(np.clip(err[~vt], 0, np.radians(10))), bins=bins, alpha=0.6, label="dropped frames")
    axs[2].axvline(2.0, color="k", ls="--", lw=1)
    axs[2].set_xlabel("worst joint angle error [deg] (clipped at 10)")
    axs[2].set_ylabel("test frames")
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "angles.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Perception checks", decision=decision)


def _to_px(pts, b):
    lo, r, bo, t = b
    return (pts[:, 0] - lo) / (r - lo) * (W - 1), (t - pts[:, 1]) / (t - bo) * (H - 1)


def _flat(te_f, i, k):
    return sum(len(f) for f in te_f[:i]) + k


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

"""Synthetic thermal-inspection data as OpenUSD scene variations: PINNeAPPle computes the physics and the labels,
PINNeAPPle World Model varies the camera and checks that every augmented image still tells the truth.

A heated plate with a subsurface void (a disk of low conductivity) runs hotter above the void. PINNeAPPle's
finite-volume solver gives the temperature field and the physical label (where the void raises the surface
temperature most). Each sample is an OpenUSD stage: the plate with its temperature and radiance primvars, a
``UsdGeom.Camera`` with the sample's pose, the variation parameters and the label as attributes. The synthetic
camera rasterizes the stage from the top and applies the variation (emissivity gain, reflected ambient, noise, blur,
zoom, rotation, offset, and sometimes a specular glint). The label check compares each image with what the same camera
sees on the defect-free plate: if the strongest anomaly is not at the label (a glint outshines the defect), the
sample is dropped. A small CNN detector is trained with and without the check and tested on fresh samples.

    python examples/nvidia/13_omniverse_worldmodel_inspection/run.py        # ~84 min on 4 CPU cores shared with other runs
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np

from pinneapple_lab import experiment
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.omniverse import render_stage_png, validate_stage
from pinneapple_tools.nvidia.runinfo import RunInfo
from pinneapple_worldmodel.inspection_data import inspection_dataset, thermal_plate
from pinneapple_worldmodel.physical_world import StateEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
H = W = 64
TOL_PX = 4.0


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("13_omniverse_worldmodel_inspection")
    architecture_diagram(
        "13 Synthetic inspection data as USD scene variations: PINNeAPPle + Omniverse (USD) + World Model",
        [Block("fv", "Heated plate, subsurface void:\nfinite-volume conduction", "pinneapple", 0, 0),
         Block("lab", "Physical label: where the\nvoid heats the surface most", "pinneapple", 0, 1),
         Block("usd", "One USD stage per variation:\nprimvars, Camera, label attrs", "omniverse", 1, 0),
         Block("cam", "Synthetic camera: gain, noise,\nblur, pose, glint", "worldmodel", 2, 0),
         Block("chk", "Label check: anomaly vs the\ndefect-free plate, same camera", "pinneapple", 2, 1),
         Block("det", "CNN hot-spot detector\n(trained on accepted samples)", "worldmodel", 3, 0),
         Block("rtx", "Omniverse Replicator / RTX\nrender of the same stages", "omniverse", 1, 1, ran=False),
         Block("cos", "Cosmos Transfer\n(photoreal augmentation)", "cosmos", 3, 1, ran=False)],
        [("fv", "usd", "T field"), ("fv", "lab", ""), ("lab", "chk", "label"), ("usd", "cam", "stage"),
         ("cam", "chk", "image"), ("chk", "det", "accepted"), ("usd", "rtx", ""), ("cam", "cos", "")],
        os.path.join(OUT, "architecture.png"),
        subtitle="Physics gives the label; the check drops augmentations that no longer show it. Hatched: not run here")
    n_train, n_test, steps = (24, 8, 60) if smoke else (600, 80, 3000)
    work = tempfile.mkdtemp(prefix="insp13_")
    try:
        with experiment("examples/nvidia/13_omniverse_worldmodel_inspection", title="Synthetic thermal-inspection "
                        "data as USD scene variations, with a physics label check", domain="physical AI",
                        methods=["finite volume", "OpenUSD", "synthetic camera", "label check", "CNN detector"],
                        out_dir=base, register=False, modules=["pinneapple_worldmodel.inspection_data",
                                                               "pinneapple_tools.nvidia.omniverse"]) as ex:
            with info.step("label resolution study (FV 48 vs 96 cells)"):
                rng = np.random.default_rng(7)
                shifts = []
                for _ in range(3 if smoke else 6):
                    d = (rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75), rng.uniform(0.06, 0.12))
                    a, b = thermal_plate(d, 48)["label_xy"], thermal_plate(d, 96)["label_xy"]
                    shifts.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
            with info.step("training samples (USD stage, camera, label check)"):
                tr = inspection_dataset(n_train, 1, work, keep_stages=4)
            with info.step("test samples"):
                te = inspection_dataset(n_test, 2, work, glint_p=0.25)
            val = [validate_stage(p, required_primvars={"/World/Plate": ["temperature", "radiance", "conductivity"]})
                   for p in tr["stages"]]
            glint = np.array([v.glint is not None for v in tr["variations"]])
            acc = tr["accepted"]
            rej_glint = float(1 - acc[glint].mean()) if glint.any() else float("nan")
            rej_clean = float(1 - acc[~glint].mean())
            sat_clean = float(tr["saturated"][~glint].mean())
            rej_unsat = float(1 - acc[~glint & ~tr["saturated"]].mean())
            dets, errs = {}, {}
            for name, mask in (("all samples", np.ones(len(acc), bool)), ("accepted samples", acc)):
                with info.step(f"detector on {name}"):
                    dets[name] = StateEncoder(H, W).fit_points(tr["images"][mask], tr["label_px"][mask],
                                                               steps=steps, batch=32)
                pred = dets[name].predict(te["images"])
                errs[name] = np.hypot(*(pred - te["label_px"]).T)
            brightest = np.array([np.unravel_index(np.argmax(im), im.shape)[::-1] for im in te["images"]], float)
            errs["brightest pixel"] = np.hypot(*(brightest - te["label_px"]).T)
            vis = te["accepted"]
            score = {k: {"median_px": float(np.median(e[vis])), "within_tol": float(np.mean(e[vis] <= TOL_PX)),
                         "within_tol_all_test": float(np.mean(e <= TOL_PX))} for k, e in errs.items()}
            best = score["accepted samples"]["within_tol"]
            checks = [
                {"name": "label independent of the mesh (48 vs 96 cells, max shift)", "value": max(shifts),
                 "limit": 1.5 / 48, "passed": max(shifts) <= 1.5 / 48, "detail": "plate units; one 48-cell = 0.021"},
                {"name": "USD stages valid (primvars, camera, UsdValidation)", "value": float(sum(not v["ok"] for v in val)),
                 "limit": 0.0, "passed": all(v["ok"] for v in val),
                 "detail": "; ".join(sum((v["problems"] for v in val), []))[:300]},
                {"name": "label check rejects glint samples", "value": rej_glint, "limit": 0.5, "kind": "min",
                 "passed": rej_glint >= 0.5, "detail": f"{int(glint.sum())} samples with a glint"},
                {"name": "label check keeps glint-free, unsaturated samples (false rejections)",
                 "value": rej_unsat, "limit": 0.1, "passed": rej_unsat <= 0.1,
                 "detail": f"{int((~glint & ~tr['saturated']).sum())} samples; {sat_clean:.0%} of glint-free samples "
                           "saturate at the label and are dropped"},
                {"name": f"detector (accepted samples) within {TOL_PX:.0f} px on test images showing the defect",
                 "value": best, "limit": 0.9, "kind": "min", "passed": best >= 0.9}]
            decision = "APPROVED" if all(c["passed"] for c in checks) else "REVIEW"
            for c in checks:
                lim = {"min": c["limit"]} if c.get("kind") == "min" else {"max": c["limit"]}
                ex.check(c["name"], c["passed"], value=c["value"], detail=c.get("detail", ""), **lim)
            for k, s in score.items():
                ex.log_metric(f"within_{TOL_PX:.0f}px_{k.replace(' ', '_')}", s["within_tol"], "",
                              "fraction of test images (defect visible) located within the tolerance")
            info.mark("PINNeAPPle (finite-volume conduction, labels, label check)", ran=True)
            info.mark("OpenUSD stages (usd-core): primvars, UsdGeom.Camera, UsdValidation", ran=True)
            info.mark("PINNeAPPle World Model (synthetic camera, CNN detector)", ran=True)
            info.mark("Omniverse Replicator / Kit RTX render", ran=False, why="needs an NVIDIA RTX GPU")
            info.mark("NVIDIA Cosmos Transfer", ran=False, why="needs a GPU or NVIDIA_API_KEY")
            _figures(OUT, tr, te, errs, checks, decision)
            summary = {"decision": decision, "label_shift_max": max(shifts), "train": int(len(acc)),
                       "train_accepted": int(acc.sum()), "train_glint": int(glint.sum()),
                       "rejected_glint": rej_glint, "rejected_clean": rej_clean, "saturated_clean": sat_clean,
                       "rejected_unsaturated_clean": rej_unsat, "test": int(len(vis)),
                       "test_defect_visible": int(vis.sum()), "detector": score,
                       "usd": [{k: v[k] for k in ("ok", "meshes", "prims")} for v in val], "checks": checks}
            with open(os.path.join(OUT, "summary.json"), "w") as f:
                json.dump(summary, f, indent=1, default=float)
            info.save(os.path.join(OUT, "run_info.json"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: summary[k] for k in ("decision", "train_accepted", "rejected_glint", "rejected_clean",
                                              "detector")}, indent=1))
    return summary


def _figures(OUT, tr, te, errs, checks, decision):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    render_stage_png(tr["stages"][0], os.path.join(OUT, "usd_preview.png"), elev=90, azim=-90, flat=True,
                     title="Sample stage read back from USD (displayColor = camera radiance)")
    acc = tr["accepted"]
    sat = tr["saturated"]
    order = (list(np.flatnonzero(acc)[:5]) + list(np.flatnonzero(~acc & sat)[:2])
             + list(np.flatnonzero(~acc & ~sat)[:3]))
    fig, axs = plt.subplots(2, 5, figsize=(12.5, 5.9))
    for a, i in zip(axs.ravel(), order, strict=False):
        a.imshow(tr["images"][i], cmap="inferno", vmin=0, vmax=1)
        a.plot(*tr["label_px"][i], "c+", ms=12, mew=2)
        v = tr["variations"][i]
        tag = "glint" if v.glint is not None else ("saturated" if tr["saturated"][i] else "no glint")
        a.set_title(f"{'accepted' if acc[i] else 'REJECTED'} ({tag})", fontsize=8,
                    color="k" if acc[i] else "#b00020")
        a.set_xticks([])
        a.set_yticks([])
    for a in axs.ravel()[len(order):]:
        a.axis("off")
    fig.suptitle("Synthetic camera images of USD scene variations (+ = physical label)", fontsize=10)
    fig.tight_layout(h_pad=1.6)
    fig.savefig(os.path.join(OUT, "samples.png"), dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    vis = te["accepted"]
    bins = np.linspace(0, 30, 31)
    for k, e in errs.items():
        ax.hist(np.clip(e[vis], 0, 30), bins=bins, histtype="step", lw=1.6, label=f"{k} (median {np.median(e[vis]):.1f} px)")
    ax.axvline(TOL_PX, color="k", ls="--", lw=1, label=f"tolerance {TOL_PX:.0f} px")
    ax.set_xlabel("hot-spot location error on test images [px]")
    ax.set_ylabel("images")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "detector.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Inspection data checks", decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

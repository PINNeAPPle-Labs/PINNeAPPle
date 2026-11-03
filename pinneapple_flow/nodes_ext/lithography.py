"""Computational lithography as a closed loop (:mod:`pinneapple_physics.lithography.design`, #1180): partially
coherent imaging (Abbe / Hopkins), a threshold resist, inverse lithography through the exact model or, when the
measured cost says it pays off, through a Fourier neural operator, and the mask re-verified by the full Abbe model
over focus and dose."""
from __future__ import annotations

from ..registry import Param, Port, node

SOURCES = ("annular", "conventional", "dipole_x", "dipole_y")


@node("litho.opc", "Mask correction (ILT), re-verified over the process window", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("wavelength_nm", "float", 193.0, minimum=1.0), Param("NA", "float", 1.35, minimum=0.1,
                                                                        maximum=1.7),
              Param("source", "enum", "annular", choices=list(SOURCES)),
              Param("sigma_out", "float", 0.9, minimum=0.05, maximum=1.0),
              Param("sigma_in", "float", 0.6, minimum=0.0, maximum=0.95),
              Param("threshold", "float", 0.3, minimum=0.05, maximum=0.95, help="resist threshold (clear = 1)"),
              Param("diffusion_nm", "float", 5.0, minimum=0.0),
              Param("focus_nm", "float", 50.0, minimum=0.0, help="process-window focus excursion (+-)"),
              Param("dose", "float", 0.05, minimum=0.0, maximum=0.3, help="process-window dose excursion (+-)"),
              Param("epe_tol_nm", "float", 2.0, minimum=0.1),
              Param("min_feature_nm", "float", 16.0, minimum=0.0, help="mask rule, wafer scale"),
              Param("clips", "int", 1, minimum=1, help="layout clips this correction runs on (decides whether a "
                                                         "learned operator pays off)"),
              Param("N", "int", 128, minimum=32, maximum=256), Param("steps", "int", 150, minimum=5),
              Param("force_surrogate", "bool", False, advanced=True),
              Param("seed", "int", 0)],
      calls="pinneapple_physics.lithography.design.correct_mask (imaging.aerial_image / socs_kernels, ilt, "
            "surrogate.train_operator, architecture_recommendation.recommend_architecture)")
def litho_opc(ctx, inputs, params):
    """Correct the test clip's mask and verify it with the exact model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_physics.lithography.design import LithoSpec, clip_lines_and_elbow, correct_mask
    from pinneapple_physics.lithography.imaging import Optics, Source, aerial_image
    N, L = int(params["N"]), 1024.0
    so = float(params["sigma_out"])
    src = Source(str(params["source"]), so, float(params["sigma_in"]), center=so - 0.2, pole=0.2, n=6)
    f, d = float(params["focus_nm"]), float(params["dose"])
    spec = LithoSpec(clip_lines_and_elbow(N, L), L, Optics(float(params["wavelength_nm"]), float(params["NA"]),
                                                           medium_index=1.44), src,
                     threshold=float(params["threshold"]), diffusion=float(params["diffusion_nm"]),
                     focus=(-f, 0.0, f) if f else (0.0,), doses=(1 - d, 1.0, 1 + d) if d else (1.0,),
                     epe_tol=float(params["epe_tol_nm"]), min_feature_nm=float(params["min_feature_nm"]),
                     clips=int(params["clips"]))
    r = correct_mask(spec, steps=int(params["steps"]), force_surrogate=bool(params["force_surrogate"]),
                     seed=int(params["seed"]))
    ctx.metric("epe.uncorrected_nm", r["epe_nm"]["uncorrected"], "full Abbe image, best focus and dose")
    ctx.metric("epe.corrected_nm", r["epe_nm"]["corrected"], "full Abbe image, best focus and dose")
    ctx.metric("pw.worst_epe_nm", max(max(row) for row in r["process_window"]["corrected"]["epe"]))
    ctx.metric("pw.dof_nm", r["process_window"]["corrected"]["dof_nm"])
    ctx.metric("mask.min_feature_nm", r["min_feature_nm"])
    ctx.metric("recommendation", r["recommendation"]["recommended"][0],
               (r["recommendation"].get("decision_path") or [r["recommendation"]["reasoning"]])[0])
    ctx.metric("cost.image_seconds", r["cost"]["image_seconds"], "measured: one exact image")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.table("assumptions", ["assumption"], [[a] for a in r["assumptions"]])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    ax[0].imshow(spec.target.T, origin="lower", cmap="gray_r")
    ax[0].set_title("target (resist)")
    ax[1].imshow(r["mask"].T, origin="lower", cmap="gray")
    ax[1].set_title("corrected mask")
    img = aerial_image(r["mask"].astype(float), L, spec.optics, spec.source).numpy()
    ax[2].imshow(img.T, origin="lower", cmap="magma")
    ax[2].contour(img.T, [spec.threshold], colors="c", linewidths=0.8)
    ax[2].set_title("aerial image + resist contour")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fg = ctx.figure(fig, "opc", "target, corrected mask and its image (full Abbe)")
    plt.close(fig)
    bad = [c["code"] for c in r["checks"] if c["status"] == "fail"]
    ctx.metric("verdict", "fail" if bad else "pass", ", ".join(bad) or "every check passes")
    ctx.summary(f"EPE {r['epe_nm']['uncorrected']:.2f} -> {r['epe_nm']['corrected']:.2f} nm; "
                + ("fail: " + ", ".join(bad) if bad else "verified over the process window"))
    return {"metrics": ctx.report["metrics"], "figure": fg}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Computational lithography: ILT mask correction over the process window"
    fb = FlowBuilder(title, "Partially coherent imaging (Abbe / Hopkins SOCS), threshold resist, inverse lithography "
                            "through the exact model or a learned operator when its measured cost pays off, and "
                            "the mask re-verified by the full Abbe model over focus and dose "
                            "(pinneapple_physics.lithography.design).")
    fb.add("opc", "litho.opc", dict(params), label="ILT mask correction")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("opc", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="litho.opc", title="Mask correction for optical lithography (closed loop)", domain="semiconductor",
        keywords={"lithography": 3, "litografia": 3, "opc": 3, "optical proximity": 3, "mask": 1, "máscara": 1,
                  "photomask": 2, "ilt": 3, "inverse lithography": 3, "process window": 2, "janela de processo": 2,
                  "aerial image": 2, "semiconductor": 1, "semicondutor": 1},
        min_score=3, nodes=("litho.opc",), build=lambda text, det: build_flow(),
        inputs=("the target layout clip", "the scanner (wavelength, NA, illumination)", "the resist threshold"),
        uncited="the method cites Born & Wolf, Hopkins (1953), Mack (2007), Cobb (1998) and Poonawala & Milanfar "
                "(2007) in pinneapple_physics.lithography; they are not in the flow reference file yet",
        description="pinneapple_physics.lithography.design.correct_mask"))


_register_capabilities()

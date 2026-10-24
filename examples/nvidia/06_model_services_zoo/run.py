"""PINNeAPPle Model Services: zoo surrogates served like NVIDIA NIM microservices, every answer with its trust report.

The server (``pinneapple_tools.model_services``, standard library only) exposes the PINNeAPPle zoo, a physics-check
service and the intent extractor on NIM's routes (``/v1/health/ready``, ``/v1/models``, ``/v1/infer``,
``/v1/chat/completions``). A client queries the airfoil surrogate (CST shape + angle of attack -> Cl, Cd, Cm, trained
on OpenFOAM RANS) across an angle sweep that leaves the validity range on purpose, and the heat-sink surrogate. Each
response carries ``pp.zoo.validate``'s trust report for that query: the card's metrics reproduced, the physics checks
(lift slope against thin-airfoil theory, monotonic lift below stall, symmetric-airfoil zero lift) and whether the
query is inside the validity range. The same client class talks to NVIDIA NIM (build.nvidia.com) with an API key.

    python examples/nvidia/06_model_services_zoo/run.py           # ~20 s, no GPU, no key
"""
import argparse
import json
import os

import numpy as np

from pinneapple_lab import experiment
from pinneapple_tools.model_services import (
    ModelServiceClient,
    ServiceError,
    default_registry,
    serve_in_thread,
)
from pinneapple_tools.model_services.client import NIM_URL
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
NACA2412 = dict(T0=0.169591, T1=0.144959, T2=0.131433, M0=0.024922, M1=0.071088, M2=0.058960)
CL_OPENFOAM_4DEG = 0.6625          # NACA 2412, Re 4e6, alpha 4 deg: OpenFOAM value quoted in AGENTS.md (zoo card)


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("06_model_services_zoo")
    architecture_diagram(
        "06 Zoo models as NIM-style services: PINNeAPPle + PINNeAPPle Model Services",
        [Block("zoo", "PINNeAPPle zoo: verified\nsurrogates + model cards", "pinneapple", 0, 0),
         Block("srv", "Model Services (NIM routes):\n/v1/models, /v1/infer, /v1/chat", "services", 1, 0),
         Block("trust", "pp.zoo.validate per query:\nmetrics, physics, validity", "pinneapple", 1, 1),
         Block("cli", "ModelServiceClient\n(same client for NIM)", "services", 2, 0),
         Block("chk", "Client-side checks: decisions,\nOOD flagged, latency, reference", "pinneapple", 2, 1),
         Block("nim", "NVIDIA NIM on build.nvidia.com\n(NVIDIA_API_KEY)", "nim", 3, 0, ran=False)],
        [("zoo", "srv", "models"), ("srv", "trust", "inputs, outputs"), ("trust", "srv", "report"),
         ("srv", "cli", "HTTP JSON"), ("cli", "chk", "responses"), ("cli", "nim", "same interface")],
        os.path.join(OUT, "architecture.png"),
        subtitle="A NIM-shaped service that answers with the physics checks of its own answer")
    with experiment("examples/nvidia/06_model_services_zoo", title="Zoo surrogates as NIM-compatible services with a "
                    "trust report per answer", domain="physics AI", methods=["model serving", "zoo validation"],
                    out_dir=base, register=False, modules=["pinneapple_tools.model_services"]) as ex:
        srv, url = serve_in_thread(default_registry())
        c = ModelServiceClient(url)
        try:
            with info.step("service discovery"):
                ready = c.ready()
                models = c.models()
            alphas = np.arange(-6.0, 21.0, 2.0 if smoke else 1.0)
            sweep, lat = [], []
            with info.step("airfoil sweep (one request per angle)"):
                for a in alphas:
                    r = c.infer("airfoil_cst_rans_mlp", {**NACA2412, "alpha_deg": [float(a)]})
                    lat.append(c.last_latency_ms)
                    sweep.append({"alpha": float(a), "Cl": r["outputs"]["Cl"][0], "Cd": r["outputs"]["Cd"][0],
                                  "decision": r["trust"]["decision"], "inside": r["trust"]["in_validity_range"],
                                  "warnings": r["warnings"], "server_ms": r["timing_ms"]})
            ref = c.infer("airfoil_cst_rans_mlp", {**NACA2412, "alpha_deg": [4.0]})
            hs = c.infer("heatsink_forced_mlp", dict(base_width_mm=100, base_depth_mm=100, base_thickness_mm=5,
                                                     n_fins=12, fin_thickness_mm=1.5, fin_height_mm=30, k_w_mk=209,
                                                     source_width_mm=40, source_depth_mm=40, t_ambient_c=25,
                                                     air_velocity_m_s=3, power_w=60))
            try:
                c.infer("no_such_model", {})
                err_ok = False
            except ServiceError as e:
                err_ok = e.status == 404
            try:
                c.infer("airfoil_cst_rans_mlp", {"T0": 0.1})
                bad_ok = False
            except ServiceError as e:
                bad_ok = e.status == 400
        finally:
            srv.shutdown()
        rng = [m for m in models if m["id"] == "airfoil_cst_rans_mlp"][0]["pinneapple"]["validity_range"]["alpha_deg"]
        inside = [s for s in sweep if rng[0] <= s["alpha"] <= rng[1]]
        outside = [s for s in sweep if not rng[0] <= s["alpha"] <= rng[1]]
        cl4 = ref["outputs"]["Cl"][0]
        checks = [{"name": "service ready, every zoo model listed", "value": float(len(models)), "limit": None,
                   "passed": ready and {"airfoil_cst_rans_mlp", "heatsink_forced_mlp"} <= {m["id"] for m in models}},
                  {"name": "in-range queries APPROVED", "value": float(np.mean([s["decision"] == "APPROVED" for s in inside])),
                   "limit": 1.0, "passed": all(s["decision"] == "APPROVED" for s in inside)},
                  {"name": "out-of-range queries flagged (not APPROVED)", "value": float(np.mean(
                      [s["decision"] != "APPROVED" for s in outside])), "limit": 1.0,
                   "passed": all(s["decision"] != "APPROVED" and s["warnings"] for s in outside)},
                  {"name": "NACA 2412, 4 deg: Cl vs OpenFOAM 0.6625", "value": abs(cl4 - CL_OPENFOAM_4DEG) / CL_OPENFOAM_4DEG,
                   "limit": 0.02, "passed": abs(cl4 - CL_OPENFOAM_4DEG) / CL_OPENFOAM_4DEG <= 0.02},
                  {"name": "heat-sink service answers with its trust report", "value": None, "limit": None,
                   "passed": hs["trust"]["decision"] in ("APPROVED", "REVIEW", "REJECT")},
                  {"name": "unknown model -> 404, missing inputs -> 400", "value": None, "limit": None,
                   "passed": err_ok and bad_ok}]
        decision = "APPROVED" if all(c_["passed"] for c_ in checks) else "REJECT"
        for c_ in checks:
            ex.check(c_["name"], c_["passed"], value=c_["value"], max=c_["limit"])
        ex.log_metric("latency_median_ms", float(np.median(lat)), "ms", "client round trip per /v1/infer request")
        ex.log_metric("cl_naca2412_4deg", cl4, "", "airfoil service, alpha 4 deg")
        info.mark("PINNeAPPle Model Services", ran=True, how=f"{len(models)} services on {url}")
        info.mark("NVIDIA NIM (build.nvidia.com)", ran=False, why="needs NVIDIA_API_KEY",
                  how=f"ModelServiceClient('{NIM_URL}', api_key=...)")
        _figures(OUT, sweep, rng, lat, checks, decision)
        summary = {"decision": decision, "services": [m["id"] for m in models], "validity_alpha": rng,
                   "sweep": sweep, "latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))},
                   "naca2412_4deg": ref["outputs"], "heatsink": {"outputs": hs["outputs"], "decision": hs["trust"]["decision"]},
                   "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "validity_alpha", "latency_ms")}, indent=1))
    return summary


def _figures(OUT, sweep, rng, lat, checks, decision):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"APPROVED": "#2e9d4f", "REVIEW": "#e0a000", "REJECT": "#c0392b"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    a = np.array([s["alpha"] for s in sweep])
    cl = np.array([s["Cl"] for s in sweep])
    axs[0].axvspan(rng[0], rng[1], color="#e8f4ea", label="validity range (model card)")
    axs[0].plot(a, cl, "-", color="#999999", lw=1)
    for d, cc in col.items():
        m = np.array([s["decision"] == d for s in sweep])
        if m.any():
            axs[0].plot(a[m], cl[m], "o", color=cc, label=f"answer: {d}")
    axs[0].plot([4.0], [CL_OPENFOAM_4DEG], "k*", ms=12, label="OpenFOAM, 4 deg")
    axs[0].set_xlabel("angle of attack [deg]")
    axs[0].set_ylabel("Cl (NACA 2412, Re 4e6)")
    axs[0].set_title("/v1/infer answers and their trust decisions", fontsize=9)
    axs[0].legend(fontsize=7)
    axs[0].grid(alpha=0.3)
    axs[1].hist(lat, bins=15, color="#1d7a50")
    axs[1].set_xlabel("client round trip per request [ms] (inference + trust report)")
    axs[1].set_ylabel("requests")
    axs[1].set_title(f"median {np.median(lat):.0f} ms on a 4-core CPU", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sweep.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Service checks", decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

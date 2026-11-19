"""End to end: an NVIDIA OpenUSD asset, CalculiX, a PhysicsNeMo FNO, the checks, and a live dashboard.

    python examples/usd_structural_surrogate/run.py all --out run/usd_pedestal

Stages (each one also runs alone, so the pipeline can be resumed at any point):

``asset``      fetch ``Props/Mounts/ur10_mount.usd`` from the Isaac content bucket, read it with OpenUSD and measure
               the mounting interface (plate diameter, plate thickness, height).
``campaign``   sample designs, mesh each with gmsh, solve each with ``ccx``, sample the fields onto the fixed grid.
``train``      train the 3-D FNO on the grid under a ``ComputeBudget``.
``validate``   score it on designs the training did not see, with the peak von Mises and the clamped face checked,
               and write the trust card.
``viz``        the deformed part with its fields in the browser, in Blender and in OpenUSD.
``dashboard``  serve the interactive page (design sliders, ``ccx`` on demand, fine-tuning).

Needs ``ccx`` (and gmsh) on PATH, plus ``numpy scipy torch`` and ``pip install usd-core nvidia-physicsnemo``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def stage_asset(out: Path) -> dict:
    from pinneapple_simulation.pipelines.usd_structural_surrogate.usd_asset import (
        NVIDIA_UR10_MOUNT,
        probe,
    )
    iface = probe(NVIDIA_UR10_MOUNT, out / "assets" / "ur10_mount.usd")
    print(json.dumps(iface, indent=1))
    return iface


def stage_campaign(out: Path, n: int, parallel: int) -> dict:
    from pinneapple_simulation.pipelines.usd_structural_surrogate import DesignSpace, build_campaign
    space = DesignSpace.from_interface(stage_asset(out))
    m = build_campaign(space, n, out / "campaign", element_size=0.022, parallel=parallel, threads=1, seed=0)
    print(f"campaign: {m['n_ok']}/{m['n_requested']} solved in {m['seconds']:.0f} s")
    return m


def stage_train(out: Path, steps: int) -> object:
    from pinneapple_analysis.cost.equal_budget import ComputeBudget
    from pinneapple_simulation.pipelines.usd_structural_surrogate import DesignSpace
    from pinneapple_simulation.pipelines.usd_structural_surrogate.campaign import (
        default_grid,
        load_samples,
    )
    from pinneapple_simulation.pipelines.usd_structural_surrogate.surrogate import train

    space = DesignSpace.from_interface()
    grid = default_grid(space)
    S = load_samples(out / "campaign", fields=("design", "occupancy", "u", "vm", "nodes", "elements", "u_nodes"))
    print(f"training on {len(S['occupancy'])} designs, {steps} steps")
    sur = train(S, grid, space, budget=ComputeBudget(steps=steps), batch_size=4, threads=10, seed=0)
    sur.provenance["campaign"] = str(out / "campaign" / "manifest.json")
    sur.provenance["asset"] = json.loads((out / "campaign" / "manifest.json").read_text())["space"].get("asset")
    sur.save(out / "surrogate")
    print("metrics:", json.dumps(sur.metrics, indent=1))
    return sur


def stage_validate(out: Path) -> dict:
    from pinneapple_simulation.pipelines.usd_structural_surrogate.campaign import load_samples
    from pinneapple_simulation.pipelines.usd_structural_surrogate.checks import report
    from pinneapple_simulation.pipelines.usd_structural_surrogate.surrogate import Surrogate

    sur = Surrogate.load(out / "surrogate")
    S = load_samples(out / "campaign", fields=("design", "occupancy", "u", "vm", "nodes", "elements", "u_nodes"))
    n = len(S["occupancy"])
    val = list(range(int(n * 0.88), n))                      # the split train() held out, by construction
    rep = report(sur, S, index=val)
    (out / "validation.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep["comparison"]["summary"], indent=1))
    print("failed checks:", rep["comparison"]["failed"] or "none")
    for card in rep["trust_cards"]:
        print("trust card:", card.get("decision"), "tier", card.get("tier"))
    return rep


def stage_viz(out: Path, blender: str | None = None) -> dict:
    from pinneapple_simulation.pipelines.usd_structural_surrogate.campaign import load_samples
    from pinneapple_simulation.pipelines.usd_structural_surrogate.surrogate import Surrogate
    from pinneapple_simulation.pipelines.usd_structural_surrogate.viz import (
        design_scene,
        export_scene,
        sweep_scene,
    )

    sur = Surrogate.load(out / "surrogate")
    S = load_samples(out / "campaign", fields=("design", "occupancy", "u", "vm", "nodes", "elements"))
    nominal = len(S["occupancy"]) // 2
    got = export_scene(design_scene(sur, {k: v[nominal] for k, v in S.items()}),
                       out / "viz" / "nominal", usd=True, blender=blender)
    sweep = list(range(0, len(S["occupancy"]), max(1, len(S["occupancy"]) // 24)))
    got["sweep"] = export_scene(sweep_scene(sur, S, sweep), out / "viz" / "sweep", usd=True)
    print(json.dumps(got, indent=1, default=str))
    return got


def stage_report(out: Path) -> dict:
    """Write ``experiment.json`` from what the run measured -- the numbers, and how each was measured."""
    from pinneapple_simulation.pipelines.usd_structural_surrogate.usd_asset import (
        RECORDED_INTERFACE,
    )
    man = json.loads((out / "campaign" / "manifest.json").read_text())
    asset = man["space"].get("asset") or RECORDED_INTERFACE
    val = json.loads((out / "validation.json").read_text())
    sur = json.loads((out / "surrogate.json").read_text())
    cmp = val["comparison"]
    s = cmp["summary"]

    def metric(name: str, value: float, unit: str, how: str) -> dict:
        return {"name": name, "value": float(value), "unit": unit, "how": how}

    metrics = [
        metric("designs solved", man["n_ok"], "", "campaign manifest, one CalculiX job per design"),
        metric("campaign wall time", man["seconds"], "s", "campaign manifest, six jobs at a time on a 10-core laptop"),
        metric("relative L2 displacement, p90", s["rel_l2_displacement"]["p90"], "",
               "predicted vs CalculiX on the held-out designs, on the part's grid cells"),
        metric("relative L2 von Mises, p90", s["rel_l2_vm"]["p90"], "",
               "predicted vs CalculiX on the held-out designs, on the part's grid cells"),
        metric("relative peak von Mises, p90", s["rel_peak_vm"]["p90"], "",
               "99.5th percentile of von Mises over the part, predicted vs CalculiX, held-out designs"),
        metric("clamped face displacement", s["clamped_displacement_fraction"]["max"], "",
               "largest predicted displacement on the bolted face over the largest displacement of the design"),
        metric("training steps", (sur.get("metrics") or {}).get("steps_total")
               or ((sur.get("metrics") or {}).get("budget") or {}).get("steps") or 0, "",
               "ComputeBudget steps summed over the initial run and its continuation chunks"),
        metric("mounting plate diameter", asset["plate_diameter_mm"], "mm",
               "measured on Props/Mounts/ur10_mount.usd by usd_asset.mount_interface"),
    ]
    checks = [{"name": "no check failed on the held-out designs", "passed": not cmp["failed"]}]
    for card in val.get("trust_cards", []):
        checks.append({"name": f"field trust card ({card.get('decision')}, tier {card.get('tier')})",
                       "passed": str(card.get("decision", "")).lower() not in ("reject", "fail")})

    exp = {
        "schema": "pinneapple.experiment/1",
        "id": "examples/usd_structural_surrogate",
        "title": "Structural surrogate of an OpenUSD asset: CalculiX + PhysicsNeMo FNO",
        "description": ("The UR10 mounting pedestal of NVIDIA's OpenUSD content, read with OpenUSD and measured for its "
                        "mounting interface, a four-variable design family solved by CalculiX, a 3-D FNO trained on the "
                        "grid, the physics checks it passes, and a dashboard that answers the sliders live."),
        "domain": "structural mechanics",
        "methods": ["FEM (CalculiX)", "FNO (PhysicsNeMo)", "OpenUSD"],
        "modules": ["pinneapple_simulation.pipelines.usd_structural_surrogate"],
        "params": {"n_designs": man["n_requested"], "element_size_m": man["solver"]["element_size_m"],
                   "grid": man["grid"]["n"], "reference_load_n": man["reference_load_n"]},
        "scripts": ["run.py"],
        "metrics": metrics,
        "checks": checks,
        "files": {"figures": [], "videos": [], "viewers": [], "data": []},
        "links": {},
        "references": ["USD: NVIDIA Isaac Sim asset Props/Mounts/ur10_mount.usd (omniverse-content-production)",
                       "CalculiX User's Manual, G. Dhondt", "Li et al., Fourier Neural Operator for parametric PDEs"],
        "status": "not curated",
        "lab": {},
        "provenance": {"source": "register", "scenes": str(out / "viz"), "campaign": str(out / "campaign"),
                       "surrogate": str(out / "surrogate")},
    }
    (Path(__file__).resolve().parent / "experiment.json").write_text(json.dumps(exp, indent=1, ensure_ascii=False) + "\n")
    print("experiment.json written:", [m["name"] for m in metrics])
    return exp


def stage_dashboard(out: Path, port: int) -> None:
    from pinneapple_simulation.pipelines.usd_structural_surrogate.dashboard import serve
    serve(out / "surrogate", port, workdir=str(out / "dashboard"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["asset", "campaign", "train", "validate", "viz", "report", "dashboard",
                                      "all"])
    ap.add_argument("--out", default="run/usd_pedestal")
    ap.add_argument("--n", type=int, default=256, help="designs in the campaign")
    ap.add_argument("--steps", type=int, default=900, help="training steps")
    ap.add_argument("--parallel", type=int, default=6, help="ccx jobs at once")
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--blender", default=None, help="write a Blender render to this path")
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.stage in ("asset", "all"):
        stage_asset(out)
    if a.stage in ("campaign", "all"):
        stage_campaign(out, a.n, a.parallel)
    if a.stage in ("train", "all"):
        stage_train(out, a.steps)
    if a.stage in ("validate", "all"):
        stage_validate(out)
    if a.stage in ("viz", "all"):
        stage_viz(out, a.blender)
    if a.stage in ("report", "all"):
        stage_report(out)
    if a.stage == "dashboard":
        stage_dashboard(out, a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

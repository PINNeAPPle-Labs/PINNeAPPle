"""Rebuild the classical CFD validation results: the published tables, and the OpenFOAM runs compared with them (#272).

    python examples/cfd/classical_cases/run_cases.py --out examples/cfd/classical_cases/results
    python examples/cfd/classical_cases/run_cases.py --only reference          # no solver needed
    python examples/cfd/classical_cases/run_cases.py --only cavity --re 100 400
    python examples/cfd/classical_cases/run_cases.py --only pinn               # the network, no solver needed
    python examples/cfd/classical_cases/run_cases.py --only cfdbench          # the benchmark's archive, once unpacked

It writes one JSON per case next to the tables it was compared with, so the README's table can be regenerated and the
numbers in it can be checked. Without OpenFOAM the ``reference`` part still runs (it is transcribed data), the ``pinn``
part still runs (a network is trained, not solved) and the solver parts say so instead of failing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pinneapple_simulation.cfd_validation import (  # noqa: E402
    cavity,
    cfdbench,
    foam,
    ladder,
    pinn,
    reference,
    step,
    tmr,
    warm_start,
)


def _dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"  wrote {path}")


def do_reference(out: str) -> None:
    print("reference tables (no solver needed)")
    _dump(os.path.join(out, "reference.json"),
          {"profiles": reference.as_dicts(),
           "nusselt": {"citation": reference.DEV_VAHL_DAVIS,
                       "values": {f"{k:g}": v for k, v in sorted(reference.DEV_VAHL_DAVIS_NU.items())}},
           "armaly_reattachment": {"citation": "Armaly et al. (1983), J. Fluid Mech. 127, 473-496",
                                   "values": {}, "note": "the published curve is a figure; not transcribed"}})


def do_warm(out: str, cases: str) -> None:
    """#494: a surrogate over Re fitted from solved cases, then a cold and a warm run at held-out Reynolds numbers."""
    print("warm start: snapshots at Re 50..800 on 64 x 64, benchmark at Re 400 and 1000 (held out)")
    snaps = warm_start.snapshots(os.path.join(cases, "warm"), (50, 100, 200, 300, 600, 800), n=64)
    surrogate = warm_start.OperatorSurrogate(modes=8).fit(snaps)
    result = {"surrogate": {"modes": 8, "fitted_on": [float(r) for r in sorted(snaps)],
                            "energy": surrogate.energy, "seconds": round(surrogate.fitted_seconds, 2)},
              "benchmarks": []}
    for re in (400, 1000):
        got = warm_start.benchmark(re, surrogate, os.path.join(cases, "warm"), n=64)
        print(f"  Re = {re}: cold {got.cold_iterations} it / {got.cold_seconds:.1f} s, "
              f"warm {got.warm_iterations} it / {got.warm_seconds:.1f} s -> {got.iteration_ratio:.2f}x iterations, "
              f"{got.speedup:.2f}x wall time; field difference {got.field_difference}")
        result["benchmarks"].append(got.to_dict())
    _dump(os.path.join(out, "warm_start.json"), result)


def do_ladder(out: str, cases: str) -> None:
    """#294: the cost-accuracy table, and the calibration's effect on the ranking."""
    print("ladder: NACA 0012 at alpha = 0, 4, 8 on three RANS levels")
    designs = [ladder.design("0012", a) for a in (0.0, 4.0, 8.0)]
    got = ladder.ladder(designs, os.path.join(cases, "ladder"), log=lambda m: print(m))
    got["calibration"] = ladder.calibrate(got["rows"], "rans-coarse", designs=[0.0, 4.0])
    for row in got["table"]:
        print(f"  {row['rung']:<14} {row['cells']:>7} cells  {row['seconds_per_design']:>7.1f} s/design  "
              f"CL err {row['CL_error_rel']}  spearman {row['spearman_CL']}")
    _dump(os.path.join(out, "ladder.json"), got)


def do_tmr(out: str, cases: str) -> None:
    """#265: the TMR flat plate with Spalart-Allmaras and k-omega SST on three grids."""
    print("NASA TMR 2DZP flat plate: SpalartAllmaras and kOmegaSST on three grids")
    got = tmr.study(os.path.join(cases, "tmr"))
    for model, entry in got["models"].items():
        print(f"  {model:<16} cf(0.97) = {[round(v, 6) for v in entry['cf']]}  order {entry.get('observed_order')}"
              f"  extrapolated {entry.get('extrapolated')}  law {entry['cf_law']:.6f}")
    _dump(os.path.join(out, "tmr_flat_plate.json"), got)


def do_cavity(out: str, reynolds: list[int], meshes: list[int], cases: str, *, reuse: bool = False) -> None:
    result = {"case": "ghia-1982", "solver": "simpleFoam", "scheme": "linearUpwind grad(U)",
              "case_dir": "system/ in pinneapple_simulation.cfd_validation.cavity", "runs": []}
    for re in reynolds:
        print(f"cavity Re = {re}: meshes {meshes}")
        t0 = time.time()
        got = cavity.study(re, os.path.join(cases, "cavity"), meshes=meshes, reuse=reuse)
        got["wall_seconds"] = round(time.time() - t0, 1)
        for rec in got["runs"]:
            print(f"  n = {rec['n']:>4}  {rec['cells']:>6} cells  {rec['iterations']:>6} it  "
                  f"{rec['seconds']:>7.1f} s  converged={rec['converged']}")
        for n, err in got["errors"].items():
            print(f"    n = {n:>4}: u l2 {err['u']['l2']:.4f} ({100 * err['u']['l2_relative']:.2f} %), "
                  f"v l2 {err['v']['l2']:.4f} ({100 * err['v']['l2_relative']:.2f} %)")
        result["runs"].append(got)
    _dump(os.path.join(out, "cavity.json"), result)


def _machine() -> dict:
    """What the run cost was paid on, so the seconds in the table have a machine next to them."""
    import platform

    import torch

    return {"device": "cpu", "threads": int(torch.get_num_threads()), "cores": os.cpu_count() or 0,
            "platform": f"{platform.system()} {platform.machine()}", "torch": torch.__version__}


def do_pinn(out: str, reynolds: list[int], cases: str, *, steps: int | None = None) -> None:
    """#272: the same cavity with a network, scored through the same stations and the same norm as the meshes.

    Two models: one network per Reynolds number, and one operator whose input carries ``Re``. Each is measured
    against the table the meshes were measured against, so the two numbers sit in one table. The finding is the
    Reynolds number at which each stops matching (:func:`pinn.stops_matching`), not the average error.
    """
    if not steps:
        steps = pinn.Config.steps
    cfg = pinn.Config(steps=int(steps), checkpoint=os.path.join(cases, "pinn", "checkpoint.pt"))
    os.makedirs(os.path.dirname(cfg.checkpoint), exist_ok=True)
    record = {"case": "ghia-1982", "method": "physics-informed network",
              "what": "the steady incompressible equations in strong form at collocation points, with the boundary "
                      "conditions as a penalty and the pressure held to zero mean. No solved field is shown to the "
                      "network: the only thing it is given is the problem statement",
              "machine": _machine(), "protocol": pinn.PROTOCOL, "config": cfg.to_dict()}

    def on_step(model, step, losses) -> None:
        if step % (cfg.steps // 10) == 0 or step == cfg.steps - 1:
            print(f"    step {int(step):>6}  pde {losses['pde']:.2e}  bc {losses['bc']:.2e}  "
                  f"{losses['seconds']:>7.0f} s", flush=True)

    record["single"] = pinn.study(tuple(reynolds), parametric=False, config=cfg, on_step=on_step)
    _dump(os.path.join(out, "cavity_pinn.json"), record)
    for row in record["single"]["rows"]:
        e = row["errors"]
        print(f"  Re = {row['reynolds']:>6.0f}: u {100 * e['u']['l2_relative']:6.2f} %  "
              f"v {100 * e['v']['l2_relative']:6.2f} %")
    solved, record["off_sample_solver"] = _off_sample_solver(cases)
    record["parametric"] = pinn.study(tuple(reynolds), parametric=True, config=cfg, on_step=on_step,
                                      probe=pinn.OFF_SAMPLE, solved=solved)
    for row in record["parametric"]["rows"]:
        e = row["errors"]
        print(f"  operator Re = {row['reynolds']:>6.0f}: u {100 * e['u']['l2_relative']:6.2f} %  "
              f"v {100 * e['v']['l2_relative']:6.2f} %")
    for row in record["parametric"]["off_sample"]:
        e = row["errors"]
        print(f"  operator Re = {row['reynolds']:>6.0f} (no table, against the solver): "
              f"u {100 * e['u']['l2_relative']:6.2f} %  v {100 * e['v']['l2_relative']:6.2f} %")
    record["comparison"] = _comparison(record, out)
    _dump(os.path.join(out, "cavity_pinn.json"), record)


def do_pinn_budget(out: str, reynolds: float, budgets: list[int], cases: str) -> None:
    """#272: the training budget, swept — the measurement that says what convergence costs, and where it stops.

    The committed measurement of :func:`do_pinn` used 10 000 Adam steps, which is the budget the machine allowed at
    the time, and reached no better than 16 % on ``u`` at the easiest Reynolds number of the table. That leaves the
    obvious question open: is the network's error a property of the method or of the budget it was given? This sweep
    answers it by training the *same* configuration — same seed, same widths, same penalty weights, same cosine decay
    — for longer, and measuring each run through :func:`pinn.measure`, which is the identical ruler the meshes and the
    committed pin are scored with.

    Two things make the sweep a measurement rather than a search for a flattering number. The cosine decay is set over
    each run's own budget and not over the longest one, because a short run under a long schedule is still at a large
    learning rate when it is stopped and would be quoted as "10 000 steps does not converge" when it means "10 000
    steps of a different schedule does not". And every run starts from the same seed, so the only thing that changes
    between rows is :attr:`pinn.Config.steps`.

    Each run writes its own row into ``results/pinn_budget.json``, merged under a file lock, so several budgets can be
    trained at once — which is how the sweep fits in a session — and an interrupted one can be restarted without
    losing the rows already measured. The checkpoint of a run is tagged with its budget: a shared path would have a
    20 000-step run resume the weights of the 10 000-step one and report the longer budget with the shorter run inside.
    """
    path = os.path.join(out, "pinn_budget.json")
    os.makedirs(out, exist_ok=True)
    lock = path + ".lock"
    re_tag = f"{reynolds:g}"
    for budget in budgets:
        budget = int(budget)
        cfg = pinn.Config(steps=budget,
                          checkpoint=os.path.join(cases, "pinn", f"budget_re{re_tag}_{budget}.pt"))
        os.makedirs(os.path.dirname(cfg.checkpoint), exist_ok=True)
        print(f"PINN budget: Re = {re_tag}, {budget} Adam steps", flush=True)

        def on_step(model, step, losses, _budget=budget) -> None:
            print(f"    step {int(step):>7}  pde {losses['pde']:.2e}  bc {losses['bc']:.2e}  "
                  f"{losses['seconds']:>7.0f} s", flush=True)

        t0 = time.time()
        model = pinn.train(reynolds, config=cfg, on_step=on_step)
        row = pinn.measure(model, reynolds)
        row["steps"] = budget
        row["seconds"] = round(model.seconds, 1)
        row["wall_seconds"] = round(time.time() - t0, 1)
        row["parameters"] = int(sum(p.numel() for p in model.parameters()))
        row["converged"] = pinn.matches(row, max(pinn.MATCH_TOLERANCES))
        row["history"] = model.history
        e = row["errors"]
        print(f"  {budget:>7} steps, {model.seconds:>7.0f} s: u {100 * e['u']['l2_relative']:6.2f} %  "
              f"v {100 * e['v']['l2_relative']:6.2f} %  converged at {100 * max(pinn.MATCH_TOLERANCES):g} %: "
              f"{row['converged']}", flush=True)

        _merge_budget_row(path, lock, row, reynolds)
        with open(path) as fh:
            merged = json.load(fh)
        print(f"  {path} now holds {len(merged['rows'])} row(s)", flush=True)


def _merge_budget_row(path: str, lock: str, row: dict, reynolds: float) -> None:
    """Add or replace one (Reynolds number, budget) row in the sweep's record, under an exclusive lock.

    The read and the write are both inside the lock, so a run that finishes while another is writing cannot be lost;
    that is what lets the sweep be trained as several processes at once.
    """
    import fcntl

    header = {"case": "ghia-1982", "method": "physics-informed network",
              "question": "what does the network cost to converge at this Reynolds number, and does it",
              "reynolds": float(reynolds),
              "protocol": pinn.PROTOCOL,
              "converged_at": f"both profiles within {100 * max(pinn.MATCH_TOLERANCES):g} % relative L2",
              "machine": _machine(), "rows": []}
    with open(lock, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        record = dict(header)
        if os.path.exists(path):
            with open(path) as old:
                record = json.load(old)
            record.setdefault("rows", [])
        key = lambda r: (float(r["reynolds"]), int(r["steps"]))  # noqa: E731
        rows = [r for r in record["rows"] if (float(r["reynolds"]), int(r["steps"])) != (float(reynolds), int(row["steps"]))]
        rows.append(row)
        record["rows"] = sorted(rows, key=key)
        record["machine"] = _machine()
        # the smallest budget that clears the tolerance, which is the answer the sweep exists to produce
        clearing = [int(r["steps"]) for r in record["rows"]
                    if float(r["reynolds"]) == float(reynolds) and r.get("converged")]
        record["cheapest_converged"] = min(clearing) if clearing else None
        tmp = path + ".tmp"
        with open(tmp, "w") as out:
            json.dump(record, out, indent=1, sort_keys=True)
            out.write("\n")
        os.replace(tmp, path)
        fcntl.flock(fh, fcntl.LOCK_UN)


def _off_sample_solver(cases: str, meshes: tuple[int, ...] = (64, 128)):
    """A function that solves the cavity at a Reynolds number the table does not publish, and its record.

    The operator is claimed to answer for every ``Re`` in its range, and the three published columns cannot test that:
    they are the points it was measured on. So the same case is solved at :data:`pinn.OFF_SAMPLE` — between the
    published columns — and the operator is scored against *that*, with the two meshes kept so the reference's own
    discretisation error is visible next to the model's. Without OpenFOAM the probe is skipped and says why.
    """
    record: dict = {"reynolds": list(pinn.OFF_SAMPLE), "meshes": list(meshes), "runs": [],
                    "skipped": None}
    if foam.bashrc() is None:
        record["skipped"] = "no OpenFOAM installation found (set FOAM_BASHRC): the operator is not probed off the table"
        print(f"  {record['skipped']}", file=sys.stderr)
        return None, record
    stations = pinn.reference_stations(100)                  # the stations are the same at every column
    wanted = {"y": [float(v) for v in stations["y"]], "x": [float(v) for v in stations["x"]]}

    def solve(reynolds: float) -> dict:
        got = None
        for n in meshes:
            case = os.path.join(cases, "cavity", f"cavity_re{int(round(reynolds))}_{n}")
            print(f"  solving Re = {reynolds:g} on {n}x{n} for the operator's off-table station", flush=True)
            cavity.write_case(case, reynolds, n)          # the dictionaries first: ``run`` only runs them
            info = cavity.run(case, reynolds, n)
            solution = cavity.sample(case, reynolds, stations=wanted)
            record["runs"].append({"reynolds": float(reynolds), "mesh": int(n), "seconds": round(info["seconds"], 1),
                                   "iterations": int(info["solver"]["iterations"]),
                                   "converged": bool(info["solver"]["converged"]),
                                   "y": [float(v) for v in solution["y"]], "u": [float(v) for v in solution["u"]],
                                   "x": [float(v) for v in solution["x"]], "v": [float(v) for v in solution["v"]]})
            got = solution
        return got

    return solve, record


def do_cfdbench(out: str, root: str) -> None:
    """#272: the CFDBench cavity family, loaded from its own archive and scored against the table ours is scored on."""
    print(f"CFDBench cavity, loaded from {root}")
    got = cfdbench.study(root)
    for name, entry in got["families"].items():
        print(f"  family {name}: {entry}")
    for row in got["rows"]:
        for case in row["cases"]:
            e = case["errors"]
            print(f"  Re = {row['reynolds']:>6.0f}  {case['case']:<20s} u {100 * e['u']['l2_relative']:6.2f} %  "
                  f"v {100 * e['v']['l2_relative']:6.2f} %  u without the lid stations "
                  f"{100 * e['u_without_lid']['l2_relative']:6.2f} %")
    _dump(os.path.join(out, "cfdbench.json"), got)


def _comparison(record: dict, out: str) -> list[dict]:
    """The one table the README prints: every method at every Reynolds number, on the same ruler.

    The solver's rows are read from ``cavity.json`` when it is there (it is a committed result), at the finest mesh of
    the ladder, so the network is quoted next to the best number the meshes produced rather than a convenient one.
    """
    rows: list[dict] = []
    path = os.path.join(out, "cavity.json")
    if os.path.exists(path):
        with open(path) as fh:
            solved = json.load(fh)
        for run in solved.get("runs", []):
            finest = max(int(n) for n in run.get("errors", {})) if run.get("errors") else None
            if finest is None:
                continue
            err = run["errors"][str(finest)]
            seconds = next((r.get("seconds") for r in run.get("runs", []) if int(r.get("n", 0)) == finest), None)
            rows.append({"method": f"simpleFoam {finest}x{finest}", "reynolds": run["reynolds"],
                         "u": err["u"]["l2_relative"], "v": err["v"]["l2_relative"], "seconds": seconds})
    for mode, label in (("single", "PINN"), ("parametric", "PINN, operator over Re")):
        models = record.get(mode, {}).get("models", [])
        for i, row in enumerate(record.get(mode, {}).get("rows", [])):
            # one network per Re in ``single`` (row i is model i), one shared network in ``parametric``
            seconds = (models[i] if i < len(models) else models[0]).get("seconds") if models else None
            rows.append({"method": label, "reynolds": row["reynolds"], "u": row["errors"]["u"]["l2_relative"],
                         "v": row["errors"]["v"]["l2_relative"], "seconds": seconds})
    return sorted(rows, key=lambda r: (float(r["reynolds"]), r["method"]))


def do_step(out: str, reynolds: list[int], meshes: list[int], cases: str) -> None:
    result = {"case": "armaly-1983", "solver": "simpleFoam", "scheme": "linearUpwind grad(U)", "runs": []}
    for re in reynolds:
        print(f"step Re = {re}: meshes {meshes}")
        t0 = time.time()
        got = step.study(re, os.path.join(cases, "step"), meshes=meshes)
        got["wall_seconds"] = round(time.time() - t0, 1)
        for rec in got["runs"]:
            print(f"  m = {rec['cells_per_step']:>3}  {rec['cells']:>6} cells  {rec['iterations']:>6} it  "
                  f"{rec['seconds']:>7.1f} s  x1/S = {rec['x1_over_S']}  converged={rec['converged']}")
        result["runs"].append(got)
    _dump(os.path.join(out, "step.json"), result)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                    help="where the JSON lands (the numbers committed next to the README)")
    ap.add_argument("--cases", default="/tmp/pinneapple-cfd-cases",
                    help="where the OpenFOAM cases are written (never committed: they are hundreds of files)")
    ap.add_argument("--only", nargs="+", default=["reference", "cavity", "step", "warm", "ladder", "tmr", "pinn"],
                    choices=["reference", "cavity", "step", "warm", "ladder", "tmr", "pinn", "pinn-budget",
                             "cfdbench"],
                    help="the parts to run; cfdbench is not in the default list because it needs the benchmark's own "
                         "786 MB archive unpacked, which this folder never fetches; pinn-budget trains one network per "
                         "budget, so it is long and asked for by name")
    ap.add_argument("--reuse", action="store_true", help="read the cases already solved under --cases")
    ap.add_argument("--re", nargs="+", type=int, default=[100, 400, 1000], help="Reynolds numbers for the cavity")
    ap.add_argument("--meshes", nargs="+", type=int, default=list(cavity.MESHES))
    ap.add_argument("--step-re", nargs="+", type=int, default=[100, 200, 400, 800])
    ap.add_argument("--step-meshes", nargs="+", type=int, default=list(step.MESHES))
    ap.add_argument("--pinn-steps", type=int, default=pinn.Config.steps,
                    help="Adam steps of the network: the training budget the machine allowed, recorded with the "
                         "result (the checkpoint under --cases resumes an interrupted run)")
    ap.add_argument("--budgets", nargs="+", type=int, default=[10000, 20000, 40000, 80000],
                    help="the training budgets of --only pinn-budget, in Adam steps; one network is trained per "
                         "budget, so the command costs the sum of them")
    ap.add_argument("--budget-re", type=float, default=float(min(pinn.REYNOLDS)),
                    help="the Reynolds number the budget sweep is run at (the cheapest column of the table)")
    ap.add_argument("--cfdbench", default=os.path.expanduser("~/cfdbench-data/cavity"),
                    help="where the CFDBench cavity archive was unzipped (never committed: 800 MB)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cases, exist_ok=True)
    if "reference" in args.only:
        do_reference(args.out)
    if "cavity" in args.only or "step" in args.only:
        found = foam.bashrc()
        if found is None:
            print("no OpenFOAM installation found (set FOAM_BASHRC): the solver cases are skipped", file=sys.stderr)
            return 1
        print(f"OpenFOAM: {found}")
    if "cavity" in args.only:
        do_cavity(args.out, args.re, args.meshes, args.cases, reuse=args.reuse)
    if "step" in args.only:
        do_step(args.out, args.step_re, args.step_meshes, args.cases)
    if "warm" in args.only:
        do_warm(args.out, args.cases)
    if "ladder" in args.only:
        do_ladder(args.out, args.cases)
    if "tmr" in args.only:
        do_tmr(args.out, args.cases)
    if "pinn" in args.only:
        do_pinn(args.out, args.re, args.cases, steps=args.pinn_steps)
    if "pinn-budget" in args.only:
        do_pinn_budget(args.out, args.budget_re, args.budgets, args.cases)
    if "cfdbench" in args.only:
        if os.path.isdir(args.cfdbench):
            do_cfdbench(args.out, args.cfdbench)
        else:
            print(f"the CFDBench archive is not at {args.cfdbench}: unzip "
                  f"{cfdbench.ARCHIVE} there, or point --cfdbench at it", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

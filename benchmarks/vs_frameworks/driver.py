"""Driver of the cross-framework benchmark: runs every runner in its own process, then scores the exports with the
one neutral evaluator.

    python -m benchmarks.vs_frameworks.driver env                       # environment record
    python -m benchmarks.vs_frameworks.driver run --track A1 --iters 2000 --seeds 0 1 2 3 4 --jobs 2
    python -m benchmarks.vs_frameworks.driver run --track A2 --cap-s 240 --seeds 0 1 2
    python -m benchmarks.vs_frameworks.driver run --track B --epochs 30 --seeds 0 1 2
    python -m benchmarks.vs_frameworks.driver run --track C --iters 10000 --seeds 0 1 2 3 4
    python -m benchmarks.vs_frameworks.driver collect                   # results/*.json from the exports
    python -m benchmarks.vs_frameworks.driver timing --work W           # results/timing.json (quiet-machine runs)

Runs are resumable (a run whose result exists with status "ok" or "failed" is not repeated; ``--rerun`` repeats
it). Raw exports (.npz) stay in ``--work`` (default: $VS_FRAMEWORKS_WORK or ./_work); the scored records, one per
run with every seed, go to ``benchmarks/vs_frameworks/results/``. Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

RUNNERS = {
    ("A", "pinneapple"): "track_a_pinneapple", ("A", "deepxde"): "track_a_deepxde",
    ("A", "physicsnemo_sym"): "track_a_physicsnemo_sym",
    ("B", "pinneapple"): "track_b_pinneapple", ("B", "physicsnemo"): "track_b_physicsnemo",
    ("B", "neuraloperator"): "track_b_neuraloperator",
    ("C", "pinneapple"): "track_c_pinneapple", ("C", "deepxde"): "track_c_deepxde",
    ("C", "physicsnemo_sym"): "track_c_physicsnemo_sym",
}
DEFAULT_PROBLEMS = {"A1": ["burgers", "poisson", "helmholtz"], "A2": ["burgers", "poisson", "helmholtz", "allen_cahn"],
                    "B": ["darcy"], "C": ["burgers_inverse"]}
DEFAULT_FRAMEWORKS = {"A1": ["pinneapple", "deepxde", "physicsnemo_sym"],
                      "A2": ["pinneapple", "deepxde", "physicsnemo_sym"],
                      "B": ["pinneapple", "physicsnemo", "neuraloperator"],
                      "C": ["pinneapple", "deepxde", "physicsnemo_sym"]}


def work_dir(args) -> Path:
    return Path(args.work or os.environ.get("VS_FRAMEWORKS_WORK", ROOT / "_work")).resolve()


def run_id(track, problem, fw, seed) -> str:
    return f"{track}_{problem}_{fw}_s{seed}"


def job_cmd(args, track, problem, fw, seed):
    kind = track[0]
    mod = RUNNERS[(kind, fw)]
    w = work_dir(args)
    cmd = [sys.executable, "-m", f"benchmarks.vs_frameworks.{mod}", "--problem", problem, "--seed", str(seed),
           "--threads", str(args.threads), "--device", args.device, "--root", str(w),
           "--out", str(w / "runs"), "--run-id", run_id(track, problem, fw, seed)]
    if track == "A1":
        cmd += ["--mode", "a1", "--iters", str(args.iters)]
    elif track == "A2":
        cmd += ["--mode", "a2", "--cap-s", str(args.cap_s), "--cap-kind", args.cap_kind]
    elif track == "B":
        cmd += ["--mode", "b", "--epochs", str(args.epochs), "--n-train", str(args.n_train)]
    elif track == "C":
        cmd += ["--mode", "c", "--iters", str(args.iters)]
    return cmd


def _env(threads: int) -> dict:
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    env["DDE_BACKEND"] = "pytorch"
    env.setdefault("PYTHONWARNINGS", "ignore")
    return env


def cmd_env(args) -> None:
    from pinneapple_tools.benchmark_suite.crossframework.records import environment
    rec = environment(args.threads)
    import subprocess as sp
    rec["git_commit"] = sp.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_run(args) -> None:
    track = args.track
    problems = args.problems or DEFAULT_PROBLEMS[track]
    fws = args.frameworks or DEFAULT_FRAMEWORKS[track]
    w = work_dir(args)
    (w / "runs").mkdir(parents=True, exist_ok=True)
    (w / "logs").mkdir(parents=True, exist_ok=True)
    jobs = []
    # seed-major order: every framework gets seed s before any framework gets seed s + 1, so a cut keeps pairs
    for seed in args.seeds:
        for problem in problems:
            for fw in fws:
                rid = run_id(track, problem, fw, seed)
                res = w / "runs" / f"{rid}.json"
                if res.exists() and not args.rerun:
                    st = json.loads(res.read_text()).get("status")
                    if st in ("ok", "failed"):
                        continue
                jobs.append((track, problem, fw, seed, rid))
    print(f"{len(jobs)} jobs, {args.jobs} at a time, {args.threads} thread(s) each", flush=True)
    running = {"n": 0}

    def one(job):
        track, problem, fw, seed, rid = job
        running["n"] += 1
        concurrent = running["n"]
        t0 = time.time()
        with open(w / "logs" / f"{rid}.log", "w") as log:
            p = subprocess.run(job_cmd(args, track, problem, fw, seed), cwd=ROOT, env=_env(args.threads),
                               stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        running["n"] -= 1
        res = w / "runs" / f"{rid}.json"
        if res.exists():
            meta = json.loads(res.read_text())
            meta["driver"] = {"returncode": p.returncode, "concurrent_jobs_at_start": concurrent,
                              "jobs_setting": args.jobs, "elapsed_s": time.time() - t0}
            res.write_text(json.dumps(meta, indent=1))
            st = meta.get("status")
        else:
            st = f"no result (exit {p.returncode})"
            res.write_text(json.dumps({"status": "failed", "error": st, "track": track, "problem": problem,
                                       "framework": fw, "seed": seed, "driver": {"returncode": p.returncode}}))
        print(f"{rid}: {st} in {time.time() - t0:.0f} s", flush=True)

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(one, jobs))


def cmd_collect(args) -> None:
    """Score every run's exports with the neutral evaluator and write results/<track>.json (all seeds, raw)."""
    import numpy as np

    from pinneapple_tools.benchmark_suite.crossframework import evaluate as E
    from pinneapple_tools.benchmark_suite.crossframework import problems as P
    from pinneapple_tools.benchmark_suite.crossframework.records import read_run, validate_result

    w = work_dir(args)
    refs = {}
    out = {}
    for f in sorted((w / "runs").glob("*.json")):
        rid = f.stem
        meta, arrays = read_run(w / "runs", rid)
        track = rid.split("_")[0]
        rec = {k: v for k, v in meta.items() if k not in ("loss_history", "traceback")}
        rec["run_id"] = rid
        rec["schema_missing"] = validate_result(meta) if meta.get("status") == "ok" else []
        lh = meta.get("loss_history") or []
        rec["loss_history_len"] = len(lh)
        if meta.get("status") == "ok":
            if track in ("A1", "A2"):
                prob = P.get_problem(meta["problem"])
                if prob.name not in refs:
                    refs[prob.name] = prob.exact(P.test_points(prob))
                rec["metrics"] = E.evaluate_pinn(prob, arrays["u_test"], arrays.get("u_stencil"),
                                                 reference=refs[prob.name])
            elif track == "B":
                from pinneapple_tools.benchmark_suite.crossframework.darcy import load_split
                ref = load_split("test", meta["config"]["resolution"])[1]
                rec["metrics"] = E.evaluate_operator(arrays["pred_test"], ref)
            elif track == "C":
                prob = P.get_problem("burgers")
                if "burgers" not in refs:
                    refs["burgers"] = prob.exact(P.test_points(prob))
                rec["metrics"] = E.evaluate_inverse(float(arrays["nu_hat"]), prob.params["nu"], arrays["u_test"],
                                                    refs["burgers"])
        out.setdefault(track, []).append(rec)
        _ = np
    RESULTS.mkdir(parents=True, exist_ok=True)
    for track, rows in out.items():
        (RESULTS / f"{track}.json").write_text(json.dumps(rows, indent=1, default=float))
        print(f"{track}: {len(rows)} runs -> {RESULTS / (track + '.json')}")


def cmd_timing(args) -> None:
    """results/timing.json from a work dir of short A1 runs taken one at a time on a quiet machine."""
    w = work_dir(args)
    keep = ("problem", "framework", "iterations", "wall_s", "cpu_s", "loadavg_before", "loadavg_after", "status")
    rows = [{k: m.get(k) for k in keep} for m in (json.loads(f.read_text()) for f in sorted((w / "runs").glob("*.json")))
            if m.get("status") == "ok"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "timing.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} timing runs -> {RESULTS / 'timing.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("env", "run", "collect", "timing"):
        p = sub.add_parser(name)
        p.add_argument("--work", default=None)
        p.add_argument("--threads", type=int, default=1)
        if name == "run":
            p.add_argument("--track", required=True, choices=("A1", "A2", "B", "C"))
            p.add_argument("--problems", nargs="*")
            p.add_argument("--frameworks", nargs="*")
            p.add_argument("--seeds", nargs="*", type=int, default=[0])
            p.add_argument("--iters", type=int, default=2000)
            p.add_argument("--cap-s", type=float, default=240.0)
            p.add_argument("--cap-kind", default="cpu", choices=("cpu", "wall"))
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--n-train", type=int, default=1000)
            p.add_argument("--jobs", type=int, default=1)
            p.add_argument("--device", default="cpu")
            p.add_argument("--timeout", type=float, default=3 * 3600)
            p.add_argument("--rerun", action="store_true")
    args = ap.parse_args(argv)
    {"env": cmd_env, "run": cmd_run, "collect": cmd_collect, "timing": cmd_timing}[args.cmd](args)


if __name__ == "__main__":
    main()

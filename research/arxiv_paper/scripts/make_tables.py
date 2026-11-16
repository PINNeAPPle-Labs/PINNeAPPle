"""Build every cross-framework table and figure of the manuscript from the committed run records.

Input: ``benchmarks/vs_frameworks/results/*.json`` -- the raw per-run records of the benchmark, each
carrying its own configuration, versions and load average. Nothing here reads a number that is not in
those files, and nothing here re-runs the benchmark.

Output: ``research/arxiv_paper/tables/*.tex`` and ``research/arxiv_paper/figures/*.pdf``.

Usage::

    python research/arxiv_paper/scripts/make_tables.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "benchmarks" / "vs_frameworks" / "results"
PAPER = ROOT / "research" / "arxiv_paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

PROBLEMS = ("burgers", "poisson", "helmholtz", "allen_cahn")
PROBLEM_LABEL = {"burgers": "Burgers 1-D", "poisson": "Poisson 2-D", "helmholtz": "Helmholtz 2-D",
                 "allen_cahn": "Allen--Cahn 1-D", "darcy": "Darcy 2-D", "burgers_inverse": "inverse Burgers"}

FRAMEWORK_LABEL = {
    "pinneapple": "PINNeAPPle", "deepxde": "DeepXDE", "physicsnemo_sym": "PhysicsNeMo-Sym",
    "physicsnemo_sym_netmatched": "Sym, net matched", "physicsnemo_sym_matched": "Sym, points matched",
    "physicsnemo": "PhysicsNeMo", "neuraloperator": "neuraloperator",
    "pinneapple_nostop": "PINNeAPPle, no early stop",
    "pinneapple_fixed": "PINNeAPPle after the fixes",
    "pinneapple_equal": "PINNeAPPle, equal weights",
}

#: A1 and A2 rows of the report, in the order the paper presents them.
A1_FRAMEWORKS = ("pinneapple", "deepxde", "physicsnemo_sym")
A2_FRAMEWORKS = ("pinneapple", "deepxde", "physicsnemo_sym", "physicsnemo_sym_netmatched",
                 "physicsnemo_sym_matched")
C_FRAMEWORKS = ("pinneapple", "pinneapple_nostop", "pinneapple_fixed", "pinneapple_equal", "deepxde",
                "physicsnemo_sym")


# --------------------------------------------------------------------------- reading
def load(name: str):
    return json.loads((RESULTS / name).read_text())


def ok(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("status") == "ok"]


def select(records: list[dict], **kwargs) -> list[dict]:
    return [r for r in ok(records) if all(r.get(k) == v for k, v in kwargs.items())]


def metric(records: list[dict], path: str) -> list[float]:
    out = []
    for r in records:
        v = r
        for part in path.split("."):
            v = (v or {}).get(part) if isinstance(v, dict) else None
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return out


def med(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    v = sorted(values)
    n = len(v)
    return {"median": v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]),
            "min": v[0], "max": v[-1], "n": n}


# --------------------------------------------------------------------------- formatting
def fmt(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-3):
        mantissa, exponent = f"{x:.{max(digits, 1) - 1}e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{x:.{digits}f}"


def cell(stats: dict | None, digits: int = 2) -> str:
    """``median [min, max]``, the form every table in the paper uses."""
    if stats is None:
        return "--"
    if stats["n"] == 1:
        return fmt(stats["median"], digits)
    return f"{fmt(stats['median'], digits)} [{fmt(stats['min'], digits)}, {fmt(stats['max'], digits)}]"


def paired_ratio(numer: list[dict], denom: list[dict], path: str) -> dict | None:
    """Per-seed ratio of ``numer`` to ``denom``, matched on (problem, seed)."""
    key = lambda r: (r.get("problem"), r.get("seed"))                       # noqa: E731
    d = {key(r): metric([r], path) for r in denom}
    ratios = []
    for r in numer:
        a, b = metric([r], path), d.get(key(r))
        if a and b and b[0]:
            ratios.append(a[0] / b[0])
    if not ratios:
        return None
    s = med(sorted(ratios))
    s["lower"] = sum(1 for x in ratios if x < 1)
    return s


# --------------------------------------------------------------------------- tables
def table_a1(records: list[dict]) -> str:
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Problem & framework & relative $L^2$ & ms / iteration & iterations \\", r"\midrule"]
    for p in PROBLEMS:
        for f in A1_FRAMEWORKS:
            rows = select(records, problem=p, framework=f)
            if not rows:
                continue
            rel = med(metric(rows, "metrics.rel_l2"))
            ms = med([1000.0 * r["cpu_s"] / r["iterations"] for r in rows if r.get("iterations")])
            it = med([float(r["iterations"]) for r in rows if r.get("iterations")])
            lines.append(f"{PROBLEM_LABEL[p]} & {FRAMEWORK_LABEL[f]} & {cell(rel, 3)} & {cell(ms, 1)} & "
                         f"{fmt(it['median'] if it else None, 0)} \\\\")
        if p != PROBLEMS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_a1_paired(records: list[dict]) -> str:
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Problem & framework & error ratio to PINNeAPPle & seeds lower \\", r"\midrule"]
    for p in PROBLEMS:
        base = select(records, problem=p, framework="pinneapple")
        for f in A1_FRAMEWORKS[1:]:
            rows = select(records, problem=p, framework=f)
            s = paired_ratio(rows, base, "metrics.rel_l2")
            if s is None:
                continue
            lines.append(f"{PROBLEM_LABEL[p]} & {FRAMEWORK_LABEL[f]} & "
                         f"{fmt(s['median'])} [{fmt(s['min'])}, {fmt(s['max'])}] & {s['lower']}/{s['n']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_a2(records: list[dict]) -> str:
    lines = [r"\begin{tabular}{l" + "r" * len(A2_FRAMEWORKS) + "}", r"\toprule",
             "Problem & " + " & ".join(FRAMEWORK_LABEL[f] for f in A2_FRAMEWORKS) + r" \\", r"\midrule"]
    for p in PROBLEMS:
        cells = []
        for f in A2_FRAMEWORKS:
            rows = select(records, problem=p, framework=f)
            cells.append(cell(med(metric(rows, "metrics.rel_l2")), 3))
        lines.append(f"{PROBLEM_LABEL[p]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for label, path, digits in (("steps taken", "iterations", 0), ("CPU seconds", "cpu_s", 0)):
        cells = []
        for f in A2_FRAMEWORKS:
            rows = [r for p in PROBLEMS for r in select(records, problem=p, framework=f)]
            cells.append(cell(med(metric(rows, path)), digits))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_b(records: list[dict]) -> str:
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"framework & test rel.\ $L^2$ (mean) & worst sample & parameters & s / epoch & "
             r"latency b=1 / b=20 (ms) \\", r"\midrule"]
    for f in ("pinneapple", "physicsnemo", "neuraloperator"):
        rows = select(records, framework=f)
        if not rows:
            continue
        mean = med(metric(rows, "metrics.rel_l2_mean"))
        worst = med(metric(rows, "metrics.rel_l2_max"))
        params = rows[0].get("n_parameters_real") or rows[0].get("n_parameters")
        epoch = med([float(np.median(r["epoch_cpu_s"])) for r in rows if r.get("epoch_cpu_s")])
        l1 = med([1000.0 * r["latency_batch1_s"] for r in rows if r.get("latency_batch1_s")])
        l20 = med([1000.0 * r["latency_batch20_s"] for r in rows if r.get("latency_batch20_s")])
        lat = "--" if l1 is None or l20 is None else f"{fmt(l1['median'], 1)} / {fmt(l20['median'], 1)}"
        params_txt = f"{int(params):,}".replace(",", "{,}") if params else "--"
        lines.append(f"{FRAMEWORK_LABEL[f]} & {cell(mean, 4)} & {cell(worst, 3)} & {params_txt} & "
                     f"{cell(epoch, 1)} & {lat} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_c(records: list[dict]) -> str:
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"framework & $\hat\nu$ & $\hat\nu$ relative error & field rel.\ $L^2$ & iterations \\",
             r"\midrule"]
    for f in C_FRAMEWORKS:
        rows = select(records, framework=f)
        if not rows:
            continue
        nu = med(metric(rows, "nu_hat"))
        nurel = med(metric(rows, "metrics.nu_rel_err"))
        field = med(metric(rows, "metrics.rel_l2"))
        it = med(metric(rows, "iterations"))
        lines.append(f"{FRAMEWORK_LABEL[f]} & {cell(nu, 4)} & {cell(nurel)} & {cell(field, 3)} & "
                     f"{cell(it, 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_c_diag(records: list[dict]) -> str:
    labels = {"equalw": "equal fixed weights (adaptive off)", "linear": r"$\nu$ linear instead of log",
              "lrp3": r"parameter lr $10^{-3}$", "equalw_linear_lrp3": "all three"}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"variant & seeds & $\hat\nu$ relative error & field rel.\ $L^2$ \\", r"\midrule"]
    for variant, label in labels.items():
        rows = [r for r in records if r.get("variant") == variant]
        if not rows:
            continue
        nurel = med(metric(rows, "nu_rel_err"))
        field = med(metric(rows, "rel_l2"))
        seeds = ", ".join(str(int(r["seed"])) for r in sorted(rows, key=lambda r: r["seed"]))
        lines.append(f"{label} & {seeds} & {cell(nurel)} & {cell(field, 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_d(records: dict) -> str:
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"framework & runner lines (A / B / C) & install size (MB) & import (s) \\", r"\midrule"]
    for f in ("pinneapple", "deepxde", "physicsnemo_sym", "physicsnemo", "neuraloperator"):
        r = records.get(f)
        if not r:
            continue
        loc = r.get("loc") or {}
        locs = " / ".join(str(loc[k]) if k in loc else "--" for k in ("A", "B", "C"))
        lines.append(f"{FRAMEWORK_LABEL[f]} & {locs} & {fmt(r.get('install_mb'), 1)} & "
                     f"{fmt(r.get('import_s'), 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_environment(env: dict) -> str:
    versions = env.get("versions") or {}
    rows = [("CPU", env.get("cpu")), ("logical CPUs", str(env.get("n_cpus"))),
            ("threads per run", str(env.get("threads"))), ("Python", env.get("python")),
            ("PyTorch", env.get("torch")), ("NumPy", env.get("numpy")),
            ("CUDA available", str(env.get("cuda_available")))]
    rows += [(k, v) for k, v in sorted(versions.items())]
    rows.append(("commit under test", str(env.get("git_commit"))[:12]))
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"item & value \\", r"\midrule"]
    lines += [f"{k} & \\texttt{{{v}}} \\\\" for k, v in rows if v is not None]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_deviations(records: list[dict]) -> str:
    """A1 / A2 / C seed counts actually present in the records, not the protocol's target."""
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"track & framework & runs & seeds \\", r"\midrule"]
    for label, data, frameworks in (("A1", load("A1.json"), A1_FRAMEWORKS),
                                    ("A2", load("A2.json"), A2_FRAMEWORKS),
                                    ("C", load("C.json"), C_FRAMEWORKS)):
        for f in frameworks:
            rows = select(data, framework=f)
            if not rows:
                continue
            seeds = sorted({int(r["seed"]) for r in rows})
            lines.append(f"{label} & {FRAMEWORK_LABEL[f]} & {len(rows)} & {len(seeds)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures
def figures(a1, a2, b, c) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200,
                         "savefig.bbox": "tight"})
    label = lambda k: PROBLEM_LABEL.get(k, k).replace("--", "\u2013")   # noqa: E731 - LaTeX vs matplotlib
    written = []

    def group(records, frameworks, problems=PROBLEMS, path="metrics.rel_l2"):
        out = {}
        for f in frameworks:
            vals = []
            for p in problems:
                rows = select(records, problem=p, framework=f)
                m = med(metric(rows, path))
                vals.append(m["median"] if m else float("nan"))
            out[f] = vals
        return out

    # Allen-Cahn was cut from A1, so its figure carries only the three problems that were run
    for name, records, frameworks, title, problems in (
            ("A1", a1, A1_FRAMEWORKS, "A1: identical configuration", PROBLEMS[:3]),
            ("A2", a2, A2_FRAMEWORKS, "A2: each framework's own recipe", PROBLEMS)):
        data = group(records, frameworks, problems)
        fig, ax = plt.subplots(figsize=(4.4, 2.4))
        x = np.arange(len(problems))
        width = 0.8 / len(frameworks)
        for i, f in enumerate(frameworks):
            ax.bar(x + (i - (len(frameworks) - 1) / 2) * width, data[f], width,
                   label=FRAMEWORK_LABEL[f])
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([label(p) for p in problems], fontsize=7)
        ax.set_ylabel("relative $L^2$ error")
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=5.5, ncol=2, frameon=False)
        f = FIGURES / f"{name.lower()}_rel_l2.pdf"
        fig.savefig(f, metadata={"CreationDate": None})  # deterministic bytes
        plt.close(fig)
        written.append(f)

    # B: error against the CPU cost of an epoch, the trade-off the text describes
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    for f in ("pinneapple", "physicsnemo", "neuraloperator"):
        rows = select(b, framework=f)
        e = med(metric(rows, "metrics.rel_l2_mean"))
        cost = med([float(np.median(r["epoch_cpu_s"])) for r in rows if r.get("epoch_cpu_s")])
        if e and cost:
            ax.scatter(cost["median"], e["median"], s=22, label=FRAMEWORK_LABEL[f])
            ax.annotate(FRAMEWORK_LABEL[f], (cost["median"], e["median"]), textcoords="offset points",
                        xytext=(4, 3), fontsize=6)
    ax.set_xlabel("CPU seconds per epoch")
    ax.set_ylabel("test relative $L^2$ (mean of samples)")
    ax.set_title("B: FNO on Darcy $64\\times64$, 12 epochs", fontsize=7)
    f = FIGURES / "b_darcy.pdf"
    fig.savefig(f, metadata={"CreationDate": None})  # deterministic bytes
    plt.close(fig)
    written.append(f)

    # C: nu estimates against the true value
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    order = [f for f in C_FRAMEWORKS if select(c, framework=f)]
    for i, f in enumerate(order):
        vals = metric(select(c, framework=f), "nu_hat")
        ax.scatter([i] * len(vals), vals, s=12, label=None)
        m = med(vals)
        if m:
            ax.plot([i - 0.3, i + 0.3], [m["median"]] * 2, color="0.2", lw=1.0)
    ax.axhline(0.01 / math.pi, color="0.4", ls="--", lw=0.8)
    ax.annotate(r"true $\nu = 0.01/\pi$", (0.02, 0.01 / math.pi), fontsize=6, color="0.35",
                xytext=(0.02, 0.012))
    ax.set_xticks(range(len(order)))
    short = {"pinneapple": "PINNeAPPle", "pinneapple_nostop": "PINN\nno early stop",
             "pinneapple_fixed": "PINN\nwith fixes", "pinneapple_equal": "PINN\nequal weights",
             "deepxde": "DeepXDE", "physicsnemo_sym": "PhysicsNeMo\n-Sym"}
    ax.set_xticklabels([short.get(f, FRAMEWORK_LABEL[f]) for f in order], fontsize=6)
    ax.set_ylabel(r"$\hat\nu$")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("C: identified viscosity, 3000 iterations", fontsize=7)
    f = FIGURES / "c_nu.pdf"
    fig.savefig(f, metadata={"CreationDate": None})  # deterministic bytes
    plt.close(fig)
    written.append(f)
    return written


# --------------------------------------------------------------------------- main
def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    a1, a2, b, c = load("A1.json"), load("A2.json"), load("B.json"), load("C.json")

    out = {
        "a1.tex": table_a1(a1), "a1_paired.tex": table_a1_paired(a1), "a2.tex": table_a2(a2),
        "b.tex": table_b(b), "c.tex": table_c(c), "c_diag.tex": table_c_diag(load("C_diagnostics.json")),
        "d.tex": table_d(load("D.json")), "environment.tex": table_environment(load("environment.json")),
        "deviations.tex": table_deviations(a1),
    }
    for name, body in out.items():
        (TABLES / name).write_text(body + "\n")
        print(f"-> tables/{name}")
    for f in figures(a1, a2, b, c):
        try:
            print(f"-> {f.relative_to(PAPER)}")
        except ValueError:                       # a caller redirected FIGURES elsewhere (the tests do)
            print(f"-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

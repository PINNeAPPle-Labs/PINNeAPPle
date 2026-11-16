"""Turn the verification corpus results into the tables and figures the manuscript uses.

Everything here reads ``results/verification.json`` (written by :mod:`run`) and writes only into
``results/`` and ``figures/``. No number is typed by hand.

Usage::

    python research/arxiv_paper/experiments/verification/analyze.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent

#: Reference-free detectors need only the governing equation and its data; reference-based ones need
#: a trusted solution for the case in hand, which a new case does not have.
REFERENCE_FREE = ("PC201", "PC202", "PC301", "PC401", "PC403")
REFERENCE_BASED = ("PC701", "PC704")

RULE_TITLES = {
    "PC201": "boundary condition", "PC202": "initial condition", "PC301": "PDE residual",
    "PC401": "declared bounds", "PC403": "non-finite values", "PC701": "regression vs reference",
    "PC704": "spectral bias vs reference",
}

#: The rule a defect is *supposed* to be caught by, declared before the run (preregistered).
EXPECTED = {
    "bc_offset": ("PC201",),
    "wrong_coefficient": ("PC301",),
    "over_smoothed": ("PC704",),
    "unit_scale": ("PC401",),
    "nonfinite": ("PC403",),
    "zero_field": ("PC202", "PC301"),
}

PROBLEM_ORDER = ("steady_heat", "heat2d", "poisson", "helmholtz", "burgers", "allen_cahn")
PROBLEM_LABEL = {
    "steady_heat": "steady heat, constant source", "heat2d": "transient heat, 2-D",
    "poisson": "Poisson 2-D", "helmholtz": "Helmholtz 2-D", "burgers": "Burgers 1-D",
    "allen_cahn": "Allen-Cahn 1-D",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- detection matrix
def detection_matrix(payload: dict) -> dict:
    """``{problem: {defect: {rule: 'error'|'warning'}}}``, plus ``valid`` as ``{rule: (fired, of)}``."""
    out: dict[str, dict[str, dict]] = {}
    counts: dict[str, dict[str, list[int]]] = {}
    for c in payload["cases"]:
        name = c["problem"]
        rows = out.setdefault(name, {})
        per = counts.setdefault(name, {r: [0, 0] for r in RULE_TITLES})
        if c["family"] == "valid":
            for r in (*c["flagged_errors"], *c["flagged_warnings"]):
                per[r][0] += 1
            for r in RULE_TITLES:
                per[r][1] += 1
            continue
        row = rows.setdefault(c["defect"], {})
        for r in c["flagged_errors"]:
            row[r] = "error"
        for r in c["flagged_warnings"]:
            row.setdefault(r, "warning")
    for name, per in counts.items():
        out[name]["valid"] = {r: tuple(v) for r, v in per.items() if v[0] > 0}
    return out


def detection_rates(payload: dict) -> dict:
    """Per rule: how many single-defect cases it caught, and how many valid controls it flagged.

    The valid controls are split by whether the problem's governing equation can be declared to the
    checker exactly. A check that fires on a valid prediction of a problem whose equation the tool
    cannot represent is a measured false positive of the *declaration*, not of the check, and the
    two are reported apart.
    """
    rows = {}
    for rule in json.loads(json.dumps(list(RULE_TITLES))):
        caught = tried = fp = n_valid = 0
        fp_exp = n_exp = fp_inexp = n_inexp = 0
        missed = []
        for c in payload["cases"]:
            if c["family"] == "valid":
                n_valid += 1
                hit = rule in c["flagged_errors"] or rule in c["flagged_warnings"]
                fp += bool(hit)
                if c["expressible"]:
                    n_exp += 1
                    fp_exp += bool(hit)
                else:
                    n_inexp += 1
                    fp_inexp += bool(hit)
            else:
                tried += 1
                if rule in c["flagged_errors"] or rule in c["flagged_warnings"]:
                    caught += 1
                else:
                    missed.append(c["case_id"])
        rows[rule] = {"caught": caught, "invalid_cases": tried, "flagged_valid": fp, "valid_cases": n_valid,
                      "flagged_valid_expressible": fp_exp, "valid_expressible": n_exp,
                      "flagged_valid_inexpressible": fp_inexp, "valid_inexpressible": n_inexp,
                      "missed": missed}
    return rows


def expected_vs_measured(payload: dict) -> list[dict]:
    """For every single-defect case, the rule the protocol expected to catch it and what did."""
    rows = []
    for c in payload["cases"]:
        if c["family"] != "invalid":
            continue
        exp = EXPECTED.get(c["defect"], ())
        fired = sorted(set(c["flagged_errors"]) | set(c["flagged_warnings"]))
        rows.append({
            "case_id": c["case_id"], "problem": c["problem"], "defect": c["defect"],
            "rel_l2": c["measured_rel_l2"], "expected": list(exp), "fired": fired,
            "expected_met": bool(set(exp) & set(fired)),
            "only_reference_based": bool(fired) and not (set(fired) & set(REFERENCE_FREE)),
        })
    return rows


def reference_free_coverage(payload: dict) -> dict:
    """How many single-defect cases any reference-free rule caught, per problem and in total."""
    caught = total = 0
    per_problem: dict[str, dict[str, int]] = {}
    missed = []
    for c in payload["cases"]:
        if c["family"] != "invalid":
            continue
        total += 1
        fired = set(c["flagged_errors"]) | set(c["flagged_warnings"])
        hit = bool(fired & set(REFERENCE_FREE))
        per_problem.setdefault(c["problem"], {"caught": 0, "cases": 0})
        per_problem[c["problem"]]["cases"] += 1
        if hit:
            caught += 1
            per_problem[c["problem"]]["caught"] += 1
        else:
            missed.append(c["case_id"])
    return {"caught": caught, "cases": total, "per_problem": per_problem, "missed": missed}


def residual_calibration(payload: dict) -> dict:
    """Relative residual against the controlled relative L2 error of the valid controls."""
    out = {}
    for c in payload["cases"]:
        if c["family"] != "valid" or c["target_rel_l2"] == 0:
            continue
        out.setdefault(c["problem"], []).append(
            (c["target_rel_l2"], c["residual"]["residual_relative"], c["residual"]["reference_residual_relative"]))
    return {p: sorted(v) for p, v in out.items()}


def amplification(payload: dict) -> dict:
    """``rho / rel L2``: how much a field error is magnified by second-order differencing."""
    out = {}
    for p, pts in residual_calibration(payload).items():
        floor = pts[0][2]
        ratios = [(rel, rho / rel) for rel, rho, _ in pts if rho > 2 * floor]
        out[p] = {"points": ratios, "median": float(np.median([r for _, r in ratios])) if ratios else None}
    return out


def floor_table(payload: dict) -> dict:
    """The residual floor: the exact reference's own relative residual on the case grid."""
    out = {}
    for c in payload["cases"]:
        if c["case_id"].endswith("valid_exact"):
            r = c["residual"]
            out[c["problem"]] = {"step": r["residual_grid_step"], "floor": r["reference_residual_relative"],
                                 "scale": r["residual_scale"]}
    return out


# --------------------------------------------------------------------------- resolution study
RESOLUTIONS = (101, 201, 401, 801)
#: Budget of the refinement study: a grid larger than this is skipped, because the reference is
#: interpolated at every point and the cost is quadratic in the number of axes.
MAX_POINTS = 300_000


def resolution_study() -> dict:
    """How the reference's own relative residual falls as the grid is refined.

    Every axis is refined together, because for a time-dependent problem the time step enters the
    residual as much as the space step. The reference is an analytic or spectral function, so it can
    be sampled at any resolution.
    """
    import corpus
    import run as runner

    out = {}
    for name in ("burgers", "helmholtz", "poisson", "allen_cahn", "heat2d"):
        cls = corpus.BY_NAME[name]
        base = _spec(cls)
        rows = []
        for count in RESOLUTIONS:
            factor = count / base[0][3]
            spec = [(axis, lo, hi, max(21, int(round(n * factor)))) for axis, lo, hi, n in base]
            if math.prod(s[3] for s in spec) > MAX_POINTS:
                continue
            g = corpus.Grid(tuple(s[0] for s in spec), tuple(np.linspace(s[1], s[2], s[3]) for s in spec))
            prob = cls()
            prob.grid = g
            ref = np.asarray(prob.exact(g.mesh()), float)
            st = runner.residual_stat(name, ref, ref, g)
            rows.append({"points_per_axis": count, "step": st["residual_grid_step"],
                         "floor": st["reference_residual_relative"]})
        out[name] = rows
    return out


def _spec(cls) -> tuple:
    g = cls.grid
    spec = []
    for axis, values in zip(g.axes, g.values, strict=False):
        spec.append((axis, float(values[0]), float(values[-1]), len(values)))
    return tuple(spec)


# --------------------------------------------------------------------------- LaTeX
def _fmt(x: float, digits: int = 2, sci_below: float = 1e-3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    if x != 0 and (abs(x) >= 1e4 or abs(x) < sci_below):
        mantissa, exponent = f"{x:.{max(digits, 1) - 1}e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{x:.{digits}f}"


def latex_detection_matrix(matrix: dict, rules: tuple[str, ...], caption: str, label: str) -> str:
    """The per-problem, per-defect matrix as a ``longtable``: it is 42 rows and has to break pages."""
    head = ("l" + "c" * (len(rules) + 1))
    cols = " & ".join(rules)
    out = [rf"\begin{{longtable}}{{{head}}}",
           rf"\caption{{{caption}}}\label{{{label}}}\\",
           r"\toprule",
           "Problem & defect & " + cols + r" \\",
           r"\midrule", r"\endfirsthead",
           rf"\multicolumn{{{len(rules) + 2}}}{{l}}{{\footnotesize\itshape continued from the previous page}}\\",
           r"\toprule",
           "Problem & defect & " + cols + r" \\",
           r"\midrule", r"\endhead",
           r"\midrule",
           rf"\multicolumn{{{len(rules) + 2}}}{{r}}{{\footnotesize\itshape continued on the next page}}\\",
           r"\endfoot",
           r"\bottomrule", r"\endlastfoot"]
    for p in PROBLEM_ORDER:
        rows = matrix.get(p, {})
        first = True
        for key in ("valid", *EXPECTED):
            if key not in rows:
                continue
            cells = []
            for r in rules:
                v = rows[key].get(r)
                if isinstance(v, str):
                    cells.append(r"\err" if v == "error" else r"\warn")
                else:
                    cells.append(r"\textcolor{black!55}{--}" if v is None else f"${v[0]}/{v[1]}$")
            name = PROBLEM_LABEL[p] if first else ""
            label_txt = "valid controls" if key == "valid" else key.replace("_", r"\_")
            out.append(f"{name} & {label_txt} & " + " & ".join(cells) + r" \\")
            first = False
        if p != PROBLEM_ORDER[-1]:
            out.append(r"\midrule")
    out.append(r"\end{longtable}")
    return "\n".join(out)


def latex_detection_rates(rates: dict) -> str:
    lines = [r"\begin{tabular}{llrrrc}", r"\toprule",
             r"Rule & detector & caught & of & \multicolumn{2}{c}{valid flagged} \\",
             r"\cmidrule(lr){5-6}",
             r" & & & & expressible & other \\", r"\midrule"]
    for r in (*REFERENCE_FREE, *REFERENCE_BASED):
        d = rates[r]
        kind = "equation" if r in REFERENCE_FREE else "reference"
        lines.append(f"{r} & {RULE_TITLES[r]} ({kind}) & {d['caught']} & {d['invalid_cases']} & "
                     f"{d['flagged_valid_expressible']}/{d['valid_expressible']} & "
                     f"{d['flagged_valid_inexpressible']}/{d['valid_inexpressible']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_floor(floors: dict, amp: dict) -> str:
    """One table for the residual check's calibration: the floor, the amplification, and the level at
    which a valid prediction starts to be rejected."""
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Problem & grid step $h$ & $\rho_0$ & $\rho/\mathrm{rel.\,L2}$ (median) & "
             r"first level with $\rho>5\%$ \\", r"\midrule"]
    for p in PROBLEM_ORDER:
        if p not in floors:
            continue
        f = floors[p]
        floor = f["floor"]
        crossing = (r"never: $\rho_0>5\%$" if floor >= 0.05 else _first_crossing(p))
        ratio = _fmt((amp.get(p) or {}).get("median"))
        lines.append(f"{PROBLEM_LABEL[p]} & {_fmt(f['step'], 2, sci_below=5e-2)} & "
                     f"{_fmt(floor, 2, sci_below=2e-2)} & {ratio} & {crossing} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


_CROSS: dict[str, str] = {}


def _first_crossing(p: str) -> str:
    return _CROSS.get(p, "--")


# --------------------------------------------------------------------------- figures
def figures(payload: dict, floors: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 200, "savefig.bbox": "tight", "text.usetex": False})
    written = []

    # 1. residual against the controlled error, log-log, one line per problem
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for p in PROBLEM_ORDER:
        pts = residual_calibration(payload).get(p)
        if not pts:
            continue
        rel = [x[0] for x in pts]
        rho = [x[1] for x in pts]
        ax.plot(rel, rho, marker="o", ms=3, lw=1.0, label=PROBLEM_LABEL[p])
    ax.axhline(0.05, color="0.35", ls="--", lw=0.8)
    ax.annotate("5% residual tolerance", (1.3e-4, 0.056), fontsize=6, color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative $L^2$ error of the field")
    ax.set_ylabel(r"relative PDE residual $\rho$")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    f = out_dir / "residual_calibration.pdf"
    fig.savefig(f, metadata={"CreationDate": None})  # deterministic bytes
    plt.close(fig)
    written.append(f)

    # 2. residual floor against the number of points per axis, log-log
    study = payload.get("resolution_study") or {}
    if study:
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        for p, rows in study.items():
            n = [r["points_per_axis"] for r in rows]
            fl = [r["floor"] for r in rows]
            ax.plot(n, fl, marker="o", ms=3, lw=1.0, label=PROBLEM_LABEL.get(p, p))
        ax.axhline(0.05, color="0.35", ls="--", lw=0.8)
        ax.annotate("5% residual tolerance", (105, 0.056), fontsize=6, color="0.35")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = sorted({r["points_per_axis"] for rows in study.values() for r in rows})
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=6)
        ax.minorticks_off()
        ax.set_xlabel("points per axis (every axis refined together)")
        ax.set_ylabel(r"reference residual $\rho_0$")
        ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
        f = out_dir / "residual_floor.pdf"
        fig.savefig(f, metadata={"CreationDate": None})  # deterministic bytes
        plt.close(fig)
        written.append(f)

    return written


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "verification.json")
    args = ap.parse_args()
    payload = load(args.results)

    floors = floor_table(payload)
    for p, pts in residual_calibration(payload).items():
        for rel, rho, _ in pts:
            if rho > 0.05:
                _CROSS[p] = f"${_fmt(rel)}$" if rel >= 1e-3 else r"$10^{-4}$"
                break

    summary = {
        "schema": "pinneapple-arxiv-verification-summary/v1",
        "n_cases": len(payload["cases"]),
        "detection_matrix": detection_matrix(payload),
        "detection_rates": detection_rates(payload),
        "expected_vs_measured": expected_vs_measured(payload),
        "reference_free_coverage": reference_free_coverage(payload),
        "residual_calibration": residual_calibration(payload),
        "amplification": amplification(payload),
        "floors": floors,
        "first_crossing": _CROSS,
        "resolution_study": resolution_study(),
    }
    out = HERE / "results" / "summary.json"
    out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"-> {out}")

    tables = PAPER / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "verification_matrix.tex").write_text(latex_detection_matrix(
        summary["detection_matrix"], (*REFERENCE_FREE, *REFERENCE_BASED),
        r"Which rule fires on which defect, per problem. \err{} = error, \warn{} = warning, $n/m$ = "
        r"the number of valid levels (of $m$) on which the rule fired. The \emph{valid controls} row is the "
        r"calibration: a rule that fires there is either badly declared or amplifying a small error into a "
        r"large one.", "tab:verification-matrix") + "\n")
    (tables / "verification_rates.tex").write_text(latex_detection_rates(summary["detection_rates"]) + "\n")
    (tables / "verification_floors.tex").write_text(
        latex_floor(floors, summary["amplification"]) + "\n")

    print(f"-> {tables}/verification_*.tex")

    figs = PAPER / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    payload["resolution_study"] = summary["resolution_study"]
    for f in figures(payload, floors, figs):
        print(f"-> {f}")

    ok = sum(1 for r in summary["expected_vs_measured"] if r["expected_met"])
    total = len(summary["expected_vs_measured"])
    only_ref = sum(1 for r in summary["expected_vs_measured"] if r["only_reference_based"])
    print(f"expected rule fired on {ok}/{total} single-defect cases; {only_ref} caught only by a "
          f"reference-based check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

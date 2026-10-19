"""Rules that turn the results an example commits into manifest metrics and checks (``manifest.from_results``).

Only values in the results files are used. A check is one of:

* ``Flag``: a verdict the study recorded itself (a boolean in its results, e.g. ``stall_margin_ok``);
* ``CheckList``: check records the study wrote (each with its own ``passed``, value, reference, tolerance or limit);
* ``Bound``: a measured value against the acceptance threshold the study's own test applies to it, cited as
  ``file:line``; the literal must appear on that line (a test checks every citation), so no threshold is invented.

Paths are dotted, ``*`` matching every key of a dict or index of a list; a name template is formatted with the
matched keys (``{0}``, ``{1}``) and the object the last ``*`` matched (``{o[aircraft]}``).

Folders curated through a Lab ``repo_results`` source (its scripts live there) need no rule: the page links them
to that item. Folders with no measured results stay "not curated".
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Metric:
    file: str
    path: str
    name: str
    unit: str
    how: str


@dataclass(frozen=True)
class Flag:
    file: str
    path: str
    name: str
    kind: str
    detail: str = ""
    invert: bool = False                 # the recorded boolean says the bad thing happened (e.g. "departed")


@dataclass(frozen=True)
class Bound:
    file: str
    path: str
    name: str
    kind: str
    literal: str                         # the threshold as written in the cited source
    source: str                          # "tests/test_x.py:51": where the study applies it
    op: str = "<"                        # "<" or ">"
    absolute: bool = False
    unit: str = ""

    @property
    def threshold(self) -> float:
        return float(self.literal)


@dataclass(frozen=True)
class CheckList:
    file: str
    path: str                            # the list of records
    name: str                            # template over the record: "{r[case]}"
    kind: str
    value_key: str | None = None
    reference_key: str | None = None
    tolerance_key: str | None = None     # relative tolerance the study used
    limit_key: str | None = None         # upper limit the study used
    passed_key: str = "passed"


Rule = Metric | Flag | Bound | CheckList

_FL = "results/verification.json"
RESULTS_RULES: dict[str, list[Rule]] = {
    "examples/flight_dynamics": [
        Bound(_FL, "conservation.*.energy_rel_error", "energy conserved, dt {0} s", "physics", "1e-8",
              "tests/test_flight_dynamics.py:51"),
        Bound(_FL, "conservation.*.momentum_rel_error", "angular momentum conserved, dt {0} s", "physics", "1e-8",
              "tests/test_flight_dynamics.py:51"),
        Bound(_FL, "trim.*.residual", "trim residual, {o[aircraft]}", "verification", "1e-8",
              "tests/test_flight_dynamics.py:68"),
        Flag("results/takeoff.json", "nominal.*.lifted_off", "take-off: {0} lifts off", "sanity"),
        Flag("results/takeoff.json", "nominal.*.stall_margin_ok", "take-off: {0} keeps its stall margin", "physics",
             "CL reached below the assumed take-off CLmax"),
        Metric(_FL, "conservation.*.energy_rel_error", "energy_rel_error_dt{0}", "", "20 s free flight, relative "
               "change of total energy (results/verification.json)"),
        Metric(_FL, "lattice.*.rel_diff", "lattice_rel_diff_{o[aircraft]}_alpha{o[alpha_deg]}", "",
               "lattice-in-the-loop CL vs steady vortex lattice (results/verification.json)"),
        Metric(_FL, "turbulence.cases.*.sigma", "turbulence_sigma_{o[generator]}_{o[model]}_{o[component]}", "m/s",
               "measured gust standard deviation; MIL-F-8785C target in sigma_target (results/verification.json)"),
        Metric("results/takeoff.json", "nominal.*.ground_roll_m", "ground_roll_{0}", "m",
               "simulated take-off ground roll (results/takeoff.json)"),
        Metric("results/takeoff.json", "nominal.*.distance_35ft_m", "distance_35ft_{0}", "m",
               "simulated distance to 35 ft (results/takeoff.json)"),
    ],
    "examples/enso_forecasting": [
        Bound("results/solver_checks.json", "kelvin.rel_error", "Kelvin wave speed vs theory", "reference", "0.01",
              "tests/test_enso.py:49", absolute=True),
        Bound("results/solver_checks.json", "rossby.rel_error", "Rossby wave speed vs theory", "reference", "0.06",
              "tests/test_enso.py:56", absolute=True),
        Metric("results/solver_checks.json", "kelvin.speed_measured", "kelvin_speed", "m/s",
               "shallow-water solver, crossing time of the basin (results/solver_checks.json)"),
        Metric("results/solver_checks.json", "kelvin.speed_theory", "kelvin_speed_theory", "m/s",
               "sqrt(g' H) of the solver's layer (results/solver_checks.json)"),
        Metric("results/solver_checks.json", "rossby.speed_measured", "rossby_speed", "m/s",
               "shallow-water solver (results/solver_checks.json)"),
        Metric("results/solver_checks.json", "gill.rows.*.rel_l2_u", "gill_rel_l2_u_nlon{o[nlon]}", "",
               "Gill (1980) steady response vs the analytic solution (results/solver_checks.json)"),
        Metric("results/solver_checks.json", "recharge_oscillator.rows.*.period_rel_error",
               "recharge_period_rel_error_sub{o[sub]}", "", "recharge oscillator period vs theory "
               "(results/solver_checks.json)"),
    ],
    "examples/pinneapple_and_physicsnemo": [
        Metric("pinneapple/benchmarks/_out/pinneapple_pure_benchmark_results.json", "*.test_mse_u",
               "test_mse_u_{o[name]}", "", "test-set MSE of u; steps and learning rate in the same record "
               "(pinneapple/benchmarks/_out/pinneapple_pure_benchmark_results.json)"),
        Metric("pinneapple/benchmarks/_out/pinneapple_pure_benchmark_results.json", "*.train_time_s",
               "train_time_{o[name]}", "s", "wall-clock training time (same file)"),
    ],
    "examples/bricks": [
        CheckList("_out/summary.json", "validation", "{r[check]}", "reference", value_key="simulated",
                  reference_key="reference", tolerance_key="tolerance"),
    ],
}


# ---------------------------------------------------------------------------------------------- extraction
def _walk(obj: Any, parts: list[str], keys: tuple = (), parent: Any = None):
    """(matched keys, value, object the last '*' matched) for every match of the dotted path."""
    if not parts:
        yield keys, obj, parent
        return
    head, rest = parts[0], parts[1:]
    if head == "*":
        items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, list) else []
        for k, v in items:
            yield from _walk(v, rest, keys + (k,), v)
    elif isinstance(obj, dict) and head in obj:
        yield from _walk(obj[head], rest, keys, parent)
    elif isinstance(obj, list) and head.isdigit() and int(head) < len(obj):
        yield from _walk(obj[int(head)], rest, keys, parent)


def _number(v) -> bool:
    import math
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def extract(folder: str, rules: list[Rule]) -> tuple[list[dict], list[dict]]:
    """(metrics, checks) from the folder's results files by ``rules``; a rule whose file or path is missing gives
    nothing (never a placeholder)."""
    metrics, checks, cache = [], [], {}

    def load(f):
        if f not in cache:
            try:
                with open(os.path.join(folder, f)) as fh:
                    cache[f] = json.load(fh)
            except (OSError, ValueError):
                cache[f] = None
        return cache[f]

    for r in rules:
        data = load(r.file)
        if data is None:
            continue
        parts = [p for p in re.split(r"\.", r.path) if p] if r.path else []
        for keys, v, o in _walk(data, parts):
            name = "" if isinstance(r, CheckList) else r.name.format(*keys, o=o if isinstance(o, dict) else {})
            if isinstance(r, Metric) and _number(v):
                metrics.append({"name": re.sub(r"[^\w.-]+", "_", name), "value": v, "unit": r.unit, "how": r.how})
            elif isinstance(r, Flag) and isinstance(v, bool):
                checks.append({"name": name, "passed": (not v) if r.invert else v, "kind": r.kind,
                               "detail": (r.detail + "; " if r.detail else "") + f"recorded in {r.file} ({r.path})"})
            elif isinstance(r, Bound) and _number(v):
                x = abs(v) if r.absolute else v
                ok = x < r.threshold if r.op == "<" else x > r.threshold
                checks.append({"name": name, "passed": bool(ok), "value": v, "kind": r.kind,
                               "max" if r.op == "<" else "min": r.threshold,
                               "detail": f"{'|value|' if r.absolute else 'value'} {r.op} {r.literal}, the threshold "
                                         f"of {r.source}; value from {r.file}"})
            elif isinstance(r, CheckList) and isinstance(v, list):
                for rec in v:
                    if not isinstance(rec, dict) or not isinstance(rec.get(r.passed_key), bool):
                        continue
                    c = {"name": r.name.format(r=rec), "passed": rec[r.passed_key], "kind": r.kind,
                         "detail": f"recorded by the study in {r.file}"}
                    if r.value_key and _number(rec.get(r.value_key)):
                        c["value"] = rec[r.value_key]
                    if r.reference_key and _number(rec.get(r.reference_key)):
                        c["reference"] = rec[r.reference_key]
                    if r.tolerance_key and _number(rec.get(r.tolerance_key)):
                        c["rtol"] = rec[r.tolerance_key]
                    if r.limit_key and _number(rec.get(r.limit_key)):
                        c["max"] = rec[r.limit_key]
                    checks.append(c)
    return metrics, checks


def check_citations(repo: str) -> list[str]:
    """Problems with the ``Bound`` citations: each literal must appear on the cited line of the cited file."""
    out = []
    for folder, rules in RESULTS_RULES.items():
        for r in rules:
            if not isinstance(r, Bound):
                continue
            path, _, line = r.source.partition(":")
            try:
                text = open(os.path.join(repo, path), encoding="utf-8").read().splitlines()[int(line) - 1]
            except (OSError, ValueError, IndexError):
                out.append(f"{folder}: {r.name}: cannot read {r.source}")
                continue
            if r.literal not in text:
                out.append(f"{folder}: {r.name}: {r.literal} is not on {r.source}: {text.strip()}")
    return out

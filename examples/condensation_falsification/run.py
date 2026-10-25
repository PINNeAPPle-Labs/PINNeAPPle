"""Condensation falsification campaign: the measured runs of docs/research/condensation_falsification.md.

Each pre-registered test is a function of :mod:`pinneapple_condense.falsify`, run in a fresh process (its peak memory
is that process's). Results go to ``results/<name>.json`` and are kept (rerun one with ``--force <name>``); a run that
fails is recorded with its error, not hidden. ``--only T1,F1`` runs a subset.

Run: ``python examples/condensation_falsification/run.py [--only ...] [--force ...]``.
"""
from __future__ import annotations

import argparse
import json
import os

from pinneapple_condense import benchmark as B

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
MOD = "pinneapple_condense.falsify"


def plan() -> dict[str, list[tuple[str, str, dict]]]:
    """Study name -> list of (result name, function, kwargs)."""
    t1 = [(f"T1_ref_{nb}", "t1_reference", {"nb": nb, "fine": nb <= 64}) for nb in (16, 32, 64, 128)]
    t1 += [(f"T1_{nb}_m{m}", "t1_point", {"nb": nb, "m": m}) for nb in (16, 32, 64, 128) for m in (6, 12, 24, 48)
           if not (nb == 128 and m == 48)]           # m = 48 at 128 x 128: dense top merges beyond 15 GB
    ms = [1, 2, 3, 4, 6, 8, 12, 16]
    cfd = {"F0": [("F0_validation", "f0_validation", {})],
           "F1": [("F1_advection", "f1_advection", {"ms": ms})],
           "F2": [("F2_stokes", "f2_stokes", {"ms": ms})],
           "F3": [("F3_cht", "f3_cht", {"ms": ms})]}
    thermal = {"T2": [("T2_contrast", "t2_contrast", {})],
               "T3": [("T3_anisotropy", "t3_anisotropy", {})],
               "T4": [("T4_no_repetition", "t4_no_repetition", {})],
               "T5": [("T5_variants", "t5_variants", {})],
               "T6": [("T6_nonlinear_k", "t6_t7_nonlinear", {"case": "k"})],
               "T7": [("T7_nonlinear_R", "t6_t7_nonlinear", {"case": "R"})],
               "T8": [("T8_adversarial", "t8_adversarial", {})],
               "T9": [("T9_thin", "t9_thin", {})],
               "T10": [("T10_pitch", "t10_pitch", {})]}
    meshless = {"M1": [("M1_advection", "m1_advection", {})],
                "M2": [("M2_stokes", "m2_stokes", {})],
                "M3": [("M3_hybrid", "m3_hybrid", {})],
                "M4": [("M4_condensed", "m4_meshless_condensed", {})],
                "E1": [("E1_cfd", "e1_cfd", {})]}
    return {"T1": t1, **cfd, **thermal, **meshless}


def run(name, func, kw, force, timeout=7200):
    path = os.path.join(RES, f"{name}.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    print(f"-- {name}: {func}({kw})", flush=True)
    try:
        r = B.isolated(func, timeout=timeout, module=MOD, **kw)
    except Exception as e:                       # recorded, not hidden
        r = {"failed": str(e)[-1500:]}
    json.dump(r, open(path, "w"), indent=1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", default="")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    force = set(filter(None, a.force.split(",")))
    for study, items in plan().items():
        if only and study not in only:
            continue
        for name, func, kw in items:
            run(name, func, kw, "all" in force or study in force or name in force)


if __name__ == "__main__":
    main()

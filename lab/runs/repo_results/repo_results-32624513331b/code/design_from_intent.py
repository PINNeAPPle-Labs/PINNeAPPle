"""Design from intent: five requests through the same pipeline.

1. "I want a toy car that does a loop"              -> loop track, clothoid entry, launch speed (Newton check)
2. "a bracket that holds 2 kN" (+ the arm length)   -> angle bracket (CalculiX check)
3. "a heat sink for a 40 W chip in still air"       -> asks for the temperature limit, then a plate-fin heat sink
4. "a wing for a 1.2 kg model glider ..."            -> glider wing (vortex-lattice check)
5. "an electric motor for my drone"                 -> out of scope: says what would be needed

Each design is written to ``_out/<case>/`` (report.html, report.pdf, design.json, geometry STEP / STL / glTF, figures,
``check/`` for ``pinneapple check``); ``_out/summary.json`` lists the verdicts and the model calls.

Run: ``python examples/design_from_intent/design_from_intent.py [--case loop] [--no-verify]``. The full solvers are
used when installed (Newton: ``pip install "pinneapple[newton]"``; CalculiX ``ccx`` and ``gmsh`` on the path);
without them the trust report says the full check did not run.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pinneapple as pp
from pinneapple_design.intent import OutOfScope, formulate

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {
    "loop": ("I want a toy car that does a loop", None),
    "bracket": ("a bracket that holds 2 kN", {"arm_length": (100, "mm")}),
    "heatsink": ("a heat sink for a 40 W chip in still air", {"t_limit": (85, "°C")}),
    "glider": ("a wing for a 1.2 kg model glider, at most 2 m span", None),
    "out_of_scope": ("an electric motor for my drone", None),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), action="append")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    a = ap.parse_args()
    summary = {}
    path = os.path.join(a.out, "summary.json")
    if os.path.exists(path):
        summary = json.load(open(path))
    for name in a.case or list(CASES):
        text, answers = CASES[name]
        print(f"\n=== {name}: {text!r}")
        t0 = time.time()
        first = formulate(text)                     # the requirement spec, or a refusal with what is needed
        print(f"  first reply: {first}" if isinstance(first, OutOfScope) else
              f"  first reply: questions {[q.text for q in first.questions]}" if first.questions else
              "  first reply: complete requirements")
        r = pp.design(text, answers=answers, out_dir=os.path.join(a.out, name), verify=not a.no_verify,
                      progress=lambda m: print("  ", m))
        d = r.to_dict()
        if hasattr(r, "summary"):
            print(r.summary())
            entry = {"text": text, "answers": {k: list(v) for k, v in (answers or {}).items()},
                     "first_reply": [q.to_dict() for q in first.questions],
                     "verdict": r.verdict, "probability": r.assessment.probability, "ci95": list(r.assessment.ci),
                     "trust": r.assessment.trust.decisao, "design": d["design"], "calls": d["calls"],
                     "verification": {"ran": r.verification.ran, "label": r.verification.label,
                                      "comparison": d["verification"]["comparison"]},
                     "savings": d["savings"]["text"], "wall_s": round(time.time() - t0, 1)}
        else:
            print(f"  {r}")
            entry = {"text": text, "reply": d}
        summary[name] = entry
        os.makedirs(a.out, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=1, default=str)


if __name__ == "__main__":
    main()

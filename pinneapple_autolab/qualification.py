"""Before a campaign: how much of the reference solver's answer is the mesh and the iteration count?

    q = solver_qualification(sim, baseline, {"coarse": {"resolution": "coarse"}, "medium": {}, "fine":
                             {"resolution": "fine"}, "medium, 1000 it": {"iterations": 1000}}, "runs/qual",
                             reference="fine", campaign="medium")

runs the same design under each variant (its own ledger, outside the mission budget) and reports, per output, the
relative difference of every variant from ``reference``. The difference between the campaign's setting and the
reference is the discretisation uncertainty every later comparison has to be read against: a design improvement
smaller than it is not resolved by the solver, and the report says so.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from .ledger import Ledger, RunRecord
from .simulator import Simulator

__all__ = ["solver_qualification"]


def solver_qualification(sim: Simulator, params: Mapping[str, float], variants: Mapping[str, Mapping[str, Any]],
                         folder: str, reference: str, campaign: str,
                         outputs: Sequence[str] | None = None) -> dict[str, Any]:
    led = Ledger(folder)
    rows: dict[str, dict[str, Any]] = {}
    for name, opts in variants.items():
        led.event("qualify", f"running variant {name!r} {dict(opts)}", running=True)
        out = sim.run(dict(params), os.path.join(folder, "cases", name.replace(" ", "_").replace(",", "")),
                      **dict(opts))
        rec = RunRecord(f"qual-{name}", "probe", dict(params), bool(out.ok), dict(out.outputs) if out.ok else {},
                        out.message, float(out.cost_s), reason=f"solver qualification: {name}", meta=dict(out.meta))
        led.add(rec)
        rows[name] = {"ok": rec.ok, "outputs": rec.outputs, "cost_s": rec.cost_s, "options": dict(opts),
                      "cells": out.meta.get("cells"), "residuals": out.meta.get("final_residuals"),
                      "iteration_std": out.meta.get("iteration_std", {}),
                      "message": rec.message}
    led.save()
    ref = rows.get(reference, {}).get("outputs") or {}
    outs = list(outputs or ref.keys())
    for _name, r in rows.items():
        r["rel_diff_vs_reference"] = {k: (r["outputs"][k] - ref[k]) / abs(ref[k]) for k in outs
                                      if k in r["outputs"] and k in ref and ref[k]}
    camp = rows.get(campaign, {}).get("rel_diff_vs_reference", {})
    return {"variants": rows, "reference": reference, "campaign": campaign, "outputs": outs,
            "campaign_vs_reference": camp, "ledger": os.path.join(folder, "ledger.json")}

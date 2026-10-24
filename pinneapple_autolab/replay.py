"""A simulator that answers from a ledger of measured runs when it holds the exact design, and runs the real
simulator otherwise.

    sim = ReplaySimulator(RaceCarAero(), "examples/autolab_aero/_out/ledger.json", fields_dir="work/fields")

A replayed outcome is the recorded measurement (outputs, cost, solver metadata, fields when kept) of a run made
earlier on the same simulator settings; its ``meta["replayed_from"]`` names the ledger and run, and the time it
cost then is kept (``cost_s``) with ``meta["replay_wall_s"]`` the time it took now. A design not in the ledger
(compared to ``atol`` in every variable) is simulated for real. Nothing is interpolated.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from .simulator import SimOutcome, Simulator

__all__ = ["ReplaySimulator"]


class ReplaySimulator(Simulator):
    def __init__(self, simulator: Simulator, ledger_path: str, fields_dir: str | None = None, atol: float = 1e-9):
        self.sim, self.ledger_path, self.fields_dir, self.atol = simulator, ledger_path, fields_dir, atol
        self.name = f"{simulator.name} (replaying {os.path.basename(ledger_path)} where it holds the design)"
        self.fidelity, self.outputs = simulator.fidelity, simulator.outputs
        self.field_names, self.regions = simulator.field_names, simulator.regions
        with open(ledger_path) as f:
            self.records = [r for r in json.load(f)["runs"] if r["ok"]]
        self.hits: list[str] = []
        self.misses = 0

    def check(self, params):
        return self.sim.check(params)

    def field_points(self, params):
        return self.sim.field_points(params)

    def _find(self, params: Mapping[str, float], options: Mapping[str, Any]) -> dict[str, Any] | None:
        for r in self.records:
            if set(r["params"]) != set(params):
                continue
            if all(abs(float(r["params"][k]) - float(params[k])) <= self.atol for k in params):
                if options.get("resolution") and r["meta"].get("resolution") != options["resolution"]:
                    continue
                return r
        return None

    def run(self, params: Mapping[str, float], workdir: str, **options: Any) -> SimOutcome:
        t0 = time.perf_counter()
        r = self._find(params, options)
        if r is None:
            self.misses += 1
            return self.sim.run(params, workdir, **options)
        self.hits.append(r["id"])
        fields = None
        if self.fields_dir:
            p = os.path.join(self.fields_dir, f"{r['id']}.npz")
            if os.path.exists(p):
                fields = {k: np.asarray(v, float) for k, v in np.load(p).items()}
        meta = dict(r["meta"]) | {"replayed_from": {"ledger": self.ledger_path, "run": r["id"], "started": r["started"]},
                                  "replay_wall_s": time.perf_counter() - t0}
        return SimOutcome(dict(r["outputs"]), True, "", float(r["cost_s"]), fields, meta)

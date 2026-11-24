"""``inverse_heat``: the convection coefficient h of a heat-spreader plate from 8 thermocouples (issue #299, case 2).

Methods: the plate PINN (h as an inverse parameter, per case), and surrogate + calibration (MLP, DeepONet trained once
on finite-volume runs over h in [5, 100] W/m2K, power in [2, 20] W). Reference: golden-section calibration through the
finite-volume model itself (computed when the problem is prepared). Cases: 3 inside the surrogates' range, 2 outside.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..registry import Check, Column, Problem, register_problem
from . import inverse_heat_impl as H


@register_problem("inverse_heat")
class InverseHeatProblem(Problem):
    title = "Inverse heat (h from 8 sensors)"
    kind = "inverse"
    space_pinn = H.SPACES["pinn"]
    space_mlp = H.SPACES["mlp"]
    space_deeponet = H.SPACES["deeponet"]
    title_mlp = "MLP surrogate + calibration"
    title_deeponet = "DeepONet surrogate + calibration"
    outputs_mlp = outputs_deeponet = outputs_pinn = "both"

    def default_options(self):
        return {"n_train": 200, "n_val": 40, "reference_seeds": 3}

    def smoke_options(self):
        return {"n_train": 40, "n_val": 10, "reference_seeds": 1}

    def _set(self, prepared):
        self._prepared = Path(prepared)
        H.set_data_dir(prepared, getattr(self, "smoke", False))

    # ------------------------------------------------------------------ data
    def prepare(self, out, n_workers=1):
        out.mkdir(parents=True, exist_ok=True)
        cost = H.generate(out, self.options["n_train"], self.options["n_val"], workers=n_workers)
        (out / "cost.json").write_text(json.dumps(cost, indent=1))
        ref = [H.classical_job(s) for s in range(self.options["reference_seeds"])]
        (out / "reference.json").write_text(json.dumps(ref, indent=1))
        return {"files": ["inverse_heat.npz", "cost.json", "reference.json"], "cost": cost}

    def artifact_files(self, out):
        return ["inverse_heat.npz", "cost.json", "reference.json"]

    def data_cost_s(self, out):
        p = Path(out) / "cost.json"
        return json.loads(p.read_text())["train_set_core_s"] if p.exists() else None

    def solver_time_per_instance_s(self, out):
        p = Path(out) / "reference.json"
        if not p.exists():
            return None
        return float(np.mean([c["calibration_core_s"] for r in json.loads(p.read_text()) for c in r["cases"]]))

    def reference_row(self, out):
        """The finite-volume model + golden-section calibration, on the same readings (seeds) as the methods."""
        p = Path(out) / "reference.json"
        if not p.exists():
            return None
        refs = json.loads(p.read_text())
        return {"title": "FV model + calibration (reference)", "seeds": [self._extract_cases(r["cases"]) for r in refs]}

    def verify_solver(self):
        from ..validation import gci_check
        hs = []
        sizes = []
        for nx, ny in ((200, 120), (100, 60), (50, 30)):              # fine, medium, coarse
            _, _, T = H.plate_2d(25.0, nx=nx, ny=ny, power=10.0, **H.FV)
            hs.append(float(T.max()))
            sizes.append(H.FV["lx"] / nx)
        c = gci_check(self.id, "the hot-spot temperature (h = 25, 10 W)", hs, sizes, max_gci=0.02)
        c.stage = "problem"
        return [c]

    def data_checks(self, out):
        self._set(out)
        z = np.load(Path(out) / "inverse_heat.npz")
        C = [Check("finite values", "data", self.id, all(np.isfinite(z[k]).all() for k in z.files))]
        sizes = {"train": len(z["train_hp"]), "val": len(z["val_hp"])}
        want = {"train": self.options["n_train"], "val": self.options["n_val"]}
        C.append(Check("split sizes as configured", "data", self.id, sizes == want, value=sizes, threshold=want))
        tr = {tuple(np.round(r, 10)) for r in z["train_hp"]}
        va = {tuple(np.round(r, 10)) for r in z["val_hp"]}
        C.append(Check("no (h, power) shared between train and val", "data", self.id, not (tr & va), value=len(tr & va)))
        ood_h = [h for _, h in H.CASES["ood"]]
        C.append(Check("OOD cases outside the surrogates' h range", "data", self.id,
                       all(h > H.H_RANGE[1] or h < H.H_RANGE[0] for h in ood_h), value=ood_h, threshold=list(H.H_RANGE)))
        id_h = [h for _, h in H.CASES["id"]]
        C.append(Check("in-range cases inside the surrogates' h range", "data", self.id,
                       all(H.H_RANGE[0] <= h <= H.H_RANGE[1] for h in id_h), value=id_h))
        C.append(Check("normalisation from the training split only", "data", self.id, True,
                       detail="Data() scales by the training fields' standard deviation"))
        return C

    def validation(self, out):
        from pinneapple_physics.physics_audit import continuum

        from ..validation import envelope_checks, physics_audit_check
        z = np.load(Path(out) / "inverse_heat.npz")
        f = lambda h, p: {"h": float(h), "power": float(p)}          # noqa: E731
        C = envelope_checks(self.id, [f(h, p) for h, p in z["train_hp"]], [f(h, p) for p, h in H.CASES["id"]],
                            [f(h, p) for p, h in H.CASES["ood"]])
        C.append(physics_audit_check(self.id, "thin-plate conduction with face convection", "solver",
                                     ["k t lap T - 2 h (T - T_air) + q = 0"], [continuum()], {"Kn": 0.0}))
        return C

    # ------------------------------------------------------------------ evaluation
    def columns(self):
        return [Column("h_id", "h rel. err, in range (n = 3)", "id", "h", kind="integral", pct=True),
                Column("h_ood", "h rel. err, out of range (n = 2)", "ood", "h", kind="integral", pct=True),
                Column("hot_id", "Hot-spot err °C, in range (n = 3)", "id", "hot", kind="integral"),
                Column("f_id", "Max field err °C, in range (n = 3)", "id", "field"),
                Column("f_ood", "Max field err °C, out of range (n = 2)", "ood", "field")]

    def validity_domain(self, prepared=None):
        """Where the case holds (#58): the ranges the case generator draws its training pairs from. The OOD cases put
        the convection coefficient at 150 and 200 W/m^2K, outside the range the model was fitted on."""
        from ..validity import ValidityDomain
        from .inverse_heat_impl import H_RANGE, P_RANGE
        return ValidityDomain(
            "inverse_heat",
            {"power [W]": (float(P_RANGE[0]), float(P_RANGE[1])),
             "h_true [W/m^2K]": (float(H_RANGE[0]), float(H_RANGE[1]))},
            source="declared by the case generator (P_RANGE and H_RANGE in inverse_heat_impl)",
            note="the OOD cases use h_true = 150 and 200 W/m^2K, outside the fitted range")

    def main_metric(self):
        return "id", "h"

    def ood_metric(self):
        return "ood", "h"

    def summary_metrics(self):
        return [("h (integral)", "integral", ("id", "h"), ("ood", "h"), True)]

    def instances(self):
        fin = [(f"{k}{i}", {"kind": k, "case": i}) for k, lst in H.CASES.items() for i in range(len(lst))]
        return {"val": [("", {"kind": "val", "case": 0})], "final": fin}

    @staticmethod
    def _extract_cases(cases):
        out = {}
        for kind in ("id", "ood"):
            cs = [c for c in cases if c["kind"] == kind]
            ids = [f"{kind}{i}" for i in range(len(cs))]
            out[kind] = {"h": {"values": [c["h_rel_err"] for c in cs], "ids": ids},
                         "hot": {"values": [c["hot_spot_err_C"] for c in cs], "ids": ids},
                         "field": {"values": [c["max_field_err_C"] for c in cs], "ids": ids}}
        return out

    def extract(self, method_id, results):
        if "" in results:
            return self._extract_cases(results[""]["cases"]) if "cases" in results[""] else {}
        cases = [dict(r, kind=r["kind"]) for _, r in sorted(results.items())]
        return self._extract_cases(cases)

    def search_score(self, method_id, result):
        return result["h_rel_err"] if method_id == "pinn" else result["val_rel_l2"]

    def time_per_instance(self, method_id, results, per_instance):
        if per_instance:
            return float(np.mean([r["train"]["wall_s"] for r in results.values()]))
        return float(np.mean([c["calibration_s"] for r in results.values() for c in r.get("cases", [])]))

    # ------------------------------------------------------------------ trainers
    def train_pinn(self, prepared, cfg, seed, budget, *, kind, case, **_):
        self._set(prepared)
        return H.pinn_job(cfg, seed, budget, kind=kind, case=case)

    def train_mlp(self, prepared, cfg, seed, budget, *, final=True, **_):
        self._set(prepared)
        return H.surrogate_job("mlp", cfg, seed, budget, final=final)

    def train_deeponet(self, prepared, cfg, seed, budget, *, final=True, **_):
        self._set(prepared)
        return H.surrogate_job("deeponet", cfg, seed, budget, final=final)

    # ------------------------------------------------------------------ gates
    def checks_pinn(self, prepared, cfg):
        return [Check("PDE residual of an exact solution ~ 0", "architecture", f"{self.id}/pinn", None,
                      detail="the plate PDE with its device source has no closed-form solution; the forward model is "
                             "verified by grid convergence instead (problem checks)")]

    def _surrogate_checks(self, mid, prepared, cfg):
        import torch

        from .fields import _overfit_check
        self._set(prepared)
        D = H.Data()
        m = H.Surrogate(mid, cfg, D.xy)
        z = D.z(np.array([10.0, 30.0, 60.0]), np.array([5.0, 10.0, 15.0]))
        y = m(z)
        T = torch.tensor(D.T["train"][:3] / D.ts, dtype=torch.float32)
        Zt = D.z(*D.hp["train"][:3].T)
        return [Check("forward shape", "architecture", f"{self.id}/{mid}", tuple(y.shape) == (3, 60, 100), value=tuple(y.shape)),
                Check("parameter count", "architecture", f"{self.id}/{mid}", True,
                      value=int(sum(p.numel() for p in m.parameters())), severity="warning"),
                _overfit_check(f"{self.id}/{mid}", lambda: ((m(Zt) - T) ** 2).mean(), list(m.parameters()), steps=1000,
                               lr=cfg["lr"])]

    def checks_mlp(self, prepared, cfg):
        return self._surrogate_checks("mlp", prepared, cfg)

    def checks_deeponet(self, prepared, cfg):
        return self._surrogate_checks("deeponet", prepared, cfg)


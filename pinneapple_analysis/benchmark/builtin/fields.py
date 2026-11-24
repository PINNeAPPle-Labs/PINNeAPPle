"""Field-operator problems (input field -> output field on a grid) and the generic methods that learn them.

``FieldOperatorProblem`` reads ``prepared/<id>/data.npz`` with ``{train,val,test,ood}_{x,y}`` arrays; subclasses
generate it (``generate``) and may add a PINN formulation (``pinn_setup``). ``ArrayProblem`` wraps a user's own arrays.
Methods: FNO (``pinneapple_neural ... fno``), DeepONet (``... deeponet``), PINN (``PINNFactory``), all trained under a
``ComputeBudget`` with ``train_within_budget``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_analysis.cost.equal_budget import (
    cosine_lr,
    relative_l2,
    set_lr,
    time_per_query,
    train_within_budget,
)

from ..registry import Check, Column, Method, Problem, register_method

_METRICS = ("rel_l2", "qoi_rel_err", "hf_bias")


def _hf_bias(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """High-wavenumber energy bias per instance (``pinneapple_analysis.validation.spectral``): about -1 for a
    prediction with its small scales smoothed away, 0 for the reference, nan where it cannot be computed."""
    from pinneapple_analysis.validation.spectral import high_wavenumber_bias
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    nd = true.ndim - 1
    out = np.full(len(true), np.nan)
    if nd < 1 or min(true.shape[1:]) < 8:
        return out
    for i in range(len(true)):
        try:
            out[i] = high_wavenumber_bias(pred[i].reshape(true.shape[1:]), true[i], nd, periodic=False)
        except ValueError:
            pass
    return out


BATCH = 20
OPERATOR_WD = 0.0          # Adam weight decay for the operators: 1e-4 held the FNO at the mean prediction on NS


def _sha(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


class FieldData:
    """The prepared arrays with pointwise Gaussian normalisers computed on the training split only."""

    def __init__(self, prepared: Path):
        z = np.load(Path(prepared) / "data.npz")
        self.raw = {k: z[k].astype(np.float32) for k in z.files}
        x, y = self.raw["train_x"], self.raw["train_y"]
        self.xm, self.xs = x.mean(0), x.std(0) + 1e-6
        self.ym, self.ys = y.mean(0), y.std(0) + 1e-6
        self.shape = x.shape[1:]
        self.norm_source = "train"

    def x(self, split):
        import torch
        return torch.from_numpy((self.raw[f"{split}_x"] - self.xm) / self.xs)

    def y(self, split):
        import torch
        return torch.from_numpy(self.raw[f"{split}_y"])

    def decode(self, yn):
        import torch
        return yn * torch.from_numpy(self.ys) + torch.from_numpy(self.ym)


class FieldOperatorProblem(Problem):
    """Base of the grid-to-grid problems. The table reports, per method, the relative L2 of the output field and the
    relative error of an integral quantity (``qoi``) on the PINN instances, the whole test set and the OOD set."""
    kind = "field_operator"
    qoi_label = "integral"

    def default_options(self) -> dict[str, Any]:
        return {"n_train": 1000, "n_val": 100, "n_test": 200, "n_ood": 100, "pinn_test_ids": [0, 1, 2],
                "pinn_ood_ids": [0, 1], "pinn_val_id": 0}

    def smoke_options(self) -> dict[str, Any]:
        return {"n_train": 40, "n_val": 8, "n_test": 8, "n_ood": 4}

    # hyper-parameter grids (each method draws n_trials of them)
    space_deeponet = {"hidden": [128, 256], "depth": [3, 4], "p": [64, 128], "lr": [1e-3, 3e-3]}
    space_pinn = {"width": [64, 128], "depth": [4, 6], "lr": [1e-3, 3e-3], "n_colloc": [2048, 8192]}
    space_fno = {"width": [24, 32], "modes": [12, 16], "lr": [1e-3, 3e-3]}

    # ------------------------------------------------------------------ data
    def generate(self, split: str, n: int, n_workers: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (inputs, outputs, solver core-seconds) for one split."""
        raise NotImplementedError

    def prepare(self, out: Path, n_workers: int = 1) -> dict[str, Any]:
        out.mkdir(parents=True, exist_ok=True)
        arrays, cost = {}, {}
        for split in ("train", "val", "test", "ood"):
            n = int(self.options[f"n_{split}"])
            x, y, s = self.generate(split, n, n_workers)
            arrays[f"{split}_x"], arrays[f"{split}_y"] = x.astype(np.float32), y.astype(np.float32)
            cost[split] = {"n": n, "solver_core_s": float(s)}
        np.savez_compressed(out / "data.npz", **arrays)
        total = sum(c["solver_core_s"] for c in cost.values())
        cost_doc = {"splits": cost, "solver_core_s_total": total,
                    "solver_core_s_per_sample": total / max(1, sum(c["n"] for c in cost.values())),
                    "train_set_core_s": cost["train"]["solver_core_s"] + cost["val"]["solver_core_s"],
                    "timing": "process CPU time of the solver call, summed over samples (one thread)"}
        (out / "cost.json").write_text(json.dumps(cost_doc, indent=1))
        return {"files": ["data.npz", "cost.json"], "cost": cost_doc}

    def artifact_files(self, out: Path) -> list[str]:
        return ["data.npz", "cost.json"]

    def data_cost_s(self, out: Path) -> float | None:
        p = Path(out) / "cost.json"
        return json.loads(p.read_text())["train_set_core_s"] if p.exists() else None

    def solver_time_per_instance_s(self, out: Path) -> float | None:
        p = Path(out) / "cost.json"
        return json.loads(p.read_text())["solver_core_s_per_sample"] if p.exists() else None

    def data_checks(self, out: Path) -> list[Check]:
        """Finite values, split sizes as configured, no input shared between splits (hashes), OOD inputs really
        outside the training distribution (PCA reconstruction error beyond the in-distribution 99th percentile),
        normalisation statistics from the training split only."""
        z = np.load(Path(out) / "data.npz")
        C = []
        finite = all(np.isfinite(z[k]).all() for k in z.files)
        C.append(Check("finite values", "data", self.id, finite, detail="all arrays finite" if finite else "NaN/inf found"))
        sizes = {s: int(z[f"{s}_x"].shape[0]) for s in ("train", "val", "test", "ood")}
        want = {s: int(self.options[f"n_{s}"]) for s in sizes}
        C.append(Check("split sizes as configured", "data", self.id, sizes == want, value=sizes, threshold=want))
        hashes = {s: {_sha(a) for a in z[f"{s}_x"]} for s in sizes}
        dup_within = {s: sizes[s] - len(h) for s, h in hashes.items()}
        C.append(Check("no duplicate inputs within a split", "data", self.id, all(v == 0 for v in dup_within.values()),
                       value=dup_within, threshold=0))
        leaks = {f"{a}&{b}": len(hashes[a] & hashes[b]) for i, a in enumerate(sizes) for b in list(sizes)[i + 1:]}
        C.append(Check("no input shared between splits (hashes)", "data", self.id,
                       all(v == 0 for v in leaks.values()), value=leaks, threshold=0))
        C.append(self._ood_check(z))
        C.append(Check("normalisation from the training split only", "data", self.id, True, value="train",
                       detail="FieldData computes means and standard deviations on train_x/train_y only"))
        return C

    def _ood_check(self, z) -> Check:
        tr = z["train_x"].reshape(len(z["train_x"]), -1).astype(np.float64)
        if len(z["ood_x"]) == 0:
            return Check("OOD set outside the training distribution", "data", self.id, None, detail="no OOD set")
        mu = tr.mean(0)
        k = int(min(20, len(tr) - 1, tr.shape[1]))
        _, _, vt = np.linalg.svd(tr - mu, full_matrices=False)
        basis = vt[:k]

        sd = (tr - mu) @ basis.T
        sd = sd.std(0) + 1e-12

        def err(a):                                               # shape novelty: relative reconstruction error
            a = a.reshape(len(a), -1) - mu
            return np.linalg.norm(a - (a @ basis.T) @ basis, axis=1) / (np.linalg.norm(a, axis=1) + 1e-12)

        def mahal(a):                                             # amplitude / mode-mix novelty in the PCA space
            c = (a.reshape(len(a), -1) - mu) @ basis.T
            return np.sqrt(((c / sd) ** 2).sum(1))
        te, oo = z["test_x"], z["ood_x"]
        t1, t2 = float(np.percentile(err(te), 99)), float(np.percentile(mahal(te), 99))
        frac = float(((err(oo) > t1) | (mahal(oo) > t2)).mean())
        return Check("OOD set outside the training distribution", "data", self.id, frac >= 0.5, value=round(frac, 3),
                     threshold=0.5, severity="warning",
                     detail=f"fraction of OOD inputs beyond the test set's 99th percentile of the PCA({k}) reconstruction "
                            f"error ({t1:.3g}) or of the Mahalanobis distance in PCA space ({t2:.3g})")

    # ------------------------------------------------------------------ evaluation
    def qoi(self, field: np.ndarray) -> np.ndarray:
        f = field.reshape(field.shape[0], -1).astype(np.float64)
        return f.mean(1)

    def scores(self, pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
        e = relative_l2(pred, true)
        q = np.abs(self.qoi(pred) - self.qoi(true)) / np.abs(self.qoi(true))
        hf = _hf_bias(pred, true)
        return {"rel_l2": e.tolist(), "rel_l2_mean": float(e.mean()), "qoi_rel_err": q.tolist(),
                "qoi_rel_err_mean": float(q.mean()), "hf_bias": hf.tolist(), "hf_bias_mean": float(np.nanmean(hf))}

    def columns(self) -> list[Column]:
        nt, no = len(self.options["pinn_test_ids"]), len(self.options["pinn_ood_ids"])
        return [Column("l2_pinn", f"Rel. L2, test (n = {nt})", "pinn_test", "rel_l2"),
                Column("l2_test", f"Rel. L2, test, all (n = {self.options['n_test']})", "test", "rel_l2"),
                Column("l2_ood", f"Rel. L2, OOD (n = {no})", "pinn_ood", "rel_l2"),
                Column("l2_ood_all", f"Rel. L2, OOD, all (n = {self.options['n_ood']})", "ood", "rel_l2"),
                Column("hf_pinn", f"HF energy bias, test (n = {nt})", "pinn_test", "hf_bias"),
                Column("qoi_pinn", f"{self.qoi_label} rel. err (n = {nt})", "pinn_test", "qoi_rel_err", kind="integral")]

    def instances(self) -> dict[str, list]:
        o = self.options
        return {"val": [("", {"split": "val", "instance": o["pinn_val_id"]})],
                "final": [(f"test{i}", {"split": "test", "instance": i}) for i in o["pinn_test_ids"]],
                "full_only": [(f"ood{i}", {"split": "ood", "instance": i}) for i in o["pinn_ood_ids"]]}

    def extract(self, method_id: str, results: dict[str, dict]) -> dict[str, dict[str, Any]]:
        o, out = self.options, {}
        if "" in results and "test" in results[""]:                       # trained once, evaluated on every split
            r = results[""]
            for split, sub, ids in (("test", "pinn_test", o["pinn_test_ids"]), ("ood", "pinn_ood", o["pinn_ood_ids"])):
                if split not in r:
                    continue
                n = len(r[split]["rel_l2"])
                ms = [m for m in _METRICS if m in r[split]]
                out[split] = {m: {"values": r[split][m], "ids": list(range(n))} for m in ms}
                out[sub] = {m: {"values": [r[split][m][i] for i in ids if i < n], "ids": [i for i in ids if i < n]}
                            for m in ms}
            return out
        for split, sub in (("test", "pinn_test"), ("ood", "pinn_ood")):  # per-instance: one result per instance
            vals = {m: [] for m in _METRICS}
            ids = []
            for suffix, r in sorted(results.items()):
                if suffix.startswith(split) and split in r:
                    ids.append(int(suffix[len(split):]))
                    for m in vals:
                        vals[m].append(r[split][m][0] if m in r[split] else float("nan"))
            if ids:
                out[sub] = {m: {"values": v, "ids": ids} for m, v in vals.items()}
        return out

    def search_score(self, method_id: str, result: dict) -> float:
        return result["val"]["rel_l2_mean"]

    def validity_domain(self, prepared=None):
        """Where the case holds (#58): the range of input values in the training split, measured from the prepared
        data. It is stated this way, rather than hardcoded, because for a grid-to-grid problem the data -- not a
        formula in this file -- is what defines the envelope; the OOD split is generated outside it."""
        from ..validity import ValidityDomain
        p = Path(prepared) / "data.npz" if prepared is not None else None
        if p is None or not p.exists():
            # No prepared data to measure yet: take a sample of the case's own training generator, so the envelope is
            # still measured rather than invented. The OOD split departs from these settings (see the note).
            try:
                sample = np.asarray(self.inputs("train", 64), dtype=float)
            except Exception:
                return None
            if not sample.size:
                return None
            ood = {k: v for k, v in self.options.items() if str(k).endswith("_ood")}
            return ValidityDomain(
                self.id, {"input field value": (float(sample.min()), float(sample.max()))},
                source=f"measured on a 64-field sample of the case's own training generator ({type(self).__name__}"
                       f".inputs('train'))",
                note=(f"the OOD split departs from the training settings ({ood})" if ood else "") +
                     "; prepared data replaces this with the training split's own range")
        train = np.load(p)["train_x"]
        return ValidityDomain(
            self.id, {"input field value": (float(train.min()), float(train.max()))},
            source=f"measured on the training split of {p}",
            note="the OOD split is generated by the case's generator; an error measured on it is an error outside "
                 "this envelope")

    def main_metric(self) -> tuple[str, str]:
        return "pinn_test", "rel_l2"

    def ood_metric(self) -> tuple[str, str]:
        return "pinn_ood", "rel_l2"

    def summary_metrics(self):
        return [("rel. L2", "field", ("pinn_test", "rel_l2"), ("pinn_ood", "rel_l2"), False)]

    # ------------------------------------------------------------------ PINN hooks (subclasses)
    def pinn_setup(self, x_in: np.ndarray, cfg: dict):
        """Return (model, spec, sample(m, g), evaluate(model), exact) for one instance; ``exact`` is a torch module
        with a known solution of the same PDE (or None), used to check the residual code."""
        raise NotImplementedError


class ArrayProblem(FieldOperatorProblem):
    """Wrap your own arrays as a benchmark problem.

    ``train``, ``val``, ``test`` (and optionally ``ood``) are ``(inputs, outputs)`` pairs of arrays shaped
    ``(n, *grid)``. ``data_cost_s``: what generating them cost (core-seconds), for the amortised cost. ``qoi``: an
    optional function field -> per-sample integral quantity. Pass ``pinn_setup`` to make PINNs applicable."""
    kind = "field_operator"

    def __init__(self, id: str, *, train, val, test, ood=None, data_cost_s: float | None = None, qoi=None,
                 pinn_setup=None, title: str | None = None, **options):
        self.id, self.title = id, title or id
        self._arrays = {"train": train, "val": val, "test": test, "ood": ood if ood is not None else
                        (np.zeros((0, *np.shape(test[0])[1:])), np.zeros((0, *np.shape(test[1])[1:])))}
        self._cost, self._qoi, self._pinn = data_cost_s, qoi, pinn_setup
        n = {f"n_{k}": len(v[0]) for k, v in self._arrays.items()}
        super().__init__(**{**n, "pinn_test_ids": [], "pinn_ood_ids": [], **options})

    def default_options(self):
        d = super().default_options()
        d.update({f"n_{k}": len(v[0]) for k, v in getattr(self, "_arrays", {}).items()})
        return d

    def smoke_options(self):
        return {}

    def generate(self, split, n, n_workers):
        x, y = self._arrays[split]
        return np.asarray(x), np.asarray(y), 0.0

    def prepare(self, out, n_workers=1):
        man = super().prepare(out, n_workers)
        doc = json.loads((Path(out) / "cost.json").read_text())
        doc.update(user_supplied=True, train_set_core_s=self._cost, solver_core_s_per_sample=None,
                   timing="supplied by the user" if self._cost is not None else "not supplied")
        (Path(out) / "cost.json").write_text(json.dumps(doc, indent=1))
        return man

    def verify_solver(self):
        return [Check("reference solver verification", "problem", self.id, None,
                      detail="user-supplied data: no reference solver to verify; only the data checks apply")]

    def qoi(self, field):
        return super().qoi(field) if self._qoi is None else np.asarray(self._qoi(field), float)

    def pinn_setup(self, x_in, cfg):
        if self._pinn is None:
            raise NotImplementedError("this ArrayProblem has no PINN formulation (pass pinn_setup=...)")
        return self._pinn(x_in, cfg)


# ====================================================================== methods
def _coords(shape):
    import torch
    if len(shape) == 1:
        return (torch.arange(shape[0], dtype=torch.float32) / shape[0]).reshape(-1, 1)
    g = torch.arange(shape[0], dtype=torch.float32) / shape[0]
    X, Y = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([X.ravel(), Y.ravel()], 1)


def _operator_model(method: str, cfg: dict, shape):
    import torch.nn as nn

    from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet
    from pinneapple_neural.architectures.neural_operators.fno import FNO2d, FourierNeuralOperator

    class OperatorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.method, self.shape = method, tuple(shape)
            if method == "fno":
                if len(shape) == 1:
                    self.net = FourierNeuralOperator(1, 1, width=cfg["width"], modes=cfg["modes"], layers=4,
                                                     use_grid=True)
                else:
                    self.net = FNO2d(1, 1, width=cfg["width"], modes1=cfg["modes"], modes2=cfg["modes"], layers=4,
                                     use_grid=True)
            else:
                self.net = DeepONet(int(np.prod(shape)), len(shape), 1, hidden=cfg["hidden"], modes=cfg["p"],
                                    depth=cfg["depth"])
                self.register_buffer("xy", _coords(shape))

        def forward(self, x):
            if self.method == "fno":
                return self.net(x.unsqueeze(1)).y.squeeze(1)
            return self.net(x.reshape(len(x), -1), self.xy).y.reshape(len(x), *self.shape)
    return OperatorModel()


def train_operator(problem: FieldOperatorProblem, prepared: Path, method: str, cfg: dict, seed: int, budget, *,
                   final: bool = True, **_) -> dict:
    import torch
    D = FieldData(prepared)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _operator_model(method, cfg, D.shape)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=OPERATOR_WD)
    Xtr, Ytr = D.x("train"), D.y("train")
    ym, ys = torch.from_numpy(D.ym), torch.from_numpy(D.ys)
    n = len(Xtr)
    bs = min(BATCH, n)
    state = {"perm": rng.permutation(n), "pos": 0}

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        if state["pos"] + bs > n:
            state["perm"], state["pos"] = rng.permutation(n), 0
        idx = torch.from_numpy(state["perm"][state["pos"]:state["pos"] + bs])
        state["pos"] += bs
        pred = model(Xtr[idx]) * ys + ym
        tgt = Ytr[idx]
        loss = (torch.linalg.vector_norm((pred - tgt).reshape(bs, -1), dim=1)
                / torch.linalg.vector_norm(tgt.reshape(bs, -1), dim=1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)
    model.eval()
    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in model.parameters()), "norm_source": D.norm_source}
    splits = ("val", "test", "ood") if final else ("val",)            # the search never touches test or OOD data
    with torch.no_grad():
        for split in splits:
            if len(D.raw[f"{split}_x"]) == 0:
                continue
            pred = torch.cat([D.decode(model(xb)) for xb in torch.split(D.x(split), 50)]).numpy()
            out[split] = problem.scores(pred, D.raw[f"{split}_y"])
        x1 = D.x("val")[:1]
        out["infer_batch1"] = time_per_query(lambda: model(x1), repeats=30, warmup=5)
    return out


def _overfit_check(subject, make_loss, params, steps=300, lr=1e-3, factor=0.1) -> Check:
    """A model that cannot overfit a tiny batch has a broken setup."""
    import torch
    opt = torch.optim.Adam(params, lr=lr)
    steps_max = steps
    first = None
    done = 0
    for done in range(1, steps + 1):  # noqa: B007 (used after the loop)
        loss = make_loss()
        if first is None:
            first = float(loss.detach())
        elif float(loss.detach()) < factor * first:              # reached: stop early
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    last = float(make_loss().detach())
    steps = done
    ratio = last / max(first, 1e-30)
    # below `factor`: fits. Between `factor` and 0.5: learns but slowly (a warning: the optimiser, not the wiring).
    # At or above 0.5, or not finite: cannot reduce the loss on 3 samples, a broken setup (an error: not trained).
    ok = bool(np.isfinite(ratio) and ratio < factor)
    broken = not np.isfinite(ratio) or ratio >= 0.5
    return Check("overfits a tiny batch", "architecture", subject, ok, value=round(ratio, 5),
                 threshold=f"< {factor} (error at >= 0.5)", severity="error" if broken else "warning",
                 detail=f"loss {first:.3g} -> {last:.3g} in {steps} Adam steps (at most {steps_max}, stops once below)")


class _OperatorMethod(Method):
    per_instance = False
    outputs = "fields"

    def supports(self, problem):
        return isinstance(problem, FieldOperatorProblem) or super().supports(problem)

    def space(self, problem):
        return getattr(problem, f"space_{self.id}")

    def train(self, problem, prepared, cfg, seed, budget, **job):
        hook = getattr(problem, f"train_{self.id}", None)
        if hook is not None:
            return hook(prepared, cfg, seed, budget, **job)
        return train_operator(problem, prepared, self.id, cfg, seed, budget, **job)

    def architecture_checks(self, problem, prepared, cfg):
        import torch
        if not isinstance(problem, FieldOperatorProblem):
            return super().architecture_checks(problem, prepared, cfg)
        subj = f"{problem.id}/{self.id}"
        D = FieldData(prepared)
        torch.manual_seed(0)
        model = _operator_model(self.id, cfg, D.shape)
        x = D.x("train")[:4]
        y = model(x)
        C = [Check("forward shape and dtype", "architecture", subj, tuple(y.shape) == tuple(x.shape) and
                   y.dtype == torch.float32, value=f"{tuple(x.shape)} -> {tuple(y.shape)} {y.dtype}"),
             Check("parameter count", "architecture", subj, True, value=int(sum(p.numel() for p in model.parameters())),
                   severity="warning")]
        C += self._specific_checks(model, D, cfg, subj)
        Y = (D.y("train")[:4] - torch.from_numpy(D.ym)) / torch.from_numpy(D.ys)
        C.append(_overfit_check(subj, lambda: ((model(x) - Y) ** 2).mean(), list(model.parameters()), steps=1000,
                                lr=cfg.get("lr", 1e-3)))
        return C

    def _specific_checks(self, model, D, cfg, subj):
        return []


@register_method("fno")
class FNOMethod(_OperatorMethod):
    title = "FNO"

    def _specific_checks(self, model, D, cfg, subj):
        import torch
        n = D.shape[-1]
        limit = n // 2 if len(D.shape) == 2 else n // 2 + 1
        C = [Check("retained Fourier modes <= Nyquist", "architecture", subj, cfg["modes"] <= limit,
                   value=cfg["modes"], threshold=limit)]
        if len(D.shape) == 1:                                            # resolution invariance on a 2x finer grid
            x = D.x("train")[:2]
            xf = torch.from_numpy(np.fft.irfft(np.fft.rfft(x.numpy(), axis=-1), n=2 * n, axis=-1) * 2).float()
            with torch.no_grad():
                yc, yf = model(x), model(xf)[:, ::2]
            rel = float(torch.linalg.vector_norm(yc - yf) / torch.linalg.vector_norm(yc))
            C.append(Check("output invariant to input resolution (2x grid, untrained)", "architecture", subj,
                           rel < 0.1, value=round(rel, 4), threshold=0.1, severity="warning",
                           detail="the grid channel spans [0, 1] inclusive, so node positions differ slightly"))
        return C


@register_method("deeponet")
class DeepONetMethod(_OperatorMethod):
    title = "DeepONet"

    def _specific_checks(self, model, D, cfg, subj):
        import torch
        net = model.net
        b_in = net.branch[0].in_features
        t_in = net.trunk[0].in_features
        ok = b_in == int(np.prod(D.shape)) and t_in == len(D.shape)
        x = D.x("train")[:3]
        b = net.branch(x.reshape(3, -1))
        t = net.trunk(model.xy)
        ok_out = tuple(b.shape) == (3, cfg["p"]) and tuple(t.shape) == (int(np.prod(D.shape)), cfg["p"])
        C = [Check("branch/trunk shapes", "architecture", subj, ok and ok_out,
                   value=f"branch {b_in}->{tuple(b.shape)}, trunk {t_in}->{tuple(t.shape)}")]
        tr = D.raw["train_y"].reshape(len(D.raw["train_y"]), -1)
        s = np.linalg.svd(tr - tr.mean(0), compute_uv=False)
        rank99 = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.99) + 1)
        C.append(Check("p >= rank of the training outputs (99 % energy)", "architecture", subj, cfg["p"] >= rank99,
                       value=cfg["p"], threshold=rank99, severity="warning"))
        _ = torch
        return C


@register_method("pinn")
class PINNMethod(Method):
    """Per-instance PINN through the problem's ``pinn_setup`` (or the problem's own ``train_pinn``)."""
    title = "PINN"
    per_instance = True
    outputs = "fields"

    def supports(self, problem):
        if hasattr(problem, "train_pinn"):
            return True
        if isinstance(problem, ArrayProblem):
            return problem._pinn is not None
        return isinstance(problem, FieldOperatorProblem) and \
            type(problem).pinn_setup is not FieldOperatorProblem.pinn_setup

    def space(self, problem):
        return problem.space_pinn

    def train(self, problem, prepared, cfg, seed, budget, **job):
        hook = getattr(problem, "train_pinn", None)
        if hook is not None:
            return hook(prepared, cfg, seed, budget, **job)
        return train_field_pinn(problem, prepared, cfg, seed, budget, **job)

    def architecture_checks(self, problem, prepared, cfg):
        hook = getattr(problem, "checks_pinn", None)
        if hook is not None:
            return hook(prepared, cfg)
        return pinn_checks(problem, prepared, cfg)


def train_field_pinn(problem, prepared, cfg, seed, budget, *, split, instance, **_):
    import torch

    from pinneapple_physics.pinn_solver.factory.pinn_factory import PINNFactory
    z = np.load(Path(prepared) / "data.npz")
    x_in, y_true = z[f"{split}_x"][instance].astype(np.float64), z[f"{split}_y"][instance]
    torch.manual_seed(seed)
    model, spec, sample, evaluate, _ = problem.pinn_setup(x_in, cfg)
    loss_fn = PINNFactory(spec).generate_loss_function()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(1000 + seed)

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        loss, _ = loss_fn(model, {"collocation": sample(cfg["n_colloc"], g)})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)
    pred = evaluate(model)
    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in model.parameters()),
           split: problem.scores(pred[None], y_true[None]), "split": split, "instance": instance}
    out["infer_eval_grid"] = time_per_query(lambda: evaluate(model), repeats=10, warmup=2)
    return out


def pinn_checks(problem, prepared, cfg) -> list[Check]:
    """Autograd derivatives against finite differences; the PDE residual of a known exact solution through the same
    residual code (catches a wrong PDE or a sign error); a tiny collocation set can be fitted."""
    import torch

    from pinneapple_physics.pinn_solver.factory.pinn_factory import PINNFactory
    subj = f"{problem.id}/pinn"
    z = np.load(Path(prepared) / "data.npz")
    x_in = z["val_x"][0].astype(np.float64)
    torch.manual_seed(0)
    model, spec, sample, evaluate, exact = problem.pinn_setup(x_in, cfg)
    C = []
    g = torch.Generator().manual_seed(0)
    pts = [t.detach().float() for t in sample(16, g)]
    pts_g = [p.clone().requires_grad_(True) for p in pts]
    out = model(*pts_g)[:, :1]
    grads = torch.autograd.grad(out.sum(), pts_g)
    worst, h = 0.0, 1e-3
    for k in range(len(pts)):
        up = [p.clone() for p in pts]
        dn = [p.clone() for p in pts]
        up[k] += h
        dn[k] -= h
        with torch.no_grad():
            fd = (model(*up)[:, :1] - model(*dn)[:, :1]) / (2 * h)
        worst = max(worst, float((grads[k] - fd).abs().max() / (fd.abs().max() + 1e-12)))
    C.append(Check("autograd derivatives match finite differences", "architecture", subj, worst < 1e-2,
                   value=f"{worst:.2e}", threshold=1e-2, detail="float32, central differences, h = 1e-3"))
    loss_fn = PINNFactory(spec).generate_loss_function()
    if exact is not None:
        batch = {"collocation": tuple(t.detach().clone().requires_grad_(True) for t in sample(512, g))}
        res = float(loss_fn(exact, batch)[1].get("pde", float("nan")))
        C.append(Check("PDE residual of an exact solution ~ 0", "architecture", subj, res < 1e-6, value=f"{res:.2e}",
                       threshold=1e-6, detail="mean squared residual of the same compiled PDE on a known solution"))
    else:
        C.append(Check("PDE residual of an exact solution ~ 0", "architecture", subj, None,
                       detail="the problem provides no exact solution"))
    fixed = tuple(t.detach().clone().requires_grad_(True) for t in sample(64, g))
    C.append(_overfit_check(subj, lambda: loss_fn(model, {"collocation": fixed})[0], list(model.parameters()),
                            steps=1000, lr=cfg["lr"], factor=0.1))
    with torch.no_grad():
        f = evaluate(model)
    C.append(Check("forward shape and dtype", "architecture", subj, f.shape == z["val_y"][0].shape,
                   value=f"{f.shape} vs data {z['val_y'][0].shape}"))
    C.append(Check("parameter count", "architecture", subj, True,
                   value=int(sum(p.numel() for p in model.parameters())), severity="warning"))
    return C


def finite(x) -> bool:
    try:
        return bool(math.isfinite(float(x)))
    except Exception:
        return False

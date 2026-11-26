"""The surrogate side of the trust study: dataset, POD field basis, an MLP ensemble, and the accuracy measures.

Everything here is a *single library call* on top of :mod:`pinneapple_autolab.domains.airfoil`, in the shape the other
end-to-end studies use. The scripts only call these.

The field output is reduced with a POD (an SVD of the training fields) before anything is learned. Thirty-two designs
against a 656-point lattice and a network that reproduces it exactly would say nothing about the model's ability to
generalise -- it would only measure how many weights the network has. Predicting a handful of POD coefficients from
the design variables is the version of the problem that has an answer, and the reconstruction error of the basis is
reported separately from the model error, because they are two different things.

The split is by **design**, never by point: two samples of the same geometry and incidence share a mesh, a solution
and a boundary layer, so a point-wise split leaks and the held-out error comes out flattering.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = ["INPUTS", "QOI", "load_dataset", "pod_basis", "PodSurrogate", "field_relative_l2",
           "accuracy_metrics", "calibration_metrics", "split_by_design", "lattice_blocks", "side_fields",
           "divergence", "LatticeBundle", "integrate_cp", "physics_checks", "input_sensitivity"]

#: design variables, in the order the models see them
INPUTS = ("alpha_deg", "speed", "thickness", "camber")
#: scalar quantities of interest
QOI = ("CL", "CD", "CD_pressure", "CD_friction", "CM")
#: fields of the lattice
FIELDS = ("u", "v", "cp")


def load_dataset(folder: str, purpose: str = "doe1") -> dict[str, Any]:
    """Everything the runs on disk measured: design variables, quantities of interest and the field lattice.

    ``folder`` is the campaign folder (``_runs/cfd``). Only successful runs are returned; a design that failed is
    reported in ``failed`` rather than silently dropped.
    """
    from pinneapple_autolab.study.campaign import load_records

    recs = load_records(folder, purposes=[purpose])
    ok = [r for r in recs if r.ok]
    failed = [{"id": r.id, "message": r.message} for r in recs if not r.ok]
    ids = [r.id for r in ok]
    X = np.array([[r.params[k] for k in INPUTS] for r in ok], float).reshape(len(ok), len(INPUTS))
    Y = np.array([[r.outputs[k] for k in QOI] for r in ok], float).reshape(len(ok), len(QOI))
    fields: dict[str, np.ndarray] = {}
    kept: list[int] = []
    for i, r in enumerate(ok):
        if not r.fields_path or not os.path.exists(r.fields_path):
            continue
        data = np.load(r.fields_path)
        if i == 0 or not fields:
            fields = {name: [] for name in FIELDS}       # type: ignore[assignment]
        for name in FIELDS:
            fields[name].append(np.asarray(data[name], float).ravel())
        kept.append(i)
    if kept:
        fields = {name: np.array(vals) for name, vals in fields.items()}
        ids = [ids[i] for i in kept]
        X, Y = X[kept], Y[kept]
    return {"ids": ids, "X": X, "Y": Y, "fields": fields, "failed": failed,
             "cost_s": float(sum(r.cost_s for r in ok))}


def pod_basis(field_arrays: Mapping[str, np.ndarray], n_modes: int = 8
              ) -> tuple[np.ndarray, dict[str, Any]]:
    """One POD basis over the stacked field of every design.

    The three fields are stacked into one vector per design and centred, so a single set of modes explains the
    velocity and the pressure together and the surrogate predicts one block of coefficients.
    """
    names = list(field_arrays)
    if not names:
        return np.zeros((0, 0)), {"names": names}
    stack = np.hstack([np.asarray(field_arrays[n], float) for n in names])      # (n_designs, n_stack)
    mean = stack.mean(axis=0)
    centred = stack - mean
    U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    k = int(min(n_modes, len(S)))
    basis = Vt[:k]
    total = float((S ** 2).sum()) or 1.0
    info = {"names": names, "sizes": {n: int(np.asarray(field_arrays[n]).shape[1]) for n in names},
            "n_modes": k, "singular_values": S[:k + 1].tolist(),
            "energy_captured": float((S[:k] ** 2).sum() / total),
            "reconstruction_rel_error": float(np.sqrt((S[k:] ** 2).sum() / total))}
    return np.vstack([mean, basis]), info


def _split_blocks(meta: Mapping[str, Any], vector: np.ndarray) -> dict[str, np.ndarray]:
    """Cut a stacked field vector back into the named fields."""
    out, at = {}, 0
    for name in meta["names"]:
        size = meta["sizes"][name]
        out[name] = np.asarray(vector)[..., at:at + size]
        at += size
    return out


def _plain(value: Any) -> Any:
    """A JSON-safe copy of a metadata mapping: the basis carries arrays, and a checkpoint that cannot be written is
    not a checkpoint."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class PodSurrogate:
    """An MLP ensemble on the design variables, predicting the quantities of interest and the POD coefficients."""

    def __init__(self, basis: Mapping[str, Any], *, hidden: int = 64, n_members: int = 5, epochs: int = 4000,
                 lr: float = 3e-3, seed: int = 0):
        self.basis_meta = dict(basis)
        self.hidden, self.n_members, self.epochs, self.lr, self.seed = hidden, n_members, epochs, lr, seed
        self.center = np.asarray(basis["center"], float)
        self.modes = np.asarray(basis["modes"], float)

    # ----------------------------------------------------------------- training
    def fit(self, X: np.ndarray, Y: np.ndarray, F: Mapping[str, np.ndarray]) -> PodSurrogate:
        import torch

        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-12
        stack = np.hstack([np.asarray(F[n], float) for n in self.basis_meta["names"]])
        coeffs = (stack - self.center) @ self.modes.T
        target = np.hstack([Y, coeffs])
        self.y_mean, self.y_std = target.mean(0), target.std(0) + 1e-12
        xt = torch.tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
        yt = torch.tensor((target - self.y_mean) / self.y_std, dtype=torch.float32)
        self.members = []
        for m in range(self.n_members):
            torch.manual_seed(self.seed + m)
            net = torch.nn.Sequential(
                torch.nn.Linear(xt.shape[1], self.hidden), torch.nn.SiLU(),
                torch.nn.Linear(self.hidden, self.hidden), torch.nn.SiLU(),
                torch.nn.Linear(self.hidden, yt.shape[1]))
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = torch.mean((net(xt) - yt) ** 2)
                loss.backward()
                opt.step()
            net.eval()
            self.members.append(net)
        return self

    # ----------------------------------------------------------------- prediction
    def predict_raw(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of the ensemble over (QOI + POD coefficients)."""
        import torch
        xt = torch.tensor((np.asarray(X, float) - self.x_mean) / self.x_std, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for net in self.members:
                preds.append(net(xt).numpy())
        preds = np.array(preds) * self.y_std + self.y_mean
        return preds.mean(axis=0), preds.std(axis=0)

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        """Quantities of interest, POD coefficients and the reconstructed fields, with the ensemble spread."""
        mean, spread = self.predict_raw(X)
        n_qoi = len(QOI)
        coeffs = mean[:, n_qoi:]
        fields = self.reconstruct(coeffs)
        return {"qoi": {name: mean[:, i] for i, name in enumerate(QOI)},
                "qoi_std": {name: spread[:, i] for i, name in enumerate(QOI)},
                "coeffs": coeffs, "coeffs_std": spread[:, n_qoi:], "fields": fields,
                "fields_std_rel": float(np.mean(spread[:, n_qoi:] / (np.abs(mean[:, n_qoi:]) + 1e-12)))}

    def reconstruct(self, coeffs: np.ndarray) -> dict[str, np.ndarray]:
        stack = np.asarray(coeffs) @ self.modes + self.center
        return _split_blocks(self.basis_meta, stack)

    # ----------------------------------------------------------------- persistence
    def save(self, folder: str) -> str:
        import torch
        os.makedirs(folder, exist_ok=True)
        for i, net in enumerate(self.members):
            torch.save(net.state_dict(), os.path.join(folder, f"member{i}.pt"))
        meta = {"basis": _plain(self.basis_meta), "hidden": self.hidden, "n_members": self.n_members, "epochs": self.epochs,
                "lr": self.lr, "seed": self.seed,
                "center": self.center.tolist(), "modes": self.modes.tolist(),
                "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                "y_mean": self.y_mean.tolist(), "y_std": self.y_std.tolist()}
        json.dump(meta, open(os.path.join(folder, "surrogate.json"), "w"), indent=1)
        return folder

    @classmethod
    def load(cls, folder: str) -> PodSurrogate:
        import torch
        meta = json.load(open(os.path.join(folder, "surrogate.json")))
        obj = cls({"center": meta["center"], "modes": meta["modes"], **meta["basis"]},
                  hidden=meta["hidden"], n_members=meta["n_members"], epochs=meta["epochs"],
                  lr=meta["lr"], seed=meta["seed"])
        obj.x_mean, obj.x_std = np.array(meta["x_mean"]), np.array(meta["x_std"])
        obj.y_mean, obj.y_std = np.array(meta["y_mean"]), np.array(meta["y_std"])
        obj.members = []
        for i in range(meta["n_members"]):
            net = torch.nn.Sequential(
                torch.nn.Linear(len(obj.x_mean), meta["hidden"]), torch.nn.SiLU(),
                torch.nn.Linear(meta["hidden"], meta["hidden"]), torch.nn.SiLU(),
                torch.nn.Linear(meta["hidden"], len(obj.y_mean)))
            net.load_state_dict(torch.load(os.path.join(folder, f"member{i}.pt")))
            net.eval()
            obj.members.append(net)
        return obj


# --------------------------------------------------------------------------- metrics
def field_relative_l2(predicted: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]) -> float:
    """Relative L2 over every field of the lattice, on the fields' own scale."""
    num = den = 0.0
    for name in predicted:
        a, b = np.asarray(predicted[name], float), np.asarray(reference[name], float)
        num += float(((a - b) ** 2).sum())
        den += float((b ** 2).sum())
    return float(np.sqrt(num / den)) if den else float("nan")


def accuracy_metrics(predicted: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]) -> dict[str, float]:
    """RMSE, MAE, relative L2 and the largest error, per quantity -- what "accurate" has to mean to be checked."""
    out: dict[str, float] = {}
    for name in predicted:
        p, r = np.asarray(predicted[name], float), np.asarray(reference[name], float)
        err = p - r
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"{name}_mae"] = float(np.mean(np.abs(err)))
        out[f"{name}_max_abs"] = float(np.max(np.abs(err)))
        den = float(np.linalg.norm(r))
        out[f"{name}_rel_l2"] = float(np.linalg.norm(err) / den) if den else float("nan")
    return out


def calibration_metrics(error: np.ndarray, sigma: np.ndarray, *, target: float = 0.9) -> dict[str, float]:
    """Empirical coverage of a +/- z sigma band and the expected calibration error over the band grid.

    A nominal confidence means nothing until the share of holds that actually landed inside it is measured; the
    difference between the two is the whole point of reporting it.
    """
    error, sigma = np.asarray(error, float), np.asarray(sigma, float)
    out: dict[str, float] = {}
    z_target = float(np.sqrt(2.0) * _erfinv(target))
    z = np.abs(error) / np.maximum(sigma, 1e-12)
    out["coverage"] = float(np.mean(z <= z_target))
    out["target_coverage"] = target
    out["sharpness"] = float(np.mean(2.0 * z_target * sigma))
    bands = np.linspace(0.1, 0.99, 10)
    ece = 0.0
    for b in bands:
        zb = float(np.sqrt(2.0) * _erfinv(b))
        ece += abs(float(np.mean(z <= zb)) - b) / len(bands)
    out["ece"] = float(ece)
    return out


def _erfinv(y: float) -> float:
    """Inverse error function by bisection -- keeps this module free of a scipy dependency."""
    lo, hi = -6.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _erf(mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _erf(x: float) -> float:
    import math
    return math.erf(x)


def split_by_design(ids: Sequence[str], splits: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    """Index arrays for each split, given the design ids each split holds."""
    index = {i: k for k, i in enumerate(ids)}
    return {name: np.array([index[d] for d in members if d in index], int) for name, members in splits.items()}


# --------------------------------------------------------------------------- the physics of a prediction
# A small accuracy error says the model reproduces what it was shown. Whether the prediction is a *flow* is a
# different question, and it is asked here -- on the same lattice, against the same reference -- so that both the
# report stage and the second campaign read the same numbers from the same code.


def lattice_blocks(params: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """The upper and lower blocks of a design's lattice, each ``(len(STATIONS), 1 + len(OFFSETS), 2)``."""
    from .airfoil import OFFSETS, STATIONS, sample_lattice

    pts, _ = sample_lattice(params)
    shape = (len(STATIONS), 1 + len(OFFSETS))
    per_side = shape[0] * shape[1]
    return tuple(pts[side * per_side:(side + 1) * per_side, :2].reshape(*shape, 2) for side in (0, 1))  # type: ignore[return-value]


def side_fields(fields: Mapping[str, np.ndarray], side: int) -> dict[str, np.ndarray]:
    """The upper (0) or lower (1) block of a design's fields, as ``(station, offset)`` arrays."""
    from .airfoil import OFFSETS, STATIONS

    shape = (len(STATIONS), 1 + len(OFFSETS))
    per = shape[0] * shape[1]
    return {k: np.asarray(v)[side * per:(side + 1) * per].reshape(shape) for k, v in fields.items()}


def divergence(u: np.ndarray, v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Divergence of (u, v) on a curvilinear (station, offset) block, in physical coordinates.

    The block is structured, so the derivatives along the two directions come from the chain rule on the local
    tangents; the two outer rings, where the differences are one-sided, are left at zero.
    """
    x, y = grid[:, :, 0], grid[:, :, 1]
    uu = np.asarray(u).reshape(grid.shape[:2])
    vv = np.asarray(v).reshape(grid.shape[:2])
    xi = np.gradient(x, axis=0)[1:-1, 1:-1], np.gradient(y, axis=0)[1:-1, 1:-1]     # along stations
    xj = np.gradient(x, axis=1)[1:-1, 1:-1], np.gradient(y, axis=1)[1:-1, 1:-1]     # along offsets
    det = xi[0] * xj[1] - xi[1] * xj[0]
    det = np.where(np.abs(det) < 1e-14, np.nan, det)
    ui = np.gradient(uu, axis=0)[1:-1, 1:-1]
    uj = np.gradient(uu, axis=1)[1:-1, 1:-1]
    vi = np.gradient(vv, axis=0)[1:-1, 1:-1]
    vj = np.gradient(vv, axis=1)[1:-1, 1:-1]
    du_dx = (ui * xj[1] - uj * xi[1]) / det
    dv_dy = (vj * xi[0] - vi * xj[0]) / det
    full = np.zeros_like(uu)
    full[1:-1, 1:-1] = np.nan_to_num(du_dx + dv_dy)
    return full


class LatticeBundle:
    """A design's fields in the shape the model gates of ``pinneapple_veriphysics.surrogate_gates`` read.

    The gate API differentiates along axes called ``t`` and ``x``; here they are the chordwise station and the
    wall-normal offset, and the wall -- where the case declares no-slip -- is therefore the ``x_low`` edge.
    """

    axes = ("t", "x")

    def __init__(self, fields: Mapping[str, np.ndarray], *, bundle_id: str, wall_speed: float = 0.0):
        from .airfoil import OFFSETS, STATIONS

        shape = (len(STATIONS), 1 + len(OFFSETS))
        self._fields = {k: np.asarray(v, float).reshape(shape) for k, v in fields.items()}
        self.id = bundle_id
        self.manifest = SimpleNamespace(
            id=bundle_id,
            physics=SimpleNamespace(boundary_conditions=[
                SimpleNamespace(kind="dirichlet", region="x_low", field="speed", value=wall_speed)]))

    def field_array(self, name: str) -> np.ndarray:
        return self._fields[name]

    def axis_array(self, name: str) -> np.ndarray:
        from .airfoil import OFFSETS, STATIONS

        return STATIONS if name == "t" else np.r_[0.0, OFFSETS]


def integrate_cp(cp_wall: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> float:
    """Lift from the wall pressure, in the direction normal to the freestream, per unit span and chord."""
    from .airfoil import STATIONS

    n_st = len(STATIONS)
    cl = 0.0
    for cp, grid in ((cp_wall[:n_st], upper), (cp_wall[n_st:], lower)):
        cl += -float(np.trapezoid(cp, grid[:n_st, 0, 0]))
    return cl


def physics_checks(predicted: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray],
                   params: Mapping[str, float], *, speed: float, alpha_deg: float, cl_predicted: float,
                   cl_reference: float | None = None, chord: float = 1.0) -> dict[str, dict]:
    """The three model gates plus the two checks the gates do not cover, each with a number and a yardstick.

    ``accuracy`` (relative L2 of the pressure coefficient) and ``boundary`` (the Dirichlet condition the case
    declares, at the wall) and ``residual`` (the continuity residual, normalised by the size of its own gradients)
    come from :func:`pinneapple_veriphysics.surrogate_gates.run_surrogate_gates`; ``freestream`` and ``cp_integral``
    are this module's own, because the gates do not cover them.
    """
    from pinneapple_veriphysics.surrogate_gates import run_surrogate_gates

    upper, lower = lattice_blocks(params)
    model_id = params.get("id", "design")
    with_speed = {**predicted, "speed": np.hypot(np.asarray(predicted["u"]), np.asarray(predicted["v"]))}
    out: dict[str, dict] = {}

    def continuity_operator(v_field, grid):
        """du/dx + dv/dy, times the chord over the freestream speed: dimensionless, on the (station, offset) block."""

        def residual(values, t, x):
            return (chord / speed) * divergence(np.asarray(values), v_field, grid)

        return residual

    reference_with_speed = {**reference, "speed": np.hypot(np.asarray(reference["u"]), np.asarray(reference["v"]))}
    gates: dict[str, list] = {}
    gates_reference: dict[str, list] = {}
    for name, field in (("accuracy", "cp"), ("boundary", "speed"), ("residual", "u")):
        predicted_results, reference_results = [], []
        for side, grid in ((0, upper), (1, lower)):
            predicted_bundle = LatticeBundle(side_fields(with_speed, side), bundle_id=f"{model_id}-{side}")
            reference_bundle = LatticeBundle(side_fields(reference_with_speed, side),
                                             bundle_id=f"{model_id}-{side}-ref")
            extra = {}
            if field == "u":
                extra = {"residual": continuity_operator(side_fields(predicted, side)["v"], grid)}
            # The tolerances are the study's, not the gate's defaults: 5 % for the wall condition, because the
            # condition declares zero velocity and the reference itself sits near 2 % of the field scale at the
            # height the sample is taken from -- a wall-function solution has no sample at the wall.
            report = run_surrogate_gates(predicted_bundle, reference=reference_bundle, field=field,
                                         model="naca-family-surrogate", gates=(name,), tolerance=0.05,
                                         boundary_tolerance=0.05, residual_tolerance=0.5, **extra)
            predicted_results.append(report.results[0])
            if name != "accuracy":            # the reference against itself is zero by construction
                ref_extra = {}
                if field == "u":
                    ref_extra = {"residual": continuity_operator(side_fields(reference, side)["v"], grid)}
                ref_report = run_surrogate_gates(reference_bundle, reference=reference_bundle, field=field,
                                                 model="reference", gates=(name,), tolerance=0.05,
                                                 boundary_tolerance=0.05, residual_tolerance=0.5, **ref_extra)
                reference_results.append(ref_report.results[0])
        gates[name] = predicted_results
        gates_reference[name] = reference_results

    def _summary(name, how, reducer):
        values = [r.value for r in gates[name] if r.value is not None]
        refs = [r.value for r in gates_reference[name] if r.value is not None]
        return {"surrogate": float(reducer(values)) if values else float("nan"),
                "reference": float(reducer(refs)) if refs else float("nan"), "how": how}

    out["gate_accuracy"] = _summary("accuracy", "relative L2 of the predicted Cp against the reference, per side, "
                                                "averaged", np.mean)
    out["gate_boundary"] = _summary("boundary", "deviation from the declared no-slip value on the wall edge, "
                                                "relative to the field's own scale, worst side", np.max)
    out["gate_residual"] = _summary("residual", "continuity residual, normalised by the size of its own gradients, "
                                                "averaged over the two sides", np.mean)

    wall, wall_ref = [], []
    for side in range(2):
        p_side, r_side = side_fields(with_speed, side), side_fields(reference, side)
        wall.append(p_side["speed"][:, 0] / speed)
        wall_ref.append(np.hypot(r_side["u"][:, 0], r_side["v"][:, 0]) / speed)
    out["wall_condition"] = {"surrogate": float(np.mean(np.concatenate(wall))),
                             "reference": float(np.mean(np.concatenate(wall_ref))),
                             "how": "mean |U| / U_inf on the wall samples; the case declares noSlip"}

    div_s, div_r = [], []
    for side, grid in ((0, upper), (1, lower)):
        p_side, r_side = side_fields(predicted, side), side_fields(reference, side)
        div_s.append(divergence(p_side["u"], p_side["v"], grid)[1:-1, 1:-1])
        div_r.append(divergence(r_side["u"], r_side["v"], grid)[1:-1, 1:-1])
    out["continuity"] = {
        "surrogate": float(np.mean(np.abs(np.concatenate([d.ravel() for d in div_s]))) * chord / speed),
        "reference": float(np.mean(np.abs(np.concatenate([d.ravel() for d in div_r]))) * chord / speed),
        "how": "mean |du/dx + dv/dy| times the chord over the freestream speed, on the lattice"}

    far, far_ref = [], []
    for side in range(2):
        p_side, r_side = side_fields(predicted, side), side_fields(reference, side)
        far.append(np.hypot(p_side["u"][:, -1] - speed * np.cos(np.radians(alpha_deg)),
                            p_side["v"][:, -1] - speed * np.sin(np.radians(alpha_deg))) / speed)
        far_ref.append(np.hypot(r_side["u"][:, -1] - speed * np.cos(np.radians(alpha_deg)),
                                r_side["v"][:, -1] - speed * np.sin(np.radians(alpha_deg))) / speed)
    out["freestream"] = {"surrogate": float(np.mean(np.concatenate(far))),
                         "reference": float(np.mean(np.concatenate(far_ref))),
                         "how": "mean |U - U_inf| / U_inf at the outermost offset"}

    cp_wall = np.concatenate([side_fields(predicted, s)["cp"][:, 0] for s in range(2)])
    cl_integrated = integrate_cp(cp_wall, upper, lower)
    cl_reference_integrated = integrate_cp(
        np.concatenate([side_fields(reference, s)["cp"][:, 0] for s in range(2)]), upper, lower)
    out["cp_integral"] = {
        "surrogate_integrated": float(cl_integrated), "surrogate_predicted": float(cl_predicted),
        "reference_integrated": float(cl_reference_integrated),
        "gap": float(abs(cl_integrated - cl_predicted)),
        # the same disagreement in the reference: integrating 41 stations of wall pressure is a crude estimator, and
        # the solver's own answer shows how crude. Without this the gate would ask the surrogate to be better than
        # the estimator it is being judged with.
        "reference_gap": (float(abs(cl_reference_integrated - cl_reference))
                          if cl_reference is not None else float("nan")),
        "how": "CL from integrating the wall Cp over the section against the CL the model predicts; the reference's "
               "own gap beside it"}
    out["gate_detail"] = {name: [{"gate": r.check_id, "passed": r.passed, "value": r.value,
                                  "tolerance": r.tolerance, "detail": r.detail} for r in results]
                          for name, results in gates.items()}
    return out


def input_sensitivity(model: PodSurrogate, X: np.ndarray, bounds: Mapping[str, tuple[float, float]],
                      *, fraction: float = 0.01) -> dict[str, dict]:
    """How far the prediction moves when one input is nudged, against how far the ensemble disagrees already.

    A model whose output swings more under a one-percent nudge of an input than its own members disagree about it
    has not learned that input -- it has memorised points between them. Camber is excluded: a dimension the model
    does not have cannot be tested by nudging it.
    """
    base = model.predict(X)
    spread = float(np.mean(np.asarray(base["qoi_std"]["CL"], float) / (np.abs(base["qoi"]["CL"]) + 1e-9)))
    rows: dict[str, dict] = {}
    for j, name in enumerate(INPUTS[:3]):
        low, high = bounds[name]
        step = fraction * (high - low)
        moved = []
        for sign in (1.0, -1.0):
            nudged = np.array(X, float)
            nudged[:, j] = np.clip(nudged[:, j] + sign * step, low, high)
            prediction = model.predict(nudged)
            moved.append(np.abs(prediction["qoi"]["CL"] - base["qoi"]["CL"]) / (np.abs(base["qoi"]["CL"]) + 1e-9))
        drift = float(np.mean(np.concatenate(moved)))
        rows[name] = {"step": step, "relative_change_CL": drift, "ensemble_spread_CL": spread,
                      "ratio": drift / spread if spread else float("inf")}
    return rows

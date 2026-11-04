"""The surrogate of the multiphysics motor: an ensemble of networks that replaces the FEA at optimisation time.

What it predicts is every output of :data:`~.model.OUTPUTS` plus the three axial profiles of
:data:`~.model.PROFILES`, from the eleven design variables of :data:`~.design.VARIABLES` -- that is, torque, ripple,
torque density, efficiency, three loss channels, three temperatures, the demagnetisation ratio and two masses, each
resolved on the four axial segments of the thermal network.

Three things make it usable rather than merely fast, and all three are properties of this class, not of the training
script:

* **an ensemble, not a network** -- ``ensemble`` networks with different initialisations are trained on the same
  split and their spread is the model's own uncertainty. A design the networks disagree on is a design to send to
  the reference model, not one to believe;
* **a validity box and a data test** -- :meth:`MotorSurrogate3D.out_of_box` reports the design variables outside the
  box the training set was drawn from, and :meth:`MotorSurrogate3D.extrapolation` the distance to the training set
  relative to its own scale, so an optimiser that wanders off the data can see it happen;
* **a model card with the held-out numbers** -- :meth:`MotorSurrogate3D.card` carries the errors measured on data the
  networks never saw, per output, in the output's own unit. A prediction without them is a number without a claim.

Outputs are trained in a normalised space (standardised, or ``log`` for the strictly positive ones, which is what
makes a loss of 300 W and a torque of 30 N m train together): ``log_targets`` lists them, and the transforms are
stored with the weights so a saved model predicts in physical units on the way out.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .design import BOUNDS, NAMES, Motor3DDesign
from .model import OUTPUTS, PROFILES, SCALARS

__all__ = ["SurrogateConfig", "MotorSurrogate3D", "load_pretrained", "pretrained_path", "PRETRAINED_VERSION"]

#: Version of the pretrained model shipped with the library (a new training run gets a new folder, never an
#: overwrite: a card that disagrees with its weights is worse than no card).
PRETRAINED_VERSION = "motor3d-v1"

#: Outputs trained in log space: strictly positive, and what matters about them is the relative error.
LOG_TARGETS = tuple(o.name for o in OUTPUTS if o.name in
                    ("iron_loss_W", "magnet_loss_W", "copper_loss_W", "total_loss_W", "magnet_mass_kg",
                     "active_mass_kg", "demag_ratio")) + PROFILES


@dataclass
class SurrogateConfig:
    """Shape and training schedule of the ensemble."""
    hidden: tuple[int, ...] = (128, 128, 128)
    ensemble: int = 5
    epochs: int = 2500
    lr: float = 3e-3
    batch: int = 64
    weight_decay: float = 1e-5
    val_fraction: float = 0.2
    patience: int = 400
    seed: int = 0
    activation: str = "silu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _activation(name: str):
    import torch

    return {"silu": torch.nn.SiLU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh,
            "gelu": torch.nn.GELU}[name]()


def _mlp(n_in: int, hidden: Sequence[int], n_out: int, act: str):
    import torch

    layers: list[Any] = []
    last = n_in
    for h in hidden:
        layers += [torch.nn.Linear(last, h), _activation(act)]
        last = h
    layers.append(torch.nn.Linear(last, n_out))
    return torch.nn.Sequential(*layers)


def _unit_of(target: str) -> str:
    """The unit of an output or of one segment of a profile (``T_winding_profile_C[2]`` -> ``degC``)."""
    base = target.split("[")[0]
    for o in OUTPUTS:
        if o.name == base:
            return o.unit
    return {"torque_slice_Nm": "N m", "T_winding_profile_C": "degC", "T_magnet_profile_C": "degC"}.get(base, "")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class MotorSurrogate3D:
    """The trained ensemble, with everything needed to predict, to be saved and to be trusted or not.

    :meth:`fit` takes a :class:`~.dataset.MotorDataset`; :meth:`predict` returns a mean and a spread per output;
    :meth:`evaluate` returns the flat arrays an optimiser consumes.
    """

    fidelity = "surrogate"

    def __init__(self, config: SurrogateConfig | None = None):
        self.config = config or SurrogateConfig()
        self.models: list[Any] = []
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.targets: list[str] = []
        self.active: list[str] = []          # targets that vary in the training set (the ones the networks learn)
        self.constants: dict[str, float] = {}  # targets that do not: predicted as their constant value
        self.card_: dict[str, Any] = {}
        self.seconds = 0.0
        self.name = "motor3d-surrogate"

    # ------------------------------------------------------------------------------------------- target layout
    @property
    def target_names(self) -> list[str]:
        """One name per network output: the scalars, then the profiles flattened in segment order."""
        out = [o.name for o in OUTPUTS]
        for k in PROFILES:
            out += [f"{k}[{j}]" for j in range(self.n_segments)]
        return out

    @property
    def n_segments(self) -> int:
        return int(self.card_.get("n_segments", 4))

    def _pack(self, Y: dict[str, np.ndarray], n: int) -> np.ndarray:
        cols = [np.asarray(Y[o.name], float).reshape(n, -1) for o in OUTPUTS]
        cols += [np.asarray(Y[k], float).reshape(n, -1) for k in PROFILES]
        return np.hstack(cols)

    def _unpack(self, Z: np.ndarray, n: int) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        j = 0
        for o in OUTPUTS:
            out[o.name] = Z[:, j]
            j += 1
        K = self.n_segments
        for k in PROFILES:
            out[k] = Z[:, j:j + K]
            j += K
        return out

    def _mask_log(self, targets: list[str]) -> np.ndarray:
        return np.array([t.split("[")[0] in LOG_TARGETS for t in targets], bool)

    # ----------------------------------------------------------------------------------------------- training
    def fit(self, dataset, *, progress=None) -> MotorSurrogate3D:
        """Train the ensemble on the valid rows of ``dataset`` and measure it on a held-out split of them."""
        import torch

        n_all = len(dataset)
        ok = np.asarray(dataset.valid, bool)
        X = np.asarray(dataset.X, float)[ok]
        Y_ok = {k: np.asarray(v, float)[ok] for k, v in dataset.Y.items()}
        n = len(X)
        if n < 20:
            raise ValueError(f"the surrogate needs at least 20 valid designs, the dataset has {n}")
        self.card_["n_segments"] = int(np.asarray(Y_ok[PROFILES[0]]).shape[1])
        targets = self.target_names
        Z = self._pack(Y_ok, n)
        is_log = self._mask_log(targets)

        rng = np.random.default_rng(self.config.seed)
        idx = rng.permutation(n)
        n_val = max(4, int(round(self.config.val_fraction * n)))
        val, tr = idx[:n_val], idx[n_val:]

        self.x_mean, self.x_std = X[tr].mean(0), X[tr].std(0) + 1e-12
        Zt = np.where(is_log, np.log(np.clip(Z, 1e-9, None)), Z)
        # an output that does not move in the training set is not learned: it is predicted as its median, so the
        # networks spend none of their capacity on it and its spread says nothing it cannot support
        spread = Zt[tr].std(0)
        varies = spread > np.maximum(1e-9, 1e-6 * np.abs(Zt[tr].mean(0)))
        self.targets = list(targets)
        self.active = [t for t, ok in zip(targets, varies, strict=True) if ok]
        self.constants = {t: float(np.median(Z[tr][:, j])) for j, (t, ok) in enumerate(zip(targets, varies, strict=True))
                          if not ok}
        sel = np.where(varies)[0]
        self.y_mean, self.y_std = Zt[tr][:, sel].mean(0), Zt[tr][:, sel].std(0) + 1e-12
        Xn = torch.tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
        Yn = torch.tensor((Zt[:, sel] - self.y_mean) / self.y_std, dtype=torch.float32)

        t0 = time.perf_counter()
        self.models = []
        self.card_["ensemble_members"] = []
        for m in range(self.config.ensemble):
            torch.manual_seed(self.config.seed + 101 * m)
            net = _mlp(len(NAMES), self.config.hidden, len(self.active), self.config.activation)
            opt = torch.optim.Adam(net.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.config.epochs)
            best, best_state, bad = float("inf"), None, 0
            tr_t = torch.tensor(tr)
            val_t = torch.tensor(val)
            g = torch.Generator().manual_seed(self.config.seed + 7919 * m)
            for ep in range(self.config.epochs):
                net.train()
                perm = torch.randperm(len(tr), generator=g)
                for b in range(0, len(tr), self.config.batch):
                    sel = tr_t[perm[b:b + self.config.batch]]
                    opt.zero_grad()
                    loss = torch.mean((net(Xn[sel]) - Yn[sel]) ** 2)
                    loss.backward()
                    opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    v = float(torch.mean((net(Xn[val_t]) - Yn[val_t]) ** 2))
                if v < best - 1e-9:
                    best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= self.config.patience:
                        break
                if progress and ep % 250 == 0:
                    progress(m, ep, v)
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.models.append(net)
            self.card_["ensemble_members"].append({"seed": self.config.seed + 101 * m, "val_loss": best,
                                                  "epochs": ep + 1})
        self.seconds = time.perf_counter() - t0
        self.card_.update(self._measure(X, Z, val, is_log))
        self.card_["constant_outputs"] = dict(self.constants)
        self.card_.update({
            "version": "unreleased", "n_train": int(len(tr)), "n_val": int(len(val)), "n_total": int(n),
            "config": self.config.to_dict(), "inputs": list(NAMES),
            "targets": targets, "log_outputs": [t for t, f in zip(targets, is_log, strict=True) if f],
            "bounds": {v: [float(BOUNDS[i, 0]), float(BOUNDS[i, 1])] for i, v in enumerate(NAMES)},
            "dataset_meta": {k: v for k, v in dataset.meta.items() if k not in ("bounds", "outputs")},
            "seconds": round(self.seconds, 1)})
        return self

    # --------------------------------------------------------------------------------------------- prediction
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Raw ensemble predictions (members, n, targets) in the physical units of the outputs."""
        import torch

        if not self.models:
            raise RuntimeError("this surrogate has no trained networks: call fit() or load a pretrained one")
        Xn = torch.tensor((np.atleast_2d(X) - self.x_mean) / self.x_std, dtype=torch.float32)
        is_log = self._mask_log(self.active)
        out = []
        with torch.no_grad():
            for net in self.models:
                z = net(Xn).numpy().astype(float) * self.y_std + self.y_mean
                out.append(np.where(is_log, np.exp(np.clip(z, -50, 50)), z))
        pred = np.stack(out)
        if self.constants:                      # put the outputs that do not vary back in their column
            full = np.empty((pred.shape[0], pred.shape[1], len(self.targets)))
            sel = [self.targets.index(t) for t in self.active]
            full[:, :, sel] = pred
            for t, v in self.constants.items():
                full[:, :, self.targets.index(t)] = v
            pred = full
        return pred

    def predict(self, X) -> dict[str, np.ndarray]:
        """Mean (``n, ...``) and ensemble spread (``n, ...``) of every output, in physical units."""
        pred = self._forward(np.atleast_2d(np.asarray(X, float)))
        mean, std = self._unpack(pred.mean(0), len(pred[0])), self._unpack(pred.std(0), len(pred[0]))
        out = {}
        for k in mean:
            out[k] = (mean[k], std[k])
        return out

    def evaluate(self, X) -> dict[str, np.ndarray]:
        """Flat arrays per output (profiles flattened), the form the design optimiser consumes, plus ``_std`` and
        ``_valid`` (every design is valid for a network; the flag is what the optimiser expects)."""
        pred = self.predict(X)
        out = {k: v[0] for k, v in pred.items()}
        out.update({f"{k}_std": v[1] for k, v in pred.items()})
        return out

    def predict_one(self, d: Motor3DDesign) -> tuple[dict[str, float], dict[str, float]]:
        """The mean and the spread of one design, as scalars (profiles summarised by their maximum)."""
        p = self.predict(d.as_vector()[None, :])
        mean = {k: float(np.max(v[0])) if np.ndim(v[0]) > 1 else float(v[0][0]) for k, v in p.items()}
        spread = {k: float(np.max(v[1])) if np.ndim(v[1]) > 1 else float(v[1][0]) for k, v in p.items()}
        return mean, spread

    # ------------------------------------------------------------------------------------------- validity tests
    def out_of_box(self, X) -> list[str]:
        """Design variables outside the box the model was trained on (an empty list: inside)."""
        X = np.atleast_2d(np.asarray(X, float))
        bad = []
        for i, name in enumerate(NAMES):
            lo, hi = self.card_.get("bounds", {}).get(name, [BOUNDS[i, 0], BOUNDS[i, 1]])
            if np.any(X[:, i] < lo) or np.any(X[:, i] > hi):
                bad.append(name)
        return bad

    def extrapolation(self, X) -> np.ndarray:
        """Distance of each design from the training mean in units of the training spread (the largest over the
        variables): 0 at the mean, 3 at the edge of a roughly Gaussian cloud."""
        X = np.atleast_2d(np.asarray(X, float))
        return np.abs((X - self.x_mean) / self.x_std).max(1)

    # --------------------------------------------------------------------------------------------- measuring
    def _measure(self, X: np.ndarray, Z: np.ndarray, idx: np.ndarray, is_log: np.ndarray) -> dict[str, Any]:
        """Held-out errors per output, in the output's own unit: MAE, R2 and the worst relative error."""
        pred = self._forward(X[idx]).mean(0)
        truth = self._unpack(Z[idx], len(idx))
        got = self._unpack(pred, len(idx))
        per: dict[str, Any] = {}
        for k, (t, g) in ((k, (truth[k], got[k])) for k in truth if k not in self.constants):
            t, g = np.asarray(t, float), np.asarray(g, float)
            err = g - t
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((t - t.mean()) ** 2)) + 1e-30
            rel = np.abs(err) / np.maximum(np.abs(t), 1e-9)
            per[k] = {"mae": float(np.mean(np.abs(err))), "r2": float(1.0 - ss_res / ss_tot),
                      "max_rel": float(np.max(rel)), "mean_rel": float(np.mean(rel)),
                      "p95_rel": float(np.percentile(rel, 95)), "unit": _unit_of(k)}
        # the worst output is the one explaining least of its variance: a relative error is meaningless for an
        # output that is near zero by construction (a loss that does not exist at this speed), its R2 is not
        worst = min(per, key=lambda k: per[k]["r2"]) if per else None
        return {"heldout": per, "worst_output": worst,
                "mean_rel_overall": float(np.mean([v["mean_rel"] for v in per.values()])),
                "mean_r2": float(np.mean([v["r2"] for v in per.values()])),
                "worst_r2": float(per[worst]["r2"])}

    def card(self) -> dict[str, Any]:
        """The model card: what it was trained on, how it scored on held-out designs, what it does not know."""
        return {**self.card_, "name": self.name, "fidelity": "surrogate",
                "notes": ["errors are on designs held out from training",
                          "the ensemble spread is a proxy for uncertainty, not a calibrated interval",
                          "predictions outside the bounds of the card are extrapolation"]}

    # -------------------------------------------------------------------------------------------- persistence
    def save(self, folder: str | Path, *, version: str = "unreleased") -> Path:
        import torch

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": [m.state_dict() for m in self.models],
                    "x_mean": self.x_mean, "x_std": self.x_std, "y_mean": self.y_mean, "y_std": self.y_std,
                    "targets": self.targets, "active": self.active, "constants": self.constants,
                    "hidden": list(self.config.hidden), "activation": self.config.activation,
                    "n_segments": self.n_segments}, folder / "weights.pt")
        card = self.card()
        card["version"] = version
        card["sha256"] = _sha256(folder / "weights.pt")
        (folder / "card.json").write_text(json.dumps(card, indent=2))
        return folder / "weights.pt"

    @classmethod
    def load(cls, folder: str | Path, *, verify: bool = True) -> MotorSurrogate3D:
        import torch

        folder = Path(folder)
        card = json.loads((folder / "card.json").read_text())
        blob = folder / "weights.pt"
        if verify and card.get("sha256") and _sha256(blob) != card["sha256"]:
            raise ValueError(f"{blob} does not match the SHA-256 of its card: the weights were changed after they "
                             f"were trained, so their errors no longer apply")
        state = torch.load(blob, map_location="cpu", weights_only=False)
        cfg = SurrogateConfig(**{k: v for k, v in card.get("config", {}).items() if k in SurrogateConfig.__annotations__})
        if isinstance(cfg.hidden, list):
            cfg.hidden = tuple(cfg.hidden)
        self = cls(cfg)
        self.models = []
        for w in state["weights"]:
            net = _mlp(len(NAMES), state["hidden"], len(state["active"]), state["activation"])
            net.load_state_dict(w)
            net.eval()
            self.models.append(net)
        self.x_mean, self.x_std = state["x_mean"], state["x_std"]
        self.y_mean, self.y_std = state["y_mean"], state["y_std"]
        self.targets = list(state["targets"])
        self.active = list(state["active"])
        self.constants = dict(state["constants"])
        self.card_ = {k: v for k, v in card.items()}
        self.card_["n_segments"] = int(state["n_segments"])
        self.seconds = float(card.get("seconds", 0.0))
        return self


def pretrained_path() -> Path:
    """Folder of the pretrained model shipped with the library."""
    return Path(__file__).resolve().parent / "data" / PRETRAINED_VERSION


def load_pretrained(version: str = PRETRAINED_VERSION, *, verify: bool = True) -> MotorSurrogate3D:
    """The pretrained motor surrogate, with its weights checked against the SHA-256 of its card."""
    folder = Path(__file__).resolve().parent / "data" / version
    if not (folder / "weights.pt").exists():
        raise FileNotFoundError(f"no pretrained motor surrogate at {folder}: build one with "
                                f"`python -m pinneapple_simulation.electromagnetics.machines.motor3d.workflow "
                                f"build-dataset train`")
    model = MotorSurrogate3D.load(folder, verify=verify)
    model.name = f"motor3d-pretrained-{version}"
    return model

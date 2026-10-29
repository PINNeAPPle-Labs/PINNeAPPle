"""Verification case for :mod:`.physics_learning`: a cross-section of a die, TIM and lid heated at the junction.

Steady conduction in ``(x, z)`` per unit depth (W/m, temperature rise above ambient):

* die (silicon, 100 um, ``k_die``), TIM (30 um, ``k_tim``), lid (copper, 500 um, ``k_lid``), ``L = 2 mm`` wide;
* a Gaussian line source of ``P`` W/m at the bottom of the die, centred at ``x_s`` (fraction of ``L``), width
  ``sigma`` (fraction of ``L``);
* Robin cooling ``h`` on the top of the lid, adiabatic sides and bottom;
* an optional contact resistance ``R_c`` (m^2 K / W) on the die/TIM interface.

The reference is the cell-centred finite-volume scheme of :mod:`pinneapple_condense.fv` on a fine grid (milliseconds
per solve), so every expected response of the verifier comes from solving the physics, not from a rule.

``theta`` is a dict over :data:`PARAMS`; :data:`TRAIN_BOX` is the domain the models under test were built for.
:func:`defective_models` builds the deliberately defective models of the report (low loss and low mean error, wrong
in one physical respect each) and :func:`good_models` the controls.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

PARAMS = ("P", "x_s", "k_tim", "h", "R_c", "k_die", "k_lid", "sigma")
DEFAULT = {"P": 100.0, "x_s": 0.5, "k_tim": 4.0, "h": 1.0e4, "R_c": 0.0, "k_die": 148.0, "k_lid": 401.0,
           "sigma": 0.05}
TRAIN_BOX = {"P": (50.0, 150.0), "x_s": (0.3, 0.7), "k_tim": (2.0, 8.0), "h": (5.0e3, 2.0e4)}
LAYERS = (("die", 100e-6), ("tim", 30e-6), ("lid", 500e-6))
WIDTH = 2e-3


@dataclass
class Section:
    """Grid and reference solver of the cross-section."""
    nx: int = 200
    nz: tuple[int, int, int] = (10, 6, 20)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def dx(self) -> float:
        return WIDTH / self.nx

    def z_edges(self) -> np.ndarray:
        e = [0.0]
        for (_, t), n in zip(LAYERS, self.nz, strict=True):
            e += list(e[-1] + t * (np.arange(1, n + 1) / n))
        return np.array(e)

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        ze = self.z_edges()
        return (np.arange(self.nx) + 0.5) * self.dx, 0.5 * (ze[1:] + ze[:-1])

    @property
    def interfaces(self) -> np.ndarray:
        """z of the die/TIM and TIM/lid interfaces (m)."""
        return np.cumsum([t for _, t in LAYERS])[:-1]

    @property
    def height(self) -> float:
        return float(sum(t for _, t in LAYERS))

    def layer_of(self, z: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.cumsum([t for _, t in LAYERS]), z, side="right").clip(0, len(LAYERS) - 1)

    def conductivity(self, z: np.ndarray, theta: dict) -> np.ndarray:
        k = np.array([theta["k_die"], theta["k_tim"], theta["k_lid"]])
        return k[self.layer_of(np.asarray(z))]

    def source(self, x: np.ndarray, theta: dict) -> np.ndarray:
        """Line-source density (W/m^2 into the bottom face, per unit depth)."""
        s = theta["sigma"] * WIDTH
        xs = theta["x_s"] * WIDTH
        g = np.exp(-0.5 * ((x - xs) / s) ** 2)
        return theta["P"] * g / (g.sum() * self.dx)

    def solve(self, theta: dict) -> np.ndarray:
        """Reference temperature rise on the cell centres, shape ``(nx, nz_total)``."""
        from pinneapple_condense import fv
        th = {**DEFAULT, **theta}
        key = tuple(sorted(th.items()))
        if key in self._cache:
            return self._cache[key]
        x, zc = self.centres()
        nzt = len(zc)
        k = np.broadcast_to(self.conductivity(zc, th)[None, None, :], (self.nx, 1, nzt)).copy()
        dz = np.diff(self.z_edges())
        R = np.zeros(nzt - 1)
        R[self.nz[0] - 1] = th["R_c"]
        sysm = fv.assemble(k, dz, R, self.dx, th["h"], 0.0, dy=1.0)
        f = np.zeros((self.nx, 1, nzt))
        f[:, 0, 0] = self.source(x, th) * self.dx           # W per cell (per metre of depth)
        T, _ = fv.solve(sysm.K, f.ravel(), method="direct")
        T = T.reshape(self.nx, nzt)
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = T
        return T

    def evaluate(self, theta: dict, X: np.ndarray) -> np.ndarray:
        """Reference at arbitrary points ``X`` (N, 2) in metres: a cubic spline through the cell centres of each
        layer, so the field is smooth inside a layer and keeps its kink at the interfaces (derivatives of the
        reference are meaningful for the checks)."""
        from scipy.interpolate import RectBivariateSpline
        th = {**DEFAULT, **theta}
        key = ("spl",) + tuple(sorted(th.items()))
        spl = self._cache.get(key)
        if spl is None:
            T = self.solve(th)
            x, zc = self.centres()
            spl, k0 = [], 0
            for n in self.nz:
                spl.append(RectBivariateSpline(x, zc[k0:k0 + n], T[:, k0:k0 + n], kx=3, ky=min(3, n - 1)))
                k0 += n
            self._cache[key] = spl
        X = np.asarray(X, float)
        lay = self.layer_of(X[:, 1])
        out = np.empty(len(X))
        for i, s in enumerate(spl):
            sel = lay == i
            if sel.any():
                out[sel] = s.ev(np.clip(X[sel, 0], 0.0, WIDTH), X[sel, 1])
        return out

    def tmax(self, theta: dict) -> float:
        return float(self.solve(theta).max())

    def outflow(self, theta: dict) -> float:
        """Reference heat leaving through the top (W/m)."""
        th = {**DEFAULT, **theta}
        T = self.solve(th)
        dzt = np.diff(self.z_edges())[-1]
        g = self.dx / (dzt / (2 * th["k_lid"]) + 1.0 / th["h"])
        return float((g * T[:, -1]).sum())


# ------------------------------------------------------------------------------------------- models under test
@dataclass
class Model:
    """A model under test: ``fn(X, theta) -> T`` with ``X`` (N, 2) metres; ``inputs``: the parameters it takes
    (others are fixed at :data:`DEFAULT` inside it); ``reinit``: a copy with untrained weights (for networks), used
    to tell properties built into the architecture from learned ones."""
    name: str
    fn: Callable[[np.ndarray, dict], np.ndarray]
    inputs: tuple[str, ...] = ("P", "x_s", "k_tim", "h")
    defect: str = ""
    reinit: Callable[[], Model] | None = None

    def __call__(self, X, theta):
        th = {**DEFAULT, **{k: v for k, v in theta.items() if k in self.inputs}}
        return np.asarray(self.fn(np.asarray(X, float), th), float)


def _smooth_z(sec: Section, X, theta, width: float):
    """Reference averaged over a z window of ``width`` (m): the kink of each interface is rounded off."""
    offs = np.linspace(-width, width, 7)
    vals = [sec.evaluate(theta, np.column_stack([X[:, 0], np.clip(X[:, 1] + o, 0, sec.height)])) for o in offs]
    return np.mean(vals, axis=0)


def good_models(sec: Section, amplitudes=(0.003,), modes=(1,)) -> list[Model]:
    """Controls: the reference itself, and the reference with smooth errors ``a cos(2 pi j x / L)`` that respect the
    boundary conditions and the mirror symmetry (good but imperfect models)."""
    out = [Model("reference", lambda X, th: sec.evaluate(th, X))]
    for a in amplitudes:
        for j in modes:
            out.append(Model(f"reference with a smooth {100 * a:g} % error (mode {j})",
                             lambda X, th, a=a, j=j: sec.evaluate(th, X) * (1 + a * np.cos(2 * np.pi * j * X[:, 0]
                                                                                             / WIDTH))))
    return out


def defective_models(sec: Section, severity: float = 1.0) -> list[Model]:
    """Each is accurate on the training distribution by the usual metrics and wrong in one physical respect.
    ``severity`` scales each defect (1: the size described in the report; the out-of-box and spurious-correlation
    models have no size)."""
    sv = float(severity)
    lo = {k: v[0] for k, v in TRAIN_BOX.items()}
    hi = {k: v[1] for k, v in TRAIN_BOX.items()}
    k0 = DEFAULT["k_tim"]

    def interface(X, th):
        return _smooth_z(sec, X, th, 25e-6 * sv)

    def ood(X, th):                                   # exact in the box, frozen at its edge outside
        c = {k: float(np.clip(v, lo.get(k, -np.inf), hi.get(k, np.inf))) for k, v in th.items()}
        return sec.evaluate(c, X)

    def reversed_k(X, th):                            # T(k) := T_ref(k0 (k0 / k)^s): exact at k0, reversed response
        return sec.evaluate({**th, "k_tim": k0 * (k0 / th["k_tim"]) ** sv}, X)

    def shortcut_h(X, th):                            # learned h from P (training data had h = 1e2 P)
        return sec.evaluate({**th, "h": 100.0 * th["P"]}, X)

    def energy(X, th):                                # 3 % too warm in the lid: heat created from nothing
        T = sec.evaluate(th, X)
        return np.where(X[:, 1] > sec.interfaces[1], (1 + 0.03 * sv) * T, T)

    def side_bc(X, th):                               # heat leaks through the "adiabatic" sides
        T = sec.evaluate(th, X)
        s = X[:, 0] / WIDTH
        return T * (1 - 0.02 * sv * (np.exp(-s / 0.03) + np.exp(-(1 - s) / 0.03)))

    def no_scaling(X, th):                            # saturates with power instead of scaling linearly
        P0 = DEFAULT["P"]
        a = 0.6 / sv
        Pe = P0 * (1 + a * np.tanh((th["P"] - P0) / (a * P0)))
        return sec.evaluate({**th, "P": Pe}, X)

    return [Model("smoothed interfaces", interface, defect="interface flux"),
            Model("frozen outside the box", ood, defect="out of distribution"),
            Model("reversed TIM response", reversed_k, defect="reversed causal response"),
            Model("cooling learned from power", shortcut_h, defect="spurious correlation"),
            Model("3 % energy created in the lid", energy, defect="energy balance"),
            Model("leaky sides", side_bc, defect="boundary condition"),
            Model("saturating power response", no_scaling, defect="scaling law")]


def training_distribution(n: int, rng: np.random.Generator, *, correlated: bool = True) -> list[dict]:
    """Parameter samples like the ones the models under test were trained on: ``k_tim`` near its nominal value,
    and (``correlated``) the cooling tied to the power, ``h = 1e2 P`` -- as in a data set taken from one product
    line. The usual held-out metrics are computed on this distribution."""
    out = []
    for _ in range(n):
        th = {k: float(rng.uniform(*TRAIN_BOX[k])) for k in ("P", "x_s")}
        th["k_tim"] = float(np.clip(rng.normal(DEFAULT["k_tim"], 0.1), *TRAIN_BOX["k_tim"]))
        th["h"] = 100.0 * th["P"] if correlated else float(rng.uniform(*TRAIN_BOX["h"]))
        out.append(th)
    return out


# ----------------------------------------------------------------------------------------------- learned models
def _field_model(sec: Section, name: str, predict: Callable[[dict], np.ndarray], inputs, defect="") -> Model:
    """A model that predicts the whole cell-centre field for ``theta`` and is read with per-layer splines."""
    from scipy.interpolate import RectBivariateSpline
    x, zc = sec.centres()
    cache: dict = {}

    def fn(X, th):
        key = tuple(sorted(th.items()))
        spl = cache.get(key)
        if spl is None:
            T = predict(th).reshape(sec.nx, -1)
            spl, k0 = [], 0
            for n in sec.nz:
                spl.append(RectBivariateSpline(x, zc[k0:k0 + n], T[:, k0:k0 + n], kx=3, ky=min(3, n - 1)))
                k0 += n
            if len(cache) > 256:
                cache.clear()
            cache[key] = spl
        lay = sec.layer_of(X[:, 1])
        out = np.empty(len(X))
        for i, s in enumerate(spl):
            m = lay == i
            if m.any():
                out[m] = s.ev(np.clip(X[m, 0], 0.0, WIDTH), X[m, 1])
        return out
    return Model(name, fn, tuple(inputs), defect)


def surrogate_model(sec: Section, n_train: int = 48, *, correlated: bool = True, seed: int = 0) -> Model:
    """A POD + Gaussian-process surrogate of the whole field over ``(P, x_s, k_tim, h)`` trained on reference
    solutions. ``correlated``: on :func:`training_distribution` (cooling tied to power, ``k_tim`` near nominal, as
    data from one product line would be); otherwise on independent uniform samples of :data:`TRAIN_BOX`."""
    from pinneapple_neural.architectures.rom.parametric import ParametricPOD
    rng = np.random.default_rng(seed)
    names = ("P", "x_s", "k_tim", "h")
    if correlated:
        thetas = training_distribution(n_train, rng)
    else:
        thetas = [{k: float(rng.uniform(*TRAIN_BOX[k])) for k in names} for _ in range(n_train)]
    P = np.array([[th[k] for k in names] for th in thetas])
    Y = np.stack([sec.solve(th).ravel() for th in thetas])
    rom = ParametricPOD(r=24, energy=1 - 1e-10, regressor="gpr", seed=seed).fit(P, Y)
    label = "POD-GPR surrogate, " + ("correlated training data" if correlated else "independent training data")
    return _field_model(sec, label, lambda th: rom.predict(np.array([[th[k] for k in names]]))[0], names,
                        "trained on correlated data" if correlated else "")


def pinn_model(sec: Section, *, hard_sides: bool = False, adam_steps: int = 3000, lbfgs_steps: int = 500,
               n_collocation: int = 3000, seed: int = 0, theta: dict | None = None):
    """A physics-informed network for the nominal case, trained with :func:`.pinn.train_pinn`: one sub-network
    output per layer, the conduction equation inside each layer, continuity of temperature and heat flux on both
    interfaces, the source flux on the bottom, Robin cooling on the top and adiabatic sides as soft constraints.
    ``hard_sides``: the lateral coordinate enters through ``cos(pi x / L)``, so the sides are adiabatic for any
    weights (a property of the architecture). Returns ``(Model, info)``; the model takes no parameters."""
    import torch

    from .pinn import PINNProblem, d, mlp, train_pinn
    th = {**DEFAULT, **(theta or {})}
    L, H = WIDTH, sec.height
    z1, z2 = (sec.interfaces / H).tolist()
    T0 = max(sec.tmax(th), 1e-6)
    kd, kt, kl = th["k_die"], th["k_tim"], th["k_lid"]
    a2 = (H / L) ** 2
    bi = th["h"] * H / kl
    s = th["sigma"]
    q0 = th["P"] / (s * L * np.sqrt(2 * np.pi))         # peak source flux (W/m^2); Gaussian normalised on the line

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = mlp(2, 3)

        def forward(self, X):
            if hard_sides:
                X = torch.cat([torch.cos(np.pi * X[:, :1]), X[:, 1:2]], 1)
            return self.body(X)

    def channel(u, X):
        z = X[:, 1:2]
        return torch.where(z < z1, u[:, 0:1], torch.where(z < z2, u[:, 1:2], u[:, 2:3]))

    def residual(u, X):
        uc = channel(u, X)
        return a2 * d(uc, X, 0, 2) + d(uc, X, 1, 2)

    def on(zv):
        def sampler(n, rng):
            return np.column_stack([rng.random(n), np.full(n, zv)])
        return sampler

    def sides(n, rng):
        return np.column_stack([rng.integers(0, 2, n).astype(float), rng.random(n)])

    def src(X):
        x = X[:, :1] * L
        return q0 * torch.exp(-0.5 * ((x - th["x_s"] * L) / (s * L)) ** 2)
    cons = [
        ("source flux", on(0.0), lambda u, X: d(u[:, 0:1], X, 1) + src(X) * H / (kd * T0)),
        ("robin top", on(1.0), lambda u, X: d(u[:, 2:3], X, 1) + bi * u[:, 2:3]),
        ("adiabatic sides", sides, lambda u, X: d(channel(u, X), X, 0)),
        ("T continuous die/TIM", on(z1), lambda u, X: u[:, 0:1] - u[:, 1:2]),
        ("flux continuous die/TIM", on(z1), lambda u, X: (kd * d(u[:, 0:1], X, 1) - kt * d(u[:, 1:2], X, 1)) / kl),
        ("T continuous TIM/lid", on(z2), lambda u, X: u[:, 1:2] - u[:, 2:3]),
        ("flux continuous TIM/lid", on(z2), lambda u, X: (kt * d(u[:, 1:2], X, 1) - kl * d(u[:, 2:3], X, 1)) / kl),
    ]
    prob = PINNProblem("die-TIM-lid section", [(0.0, 1.0), (0.0, 1.0)], residual, constraints=cons,
                       outputs=("T_die", "T_tim", "T_lid"), grid=(64, 64))
    torch.manual_seed(seed)
    net, info = train_pinn(prob, Net(), n_collocation=n_collocation, adam_steps=adam_steps,
                           lbfgs_steps=lbfgs_steps, seed=seed)

    def wrap(module, name):
        def fn(X, _th):
            Xs = torch.tensor(np.column_stack([X[:, 0] / L, X[:, 1] / H]), dtype=torch.float32)
            with torch.no_grad():
                u = module(Xs).double().numpy()
            lay = sec.layer_of(X[:, 1])
            return T0 * u[np.arange(len(X)), lay]

        def reinit():
            torch.manual_seed(seed + 1000)
            return Model(name + " (untrained weights)", wrap(Net(), name)[0], ())
        return fn, reinit
    label = "PINN, hard adiabatic sides" if hard_sides else "PINN, soft constraints"
    fn, reinit = wrap(net, label)
    return Model(label, fn, (), "", reinit), {"loss": info["loss"], "steps": info["steps"]}

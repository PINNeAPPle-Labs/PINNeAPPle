"""JSON-argument entry points for library capabilities whose own API takes objects (a soil, a wheel, a snapshot
set, a preset problem), so they can be library-call nodes. Each function builds the objects with the library's own
constructors and calls the library; nothing here computes physics itself.
"""
from __future__ import annotations

import importlib.util
import shutil
from typing import Literal

import numpy as np

from pinneapple_core.workflows import StepFailed

PINN_ARCHITECTURES = Literal["modified_mlp", "vanilla_pinn", "siren", "hash_grid_mlp", "vpinn", "xtfc", "pielm",
                             "elm", "rbf", "neural_gp", "bench_res_mlp", "bench_fourier_mlp"]
SOILS = Literal["GRC-1", "JSC-1A", "Ding2011-TYII", "FJS-1/Hurrell2025", "KLS-1", "Apollo-lunar"]
MESHLESS_DOMAINS = Literal["square", "lshape", "disk", "annulus", "plate_hole"]


def _solver(name: str, **kw):
    """A solver of the numerical-solver registry (all solver modules registered first)."""
    from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all
    register_all()
    return SolverRegistry.build(name, **kw)


def _sol_summary(sol) -> dict:
    info = {k: v for k, v in (sol.info or {}).items() if isinstance(v, (int, float, str, bool))}
    return {"method": sol.method, "wall_time_s": float(sol.wall_time_s), **info}


def solve_preset(preset: str = "burgers_1d", method: Literal["analytic", "reference", "pinn", "fem"] = "analytic",
                 architecture: PINN_ARCHITECTURES = "modified_mlp", epochs: int = 50, hidden_dim: int = 32,
                 n_layers: int = 3, n_collocation: int = 512, seed: int = 0) -> dict:
    """``pp.solve`` on a library preset; for ``pinn`` the architecture is any PINN variant of the model registry."""
    import pinneapple as pp
    if preset not in pp.list_presets():
        raise StepFailed(f"unknown preset {preset!r}")
    opts = {}
    if method == "pinn":
        opts = dict(architecture=architecture, epochs=epochs, hidden_dim=hidden_dim, n_layers=n_layers,
                    n_collocation=n_collocation, seed=seed)
    return _sol_summary(pp.solve(preset, method, **opts))


def compare_preset(preset: str = "burgers_1d", methods: list = ("analytic", "pinn"), reference: str = "analytic",
                   architecture: PINN_ARCHITECTURES = "modified_mlp", epochs: int = 50, n_points: int = 2048) -> dict:
    """``pp.compare`` of methods on a preset against a reference, on the same points."""
    import pinneapple as pp
    opts = {"pinn": {"architecture": architecture, "epochs": epochs, "hidden_dim": 32, "n_layers": 3,
                     "n_collocation": 512}}
    return pp.compare(preset, list(methods), reference=reference, options=opts, n_points=n_points).to_dict()


def lbm_channel(nx: int = 64, ny: int = 32, Re: float = 100.0, u_in: float = 0.05, steps: int = 200,
                save_every: int = 50) -> dict:
    """Lattice Boltzmann channel flow (``SolverRegistry.build('lbm')``)."""
    s = _solver("lbm", nx=nx, ny=ny, Re=Re, u_in=u_in)
    out = s(steps=steps, save_every=save_every)
    r = out.result.detach().cpu().numpy()
    return {"shape": list(r.shape), "max": float(np.nanmax(r)), "min": float(np.nanmin(r)),
            "mean": float(np.nanmean(r))}


def beam_deflection(L_m: float = 1.0, E_Pa: float = 2.0e11, I_m4: float = 8.33e-6, q_N_per_m: float = 1000.0,
                    nx: int = 200, bc_type: int = 0) -> dict:
    """Euler-Bernoulli beam boundary-value problem by finite differences (``beam_bvp_fdm``)."""
    s = _solver("beam_bvp_fdm", nx=nx, bc_type=bc_type)
    out = s(L_m, E_Pa, I_m4, q_N_per_m)
    w = out.result.detach().cpu().numpy().ravel()
    return {"max_abs_deflection_m": float(np.abs(w).max()), "nodes": int(w.size),
            **{k: float(v) for k, v in (out.extras or {}).items() if isinstance(v, (int, float))}}


def xtfc_pde(pde: str = "burgers_1d", n_basis: int = 60, n_query: int = 40) -> dict:
    """Extreme theory of functional connections solver (``xtfc_pde``)."""
    s = _solver("xtfc_pde", pde=pde, n_basis=n_basis)
    out = s(n_query=n_query)
    return {k: float(v) for k, v in (out.extras or {}).items() if isinstance(v, (int, float))} | {
        "result_shape": list(out.result.shape)}


def wheel_on_soil(soil: SOILS = "GRC-1", radius_m: float = 0.25, width_m: float = 0.2, load_N: float = 100.0,
                  slip: float = 0.2, gravity: float = 1.62) -> dict:
    """Wong-Reece wheel-soil model at steady state (``pinneapple_simulation.terramechanics``)."""
    from pinneapple_simulation.terramechanics import Wheel, WongReeceModel, get_soil
    m = WongReeceModel(get_soil(soil), Wheel(radius_m, width_m), g=gravity)
    return {k: v for k, v in m.steady_state(load_N, slip).items() if isinstance(v, (int, float, str))}


def meshless_compare(kind: Literal["poisson", "heat", "helmholtz"] = "poisson",
                     exact: str = "sin(pi*x)*sin(pi*y)", domain: MESHLESS_DOMAINS = "square",
                     methods: list = ("kansa", "rbf_fd", "fd", "fem"), k: float = 4.0, alpha: float = 1.0) -> dict:
    """Meshless vs mesh-based methods on one problem (``meshless_study``); expressions go through the library's
    own expression checks."""
    from pinneapple_simulation.meshless import Problem, meshless_study
    from pinneapple_simulation.meshless import domain as make_domain
    st = meshless_study(Problem(kind=kind, exact=exact, k=k, alpha=alpha), make_domain(domain), methods=tuple(methods))
    return st.to_dict() if hasattr(st, "to_dict") else {"result": str(st)}


def rom_compare(Y=None, A=None, methods: list = (), ranks: list = (), budget_s: float = 5.0, seed: int = 0) -> dict:
    """Reduced-order models compared on snapshots (``pinneapple_neural.architectures.rom.study.rom_study``):
    parametric snapshots ``Y`` (samples, dofs) for parameters ``A``."""
    from pinneapple_data.snapshots import snapshots_from_arrays
    from pinneapple_neural.architectures.rom.study import rom_study
    if Y is None:
        raise StepFailed("connect a dataset")
    snaps = snapshots_from_arrays({"u": np.asarray(Y, float)}, params=np.asarray(A, float),
                                  param_names=[f"p{i}" for i in range(np.asarray(A).shape[1])])
    res = rom_study(snaps, methods=list(methods) or None, ranks=list(ranks) or None, budget_s=budget_s, seed=seed)
    return res.to_dict()


def parametric_pod(A=None, Y=None, r: int = 16, regressor: Literal["rbf", "gpr", "nearest", "linear"] = "rbf",
                   test_fraction: float = 0.2, seed: int = 0) -> dict:
    """Parametric POD (``ParametricPOD``) fitted on a dataset, relative L2 on held-out samples."""
    from pinneapple_neural.architectures.rom import ParametricPOD
    A, Y = np.asarray(A, float), np.asarray(Y, float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(A))
    nt = max(1, int(round(test_fraction * len(A))))
    te, tr = idx[:nt], idx[nt:]
    m = ParametricPOD(r=r, regressor=regressor, seed=seed).fit(A[tr], Y[tr])
    pred = np.asarray(m.predict(A[te]))
    rel = np.linalg.norm(pred - Y[te], axis=1) / np.maximum(np.linalg.norm(Y[te], axis=1), 1e-12)
    return {"test_rel_l2_mean": float(rel.mean()), "test_rel_l2_max": float(rel.max()), "train": int(len(tr)),
            "test": int(len(te)), "rank": r, "regressor": regressor}


def surrogate_study(X=None, Y=None, example: Literal["none", "burgers_fields", "airfoil_rans"] = "none",
                    methods: list = ("ridge", "gp", "gbt"), budget: float = 10.0, seeds: int = 1) -> dict:
    """Classical ML / GP / tree / neural surrogates compared (``pinneapple_analysis.surrogate.run_study``)."""
    from pinneapple_analysis import surrogate as su
    if example != "none":
        data = getattr(su.examples, example)(**({"n": 32} if example == "burgers_fields" else {}))
    else:
        if X is None:
            raise StepFailed("connect a dataset or pick a built-in example")
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        arrays = {**{f"x{i}": X[:, i] for i in range(X.shape[1])}, **{f"y{j}": Y[:, j] for j in range(Y.shape[1])}}
        data = su.from_arrays(arrays, input_names=[f"x{i}" for i in range(X.shape[1])],
                              output_names=[f"y{j}" for j in range(Y.shape[1])])
    res = su.run_study(data, methods=list(methods), budget=budget, seeds=seeds)
    out = {"best": str(getattr(res, "best", ""))}
    for k in ("trust", "limitations", "methods"):
        v = getattr(res, k, None)
        if v is not None:
            out[k] = v if isinstance(v, (str, int, float, list, dict)) else str(v)[:2000]
    return out


def calculix_cantilever(length_m: float = 1.0, b_m: float = 0.1, h_m: float = 0.1, nx: int = 20,
                        element_type: Literal["C3D20R", "C3D8", "C3D8R"] = "C3D20R", run: bool = True) -> dict:
    """A cantilever deck written with the library (``cantilever_hex_mesh`` + ``write_inp``); solved with CalculiX
    when ``ccx`` (or its Docker image) is available, otherwise the deck is written and the node says so."""
    import tempfile
    from pathlib import Path

    from pinneapple_simulation.external_solvers import calculix as cx
    model = cx.cantilever_hex_mesh(L=length_m, b=b_m, h=h_m, nx=nx, element_type=element_type)
    d = Path(tempfile.mkdtemp(prefix="ccx_"))
    cx.write_inp(model, d / "beam.inp")
    out = {"deck": str(d / "beam.inp"), "nodes": int(len(getattr(model, "nodes", []) or [])),
           "ccx_available": bool(cx.ccx_backend())}
    if run and cx.ccx_backend():
        r = cx.run_ccx(d, "beam")
        out["run"] = str(r)[:500]
    elif run:
        out["note"] = "ccx not available: deck written, not solved"
    return out


def fmu_simulate(file: str = "", stop_time: float = 1.0, step_size: float = 0.01) -> dict:
    """Simulate an FMU from the data folder (``pinneapple_simulation.external_solvers.modelica.simulate_fmu``;
    needs fmpy)."""
    from pinneapple_simulation.external_solvers.modelica import FMUSimConfig, simulate_fmu

    from .config import safe_data_path
    if importlib.util.find_spec("fmpy") is None:
        raise StepFailed("needs fmpy (pip install fmpy)")
    res = simulate_fmu(str(safe_data_path(file)), FMUSimConfig(stop_time=stop_time, step_size=step_size))
    return {k: {"final": float(np.asarray(v)[-1]), "n": int(np.asarray(v).size)} for k, v in res.items()}


def newton_validate(solver: Literal["xpbd", "semi_implicit", "featherstone", "mujoco"] = "xpbd",
                    cases: list = ("free_fall", "pendulum_period", "bounce")) -> dict:
    """Rigid-body solver validation against closed forms (``newton_physics.validate``; needs NVIDIA Newton)."""
    if importlib.util.find_spec("newton") is None:
        raise StepFailed("needs NVIDIA Newton (pip install newton)")
    from pinneapple_simulation.external_solvers import newton_physics as nwt
    return nwt.validate(solver=solver, cases=list(cases))


def openfoam_available() -> dict:
    """Is OpenFOAM installed (blockMesh on PATH)?"""
    return {"openfoam": bool(shutil.which("blockMesh")), "calculix": bool(shutil.which("ccx"))}


def model_architecture(name: str = "fno", in_dim: int = 1, out_dim: int = 1, hidden: int = 64) -> dict:
    """Build any model of the registry (operators, GNNs, PINN variants, ROMs, recurrent, transformers) with its
    default arguments plus sizes, and count its parameters: a preview before wiring data to it."""
    import inspect

    from pinneapple_neural.architectures.registry import ModelRegistry

    from . import learn
    cls_kw = {}
    spec = ModelRegistry.spec(name) if hasattr(ModelRegistry, "spec") else None
    fn = getattr(spec, "builder", None) or getattr(spec, "cls", None)
    sig = inspect.signature(fn) if fn is not None else None
    for k, v in (("in_dim", in_dim), ("out_dim", out_dim), ("in_channels", in_dim), ("out_channels", out_dim),
                 ("branch_dim", in_dim), ("trunk_dim", 1), ("node_in_dim", in_dim), ("node_dim", in_dim),
                 ("hidden", hidden), ("hidden_dim", hidden), ("width", hidden), ("r", min(hidden, 16))):
        if sig is None or k in sig.parameters:
            cls_kw[k] = v
    try:
        m = ModelRegistry.build(name, **cls_kw)
    except TypeError as exc:
        raise StepFailed(f"{name}: {exc}") from None
    g = learn.layer_graph(m)
    return {"model": name, "parameters": g["total_params"], "layers": len(g["nodes"]), "arguments": cls_kw}


DOE_METHODS = Literal["lhs", "sobol", "halton", "random", "maximin", "full_factorial", "fractional_factorial"]


def doe_design(parameters: dict = None, n: int = 16, method: DOE_METHODS = "lhs", seed: int = 0,
               out_dir: str = "") -> dict:
    """``pinneapple_data.doe.design`` over a parameter space given as ``{"name": [low, high], ...}`` (or a
    ``Parameter`` dict per name). Writes the cases as CSV in the node's folder."""
    from pathlib import Path

    from pinneapple_data.contracts import ParameterSpace
    from pinneapple_data.doe.designs import METHODS, design
    if method not in METHODS:
        raise StepFailed(f"unknown method {method!r}; one of {METHODS}")
    parameters = parameters or {"x1": [0.0, 1.0], "x2": [0.0, 1.0]}
    specs = []
    for name, v in parameters.items():
        spec = dict(v) if isinstance(v, dict) else {"low": float(v[0]), "high": float(v[1])}
        specs.append({"name": name, **spec})
    try:
        space = ParameterSpace(name="flow_doe", parameters=specs)
    except Exception as exc:  # noqa: BLE001 - pydantic validation message is the useful part
        raise StepFailed(f"parameter space: {exc}") from exc
    d = design(space, n, method, seed=seed)
    path = ""
    if out_dir:
        path = str(Path(out_dir) / "cases.csv")
        d.cases.to_csv(path, index=False)
    from pinneapple_data.doe.coverage import coverage
    cov = coverage(d.unit, space.names)
    return {"method": d.method, "n_cases": len(d), "min_distance": float(_min_distance(d.unit)),
            "cases_csv": path, "coverage": {k: v for k, v in cov.items() if isinstance(v, (int, float, str))}}


def _min_distance(unit):
    from pinneapple_data.doe.designs import min_distance
    return min_distance(unit)


GEOMETRY_FORMATS = Literal["stl", "obj", "glb", "usda", "step"]


def geometry_export(name: str = "caarc_building", fmt: GEOMETRY_FORMATS = "stl", lod: str = "light",
                    out_dir: str = "") -> dict:
    """Build a parametric geometry of ``pinneapple_design.geometry_library`` and write it (STL / OBJ / glTF binary /
    USD ASCII / STEP) in the node's folder."""
    from pathlib import Path

    from pinneapple_design import geometry_library as gl
    if name not in gl.list():
        raise StepFailed(f"unknown geometry {name!r}; one of {gl.list()}")
    geom = gl.build(name, lod=lod)
    path = geom.export(str(Path(out_dir or ".") / f"{name}.{fmt}"), fmt)
    info = gl.info(name)
    return {"geometry": name, "format": fmt, "file": path, "bytes": Path(path).stat().st_size,
            **{k: v for k, v in info.items() if isinstance(v, (int, float, str))}}


def design_intent(text: str = "a heat sink for a 20 W chip in still air", seed: int = 0, budget: int = 200,
                  n_screen: int = 500, out_dir: str = "") -> dict:
    """``pinneapple_design.intent.design``: a plain-language goal to a verified design package (files in the node's
    folder), or the questions it still needs, or why it is out of scope."""
    from pinneapple_design.intent.api import design
    res = design(text, seed=seed, budget=budget, n_screen=n_screen, out_dir=out_dir or None)
    kind = type(res).__name__
    out = res.to_dict() if hasattr(res, "to_dict") else {}
    return {"outcome": kind, **{k: v for k, v in out.items() if k not in ("candidates",)}}

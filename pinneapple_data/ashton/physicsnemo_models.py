"""PhysicsNeMo's DoMINO and X-MeshGraphNet as members of the Ashton benchmark and of the trust layer.

NVIDIA's PhysicsNeMo (formerly Modulus) publishes two external-aerodynamics surrogates trained on DrivAerML: DoMINO,
a local multi-scale point-cloud model, and X-MeshGraphNet, MeshGraphNet scaled with halo regions. This module is
how they enter PINNeAPPle -- as *members*, on the same splits and the same metrics as the library's own models,
and on data the reader already serves (:mod:`pinneapple_data.ashton.loader`), never as a separate pipeline.

Three things this module is careful about
-----------------------------------------
**What is measured here, and what is not.** The three datasets are terabytes and are not downloaded in a test or
on a laptop, so DoMINO runs on the *fixture* in :mod:`pinneapple_data.ashton.fixture` -- two parametric body
families in the same file layout -- and every row it fills says so. A number produced on synthetic geometry is a
statement about the adapter and the trust layer, not about DoMINO's accuracy on DrivAerML, and the row's ``model``
and ``note`` carry the step budget and the fixture it was trained on.

**The mesh model needs a wheel this install does not have.** X-MeshGraphNet is a PhysicsNeMo *recipe* -- a
halo-sharded MeshGraphNet built on ``physicsnemo.domain_parallel`` and on PyTorch Geometric -- rather than a model
class that the installed ``physicsnemo`` 2.2.2 ships, and PyG (``torch-geometric`` with ``torch-scatter``) is not
installed here. :class:`XMeshGraphNetMember` therefore raises at construction, naming what to install and what is
merely absent, and the benchmark table gets a row that says ``awaiting_data`` with that reason -- never a number.

**No retraining to join.** A PhysicsNeMo model is a ``torch.nn.Module``. All a member needs is the case-to-inputs
mapping (:meth:`DoMINOMember.inputs`), the output-to-field mapping (:meth:`DoMINOMember.predict`) and, for the
trust layer, an embedding (:meth:`DoMINOMember.embedding`). Weights trained anywhere -- PhysicsNeMo's own recipe,
a checkpoint, this module's ``fit`` -- plug in unchanged.

    >>> from pinneapple_data.ashton.physicsnemo_models import availability
    >>> availability()["domino"]                     # True on a machine with physicsnemo and its domino module
    True
"""
from __future__ import annotations

import copy
import importlib.util
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import AWAITING, BenchmarkRow, BenchmarkTable, benchmark_table, declared
from .catalog import DATASETS
from .catalog import dataset as _dataset
from .loader import _DIMENSIONLESS, AshtonData, Case, resolve_field
from .vtk_xml import face_area_normals

__all__ = ["PHYSICSNEMO_LICENCE", "DEFAULT_GRID", "DEFAULT_NEIGHBORS", "availability",
           "x_meshgraphnet_availability", "require_domino", "DoMINOMember", "XMeshGraphNetMember",
           "SurfaceResidual", "admissibility_residual", "physicsnemo_table"]

#: PhysicsNeMo is Apache-2.0 -- the licence that travels with a member's provenance, next to the data's CC BY-SA.
PHYSICSNEMO_LICENCE = {"spdx": "Apache-2.0", "name": "Apache License 2.0",
                       "url": "https://github.com/NVIDIA/physicsnemo/blob/main/LICENSE"}

#: The grid DoMINO's geometry encoder evaluates the body on. PhysicsNeMo's default is ``[128, 64, 64]``, sized for a
#: 160-million-cell DrivAerML volume; on a fixture case whose whole body is 728 faces it only costs time, so the
#: adapter's default is small and stated in every row it writes.
DEFAULT_GRID = (24, 16, 16)

#: Neighbours per surface point in DoMINO's local geometry encoding (PhysicsNeMo's default is 7).
DEFAULT_NEIGHBORS = 6


# ---------------------------------------------------------------------------
# what this machine can run
# ---------------------------------------------------------------------------

def _has(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it (a dotted name imports only its parent)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def availability() -> dict[str, Any]:
    """What of PhysicsNeMo this machine can actually run, and the reason when it cannot.

    The keys are the facts a table row and a test want: ``physicsnemo`` (the installed version or ``None``),
    ``domino`` (the model class PhysicsNeMo ships for it), ``torch_geometric`` and ``torch_scatter`` (what the graph
    models need), and one ``reason`` per thing that cannot run. Nothing here imports ``physicsnemo``, so asking the
    question is free on a machine that does not have it.
    """
    has_pnemo = _has("physicsnemo")
    version = None
    if has_pnemo:
        try:
            import physicsnemo

            version = str(getattr(physicsnemo, "__version__", "")) or None
        except Exception:                                      # an install that cannot even import: say so
            has_pnemo = False
    domino = has_pnemo and _has("physicsnemo.models.domino")
    geometric = _has("torch_geometric")
    scatter = _has("torch_scatter")
    reasons: dict[str, str] = {}
    if not has_pnemo:
        reasons["domino"] = ("physicsnemo is not installed: pip install nvidia-physicsnemo "
                             "(https://github.com/NVIDIA/physicsnemo)")
    elif not domino:
        reasons["domino"] = (f"physicsnemo {version} is installed but carries no models.domino module; DoMINO "
                             f"arrived in a later release")
    if not (geometric and scatter):
        missing = [name for name, here in (("torch-geometric", geometric), ("torch-scatter", scatter)) if not here]
        reasons["x_meshgraphnet"] = (
            f"the graph model needs {' and '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} not "
            f"installed (pip install torch-geometric torch-scatter); no PINNeAPPle extra ships them")
    elif not _has("physicsnemo.domain_parallel"):
        reasons["x_meshgraphnet"] = ("the halo exchange X-MeshGraphNet is built on needs physicsnemo.domain_parallel, "
                                     "which this install does not have")
    return {"physicsnemo": version, "domino": domino, "torch_geometric": geometric, "torch_scatter": scatter,
            "x_meshgraphnet": not reasons.get("x_meshgraphnet"), "reasons": reasons}


def x_meshgraphnet_availability() -> dict[str, Any]:
    """The X-MeshGraphNet half of :func:`availability`: ``{"available", "reason", "requires"}``."""
    facts = availability()
    return {"available": bool(facts["x_meshgraphnet"]),
            "reason": facts["reasons"].get("x_meshgraphnet", ""),
            "requires": ["torch-geometric", "torch-scatter", "physicsnemo domain-parallel MeshGraphNet",
                         "nvidia-physicsnemo (the xaeronet recipe, not a model class in models/)"]}


def require_domino() -> Any:
    """Import PhysicsNeMo's DoMINO, or raise the actionable ``ImportError`` this module's callers rely on."""
    facts = availability()
    if not facts["domino"]:
        raise ImportError(f"DoMINO needs PhysicsNeMo's domino module. {facts['reasons'].get('domino', '')}. "
                          f"A PhysicsNeMo model is a plain torch.nn.Module once trained, so anything else in "
                          f"pinneapple_data.ashton.physicsnemo_models works without it.")
    from physicsnemo.models.domino.config import DEFAULT_MODEL_PARAMS, Config
    from physicsnemo.models.domino.model import DoMINO

    return DoMINO, Config, DEFAULT_MODEL_PARAMS


# ---------------------------------------------------------------------------
# a case as DoMINO's inputs
# ---------------------------------------------------------------------------

def _merge(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _small_parameters(filters: int = 8, width: int = 32, grid: Sequence[int] = DEFAULT_GRID,
                      neighbors: int = DEFAULT_NEIGHBORS) -> dict[str, Any]:
    """PhysicsNeMo's default model parameters with the layers a fixture-sized case can afford.

    Every number is a size, not a change of method: the same convolutions, the same Stachenfeld-style processor (as
    a plain ``conv`` stack instead of a ``unet``, because a 24 x 16 x 16 grid has nothing for a unet to downsample),
    the same two-loop surface solution calculator. The defaults it overrides are sized for 160-million-cell volumes.
    """
    return {
        "interp_res": list(grid),
        "num_neighbors_surface": int(neighbors),
        "num_neighbors_volume": 4,
        "geometry_rep": {
            "base_filters": int(filters),
            "geo_conv": {"base_neurons": 16, "surface_radii": [0.05, 1.0], "surface_neighbors_in_radius": [4, 32],
                         "volume_radii": [0.1, 1.0], "volume_neighbors_in_radius": [4, 32]},
            "geo_processor": {"base_filters": max(4, int(filters) // 2), "processor_type": "conv"},
        },
        "geometry_local": {"base_layer": int(width), "surface_radii": [0.05, 0.25],
                           "surface_neighbors_in_radius": [4, 16],
                           "volume_radii": [0.1, 0.25], "volume_neighbors_in_radius": [4, 16]},
        "nn_basis_functions": {"base_layer": int(width), "num_modes": 4},
        "aggregation_model": {"base_layer": int(width)},
        "position_encoder": {"base_neurons": int(width), "num_modes": 4},
        "parameter_model": {"base_layer": int(width), "num_modes": 4},
    }


def _surface_and_field(case: Case) -> tuple[Any, np.ndarray, str]:
    """The case's surface and its Cp field: ``(surface, values, "point" | "cell")``, in two reads of one file."""
    names_only = case.surface(arrays=())
    name, location = resolve_field(names_only, case.dataset, "cp", path=case.surface_path)
    read = case.surface(arrays=[name])
    values = read.point_data.get(name, read.cell_data.get(name))
    return read, np.asarray(values, dtype=np.float64), location


def _face_values(surface: Any, values: np.ndarray, location: str) -> np.ndarray:
    """A surface field on the mesh's faces, whichever quadrature the dataset publishes it on."""
    if location == "cell":
        return np.asarray(values, dtype=np.float64)
    cells = np.asarray(surface.connectivity).reshape(-1, 3)
    return np.asarray(values, dtype=np.float64)[cells].mean(axis=1)


def _to_nodes(face_values: np.ndarray, areas: np.ndarray, connectivity: np.ndarray, n_points: int) -> np.ndarray:
    """Fold a per-face field onto the nodes, weighted by each face's share of a node's area.

    The same operation :mod:`pinneapple_data.ashton.fixture` uses to write WindsorML's point quadrature, run in the
    other direction: DoMINO predicts one Cp per face, and WindsorML publishes one per point.
    """
    cells = np.asarray(connectivity).reshape(-1, 3)
    numerator = np.zeros(int(n_points), dtype=np.float64)
    denominator = np.zeros(int(n_points), dtype=np.float64)
    share = np.asarray(areas, dtype=np.float64) / 3.0
    for k in range(3):
        np.add.at(numerator, cells[:, k], face_values * share)
        np.add.at(denominator, cells[:, k], share)
    return numerator / np.maximum(denominator, 1e-30)


class DoMINOMember:
    """PhysicsNeMo's DoMINO on an Ashton case: the inputs it needs, the field it returns, and its embedding.

    One member is one set of weights. ``fit`` trains it on cases the reader opened (a step budget, not an epoch
    count: the fixture is a dozen cases), ``predict`` answers with a Cp field on the case's own quadrature -- one
    value per face, or folded onto the points where the dataset publishes one per point -- and ``embedding``
    returns the case's own representation inside the model, which is what the trust layer scores against the
    training distribution.

    Nothing is downloaded and nothing is retrained to *use* a model: assign ``member.model`` a trained PhysicsNeMo
    ``DoMINO`` and every other method works unchanged (see ``docs/guides/physicsnemo_baselines.md``).
    """

    def __init__(self, *, grid: Sequence[int] = DEFAULT_GRID, neighbors: int = DEFAULT_NEIGHBORS,
                 filters: int = 8, width: int = 32, seed: int = 0, model_parameters: dict | None = None) -> None:
        import torch

        DoMINO, Config, defaults = require_domino()
        self.grid = tuple(int(g) for g in grid)
        self.neighbors = int(neighbors)
        self.seed = int(seed)
        self.steps = 0
        self.history: list[float] = []
        self.trained_on = ""
        parameters = _small_parameters(filters, width, self.grid, self.neighbors)
        if model_parameters is not None:
            _merge(parameters, model_parameters)
        merged = _merge(copy.deepcopy(defaults), parameters)
        torch.manual_seed(self.seed)
        self.model = DoMINO(input_features=3, output_features_vol=None, output_features_surf=1, global_features=2,
                            model_parameters=Config(merged))
        self._cases: dict[str, dict[str, Any]] = {}

    # -- the case -> inputs mapping -------------------------------------------------------------
    def inputs(self, case: Case) -> dict[str, Any]:
        """The tensors DoMINO's ``forward`` requires, built from one case's surface file (cached per case).

        PhysicsNeMo's recipe computes them from a mesh with VTK and its own SDF utilities; here they come from
        :func:`pinneapple_data.ashton.vtk_xml.read_surface` and from a signed distance taken to the case's own faces,
        so a case is one read of one file. The keys are PhysicsNeMo's names, spelled as its ``forward`` validates
        them: ``geometry_coordinates``, ``surf_grid``, ``sdf_surf_grid``, ``surface_mesh_centers``,
        ``surface_mesh_neighbors`` (the neighbour *coordinates*, as in PhysicsNeMo's own docstring example),
        ``surface_normals``, ``surface_neighbors_normals``, ``surface_areas``, ``surface_neighbors_areas``,
        ``pos_surface_center_of_mass`` and the two global-parameter tensors.
        """
        key = str(getattr(case, "surface_path", case))
        if key not in self._cases:
            self._cases[key] = self._build(case)
        return self._cases[key]["data"]

    def truth(self, case: Case) -> np.ndarray:
        """The case's own Cp field on its own quadrature: what a benchmark row scores a prediction against."""
        key = str(getattr(case, "surface_path", case))
        if key not in self._cases:
            self._cases[key] = self._build(case)
        return self._cases[key]["truth"]

    def _build(self, case: Case) -> dict[str, Any]:
        import torch
        from scipy.spatial import cKDTree

        surface, values, location = _surface_and_field(case)
        areas, normals = face_area_normals(surface)
        cells = np.asarray(surface.connectivity).reshape(-1, 3)
        centers = np.asarray(surface.points, dtype=np.float64)[cells].mean(axis=1)
        tree = cKDTree(centers)
        _distances, index = tree.query(centers, k=self.neighbors + 1)
        index = np.atleast_2d(index)[:, 1:]                       # drop the point itself
        lower, upper = np.asarray(surface.points).min(axis=0), np.asarray(surface.points).max(axis=0)
        axes = [np.linspace(lower[i], upper[i], self.grid[i]) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        distance, nearest = tree.query(grid.reshape(-1, 3), k=1)
        outward = np.einsum("ij,ij->i", grid.reshape(-1, 3) - centers[nearest], normals[nearest])
        sdf = (np.sign(outward) * distance).reshape(self.grid)
        center_of_mass = centers.mean(axis=0)
        data = {
            "geometry_coordinates": torch.as_tensor(np.asarray(surface.points)[None], dtype=torch.float32),
            "surf_grid": torch.as_tensor(grid[None], dtype=torch.float32),
            "sdf_surf_grid": torch.as_tensor(sdf[None], dtype=torch.float32),
            "pos_surface_center_of_mass": torch.as_tensor((centers - center_of_mass)[None], dtype=torch.float32),
            "surface_mesh_centers": torch.as_tensor(centers[None], dtype=torch.float32),
            "surface_mesh_neighbors": torch.as_tensor(centers[index][None], dtype=torch.float32),
            "surface_normals": torch.as_tensor(normals[None], dtype=torch.float32),
            "surface_neighbors_normals": torch.as_tensor(normals[index][None], dtype=torch.float32),
            "surface_areas": torch.as_tensor(areas[None], dtype=torch.float32) + 1e-6,
            "surface_neighbors_areas": torch.as_tensor(areas[index][None], dtype=torch.float32) + 1e-6,
            "global_params_values": torch.zeros(1, 2, 1),
            "global_params_reference": torch.ones(1, 2, 1),
        }
        return {"data": data, "location": location, "values": values, "surface": surface, "areas": areas,
                "normals": normals, "face_truth": _face_values(surface, values, location),
                "truth": values}

    # -- training ------------------------------------------------------------------------------
    def fit(self, cases: Iterable[Case], *, steps: int = 240, lr: float = 3e-3) -> DoMINOMember:
        """Train the member on cases, for a *step budget* rather than an epoch count.

        One case per step, cyclically, on the mean squared Cp error over that case's faces -- the same loss the
        published recipes use for the surface field. A budget because the honest unit here is compute, not
        epochs: a dozen fixture cases are one epoch, and a measurement has to say how much training it saw.
        """
        import torch

        cases = list(cases)
        if not cases:
            raise ValueError("fit needs at least one case to train on")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        for step in range(int(steps)):
            bundle = self._bundle(cases[step % len(cases)])
            optimizer.zero_grad()
            _volume, surface = self.model(bundle["data"])
            target = torch.as_tensor(bundle["face_truth"], dtype=torch.float32)[None]
            loss = torch.mean((surface[:, :, 0] - target) ** 2)
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach()))
        self.steps += int(steps)
        self.model.eval()
        return self

    def _bundle(self, case: Case) -> dict[str, Any]:
        key = str(getattr(case, "surface_path", case))
        if key not in self._cases:
            self._cases[key] = self._build(case)
        return self._cases[key]

    # -- the output ----------------------------------------------------------------------------
    def predict(self, case: Case) -> np.ndarray:
        """The Cp field for one case, on the quadrature the dataset publishes: ``model(case)`` for a benchmark row."""
        import torch

        bundle = self._bundle(case)
        self.model.eval()
        with torch.no_grad():
            _volume, surface = self.model(bundle["data"])
        cp = surface[0, :, 0].detach().cpu().numpy().astype(np.float64)
        if bundle["location"] == "point":
            surface_geometry = bundle["surface"]
            cp = _to_nodes(cp, bundle["areas"], surface_geometry.connectivity, surface_geometry.n_points)
        return cp

    def embedding(self, case: Case) -> np.ndarray:
        """The case's own representation inside the model: its geometry encoding, pooled to one vector.

        DoMINO's first stage encodes the body onto a fixed grid; the channels of that encoding are what the rest of
        the network reads as "this geometry". Pooling them (mean and standard deviation over the grid) gives a short
        fixed-length vector per case -- six numbers on this adapter's default, because the encoding has three
        channels -- which is what :func:`pinneapple_analysis.trust.trust_gate.TrustGate` can score a Mahalanobis
        distance on with a dozen training cases. A per-face encoding would be 20 x 728 numbers per case and a
        singular covariance; this is the same information at a dimension the calibration can carry.
        """
        import torch

        bundle = self._bundle(case)
        captured: dict[str, Any] = {}

        def hook(_module, _inputs, output):                     # a reading, not a substitution: return None
            captured["encoding"] = output.detach()

        handle = self.model.geo_rep_surface.register_forward_hook(hook)
        try:
            self.model.eval()
            with torch.no_grad():
                self.model(bundle["data"])
        finally:
            handle.remove()
        encoding = captured["encoding"][0].cpu().numpy().astype(np.float64)
        flat = encoding.reshape(encoding.shape[0], -1)
        return np.concatenate([flat.mean(axis=1), flat.std(axis=1)])

    # -- the trust layer -----------------------------------------------------------------------
    def expert(self, name: str | None = None) -> Any:
        """This member as a :class:`pinneapple_physics.ensemble.Expert`, provenance included.

        The provenance is what lets a report say where an ensemble member came from: the PhysicsNeMo release and its
        Apache-2.0 licence, how many steps of training this instance saw, and on what.
        """
        from pinneapple_physics.ensemble import Expert

        facts = availability()
        return Expert(name or self.model_name, self.predict, kind="neural",
                      provenance={"source": f"physicsnemo {facts.get('physicsnemo') or '?'} DoMINO",
                                  "licence": dict(PHYSICSNEMO_LICENCE),
                                  "steps": self.steps, "trained_on": self.trained_on,
                                  "notes": "inputs built by pinneapple_data.ashton.physicsnemo_models.DoMINOMember"})

    @property
    def model_name(self) -> str:
        """How a row names this member: the model, the release, the step budget."""
        facts = availability()
        steps = f"{self.steps} steps" if self.steps else "untrained weights"
        return f"DoMINO (PhysicsNeMo {facts.get('physicsnemo') or '?'}, {steps})"

    def describe(self) -> dict[str, Any]:
        return {"model": self.model_name, "trained_on": self.trained_on, "steps": self.steps,
                "grid": list(self.grid), "neighbors": self.neighbors, "seed": self.seed,
                "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                "final_loss": (self.history[-1] if self.history else None)}


class XMeshGraphNetMember:
    """PhysicsNeMo's X-MeshGraphNet as a member -- *when this install can run it*.

    Constructing it on a machine without PyTorch Geometric raises, before anything is built, with the names of what
    to install and what is merely absent (the recipe is not a model class in ``physicsnemo.models``; it is a
    halo-sharded MeshGraphNet in PhysicsNeMo's external-aerodynamics recipe). Nothing else in this module depends on
    it, and the benchmark table gets its row from :func:`x_meshgraphnet_availability` with the same reason.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        facts = x_meshgraphnet_availability()
        if not facts["available"]:
            raise ImportError(
                "X-MeshGraphNet cannot run on this installation: " + facts["reason"] + ". It requires "
                + ", ".join(facts["requires"]) + ". Its row in the benchmark table is 'awaiting_data' for this "
                "reason rather than carrying a number nothing measured.")
        raise ImportError(
            "X-MeshGraphNet's halo-sharded MeshGraphNet is a training recipe in PhysicsNeMo's external-aerodynamics "
            "examples, not an importable model class: drive it through the recipe and wrap the trained "
            "torch.nn.Module with pinneapple_physics.ensemble.from_torch, or pass a member with the same "
            "fit/predict/embedding shape as DoMINOMember to the cross-family experiment.")

    @staticmethod
    def availability() -> dict[str, Any]:
        return x_meshgraphnet_availability()


# ---------------------------------------------------------------------------
# the residual: a physics check that needs no truth
# ---------------------------------------------------------------------------

class SurfaceResidual:
    """A label-free physics residual on a predicted surface-pressure field, callable as ``residual_fn``.

    Three things a Cp field on a closed bluff body must satisfy whatever the case is, none of which needs the truth:

    * **the Bernoulli bound** -- ``Cp <= 1``: stagnation is the maximum of the pressure coefficient, so an area
      where the prediction exceeds it is not an incompressible flow. Reported as the area-weighted mean excess;
    * **no side force on a body symmetric about its centre plane** -- Ahmed, DrivAer and Windsor bodies all are,
      so ``|F_y| / |F_x|`` is a residual that a correct prediction keeps near zero;
    * **non-negative drag** -- a closed body cannot be pushed upstream by its own pressure, so a negative ``F_x`` is
      physical nonsense, reported as ``max(0, -F_x) / |F_x|``.

    Together they are one number: the sum of the three, in coefficient units, zero on a physical field. It is the
    signal a trust layer can compute in operation, where no reference field exists -- and, as
    :mod:`pinneapple_data.ashton.physicsnemo_trust` measures, it catches *unphysical* predictions while it does not
    catch a systematic out-of-family error. Both facts are reported there rather than one standing in for the other.

    ``Geometry`` is cached per case (a real case's surface is 612 MB), and a field in pascals is converted with the
    dataset's own declared freestream before any of the three is computed.
    """

    def __init__(self, *, bound: float = 1.0) -> None:
        self.bound = float(bound)
        self._geometry: dict[str, dict[str, Any]] = {}

    def _of(self, case: Case) -> dict[str, Any]:
        key = str(getattr(case, "surface_path", case))
        if key not in self._geometry:
            surface, values, location = _surface_and_field(case)
            del values
            areas, normals = face_area_normals(surface)
            name = case.field_of("cp")[0]
            dimensionless = bool(_DIMENSIONLESS.match(name))
            data = case.dataset
            dynamic = 0.5 * float(data.density) * float(data.velocity) ** 2
            self._geometry[key] = {"areas": areas, "normals": normals, "location": location,
                                   "dimensionless": dimensionless, "dynamic": dynamic}
        return self._geometry[key]

    def terms(self, prediction: Any, case: Case) -> dict[str, float]:
        """The three residuals and their sum, on one prediction for one case."""
        geometry = self._of(case)
        field = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if geometry["location"] == "point":
            return {"bernoulli": float("nan"), "side": float("nan"), "thrust": float("nan"), "total": float("nan"),
                    "note": "a per-point field cannot be integrated on the faces without folding it back; the "
                            "residual is measured on the cell quadrature"}
        coefficient = field if geometry["dimensionless"] else field / max(geometry["dynamic"], 1e-30)
        weight = geometry["areas"]
        force = -(coefficient * weight) @ geometry["normals"]          # (3,), its first component is the drag
        drag, side = float(force[0]), float(force[1])
        scale = abs(drag) + 1e-300
        bernoulli = float(np.mean(np.clip(coefficient - self.bound, 0.0, None)))
        terms = {"bernoulli": bernoulli, "side": abs(side) / scale, "thrust": max(0.0, -drag) / scale}
        terms["total"] = terms["bernoulli"] + terms["side"] + terms["thrust"]
        return terms

    def __call__(self, prediction: Any, case: Case) -> float:
        """``residual_fn`` for :class:`pinneapple_physics.ensemble.PhysicsEnsemble`: one non-negative number."""
        return float(self.terms(prediction, case)["total"])


def admissibility_residual(prediction: Any, case: Case) -> float:
    """The residual of :class:`SurfaceResidual` on one prediction, building the geometry cache for this call.

    Convenience for a one-off check; a loop over cases should hold one :class:`SurfaceResidual` so the surface is
    read once per case.
    """
    return float(SurfaceResidual()(prediction, case))


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

def physicsnemo_table(sources: dict[str, str | Path | AshtonData] | None = None, *,
                      datasets: Iterable[str] | None = None, limit: int | None = 8,
                      member: DoMINOMember | None = None, model_name: str | None = None) -> BenchmarkTable:
    """The Ashton benchmark table with the PhysicsNeMo members in it: measured where a member ran, declared where not.

    ``member`` fills the surface-pressure column of every dataset that has a case on disk, through
    :func:`pinneapple_data.ashton.benchmark.benchmark_table` (so the metrics, the quadrature and the declared
    columns are the existing ones, not a second implementation). X-MeshGraphNet gets a row per dataset that says
    ``awaiting_data`` and why -- a row with a reason, never a borrowed number.
    """
    names = list(datasets if datasets is not None else DATASETS)
    table = benchmark_table(sources, datasets=names, limit=limit,
                            model=(member.predict if member is not None else None),
                            model_name=(model_name or (member.model_name if member is not None else
                                                       "constant baseline")))
    facts = x_meshgraphnet_availability()
    for name in names:
        data = _dataset(name)
        note = (f"not run: {facts['reason']}. Its inputs are a halo-sharded PyG graph, so the row stays empty until "
                f"the wheels are installed and the recipe is run on the real data") if not facts["available"] else (
                "not run: the wheels are installed but this adapter drives the recipe's model class, which this "
                "release does not ship")
        table.add(BenchmarkRow(dataset=data.id, quantity="surface_pressure", status=AWAITING, model="X-MeshGraphNet",
                               declared=declared(data.id, "surface_pressure"), note=note))
    return table

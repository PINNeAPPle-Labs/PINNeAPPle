"""Stage 5's second half: a model from another framework behind the interface this pipeline already speaks.

``lpm.fit`` trains :class:`~pinneapple_neural.lpm.LargePhysicsModel` and the campaign is built around it. A comparison
that only ever ran its own model would answer "does our model learn" and not "is our model the right choice", so this
module puts a PhysicsNeMo model behind the same three calls the pipeline uses — ``fit(designs)``, ``predict(design)``,
``describe()`` — on the **same** :class:`~pinneapple_neural.lpm.Design` objects, with the same split and the same card.

The member here is **DoMINO**, because it is the model Luminary's SHIFT datasets were published for and the one this
environment can import (PhysicsNeMo 2.2.2; X-MeshGraphNet needs ``torch-geometric`` and ``torch-scatter``, which are not
installed — :func:`availability` says so rather than failing at import time). Its inputs are built from the design's own
point cloud: the signed-distance grid over the body's bounding box, the k-nearest neighbours among the query points, and
the surface areas. PhysicsNeMo's own recipe computes the same quantities from a mesh with VTK
(:mod:`pinneapple_data.ashton.physicsnemo_models` does exactly that for the Ashton cases); a design here is a point
cloud with no connectivity, so the areas are **uniform** and the normals are the ones the family produced. That is a
real difference between the two bundles and it is reported by :meth:`DoMINOMember.describe`, not hidden: a DoMINO row in
a comparison is DoMINO on a point cloud, not on a mesh.

What this does **not** claim: that DoMINO trained for 300 steps on a few hundred designs is DoMINO as Luminary trains
it. The step budget goes into every row (``budget``), because a baseline whose budget is invisible invites a comparison
that is not one.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["DoMINOMember", "availability", "compare_backends", "finetune"]

#: the point-cloud grid DoMINO interpolates on: a size, not a method (see ``ashton.physicsnemo_models._small_parameters``)
DEFAULT_GRID = (24, 16, 16)
DEFAULT_NEIGHBORS = 8


def availability() -> dict[str, Any]:
    """Which third-party backends can run here, and the reason for each one that cannot."""
    out: dict[str, Any] = {"torch": torch.__version__, "mps": bool(torch.backends.mps.is_available())}
    try:
        import physicsnemo

        out["physicsnemo"] = getattr(physicsnemo, "__version__", "unknown")
        out["domino"] = True
    except Exception as exc:                                     # noqa: BLE001 - the import is the probe
        out["physicsnemo"] = None
        out["domino"] = False
        out["reason"] = f"physicsnemo is not importable: {exc}"
    for name, module in (("torch_geometric", "torch_geometric"), ("torch_scatter", "torch_scatter")):
        try:
            __import__(module)
            out[name] = True
        except Exception:                                        # noqa: BLE001 - same probe
            out[name] = False
    out["x_meshgraphnet"] = out.get("torch_geometric", False) and out.get("torch_scatter", False)
    return out


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


@dataclass
class DoMINOMember:
    """PhysicsNeMo's DoMINO answering on a point-cloud design, in the units the design's target is in.

    ``predict`` returns one value per query point — the same quadrature the target is on — so a comparison with the
    LPM needs no interpolation between them. ``fit`` takes a step budget, not epochs: on a design space of a few
    hundred designs the honest unit is compute, and the budget is part of :meth:`describe`.
    """

    n_params: int = 0
    n_fields: int = 1
    grid: tuple[int, int, int] = DEFAULT_GRID
    neighbors: int = DEFAULT_NEIGHBORS
    filters: int = 8
    width: int = 32
    seed: int = 0
    device_name: str = "auto"
    name: str = "DoMINO (PhysicsNeMo)"
    budget: str = "0 steps"
    model: Any = None
    history: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    trained_on: str = ""

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = self._build()
        self.budget = "0 steps of DoMINO so far"

    # -- construction ---------------------------------------------------------------------------
    def _build(self) -> Any:
        import copy

        from pinneapple_data.ashton.physicsnemo_models import (
            merge_config,
            require_domino,
            small_parameters,
        )

        DoMINO, Config, defaults = require_domino()
        parameters = small_parameters(self.filters, self.width, self.grid, self.neighbors)
        merged = merge_config(copy.deepcopy(defaults), parameters)
        torch.manual_seed(self.seed)
        # one global parameter channel at least: DoMINO's parameter encoder is sized at construction and a family with
        # no conditioning (a shape-only space) still has to be representable
        global_features = max(1, int(self.n_params))
        model = DoMINO(input_features=3, output_features_vol=None, output_features_surf=int(self.n_fields),
                       global_features=global_features, model_parameters=Config(merged))
        return model.to(_device(self.device_name))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # -- the design -> inputs mapping -----------------------------------------------------------
    def bundle(self, design: Any) -> dict[str, torch.Tensor]:
        """DoMINO's forward inputs from one design's point cloud.

        The signed distance is the one its own recipe uses — distance to the nearest surface point, signed by that
        point's outward normal — with the query points standing in for the mesh faces, since a design has no
        connectivity. ``surface_areas`` is therefore uniform, and the note that says so travels in
        :meth:`describe`.
        """
        from scipy.spatial import cKDTree

        centers = np.asarray(design.points.detach().cpu(), dtype=np.float64)
        normals = np.asarray(design.normals.detach().cpu(), dtype=np.float64)
        # a 2-D design space (a section, a meridian) is embedded in its own plane with a zero third coordinate:
        # DoMINO's grid is three-dimensional, and this is the same embedding the family's points already imply
        if centers.shape[1] == 2:
            centers = np.concatenate([centers, np.zeros((centers.shape[0], 1))], axis=1)
            normals = np.concatenate([normals, np.zeros((normals.shape[0], 1))], axis=1)
        elif centers.shape[1] != 3:
            raise ValueError(f"a {centers.shape[1]}-dimensional design cannot be embedded in DoMINO's 3-D grid")
        n_centers = _unit(normals)
        tree = cKDTree(centers)
        k = min(self.neighbors + 1, len(centers))
        _distance, index = tree.query(centers, k=k)
        index = np.atleast_2d(index)[:, 1:]
        lower, upper = centers.min(axis=0), centers.max(axis=0)
        # a flat body (a plate, an airfoil section) has one degenerate bounding-box axis: give it thickness, or the
        # grid has no cells to interpolate on
        span = np.maximum(upper - lower, 1e-6)
        upper = lower + np.where(span < 1e-3 * max(span.max(), 1e-12), span.max() * 1e-3, span)
        axes = [np.linspace(lower[i], upper[i], self.grid[i]) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        distance, nearest = tree.query(grid.reshape(-1, 3), k=1)
        outward = np.einsum("ij,ij->i", grid.reshape(-1, 3) - centers[nearest], n_centers[nearest])
        sdf = np.sign(outward) * distance
        areas = np.full(len(centers), 1.0 / max(len(centers), 1))
        params = np.asarray(design.params.detach().cpu(), dtype=np.float64).ravel()
        if params.size != self.n_params:
            raise ValueError(f"the design carries {params.size} conditioning parameter(s) and this member was built "
                             f"for {self.n_params}")
        values = np.zeros(max(1, self.n_params), dtype=np.float64)
        values[: params.size] = params
        center_of_mass = centers.mean(axis=0)
        data = {
            "geometry_coordinates": centers[None],
            "surf_grid": grid[None],
            "sdf_surf_grid": sdf.reshape(self.grid)[None],
            "pos_surface_center_of_mass": (centers - center_of_mass)[None],
            "surface_mesh_centers": centers[None],
            "surface_mesh_neighbors": centers[index][None],
            "surface_normals": n_centers[None],
            "surface_neighbors_normals": n_centers[index][None],
            "surface_areas": areas[None],
            "surface_neighbors_areas": areas[index][None],
            "global_params_values": torch.as_tensor(values[None, :, None], dtype=torch.float32),
            "global_params_reference": torch.ones(1, values.size, 1),
        }
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def _forward(self, design: Any) -> tuple[Any, Any]:
        data = {k: v.to(self.device) for k, v in self.bundle(design).items()}
        try:
            return self.model(data)
        except RuntimeError as exc:
            if self.device.type == "mps":                       # honest fallback: record it and keep the answer
                self.notes.append(f"mps fell back to cpu: {str(exc).splitlines()[0][:160]}")
                self.model = self.model.to("cpu")
                data = {k: v.to("cpu") for k, v in data.items()}
                return self.model(data)
            raise

    # -- training -------------------------------------------------------------------------------
    def fit(self, designs: Iterable[Any], *, steps: int = 200, lr: float = 3e-3) -> DoMINOMember:
        """Train on designs for a step budget, one design per step, on the mean squared error of the target."""
        designs = list(designs)
        if not designs:
            raise ValueError("fit needs at least one design")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        for step in range(int(steps)):
            design = designs[step % len(designs)]
            _volume, surface = self._forward(design)
            target = design.target.to(self.device)[None]
            loss = torch.mean((surface[:, :, : self.n_fields] - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach()))
        self.model.eval()
        self.budget = f"{len(self.history)} steps of DoMINO at lr {lr:g} on {len(designs)} design(s)"
        self.trained_on = f"{len(designs)} design(s)"
        return self

    # -- the answer -----------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, design: Any) -> np.ndarray:
        """The field on the design's own query points, ``(Q, n_fields)``, in the target's units."""
        self.model.eval()
        _volume, surface = self._forward(design)
        return surface[0, :, : self.n_fields].detach().cpu().numpy().astype(np.float64)

    def describe(self) -> dict[str, Any]:
        state = availability()
        return {
            "model": "DoMINO",
            "framework": f"physicsnemo {state.get('physicsnemo')} (Apache-2.0)",
            "budget": self.budget,
            "grid": list(self.grid),
            "neighbors": int(self.neighbors),
            "device": str(self.device),
            "n_params": int(self.n_params),
            "n_fields": int(self.n_fields),
            "inputs": "signed-distance grid over the body's bounding box, k-nearest query points, unit normals and "
                      "uniform point areas — a point cloud, not a mesh",
            "notes": list(self.notes),
            "x_meshgraphnet": state.get("x_meshgraphnet"),
        }


def _unit(normals: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-12)


def compare_backends(designs: Sequence[Any], *, train: int = 200, seed: int = 0) -> dict[str, Any]:
    """Fit DoMINO on a design list and describe what it took: the shape a report needs to quote its budget."""
    if not designs:
        raise ValueError("nothing to fit")
    member = DoMINOMember(n_params=int(designs[0].params.numel()), n_fields=int(designs[0].target.shape[-1]),
                          seed=seed)
    member.fit(designs, steps=train)
    return {"member": member, "describe": member.describe(), "final_loss": member.history[-1] if member.history else None}


def finetune(source_run: str | Path, dataset_dir: str | Path, out_dir: str | Path, **train_kwargs: Any) -> dict[str, Any]:
    """Warm-start a fit from another design space's weights: the cheap half of a new model.

    The design spaces in this repository share a body model and a conditioning width, so what a fit on the fin family
    learnt about "a surface with a field on it" is not wasted on the exchanger — only the conditioning and the specific
    field have to be learnt again. The weights are only transferable when the two spaces agree on the point dimension,
    the conditioning width and the neighbourhood radii, and :func:`~pinneapple_neural.lpm_pipeline.train` refuses the
    rest instead of loading a state dict into the wrong first layer.

    This is a *convenience over the existing training path*, not a second trainer: everything goes through ``train``,
    which keeps the split, the card and the checkpoint behaviour identical to a cold fit — so a fine-tuned model is
    measured and reported exactly like any other, which is the only way the comparison between the two means anything.
    """
    from ..lpm_pipeline import train

    result = train(dataset_dir, out_dir, init_from=source_run, **train_kwargs)
    result.setdefault("notes", []).append(
        f"warm-started from {source_run}; the split, the card and the measurement are the ones a cold fit would produce")
    return result

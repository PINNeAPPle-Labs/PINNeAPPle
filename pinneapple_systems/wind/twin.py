"""The wind farm as a ``pinneapple_twin3d`` scene: a wake footprint you can orbit.

One scene carries everything the demo needs, and nothing that the physics did not
produce:

* a **ground grid** whose per-vertex `velocity_ratio` field is the plan-view wake
  footprint of the farm, animated over a full sweep of wind direction;
* **towers and rotor discs** at their published positions, at hub height;
* one **sensor per turbine**, carrying that machine's modelled power at each
  direction, so the viewer can read the array while it turns.

The field comes from :func:`pinneapple_systems.wind.wake.wake_field`, so the
picture and the farm model share the same wake constants. The scene is a
*visualisation of a model*: it is not a flow solution, and the caption on the
exported twin says so.

Because the scene is a ``pinneapple_twin3d.Scene``, the same object exports to
the bundled web viewer, to OpenUSD, and to the frame-by-frame renderer in
:mod:`pinneapple_systems.wind.video` that produces the demo film.
"""
from __future__ import annotations

import numpy as np

from .layout import WindFarmLayout
from .wake import WakeConfig, farm_power, wake_field

__all__ = ["build_farm_scene", "export_twin", "ground_grid"]

CAPTION = (
    "Analytical wake footprint (Jensen/PARK or Gaussian), not CFD. "
    "Every wake is seeded with the free-stream speed, so overlapping wakes "
    "superpose rather than cascade."
)


def ground_grid(
    layout: WindFarmLayout,
    *,
    n_x: int = 96,
    n_y: int = 72,
    margin: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan-view grid around the farm, ``(V, 3)`` vertices and ``(F, 3)`` faces.

    The extent is the layout's bounding box grown by ``margin`` on each side, so
    the wakes leave the frame instead of being cut at the farm boundary.
    """
    positions = layout.positions
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    x = np.linspace(lo[0] - margin * span[0], hi[0] + margin * span[0], n_x)
    y = np.linspace(lo[1] - margin * span[1], hi[1] + margin * span[1], n_y)
    xx, yy = np.meshgrid(x, y)                      # (n_y, n_x), row-major
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)]).astype(np.float32)

    idx = np.arange(xx.size).reshape(n_y, n_x)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.column_stack([np.concatenate([a, b]), np.concatenate([b, d]),
                             np.concatenate([c, c])]).astype(np.int64)
    return vertices, faces


def build_farm_scene(
    layout: WindFarmLayout,
    directions,
    *,
    wind_speed: float = 8.0,
    config: WakeConfig | None = None,
    power_curves=None,
    n_x: int = 96,
    n_y: int = 72,
    tower_radius: float = 1.8,
) -> object:
    """Build the ``pinneapple_twin3d.Scene`` for one farm at a swept wind direction.

    ``directions`` is the animation axis: one scene time step per direction. The
    ground field, the rotor discs and the per-turbine power series all correspond
    to that same direction at each step.
    """
    from pinneapple_twin3d import Scene
    from pinneapple_twin3d.primitives import cylinder, merge, transform

    cfg = config or WakeConfig()
    directions = np.asarray(directions, dtype=float)
    vertices, faces = ground_grid(layout, n_x=n_x, n_y=n_y)

    scene = Scene(
        f"{layout.name}: wake footprint at {wind_speed:g} m/s",
        times=[float(d) for d in directions],
        time_unit="deg",
        source=CAPTION,
    )
    scene.step_labels = [f"{d:.0f} deg" for d in directions]

    scene.add_part("ground", vertices, faces, group="terrain", color=(0.32, 0.34, 0.36))

    # Plan-view footprint, one slice per direction.
    ratios = np.empty((directions.size, vertices.shape[0]), dtype=np.float32)
    powers = np.zeros((directions.size, len(layout)))
    for k, direction in enumerate(directions):
        field = wake_field(layout, vertices[:, :2], wind_speed, float(direction), cfg)
        ratios[k] = field["velocity_ratio"]
        if power_curves is not None:
            result = farm_power(layout, wind_speed, float(direction), power_curves, cfg)
            powers[k] = result.power_kw
    scene.add_field("ground", "velocity_ratio", ratios, unit="-")

    towers, rotors = [], []
    for turbine in layout.turbines:
        towers.append(cylinder((turbine.x, turbine.y, 0.0), (turbine.x, turbine.y, turbine.hub_height),
                               tower_radius, 16))
        rotor = cylinder((turbine.x, turbine.y, turbine.hub_height - 1.0),
                         (turbine.x, turbine.y, turbine.hub_height + 1.0),
                         turbine.rotor_diameter / 2.0, 48)
        rotors.append(transform(rotor, scale=1.0))
    tower_mesh = merge(towers)
    rotor_mesh = merge(rotors)
    scene.add_part("towers", tower_mesh[0], tower_mesh[1], group="turbines", color=(0.86, 0.87, 0.88))
    scene.add_part("rotors", rotor_mesh[0], rotor_mesh[1], group="turbines", color=(0.18, 0.30, 0.38))

    for i, turbine in enumerate(layout.turbines):
        series = powers[:, i] if power_curves is not None else None
        scene.add_sensor(
            f"{turbine.name}-power",
            (turbine.x, turbine.y, turbine.hub_height),
            label=f"{turbine.name} power",
            unit="kW",
            quantity="power",
            series=series,
            envelope=(0.0, turbine.rated_power_kw),
        )
    return scene


def export_twin(scene, folder, *, usd: bool = False) -> str:
    """Write the scene (and the bundled viewer) to ``folder``; returns scene.json's path."""
    return scene.export(str(folder), with_viewer=True, usd=usd)

"""The USD -> CalculiX -> FNO pipeline: the parts that run without a solver, a network or a GPU.

What is covered here is everything the pipeline's correctness rests on and that can be checked in a second: the
design space's sampling and envelope, the revolved surface against the analytic solid, the grid's occupancy and the
two maps between the grid and a mesh (including the fill that the 6 mm plate depends on), the operator's save/load
round-trip and its refusal outside the envelope, and the reading of a mounting interface from a synthetic USD mesh.

What is not covered, and why: ``ccx`` and gmsh are external binaries, ``usd-core`` and ``nvidia-physicsnemo`` are
optional installs, and a campaign is minutes of solving. Those paths are exercised by
``examples/usd_structural_surrogate/run.py``, whose measured numbers land in its ``experiment.json``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.pipelines.usd_structural_surrogate import NOMINAL, DesignSpace
from pinneapple_simulation.pipelines.usd_structural_surrogate.campaign import default_grid
from pinneapple_simulation.pipelines.usd_structural_surrogate.design import (
    BOUNDS,
    PLATE_DIAMETER_M,
    profile,
    revolve_surface,
)
from pinneapple_simulation.pipelines.usd_structural_surrogate.grid import (
    GridSpec,
    fill_outside,
    occupancy,
    rasterize,
    signed_distance,
)


# ------------------------------------------------------------------ design space
def test_design_space_samples_inside_its_bounds():
    space = DesignSpace.from_interface()
    designs = space.sample(24, seed=3)
    assert len(designs) == 24
    for d in designs:
        assert set(d) == set(BOUNDS)
        space.validate(d)                                    # raises outside the bounds
    for k, (lo, hi) in space.bounds.items():
        values = [d[k] for d in designs]
        assert min(values) >= lo and max(values) <= hi


def test_design_space_rejects_out_of_bounds_and_round_trips_normalisation():
    space = DesignSpace.from_interface()
    with pytest.raises(ValueError):
        space.validate({**NOMINAL, "t": 10.0})
    u = space.normalized(space.denormalized([0.1, 0.4, 0.7, 0.9]))
    assert np.allclose(u, [0.1, 0.4, 0.7, 0.9])


def test_interface_anchor_sets_plate_diameter_and_thinnest_plate():
    space = DesignSpace.from_interface({"plate_diameter_mm": 300.0, "plate_thickness_mm": 8.0})
    assert space.plate_diameter == pytest.approx(0.300)
    assert space.bounds["t"][0] == pytest.approx(0.008)       # a plate thinner than the asset is not in the family


def test_the_envelope_is_the_bounds():
    space = DesignSpace.from_interface()
    assert space.contains(NOMINAL) is True
    assert space.contains({**NOMINAL, "h": 0.9}) is False


# ------------------------------------------------------------------ geometry
def test_profile_is_the_analytic_solid_of_revolution():
    p = profile(NOMINAL)
    t, dc, r, h = (NOMINAL[k] for k in ("t", "dc", "r", "h"))
    assert p[0].tolist() == [0.0, 0.0]                        # axis at the bottom
    assert p[-1].tolist() == [0.0, h]                         # axis at the top
    radii = p[:, 0]
    assert radii.max() == pytest.approx(PLATE_DIAMETER_M / 2)
    assert radii.max() != pytest.approx(dc / 2)               # the plate is wider than the column
    assert p[:, 1].max() == pytest.approx(h)


def test_revolved_surface_closes_and_matches_the_analytic_volume():
    V, F = revolve_surface(NOMINAL, n_theta=64)
    assert V.shape[0] == len(V) and F.shape[1] == 3
    # revolution volume by the conical-frustum rule, against the closed polyline that generated the surface
    p = profile(NOMINAL)
    vol = sum(np.pi * (z1 - z0) * (r0 * r0 + r0 * r1 + r1 * r1) / 3
              for (r0, z0), (r1, z1) in zip(p[:-1], p[1:], strict=True))
    # the surface is watertight: every edge is shared by exactly two triangles
    edges: dict = {}
    for a, b, c in F:
        for e in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(e))
            edges[key] = edges.get(key, 0) + 1
    assert all(v == 2 for v in edges.values()), "surface has a boundary or a non-manifold edge"
    assert vol > 0 and np.ptp(V[:, 2]) == pytest.approx(p[-1, 1])


def test_a_flare_that_does_not_fit_is_refused():
    from pinneapple_simulation.pipelines.usd_structural_surrogate.design import bracket_geo
    with pytest.raises(ValueError):
        bracket_geo({**NOMINAL, "r": 0.30})                   # flare taller than the part
    with pytest.raises(ValueError):
        bracket_geo({**NOMINAL, "r": 0.10, "dc": 0.30})       # flare wider than the plate


# ------------------------------------------------------------------ the grid
def _box(nodes_per_axis: int = 4):
    """A unit cube of tetrahedra: the simplest mesh whose inside is known analytically."""
    lin = np.linspace(0.0, 1.0, nodes_per_axis + 1)
    X = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)
    n = nodes_per_axis
    idx = lambda i, j, k: i * (n + 1) * (n + 1) + j * (n + 1) + k   # noqa: E731
    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx(i, j, k), idx(i + 1, j, k), idx(i, j + 1, k), idx(i, j, k + 1),
                     idx(i + 1, j + 1, k), idx(i + 1, j, k + 1), idx(i, j + 1, k + 1), idx(i + 1, j + 1, k + 1)]
                tets += [[c[0], c[1], c[3], c[5]], [c[1], c[2], c[3], c[7]],
                         [c[1], c[3], c[5], c[7]], [c[3], c[5], c[6], c[7]],
                         [c[1], c[4], c[2], c[7]], [c[1], c[5], c[4], c[7]]]
    return X, np.asarray(tets, int)


def test_occupancy_finds_the_inside_of_a_unit_cube():
    X, T = _box(4)
    g = GridSpec([-0.2, -0.2, -0.2], [1.4, 1.4, 1.4], [14, 14, 14])
    inside = occupancy(X, T, g, "C3D4")
    C = g.centers()
    truth = np.all((C >= 0.0) & (C <= 1.0), axis=1).reshape(g.n)
    # only cells whose centre is within one cell of the surface may disagree
    disagree = (inside != truth)
    assert disagree.sum() <= 2 * (14 * 14) * 3
    assert inside.sum() > 0 and inside.sum() < inside.size


def _linear(X):
    return 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 2]


def test_barycentric_sampling_is_exact_for_a_field_linear_over_each_element():
    from pinneapple_simulation.pipelines.usd_structural_surrogate.grid import rasterize_barycentric
    X, T = _box(4)
    g = GridSpec([-0.05, -0.05, -0.05], [1.1, 1.1, 1.1], [24, 24, 24])
    inside = occupancy(X, T, g, "C3D4")
    field = rasterize_barycentric(X, _linear(X), g, T, "C3D4", inside=inside)
    # at every cell that is inside the part, the interpolated value is the analytic one
    C = g.centers().reshape(g.n + (3,))
    got, want = field[inside], _linear(C[inside])
    assert got.size and np.abs(got - want).max() < 1e-12


def test_inverse_distance_sampling_is_the_coarser_alternative():
    X, T = _box(4)
    g = GridSpec([-0.05, -0.05, -0.05], [1.1, 1.1, 1.1], [24, 24, 24])
    inside = occupancy(X, T, g, "C3D4")
    field = rasterize(X, _linear(X), g, inside)
    C = g.centers().reshape(g.n + (3,))
    err = np.abs(field[inside] - _linear(C[inside])).max() / np.ptp(_linear(X))
    assert 0.0 < err < 0.5                      # it works, and it is visibly less exact than the barycentric one


def test_fill_outside_removes_the_cliff_at_the_surface():
    g = GridSpec([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [10, 10, 10])
    inside = np.zeros(g.n, bool)
    inside[4:6, 4:6, :] = True                                  # a narrow column through the box
    raw = np.where(inside, 1.0, 0.0)
    filled = fill_outside(raw, inside, g)
    assert np.isclose(filled[0, 0, 0], 1.0)                     # every outside cell takes the nearest inside value
    assert np.isclose(filled[inside].min(), 1.0)


def test_signed_distance_is_negative_inside():
    g = GridSpec([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [9, 9, 9])
    inside = np.zeros(g.n, bool)
    inside[4, 4, 4] = True
    d = signed_distance(inside, g)
    assert d[4, 4, 4] < 0.0
    assert d[0, 0, 0] > 0.0


def test_default_grid_holds_every_design_of_the_family():
    space = DesignSpace.from_interface()
    g = default_grid(space)
    r_plate = space.plate_diameter / 2
    assert g.origin[0] <= -r_plate and g.origin[1] <= -r_plate
    assert g.origin[2] + g.size[2] >= space.bounds["h"][1]


# ------------------------------------------------------------------ the operator
def test_features_are_two_scaled_channels():
    pytest.importorskip("torch")
    from pinneapple_simulation.pipelines.usd_structural_surrogate.surrogate import features
    g = GridSpec([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [8, 8, 8])
    inside = np.zeros(g.n, bool)
    inside[2:6, 2:6, 2:6] = True
    x = features(inside, signed_distance(inside, g), g)
    assert x.shape == (2, 8, 8, 8)
    assert set(np.unique(x[0])) <= {0.0, 1.0}
    assert np.abs(x[1]).max() <= 1.0 + 1e-6


def test_operator_saves_and_loads_and_keeps_its_envelope(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("physicsnemo")
    from pinneapple_simulation.pipelines.usd_structural_surrogate.surrogate import (
        GeometryFNO,
        Surrogate,
    )

    space = DesignSpace.from_interface()
    g = GridSpec([-0.15, -0.15, 0.0], [0.30, 0.30, 0.51], [8, 8, 8])
    net = GeometryFNO(g.n, latent_channels=4, num_fno_layers=1, num_fno_modes=3, padding=2)
    sur = Surrogate(net=net, grid=g, space=space, out_scale=np.array([1e-4, 1e-7, 1e-5, 1e7]))
    sur.save(tmp_path / "sur")
    back = Surrogate.load(tmp_path / "sur")
    assert back.grid.n == g.n
    assert np.allclose(back.out_scale, sur.out_scale)
    assert back.valid(NOMINAL) is True
    inside = np.zeros(g.n, bool)
    inside[2:6, 2:6, 2:6] = True
    out = back.predict_fields(back.features_of({"occupancy": inside}))
    assert set(out) == {"ux", "uy", "uz", "vm"}
    assert out["vm"].shape == g.n
    with pytest.raises(ValueError):
        back.predict_design({**NOMINAL, "h": 0.9})            # outside the envelope


# ------------------------------------------------------------------ the asset
def test_mount_interface_is_measured_from_the_mesh():
    from pinneapple_simulation.pipelines.usd_structural_surrogate.usd_asset import (
        UsdMesh,
        mount_interface,
    )
    # a plate of radius 100 at z = -20 .. -8 (thickness 12) plus a column up to z = 0
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 400)
    r = 100.0                                  # the asset's units, read as millimetres
    plate = np.concatenate([np.stack([r * np.cos(th), r * np.sin(th), np.full(400, -20.0)], 1),
                            np.stack([r * np.cos(th), r * np.sin(th), np.full(400, -8.0)], 1)])
    column = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    pts = np.concatenate([plate, column])
    mesh = UsdMesh(points=pts, face_counts=np.zeros(0, int), face_indices=np.zeros(0, int),
                   meters_per_unit=1.0, up_axis="Z", path="synthetic.usda", sha256="")
    iface = mount_interface(mesh)
    assert iface["plate_diameter_mm"] == pytest.approx(200.0, abs=1e-6)
    assert iface["plate_thickness_mm"] == pytest.approx(12.0, abs=1e-6)
    assert iface["height_mm"] == pytest.approx(20.0, abs=1e-6)
    assert iface["up_axis"] == "Z"


def test_recorded_interface_matches_the_shape_the_family_expects():
    from pinneapple_simulation.pipelines.usd_structural_surrogate.usd_asset import (
        RECORDED_INTERFACE,
    )
    space = DesignSpace.from_interface(RECORDED_INTERFACE)
    assert space.plate_diameter == pytest.approx(RECORDED_INTERFACE["plate_diameter_mm"] / 1000.0)
    assert space.bounds["t"][0] == pytest.approx(RECORDED_INTERFACE["plate_thickness_mm"] / 1000.0)


def test_the_dashboard_occupancy_matches_the_revolved_solid():
    """The sliders' occupancy must be the same solid the 3-D view draws, plate included."""
    pytest.importorskip("torch")
    pytest.importorskip("physicsnemo")
    from pinneapple_simulation.pipelines.usd_structural_surrogate.dashboard import Predictor
    from pinneapple_simulation.pipelines.usd_structural_surrogate.surrogate import (
        GeometryFNO,
        Surrogate,
    )

    space = DesignSpace.from_interface()
    g = GridSpec([-0.15, -0.15, 0.0], [0.30, 0.30, 0.51], [24, 24, 40])
    net = GeometryFNO(g.n, latent_channels=4, num_fno_layers=1, num_fno_modes=3, padding=2)
    sur = Surrogate(net=net, grid=g, space=space, out_scale=np.ones(4))
    pred = Predictor(sur, n_theta=32)
    occ = pred._occupancy(NOMINAL)

    C = g.centers().reshape(g.n + (3,))
    t, dc, r, h = (NOMINAL[k] for k in ("t", "dc", "r", "h"))
    radius = np.where(C[..., 2] < t, space.plate_diameter / 2,
                      np.where(C[..., 2] < t + r, (dc / 2 + r) - r * (C[..., 2] - t) / r, dc / 2))
    truth = (np.hypot(C[..., 0], C[..., 1]) <= radius) & (C[..., 2] <= h)
    assert np.array_equal(occ, truth)

    # the failure this guards: with the plate thick enough to hold a layer of cells, those cells are inside the part
    thick = {**NOMINAL, "t": 0.014}
    occ2 = pred._occupancy(thick)
    plate = ((C[..., 2] > 0.0) & (C[..., 2] < thick["t"])
             & (np.hypot(C[..., 0], C[..., 1]) < space.plate_diameter / 2))
    assert plate.any() and occ2[plate].all()

    # and the reason fill_outside exists: at the asset's own 6 mm the plate is thinner than a cell, so no cell centre
    # is inside it at all -- the outer cells touching it carry no value of their own
    assert not ((C[..., 2] > 0.0) & (C[..., 2] < NOMINAL["t"])
                & (np.hypot(C[..., 0], C[..., 1]) < space.plate_diameter / 2)).any()


def test_the_sliders_speak_millimetres_and_the_model_metres():
    """The slider values are mm; a missing conversion made every design 1000x too big and out of envelope."""
    pytest.importorskip("torch")
    pytest.importorskip("physicsnemo")
    from pinneapple_simulation.pipelines.usd_structural_surrogate.dashboard import Predictor
    from pinneapple_simulation.pipelines.usd_structural_surrogate.surrogate import (
        GeometryFNO,
        Surrogate,
    )

    space = DesignSpace.from_interface()
    g = GridSpec([-0.15, -0.15, 0.0], [0.30, 0.30, 0.51], [8, 8, 8])
    net = GeometryFNO(g.n, latent_channels=4, num_fno_layers=1, num_fno_modes=3, padding=2)
    pred = Predictor(Surrogate(net=net, grid=g, space=space, out_scale=np.ones(4)))
    query = {"t": "11.0", "dc": "70", "r": "20", "h": "420"}
    design = pred.design_from_query(query)
    assert design == pytest.approx({"t": 0.011, "dc": 0.070, "r": 0.020, "h": 0.420})
    assert space.contains(design), "the sliders' own nominal must land inside the envelope"
    with pytest.raises(ValueError):
        pred.payload({**design, "t": 11.0})          # a metre value passed where millimetres belong

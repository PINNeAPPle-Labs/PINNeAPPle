"""``{"tag": "boundary"}`` on a ProblemSpec is the boundary of its box: solve_pde samples it on the spatial faces
instead of refusing (it used to drop it silently, then to refuse it, so laplace_2d and poisson_2d could not be
solved without hand-built masks). Any other tag still needs real geometry.

References: Laplace's equation with u = 0 on the boundary of the square has the unique solution u = 0, while any
constant is harmonic; so a model starting at u = 1 reaches u = 0 on the boundary only if the boundary condition
is enforced.
"""
import numpy as np
import pytest
import torch

import pinneapple_physics as pp
from pinneapple_physics.pde_environment import condition_sampling as CS


class _Offset(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.fill_(1.0)                                                   # u = 1 everywhere

    def forward(self, X):
        return self.net(X)


def _boundary_points(n=400):
    s = np.linspace(-1, 1, n // 4, dtype=np.float32)
    o = np.ones_like(s)
    return torch.as_tensor(np.concatenate([np.c_[s, -o], np.c_[s, o], np.c_[-o, s], np.c_[o, s]]))


def test_laplace_2d_trains_with_its_boundary_condition():
    spec = pp.get_preset("laplace_2d")
    lo = spec.domain_bounds["x"]
    assert lo == (-1.0, 1.0) or lo == (0.0, 1.0)
    torch.manual_seed(0)
    model = _Offset()
    xb = _boundary_points()
    if lo == (0.0, 1.0):
        xb = (xb + 1) / 2
    before = model(xb).abs().mean().item()
    pp.solve_pde(spec, model, epochs=400, n_collocation=256, lr=1e-2)
    after = model(xb).detach().abs().mean().item()
    assert before == pytest.approx(1.0, abs=1e-6) and after < 0.1, (before, after)


def test_a_transient_box_samples_the_spatial_faces_not_the_time_faces(monkeypatch):
    spec = pp.get_preset("reaction_diffusion_2d")
    cond = next(c for c in spec.conditions if c.selector_type == "tag" and c.selector.get("tag") == "boundary")
    seen = []
    real = CS.sample_condition_points

    def spy(select, bounds, coords, n, rng):
        out = real(select, bounds, coords, n, rng)
        if getattr(select, "__name__", "") == "_box_boundary":
            seen.append((list(coords), out))
        return out
    monkeypatch.setattr(CS, "sample_condition_points", spy)
    m = torch.nn.Sequential(torch.nn.Linear(len(spec.coords), 8), torch.nn.Tanh(),
                            torch.nn.Linear(8, len(spec.fields)))
    try:
        pp.solve_pde(spec, m, epochs=1, n_collocation=64)
    except pp.TagConditionsUnresolved as e:                    # another tag of this preset needs geometry
        assert cond.name not in str(e)
        return
    coords, pts = next((c, p) for c, p in seen if len(p))
    b = spec.domain_bounds
    spatial = [i for i, c in enumerate(coords) if c != "t"]
    on_face = np.zeros(len(pts), dtype=bool)
    for i in spatial:
        lo, hi = b[coords[i]]
        on_face |= np.isclose(pts[:, i], lo) | np.isclose(pts[:, i], hi)
    assert on_face.all()


def test_a_geometry_tag_is_still_refused():
    spec = _first_with_tag("inlet")
    m = torch.nn.Linear(len(spec.coords), len(spec.fields))
    with pytest.raises(pp.TagConditionsUnresolved, match="inlet"):
        pp.solve_pde(spec, m, epochs=1, n_collocation=16)


def _first_with_tag(tag):
    from pinneapple_physics.pde_environment.presets.registry import list_presets
    for n in list_presets():
        try:
            s = pp.get_preset(n)
        except Exception:
            continue
        if any(c.selector_type == "tag" and c.selector.get("tag") == tag for c in s.conditions):
            return s
    pytest.skip(f"no preset with a {tag!r} tag")

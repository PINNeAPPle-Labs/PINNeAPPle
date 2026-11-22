"""``pp.tensor``: a torch tensor that keeps its name, unit, domain and components (#186, item X2).

The three things the issue asks for are the three things tested here, in this order: it **exists** and carries the
labels; it **round-trips through autograd** (a labelled leaf gets its gradient, and a labelled computation keeps the
labels); and it **rejects incompatible operations** -- ``m/s + Pa`` raises instead of producing a number whose unit
nobody writes down. Plus the two traps that make the first two true: the leaf really is a leaf, and the operations
return the same class, not a plain tensor that silently lost everything.
"""
from __future__ import annotations

import pytest
import torch

import pinneapple as pp
from pinneapple_data.physical_units import UnitError


def test_the_labels_are_kept_and_read_back():
    t = pp.tensor([[1.0, 2.0], [3.0, 4.0]], name="velocity", units="m/s", coords=("x", "y"),
                  components=("u", "v"), domain={"x": (0, 1), "y": (0, 1)})
    assert isinstance(t, pp.PhysicsTensor) and isinstance(t, torch.Tensor)
    assert (t.name, t.units) == ("velocity", "m/s")
    assert t.unit is not None and t.unit.dim != pp.tensor([1.0], units="Pa").unit.dim
    assert t.coords == ("x", "y") and t.components == ("u", "v")
    assert t.domain == {"x": (0, 1), "y": (0, 1)}
    assert t.semantics["units"] == "m/s" and t.semantics["shape"] == (2, 2)
    assert t.labelled() and not pp.tensor([1.0, 2.0]).labelled()


def test_a_labelled_leaf_is_a_leaf_so_autograd_fills_its_gradient():
    t = pp.tensor([[1.0, 2.0], [3.0, 4.0]], name="velocity", units="m/s", requires_grad=True)
    assert t.is_leaf and t.requires_grad
    loss = (t * 2.0).sum()
    loss.backward()
    assert t.grad is not None
    assert torch.equal(t.grad, torch.full((2, 2), 2.0))


def test_a_labelled_computation_comes_back_labelled():
    u = pp.tensor([1.0, 2.0, 3.0], name="velocity", units="m/s")
    for result in (u * 2.0, u + 1.0, u.sum(), u.mean(), u.detach(), u.clone(), torch.stack([u, u]),
                   torch.relu(u - 5.0)):
        assert isinstance(result, pp.PhysicsTensor), type(result)
        assert result.units == "m/s" and result.name == "velocity"


def test_the_units_have_to_agree_to_be_added_or_compared():
    u = pp.tensor([1.0, 2.0], name="velocity", units="m/s")
    p = pp.tensor([1.0, 2.0], name="pressure", units="Pa")
    for op in (lambda: u + p, lambda: u - p, lambda: u < p, lambda: u >= p):
        with pytest.raises(UnitError, match="dimensions differ"):
            op()
    same = pp.tensor([3.0, 4.0], name="wind", units="m/s")
    assert torch.allclose((u + same), torch.tensor([4.0, 6.0]))
    assert (u + same).units == "m/s", "the result carries the unit of what it was made from"
    assert torch.allclose(u + 1.0, torch.tensor([2.0, 3.0])), "a number means 'in the same unit'"
    assert torch.allclose(u + torch.tensor([1.0, 1.0]), torch.tensor([2.0, 3.0])), "an unlabelled tensor too"


def test_a_product_of_two_dimensioned_tensors_is_refused_not_guessed():
    """m/s * Pa = ? A unit algebra would answer; this class would invent one, so it says what to do instead."""
    u = pp.tensor([2.0, 4.0], name="velocity", units="m/s")
    p = pp.tensor([3.0, 5.0], name="pressure", units="Pa")
    with pytest.raises(UnitError, match="unit of a product"):
        u * p
    with pytest.raises(UnitError, match="unit of a product"):
        u / p
    assert torch.allclose(u * 2.0, torch.tensor([4.0, 8.0])), "a plain number is not a dimensioned operand"
    raw = pp.tensor(u.without_labels() * p.without_labels(), name="power", units="W")
    assert torch.allclose(raw, torch.tensor([6.0, 20.0])) and raw.units == "W"


def test_a_conversion_stays_inside_the_dimension():
    u = pp.tensor([1.0, 10.0], name="velocity", units="m/s")
    kmh = u.convert("km/h")
    assert kmh.units == "km/h"
    assert torch.allclose(kmh.without_labels(), torch.tensor([3.6, 36.0]), rtol=1e-6)
    assert u.convert("m/s").units == "m/s"
    with pytest.raises(UnitError, match="dimensions differ"):
        u.convert("Pa")
    with pytest.raises(UnitError, match="no unit"):
        pp.tensor([1.0], name="bare").convert("m/s")


def test_an_unlabelled_tensor_is_just_a_tensor():
    t = pp.tensor([1.0, 2.0])
    assert not t.labelled() and t.unit is None
    assert torch.allclose(t * 3.0, torch.tensor([3.0, 6.0]))
    assert "PhysicsTensor(tensor([1., 2.])" in repr(t)


def test_the_labels_are_metadata_not_a_type_system():
    """Stated in the docstring, so it is asserted: a trip through numpy loses them, and that is not a bug."""
    u = pp.tensor([1.0, 2.0], name="velocity", units="m/s")
    assert not hasattr(u.numpy(), "units")
    assert isinstance(u.without_labels(), torch.Tensor) and not isinstance(u.without_labels(), pp.PhysicsTensor)

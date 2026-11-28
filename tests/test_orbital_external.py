import numpy as np
import pytest
from pinneapple_simulation.orbital_intent import propagate
from pinneapple_simulation.orbital_external import propagate_pyastrodyn
from pinneapple_veriphysics.orbital_gates import verify_orbit

def test_pyastrodyn():
    pytest.importorskip("pyastrodyn_edu")
    t = np.linspace(10, 16, 101)
    y0 = [1, 0, 0, 0, 1, 0]
    np.testing.assert_allclose(propagate_pyastrodyn(1,y0,t), propagate(1,y0,t), atol=1e-8)

def test_gates():
    t = np.linspace(0, 6, 101)
    y = propagate(1,[1,0,0,0,1,0],t)
    opts = dict(energy_tolerance=1e-8, momentum_tolerance=1e-8,
                position_tolerance=1e-8, velocity_tolerance=1e-8)
    report = verify_orbit(1,y,reference=y,**opts)
    assert report.ok
    assert verify_orbit(1,y,**opts).get("orbital_position").passed is None
    bad = y.copy()
    bad[-1,3] += 1
    assert not verify_orbit(1,bad,reference=y,**opts).ok

import numpy as np
import pytest
from pinneapple_simulation.orbital_benchmark import kepler_reference, compare_solvers
from pinneapple_simulation.orbital_intent import propagate

@pytest.mark.parametrize("e",[0.,0.35,0.8])
def test_analytic(e):
    t=np.linspace(10,10+4*np.pi,401)
    ref=kepler_reference(1,1,e,t)
    np.testing.assert_allclose(ref[0],ref[-1],atol=1e-12)
    np.testing.assert_allclose(propagate(1,ref[0],t),ref,atol=2e-7)

@pytest.mark.parametrize("e",[0.,0.35,0.8])
def test_crosscheck(e):
    pytest.importorskip("pyastrodyn_edu")
    reports=compare_solvers(1,1,e,np.linspace(0,4*np.pi,401),
        energy_tolerance=2e-7,momentum_tolerance=2e-7,
        position_tolerance=2e-7,velocity_tolerance=2e-7)
    assert all(r.ok for r in reports.values())

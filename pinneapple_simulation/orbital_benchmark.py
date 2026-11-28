"""Independent elliptic Kepler reference and cross-backend verification."""
import numpy as np
from scipy.optimize import brentq
from .orbital_intent import propagate
from .orbital_external import propagate_pyastrodyn
from pinneapple_veriphysics.orbital_gates import verify_orbit

def kepler_reference(mu, a, e, times):
    """Periapsis initial state in XY plane, SI units; elliptic orbits only."""
    t = np.asarray(times, dtype=float)
    if not np.isfinite([mu,a,e]).all() or mu <= 0 or a <= 0 or not 0 <= e < 1:
        raise ValueError("Invalid elliptic orbit")
    if t.ndim != 1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError("Invalid times")
    n = np.sqrt(mu/a**3)
    m = np.remainder(n*(t-t[0]), 2*np.pi)
    E = np.array([brentq(lambda x: x-e*np.sin(x)-q, 0, 2*np.pi, xtol=1e-14) for q in m])
    c,s = np.cos(E),np.sin(E)
    b = np.sqrt(1-e*e)
    rate = n/(1-e*c)
    z = np.zeros_like(t)
    return np.column_stack((a*(c-e),a*b*s,z,-a*s*rate,a*b*c*rate,z))

def compare_solvers(mu,a,e,times,*,energy_tolerance,momentum_tolerance,
                    position_tolerance,velocity_tolerance):
    ref = kepler_reference(mu,a,e,times)
    native = propagate(mu,ref[0],times)
    external = propagate_pyastrodyn(mu,ref[0],times)
    opts = dict(energy_tolerance=energy_tolerance,momentum_tolerance=momentum_tolerance,
                position_tolerance=position_tolerance,velocity_tolerance=velocity_tolerance)
    reports = {
        "native_vs_analytic": verify_orbit(mu,native,reference=ref,model="native",**opts),
        "external_vs_analytic": verify_orbit(mu,external,reference=ref,model="pyastrodyn",**opts),
        "external_vs_native": verify_orbit(mu,external,reference=native,model="crosscheck",**opts),
    }
    return reports

"""Optional PyAstroDyn-edu adapter; install pyastrodyn-edu on Python >=3.11."""
import numpy as np
from .orbital_intent import propagate

def pyastrodyn_backend(mu, initial, times):
    from pyastrodyn_edu.twobody import propagate_numerical
    r, v = propagate_numerical(initial[:3], initial[3:], times-times[0], mu)
    return np.concatenate((r, v), axis=1)

def propagate_pyastrodyn(mu, initial, times):
    return propagate(mu, initial, times, backend=pyastrodyn_backend)

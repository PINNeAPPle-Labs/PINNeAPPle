"""The pin-fin closed forms (Incropera, Table 3.4, case A): base condition, the fin equation by finite differences,
the convective tip condition, and the heat balance (base conduction = the fin's convective loss)."""
import math

import numpy as np
from scipy.integrate import trapezoid

from pinneapple_physics.closed_form.heat import pin_fin_heat, pin_fin_theta

K, D, L, H = 16.0, 0.005, 0.05, 25.0


def test_profile_satisfies_the_fin_equation_and_its_conditions():
    xi = np.linspace(0, 1, 4001)
    th = pin_fin_theta(xi, H, K, D, L)
    assert th[0] == 1.0
    dx = L / (len(xi) - 1)
    m2 = 4 * H / (K * D)
    d2 = (th[2:] - 2 * th[1:-1] + th[:-2]) / dx ** 2
    assert np.max(np.abs(d2 - m2 * th[1:-1])) < 1e-4 * m2
    dtip = (3 * th[-1] - 4 * th[-2] + th[-3]) / (2 * dx)
    assert abs(-K * dtip - H * th[-1]) < 1e-3 * H


def test_heat_equals_the_convective_loss():
    xi = np.linspace(0, 1, 20001)
    th = pin_fin_theta(xi, H, K, D, L)
    lateral = H * math.pi * D * L * trapezoid(th, xi) * 55.0
    tip = H * math.pi * D ** 2 / 4 * th[-1] * 55.0
    assert math.isclose(pin_fin_heat(H, K, D, L, 80.0, 25.0), lateral + tip, rel_tol=1e-6)

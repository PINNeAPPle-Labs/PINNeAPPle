"""Physical constants of electromagnetics, SI, CODATA 2022 (NIST, https://physics.nist.gov/cuu/Constants/).

Since the 2019 SI redefinition ``c``, ``e``, ``h`` and ``k_B`` are exact and ``mu0`` is measured
(``mu0 = 4 pi alpha hbar / (e^2 c)``); ``eps0 = 1 / (mu0 c^2)`` follows from it. ``mu0`` differs from the
pre-2019 value ``4 pi 1e-7`` by about 5e-10 relative, far below any discretisation error in this package.
"""
from __future__ import annotations

import math

__all__ = ["C0", "MU0", "EPS0", "ETA0", "Q_E", "M_E", "M_P", "K_COULOMB"]

#: speed of light in vacuum, m/s (exact)
C0 = 299_792_458.0
#: vacuum magnetic permeability, N/A^2 (CODATA 2022, relative uncertainty 1.6e-10)
MU0 = 1.25663706127e-6
#: vacuum electric permittivity, F/m
EPS0 = 1.0 / (MU0 * C0 * C0)
#: impedance of free space, ohm
ETA0 = MU0 * C0
#: elementary charge, C (exact)
Q_E = 1.602176634e-19
#: electron mass, kg (CODATA 2022)
M_E = 9.1093837139e-31
#: proton mass, kg (CODATA 2022)
M_P = 1.67262192595e-27
#: Coulomb constant 1 / (4 pi eps0), N m^2 / C^2
K_COULOMB = 1.0 / (4.0 * math.pi * EPS0)

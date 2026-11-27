- **`pinneapple_physics.maritime`** (#364): the sea state, the vessel's answer to it, the mooring that holds it, and
  the operability window with a measured confidence on it.
  - *the sea state*: JONSWAP and Pierson-Moskowitz with the significant height fixed by the spectrum's own integral, the
    sea state read back off a measured spectrum, a fit of the shape to a measurement, and an evolution with a horizon --
    the swell decays faster at high frequency (so `Tp` rises while `Hs` falls, which persistence cannot say), the wind
    sea is carried with the wind and only decays when it exceeds what that wind can hold. Two coefficients, both fitted
    on one half of a record and scored on the other.
  - *the section*: added mass, damping and excitation from the **two-dimensional free-surface problem solved on a graded
    grid** -- complex-frequency sponge, Neumann hull, pressure integration -- not a table. Heave and sway to 1 in the
    long-wave limit, roll to 1.06, oscillating damping that is positive everywhere, and a roll resonance that lands where
    `C44/(I44 + A44)` says with the `A44` the solve measured (0.48 of the roll inertia, which the `0.35 B` rule of thumb
    does not contain). Pitch comes from the longitudinal section and says so.
  - *the mooring*: the quasi-static catenary with bottom contact, the tangency point where the line lifts off, tension
    that rises faster than the offset, and an equilibrium solve so that losing a line is compared at the same
    environmental load rather than the same offset (598 kN against 512 kN for two lines gone).
  - *the residual*: Ochi's threshold on the relative velocity at the bow, a tension step that is called the mooring, and
    a redundant accelerometer pair so that an instrument that lies is told apart from a hull that was hit -- five
    scenarios, five correct diagnoses, on records with the noise of the instruments in them.
  - *the window*: `Criteria.allows`, contiguous windows, and a confidence that is the measured fraction of hours the
    forecast's verdict matched the observation's, at each lead time.
- **`pinneapple_physics.maritime.ndbc`**: NOAA NDBC's standard meteorological, spectral density and swell/wind-sea
  files parsed without a client library, with the missing-value marker kept missing and the compressed slice the example
  ships written by `build_dataset.py` (station 41010, 45 days, 2080 measured spectra).
- **waves against a buoy, measured**: on 22 days the coefficients never saw, `Tp` beats persistence by 17 % at 12 h and
  10 % at 24 h; `Hs` does not (3.6 % at 24 h, nothing at 6 h) -- a single buoy's local spectrum carries very little
  about tomorrow's height, and the guide says so with the numbers. The operability verdict the forecast gives matches the
  observed one on 93 % of hours at 1 h and 76 % at 24 h.

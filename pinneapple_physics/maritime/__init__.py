"""Maritime and offshore (#364): the sea state, how the vessel answers it, the mooring that holds it, and a window of
operation with the confidence of the forecast written on it.

The chain, in the order an operability window is built:

* :mod:`.ndbc` -- a NOAA NDBC buoy's files read into arrays: the significant height, the peak period, the wind, the
  measured spectral density and NDBC's own swell/wind-sea split;
* :mod:`.spectrum` -- JONSWAP and Pierson-Moskowitz, the sea state read back off a spectrum, the fit of a spectrum to a
  measurement, and the time series a spectrum implies;
* :mod:`.seastate` -- the evolution of a measured spectrum with a horizon: the swell decays faster at higher frequency,
  the wind sea relaxes toward what the wind can hold, and the group velocity says how far each band travelled;
* :mod:`.rao` -- the answer of a section to a regular wave, from the two-dimensional free-surface problem solved on a
  grid: added mass, damping, excitation, and the transfer functions of heave, sway, roll and -- from the longitudinal
  section -- pitch;
* :mod:`.motion` -- the motion in time out of a spectrum and the transfer functions, the vertical acceleration at a
  point, and the relative velocity at the bow;
* :mod:`.mooring` -- the quasi-static catenary of a mooring line, and what a lost line does to the fairlead load;
* :mod:`.confidence` -- the residual of the motion model against the instruments: a regime change (a slam, a line gone)
  told apart from a sensor that is lying;
* :mod:`.operability` -- the criterion, the window it opens and closes, and the confidence of that verdict at the
  horizon it was issued for.
"""
from __future__ import annotations

from . import confidence, mooring, motion, ndbc, operability, rao, seastate, spectrum
from .confidence import Diagnosis, Event, evaluate
from .mooring import Line, Mooring, catenary, fairlead_tension
from .motion import Motion, Point, response, slamming_threshold
from .ndbc import BuoyRecord, load
from .operability import Criteria, Window, decision_skill, opening, windows
from .rao import Section, TransferFunction, pitch_transfer_function, solve, sweep
from .seastate import Forecast, SeaStateModel, arrival_hours, forecast, inspected_radius, skill

__all__ = ["ndbc", "spectrum", "seastate", "rao", "motion", "mooring", "confidence", "operability",
           "BuoyRecord", "load", "SeaStateModel", "Forecast", "forecast", "skill", "arrival_hours",
           "inspected_radius", "Section", "TransferFunction", "solve", "sweep", "pitch_transfer_function", "Motion",
           "Point", "response", "slamming_threshold", "Line", "Mooring", "catenary",
           "fairlead_tension", "Diagnosis", "Event", "evaluate", "Criteria", "Window", "windows", "opening",
           "decision_skill"]

"""Positional astronomy and astrophysics for the PINNeAPPle observatory.

An independent NumPy implementation, validated against USNO NOVAS, JPL Horizons and skyfield
(``tests/test_astro_accuracy.py``):

* :mod:`.timescales`  UTC / TAI / TT / TDB / UT1, sidereal time
* :mod:`.spk`         JPL SPK kernel reader (DE421 and later, public domain ephemerides)
* :mod:`.frames`      bias / precession (IAU 2006) / nutation, geodetic, horizon, ecliptic
* :mod:`.astrometry`  apparent places (light time, deflection, aberration), refraction, airmass, extinction
* :mod:`.phenomena`   rise / transit / set, twilight, phases, magnitudes, elongations, eclipses, transits
* :mod:`.satellites`  TLE + SGP4, look angles, visible passes and magnitudes
* :mod:`.stars`       Yale Bright Star Catalogue, stellar luminosity / temperature / radius, Planck spectra
* :mod:`.orbits`      Keplerian elements, propagation, light time and Doppler
* :mod:`.nbody`       symplectic and Runge-Kutta N-body integrators with conservation diagnostics
* :mod:`.surrogate`   physics-informed neural orbit surrogate with conformal bands and a trust verdict
* :mod:`.lensing`     black-hole lensing of the real sky, Einstein radius, shadow size vs the EHT
* :mod:`.journey`     a trip to a target: heliocentric / galactic positions, distance, light time, magnitude vs distance
* :mod:`.observing`   observing sessions: visibility, air mass, extinction, sky brightness, optics, slews, field rotation
* :mod:`.facts`       info cards in English and Portuguese: numbers with sources, curiosities, history, quiz

    from pinneapple_physics.astro import Observer, Time, apparent
    p = apparent("jupiter", Time.now(), Observer(-23.55, -46.63, 760))
"""
from .astrometry import (
                        Apparent,
                        Observer,
                        airmass,
                        apparent,
                        extinction_mag,
                        refraction_deg,
                        star_apparent,
)
from .ephemeris import AU_KM, BODIES, PLANETS, Ephemeris, default_ephemeris
from .frames import angular_separation, position_angle
from .phenomena import (
                        lunar_eclipses,
                        lunar_phase_events,
                        magnitude,
                        moon_phase,
                        planet_events,
                        planetary_transits,
                        rise_transit_set,
                        seasons,
                        sky_state,
                        solar_eclipses,
                        twilight_events,
)
from .timescales import Time, gast_hours

__all__ = ["Apparent", "Observer", "airmass", "apparent", "extinction_mag", "refraction_deg", "star_apparent",
           "AU_KM", "BODIES", "PLANETS", "Ephemeris", "default_ephemeris", "angular_separation", "position_angle",
           "lunar_eclipses", "lunar_phase_events", "magnitude", "moon_phase", "planet_events", "planetary_transits",
           "rise_transit_set", "seasons", "sky_state", "solar_eclipses", "twilight_events", "Time", "gast_hours"]

# Facts dataset (`pinneapple_physics.astro.facts`), version 1.0

Info cards for a general audience, in English and Portuguese: what an object is, key numbers with everyday
comparisons, curiosities, a short history, missions and a quiz. Read with `pinneapple_physics.astro.facts`
(`card`, `search`, `quiz`, `did_you_know`, `tonight`, `validate`).

| File | Content | Numbers from |
|---|---|---|
| `sources.json` | every source a number or a curiosity cites (title, publisher, URL, licence) | — |
| `solar_system.json` | the Sun, the 8 planets, the Moon, Pluto, Ceres, Eris, Haumea, Makemake, Io, Europa, Ganymede, Callisto, Titan, Enceladus, Triton | NASA NSSDCA fact sheets (planets, Sun, Moon, Pluto, Jovian / Saturnian / Neptunian satellites), JPL SBDB; Eris, Haumea, Makemake sizes from stellar occultations (Sicardy 2011, Ortiz 2012, 2017) |
| `stars.json` | the brightest stars, down to V = 2.05 | Hipparcos new reduction parallaxes (van Leeuwen 2007); V, B−V and spectral types from the Yale BSC in `../bsc5.json.gz`; temperature, luminosity and radius computed by `stars` |
| `constellations.json` | who introduced each of the 88 constellations | IAU; history of the constellations |
| `constellation_areas.json` | area of each constellation (deg²), total 41 252.9 | computed by `scripts/astro/build_facts.py` from the IAU boundaries (Delporte 1930) in d3-celestial (BSD-3) |
| `messier_catalog.json` | the 110 Messier objects: J2000 position, type, V magnitude, size, NGC number, constellation | d3-celestial `messier.json` (BSD-3, O. Frohn); constellation from the IAU boundaries |
| `messier.json` | common names, distances (with their source), hand-written text for some objects | distances: EHT 2019 (M87), Menten 2007 (M42), Gaia 2018 (M44, M45), Trimble 1973 (M1), galaxy distances (Freedman & Madore; McQuinn 2016), Harris 2010 (globular clusters) |
| `satellites.json` | ISS, Hubble, Tiangong | NASA, CMSA |

Rules: every number has a `source` (a key of `sources.json`); text is written for this project in both languages,
never copied; comparisons and derived curiosities are computed by the library from the numbers and cite `computed`
or the number's own source. `tests/test_astro_facts.py` checks that every card has a source per number, text in
both languages and three to five curiosities; that the planet numbers are the NASA fact sheet's and agree with the
library's independent data (DE421 masses and orbits, IAU radii); and that the constellation areas add up to the
whole sky and match the IAU's to 0.5 %.

Precision: Hipparcos distances of the brightest stars are good to a few per cent within ~300 light-years and
uncertain beyond (Deneb, Alnilam, Wezen); the Messier distances are rounded literature values; the luminosities and
radii of stars are black-body estimates (`stars.star_properties`), within ~30 % for cool supergiants.

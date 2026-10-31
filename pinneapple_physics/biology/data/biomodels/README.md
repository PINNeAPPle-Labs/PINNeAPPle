# BioModels entries shipped with PINNeAPPle

Six curated BioModels entries in SBML Level 2, used by `pinneapple_physics.biology.biomodel(...)`, the SBML tests
and the guide. Source: the BioModels mirror github.com/sys-bio/temp-biomodels at commit
6a09daf46af1bb89e4857436b623a7b8720863ad, `final/<id>/<id>_url.xml` (www.ebi.ac.uk/biomodels itself is refused by
the network of the session that added them). Licence: CC0 1.0 (the mirror's LICENSE, as BioModels). The only change:
`<annotation>` and `<notes>` blocks removed (MIRIAM / RDF metadata and HTML notes); the model content is untouched.

| File | Model |
|---|---|
| BIOMD0000000005.xml | Tyson 1991, cell cycle, 6 variables |
| BIOMD0000000010.xml | Kholodenko 2000, MAPK cascade with negative feedback (oscillations) |
| BIOMD0000000012.xml | Elowitz and Leibler 2000, repressilator |
| BIOMD0000000021.xml | Leloup and Goldbeter 1999, Drosophila circadian clock |
| BIOMD0000000028.xml | Markevich et al. 2004, MAPK double phosphorylation (random, elementary steps) |
| BIOMD0000000035.xml | Vilar et al. 2002, circadian oscillator (deterministic) |

`roadrunner_reference.json` holds each model's states on t = 0, 5, ..., 100 computed by libRoadRunner 2.11.0
(CVODE, relative tolerance 1e-10, absolute 1e-12) from these files; the tests compare the SciPy integration of the
imported ODE system with it when libRoadRunner is not installed, and with a live libRoadRunner run when it is.

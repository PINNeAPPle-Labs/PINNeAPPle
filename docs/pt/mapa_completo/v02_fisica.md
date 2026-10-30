# Volume 2 — Física: problemas, equações e ciência

> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a partir do inventário do commit `e53cdcfb04`.

Como um problema físico é descrito e resolvido (`PhysicalProblem`, `pp.solve`), as equações, os presets industriais, o desenho de problemas, os experimentos por área, a condensação e as consultas físicas.

## Sumário

1. [`pinneapple_physics`](#1-pinneapple_physics)
2. [`pinneapple_presets`](#2-pinneapple_presets)
3. [`pinneapple_problemdesign`](#3-pinneapple_problemdesign)
4. [`pinneapple_science`](#4-pinneapple_science)
5. [`pinneapple_condense`](#5-pinneapple_condense)
6. [`pinneapple_query`](#6-pinneapple_query)

## 1. `pinneapple_physics`

### Objetivo

`pinneapple_physics` é o pacote onde um problema físico é definido e resolvido. Ele junta a especificação de EDPs (`pde_environment`: `ProblemSpec`, condições de contorno e iniciais, presets como Navier-Stokes, calor, Burgers, elasticidade, modelos RANS e uma base de famílias de EDP), o compilador de PINN (`pinn_solver`), o compilador SymPy para autograd (`symbolic_pde`) e modelos de forma fechada (`closed_form`). O `__init__` reexporta esses nomes e mantém apelidos antigos (`environment`, `pinn`, `symbolic`) por compatibilidade.

O caminho principal é: montar um `ProblemSpec` (à mão, com `define_problem` ou com `get_preset`), compilar as perdas com `compile_physics`/`compile_problem` e treinar com `solve_pde`, um laço Adam próprio que amostra pontos de colocação a cada passo. `pipeline(task)` junta preset, modelo do registro de `pinneapple_neural` e `solve_pde` numa chamada e devolve um `PhysicsPipeline` com `.predict()`. Condições com `selector_type="tag"` exigem geometria real (por exemplo `STLDomainBatchBuilder` de `pinneapple_design`); sem ela, `solve_pde` levanta `TagConditionsUnresolved` antes de treinar. Os pontos de entrada `solve`/`compare` (de `solving.py`) e `Experiment` também são reexportados: é por aqui que `pp.solve(problem, "analytic")` usa as soluções de `closed_form`.

Além do núcleo PINN, o pacote traz camadas que julgam previsões pela física, sem solução de referência: `darcy_checks` e `maxwell_checks` (verificações usadas pelo comando `pinneapple check` e por relatórios de confiança APPROVED/REVIEW/REJECT), `physics_audit` (premissas de cada modelo, números adimensionais, monitor de validade, atribuição de falhas, recomendações e roteamento) e `multiscale` (filtros LES, fechamentos, resíduos multiescala e seleção de modelos sensível à escala). `non_hermitian` cobre dímeros PT-simétricos e pontos excepcionais; `weather` é um caso de uso de previsão global com ERA5 do WeatherBench2.

Esta parte do mapa cobre o `__init__` do pacote e os submódulos `closed_form`, `darcy_checks`, `maxwell_checks`, `multiscale`, `non_hermitian`, `physics_audit`, `pinn_solver` e `weather`. Os demais submódulos (`pde_environment`, `symbolic_pde`, `solving`, `ensemble`, etc.) são descritos nas outras partes.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 166 | 43.959 | 169 | 763 | 1.075 (105 arquivos) | 52 |

Pasta de testes dedicada: `tests/pinneapple_physics` (48 funções de teste).

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `advection_diffusion_1d` |  | 1 | 1 | 13 |
| `astro` |  | 24 | 216 | 71 |
| `blackhole` |  | 12 | 60 | 19 |
| `closed_form` | Modelos de engenharia em forma fechada (aletas, vigas e fadiga, difusão em membrana, Helmholtz, dissipadores, PCB, escoamento em tubos, bocal de foguete, Burgers exato, etc.) para verdade de referência rápida sem resolver EDP. | 19 | 91 | 103 |
| `codegen` |  | 3 | 4 | 13 |
| `compile_api` |  | 1 | 3 | 5 |
| `conservation` |  | 1 | 3 | 6 |
| `darcy_checks` | Verificações físicas sem referência de campos de Darcy 2-D (resíduo de volumes finitos, limites certificados de erro, balanço de fluxo, contorno, envelope de coeficientes) e decisão de implantação. | 1 | 31 | 26 |
| `deepxde_bridge` |  | 2 | 4 | 9 |
| `ensemble` |  | 1 | 9 | 68 |
| `ensemble_viz` |  | 1 | 1 | 3 |
| `experiment` |  | 1 | 3 | 0 |
| `maxwell_checks` | Verificações das equações de Maxwell e das leis de balanço (Gauss, Faraday, Ampère-Maxwell, continuidade, Poynting, PEC, reciprocidade, passividade) em campos de qualquer origem. | 1 | 19 | 51 |
| `metrics` |  | 1 | 11 | 36 |
| `multiscale` | Física multiescala para ensembles adaptativos: filtros LES, solver de vorticidade 2-D, fechamentos de subfiltro, resíduos multiescala, consistência entre escalas e roteamento sensível à escala. | 11 | 76 | 45 |
| `non_hermitian` | Dímeros e cadeias PT-simétricos, pontos excepcionais e a verificação simbólica de leis de sintonia do EP contra um teorema de impossibilidade. | 4 | 19 | 8 |
| `online_learning` |  | 1 | 4 | 0 |
| `paper_problems` |  | 1 | 7 | 5 |
| `paper_problems_xtfc` |  | 1 | 4 | 10 |
| `pcb` |  | 3 | 10 | 16 |
| `pde_environment` |  | 28 | 154 | 154 |
| `physical_problem` |  | 1 | 3 | 12 |
| `physics_audit` | Auditoria física de modelos: premissas e condições de validade, números adimensionais, monitor de validade sem verdade de referência, atribuição de falhas, recomendações, grafo de conhecimento e roteamento. | 8 | 65 | 37 |
| `pinn_solver` | Compilador de PINN: traduz um `ProblemSpec` em funções de perda, mais DoMINO (decomposição de domínio), PINN estocástico, fábrica SymPy legada e leitura de dados UPD. | 19 | 41 | 121 |
| `solid` |  | 2 | 6 | 8 |
| `solving` |  | 1 | 12 | 71 |
| `symbolic_pde` |  | 6 | 24 | 17 |
| `tracking` |  | 1 | 3 | 3 |
| `tribology` |  | 2 | 3 | 20 |
| `weather` | Caso de uso de previsão do tempo global: dados ERA5 do WeatherBench2, U-Net esférica, treino por rollout, métricas de habilidade, eventos extremos e animações. | 7 | 38 | 8 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_physics`

Ponto de entrada do pacote: reexporta `pde_environment`, `pinn_solver`, `symbolic_pde`, `closed_form`, `solve`/`compare` e `Experiment`, mantém apelidos antigos e define os auxiliares `define_problem`, `compile_physics`, `solve_pde` e `pipeline`.

<small>`pinneapple_physics/__init__.py` · 604 linhas · testes: 206</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TagConditionsUnresolved` | classe | `()` | Erro (`ValueError`) levantado por `solve_pde` quando há condições `selector_type="tag"` sem pontos e máscaras fornecidos pelo chamador, para não treinar sem essas condições. |
| `PhysicsPipeline` | classe | `(model, spec, history)` | Resultado de `pipeline`: guarda modelo treinado, `ProblemSpec` e histórico, e oferece `.predict(x)` que devolve um array numpy. |
| `compile_physics` | função | `(spec, **kwargs)` | Atalho para `compile_problem(spec, **kwargs)`: compila o `ProblemSpec` em perdas PINN ponderadas. |
| `identify` | função | `(description)` | Atalho para `identify_pde`: identifica a família de EDP a partir de uma descrição em linguagem natural. |
| `define_problem` | função | `(preset=None, **spec_kwargs) -> 'ProblemSpec'` | Monta um `ProblemSpec`: a partir de um preset nomeado (sobrescrevendo atributos) ou diretamente dos argumentos. |
| `solve_pde` | função | `(spec, model, *, epochs=5000, device='cpu', lr=0.001, n_collocation=2048, seed=0, weights…)` | Treina um modelo num `ProblemSpec` com um laço Adam próprio, amostrando colocação e pontos de condição (`all`/`callable`) a cada passo; aceita pontos fixos `x_col` e máscaras de tags. |
| `pipeline` | função | `(task, *, architecture='modified_mlp', hidden_dim=64, n_layers=4, epochs=2000, device='cp…)` | Uma chamada para tarefa nomeada: `get_preset` + `ModelRegistry.build` + `solve_pde`, devolvendo um `PhysicsPipeline`. |

#### `pinneapple_physics.advection_diffusion_1d`

Periodic 1-D advection-diffusion u_t + c u_x = nu u_xx on [0, 2 pi): exact solution, classical finite-difference schemes and the PDE residual of any predicted space-time field.

<small>`pinneapple_physics/advection_diffusion_1d.py` · 80 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AdvectionDiffusion1D` | classe | `(nx, nt, t_end)` |  |

#### `pinneapple_physics.astro`

Positional astronomy and astrophysics for the PINNeAPPle observatory.

<small>`pinneapple_physics/astro/__init__.py` · 54 linhas · testes: 71</small>

#### `pinneapple_physics.astro.astrometry`

Apparent places of the Sun, Moon and planets (and catalogue stars) for an observer on Earth.

<small>`pinneapple_physics/astro/astrometry.py` · 194 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Observer` | classe | `(lat, lon, elevation_m, pressure_mbar, temperature_c, name)` | A place on Earth: geodetic latitude/longitude (deg, east positive), height (m), and the weather used by the refraction model. |
| `Apparent` | classe | `(body, ra, dec, ra_astrometric, dec_astrometric, distance_km, light_time_s, range_rate_km…)` |  |
| `refraction_deg` | função | `(alt_true_deg, pressure_mbar=1010.0, temperature_c=10.0)` | Atmospheric refraction (deg) for a true altitude, Saemundsson (1986) with the usual P/T scaling. Zero below -1 deg, where the formula is not meaningful. |
| `airmass` | função | `(alt_apparent_deg)` | Relative air mass (Kasten & Young 1989), finite down to the horizon; ``inf`` below it. |
| `extinction_mag` | função | `(alt_apparent_deg, k_v=0.25)` | Atmospheric extinction (mag) at V for an extinction coefficient ``k_v`` (mag/airmass; 0.2-0.3 typical). |
| `apparent` | função | `(body, t, observer=None, eph=None, *, deflection=True, refraction=True) -> Apparent` | Apparent place of a solar-system ``body`` at ``t`` (scalar or array), geocentric if ``observer`` is None. |
| `star_apparent` | função | `(ra_j2000, dec_j2000, t, observer=None, *, pm_ra_mas=0.0, pm_dec_mas=0.0, parallax_mas=0.…)` | Apparent places of catalogue stars (ICRS J2000 epoch 2000.0) at a single instant ``t``. |

#### `pinneapple_physics.astro.conditions`

Observing conditions for tonight from open numerical weather prediction (Open-Meteo, CC BY 4.0).

<small>`pinneapple_physics/astro/conditions.py` · 126 linhas · testes: 25</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fetch_ensemble` | função | `(lat, lon, model='icon_seamless', hours=48, timeout=15.0)` |  |
| `members` | função | `(resp, var) -> np.ndarray` | (n_members, n_hours) array for ``var`` (the control run and every ``var_memberNN`` column). |
| `summarize` | função | `(resp, clear_threshold=30.0) -> dict` |  |
| `calibrate` | função | `(p_forecast, observed_clear, bins=10) -> dict` | Reliability of past clear-sky probabilities against what was observed, and an isotonic-style correction map: for each probability bin, the observed frequency (needs a history of forecasts and observations). |
| `fetch_cloud_forecast` | função | `(lat, lon, days=3, timeout=8.0) -> dict` | Hourly total cloud cover (%) for the next ``days`` from Open-Meteo's forecast API (no key; CC BY 4.0): {"time": [ISO UTC], "cloud_cover": [%], "source": ...}. Raises on network errors (callers fall back). |
| `cloud_at` | função | `(forecast, iso_utc) -> float \| None` | Cloud cover (%) at a time, linearly interpolated in the hourly forecast; None outside it. |
| `sky_view` | função | `(mode, forecast, iso_utc) -> dict` | What the sky lets through for an observing session. ``mode`` "perfect" removes the weather (any target, any night); "real" uses the cloud-cover forecast and hides the view at or above ``CLOUDY_PCT``. Without a forecast "real" falls back to "perfect" and says so. |

#### `pinneapple_physics.astro.conjunction`

Conjunction screening: closest approach in Clohessy-Wiltshire relative motion, collision probability, J2 orbits.

<small>`pinneapple_physics/astro/conjunction.py` · 160 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ClosestApproach` | classe | `(t_tca_s, position_km, velocity_km_s, miss_distance_km, relative_speed_km_s, perpendicula…)` |  |
| `cw_mean_motion` | função | `(a_km, mu_km3_s2=398600.4418) -> float` | Mean motion n = sqrt(mu / a^3) of the (circular) reference orbit, rad/s. |
| `cw_closest_approach` | função | `(n, state0, duration_s, n_samples=2000) -> ClosestApproach` | Closest approach over [0, duration_s] for the CW deputy state ``state0 = (x0, y0, z0, vx0, vy0, vz0)`` (km, km/s) about a reference orbit of mean motion ``n`` (rad/s). |
| `collision_probability_2d` | função | `(xm, zm, sigma_x, sigma_z, r_c) -> float` | Probability mass of N((xm, zm), diag(sigma_x^2, sigma_z^2)) inside the disk x^2 + z^2 <= r_c^2. |
| `collision_probability_centered` | função | `(r_c, sigma) -> float` | Closed form for a centred isotropic Gaussian: 1 - exp(-r_c^2 / (2 sigma^2)) (Rayleigh CDF). |
| `screen_cw_conjunction` | função | `(n, state0, *, duration_s=None, n_samples=2000, hard_body_radius_km=0.01, sigma_km=1.0) -…)` | Closest approach + collision probability for one CW encounter (one reference period by default). |
| `two_body_j2_rhs` | função | `(mu, J2, Re)` | Right-hand side f(t, [x, y, z, vx, vy, vz]) of two-body motion plus the J2 acceleration. |
| `propagate_two_body_j2` | função | `(state0, t_eval, *, mu=398600.4418, J2=0.00108262668, Re=6378.137, rtol=1e-11, atol=1e-08…)` | States (len(t_eval), 6) of two-body + J2 motion from ``state0`` (km, km/s), DOP853. |

#### `pinneapple_physics.astro.decay`

Orbit decay of low-Earth-orbit satellites by atmospheric drag, with a physics-informed model and its checks.

<small>`pinneapple_physics/astro/decay.py` · 258 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DensityTable` | classe | `(h_km=None, **kw)` | log-density on an altitude grid, interpolated (NumPy and PyTorch) so models can be evaluated fast. |
| `exponential_density` | função | `(h_km)` |  |
| `msis_available` | função | `() -> bool` |  |
| `orbit_mean_density` | função | `(h_km, date='2026-10-13T00:00', f107=150.0, f107a=150.0, ap=15.0, inclination_deg=51.6, n…)` | Orbit-averaged NRLMSIS 2.0 total mass density (kg/m^3) at altitudes ``h_km``: mean over latitudes sampled along the orbit (\|lat\| <= inclination, weighted by the time spent there) and over all local times. |
| `ballistic_from_bstar` | função | `(bstar_per_er) -> float` | B = C_D A / m (m^2/kg) implied by an SGP4 B* (1/Earth radii): B* = B rho0 / 2. |
| `reference_decay` | função | `(h0_km, B, days, table, n=200, reentry_km=150.0)` | Altitude (km) vs time (days) from an adaptive Runge-Kutta integration of the decay ODE. |
| `envelope` | função | `(h0_km, B, days, *, f107=150.0, f107a=None, ap=15.0, date='2026-10-13T00:00', inclination…)` | Monte-Carlo spread of the reference decay over the uncertain inputs: solar flux (log-normal, 20 %), geomagnetic activity (Ap x 0.5..2) and the ballistic coefficient (log-normal, 15 %). Returns t, median, 5 %, 95 % altitude curves. |
| `train_decay_pinn` | função | `(h0_km=415.0, B=0.005, days=60.0, *, f107=150.0, ap=15.0, date='2026-10-13T00:00', inclin…)` | Train the decay PINN and return its report: prediction, reference, envelope, checks and verdict. |
| `sgp4_decay` | função | `(line1, line2, days=30.0, step_min=10.0)` | Mean altitude (km) vs time (days) predicted by SGP4 from a TLE: the geocentric radius averaged over each orbit (removes the eccentricity and J2 oscillations), minus the equatorial radius. |
| `compare_with_sgp4` | função | `(line1, line2, days=30.0, f107=70.0, ap=8.0, B=ISS_B) -> dict` | Decay predicted by this model (NRLMSIS + a physical ballistic coefficient B) next to SGP4's own prediction from the same TLE. SGP4's B* is a fitted drag term for its own power-law atmosphere, not a physical ballistic coefficient (converted with B* = B rho0 / 2 it is typically an order of magnitude … |

#### `pinneapple_physics.astro.ephemeris`

Solar-system ephemeris on top of a JPL DE kernel (public domain), read by :mod:`.spk`.

<small>`pinneapple_physics/astro/ephemeris.py` · 77 linhas · testes: 71</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Ephemeris` | classe | `(path=None)` |  |
| `find_kernel` | função | `() -> str` |  |
| `default_ephemeris` | função | `(path=None) -> Ephemeris` |  |

#### `pinneapple_physics.astro.facts`

Facts and curiosities about the objects in the sky, for a general audience, in English and Portuguese.

<small>`pinneapple_physics/astro/facts.py` · 746 linhas · testes: 30</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sources` | função | `() -> dict` | Every source a number or a curiosity can cite: {key: {title, publisher, url, licence}}. |
| `ids` | função | `() -> dict` | Every object with a card: {id: kind}. Ids: Solar System bodies by name ('saturn'), stars 'star-<HR>', Messier objects 'M31', constellations 'con-<IAU abbreviation>' ('con-Ser1', 'con-Ser2' for Serpens), satellites 'sat-<name>' ('sat-iss'). |
| `resolve` | função | `(ref) -> str` | The card id for an id or an observatory reference: 'saturn', 'body:saturn', 'star:2061', 'HR 2061', 'M 31', 'sat:25544', 'con:Ori'. Raises KeyError if there is no card. |
| `names` | função | `(oid) -> dict` | {en, pt} display names of a card. |
| `numbers` | função | `(oid) -> dict` | The key numbers of a card: {key: {value, unit, source}}. |
| `comparison` | função | `(key, number, lang='en') -> str \| None` | An everyday comparison for a number, computed from it (source: computed). |
| `quiz` | função | `(oid, lang='en', seed=0, n=3) -> list[dict]` | A short multiple-choice quiz from the card's numbers: [{question, options, answer (index), source}]. |
| `card` | função | `(ref, lang='en') -> dict` | The info card of an object in ``lang``: name, kind, summary, numbers (value, unit, label, comparison, source), three to five curiosities with sources, a short history, missions, a quiz and the sources cited. |
| `list_cards` | função | `(kind=None, lang='en') -> list[dict]` | [{id, kind, name}] of every card (or of one kind: 'planet', 'star', 'constellation', 'messier', ...). |
| `search` | função | `(query, lang=None, limit=12) -> list[dict]` | Cards whose name (in English or Portuguese), id or designation matches ``query`` (accents and case ignored); prefix matches first. |
| `did_you_know` | função | `(lang='en', seed=None) -> dict` | A random curiosity from the hand-written cards, for a start-up tip: {id, name, text, source}. |
| `validate` | função | `() -> list[str]` | Problems in the dataset (an empty list means every card has a source per number, text in both languages, three to five curiosities, a history and known sources). |
| `position` | função | `(ref)` | J2000 (ICRS) RA/Dec in degrees of a fixed object (star, Messier object, constellation label point), or None for Solar System bodies and satellites, whose positions come from the ephemeris / TLE. |
| `tonight` | função | `(ref, observer, t, hours=24.0, step_min=10.0, eph=None) -> dict \| None` | How to see an object from ``observer`` in the ``hours`` after ``t``: altitude now, rise, culmination (highest point) and set (the next rise only if it is below the horizon now), the highest altitude while the sky is dark (Sun below -12 deg) and when, and whether it never rises or never sets. None … |

#### `pinneapple_physics.astro.frames`

Reference-frame rotations: frame bias, precession (IAU 2006), nutation (IAU 1980, terms above 5 mas), geodetic observer positions (WGS84), horizon (alt-az) and ecliptic coordinates.

<small>`pinneapple_physics/astro/frames.py` · 202 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fundamental_args` | função | `(T)` | Delaunay arguments D, M, M', F, Omega (rad) for Julian centuries TT since J2000 (Meeus ch. 22). |
| `nutation` | função | `(jd_tt)` | Nutation in longitude and obliquity (rad). |
| `mean_obliquity` | função | `(jd_tt)` |  |
| `rot_x` | função | `(a)` |  |
| `rot_y` | função | `(a)` |  |
| `rot_z` | função | `(a)` |  |
| `precession_matrix` | função | `(jd_tt)` | Mean J2000 -> mean equator/equinox of date: IAU 2006 (P03) angles, IERS Conventions 2010 eq. 5.40. |
| `nutation_matrix` | função | `(jd_tt)` | Mean of date -> true equator/equinox of date. |
| `npb_matrix` | função | `(jd_tt)` | GCRS -> true equator and equinox of date (N P B). |
| `ecliptic_matrix` | função | `(jd_tt=None)` | GCRS -> ecliptic. ``None`` gives the J2000 mean ecliptic (IAU 2006 obliquity 84381.406"). |
| `geodetic_to_itrs` | função | `(lat_deg, lon_deg, height_m=0.0)` | Observer position (km) in the terrestrial frame. |
| `itrs_to_geodetic` | função | `(r)` | Inverse of :func:`geodetic_to_itrs` (Bowring iteration); returns lat, lon (deg), height (km). |
| `radec` | função | `(v)` | Unit or full vectors (..., 3) -> RA (deg, 0..360), Dec (deg), norm. |
| `from_radec` | função | `(ra_deg, dec_deg)` |  |
| `equatorial_to_horizontal` | função | `(ha_rad, dec_rad, lat_rad)` | Hour angle, declination, latitude (rad) -> altitude, azimuth (deg; azimuth from north through east). |
| `horizontal_to_equatorial` | função | `(alt_deg, az_deg, lat_deg)` | Altitude/azimuth -> hour angle (deg) and declination (deg). |
| `angular_separation` | função | `(ra1, dec1, ra2, dec2)` | Great-circle separation (deg), Vincenty form (stable at all angles). |
| `position_angle` | função | `(ra1, dec1, ra2, dec2)` | Position angle (deg, north through east) of point 2 seen from point 1. |

#### `pinneapple_physics.astro.journey`

A journey from your place on Earth to a planet, a moon, the Sun or a star: the geometry of the trip.

<small>`pinneapple_physics/astro/journey.py` · 193 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `galactic` | função | `(v)` | Galactic (l, b) in degrees of an ICRS (J2000) vector. |
| `body_position` | função | `(body, t, eph=None) -> np.ndarray` | Heliocentric ICRS position (km) of a Solar System body at t (geometric, no light time). |
| `observer_position` | função | `(observer, t, eph=None) -> np.ndarray` | Heliocentric ICRS position (km) of a place on Earth at t (Earth from the ephemeris + the site in GCRS). |
| `star_position` | função | `(ra_deg, dec_deg, parallax_mas) -> np.ndarray` | Heliocentric ICRS position (km) of a star from its J2000 direction and parallax. |
| `travel_times` | função | `(distance_km) -> dict` | Seconds needed to cover a distance at the speeds in ``SPEEDS``. |
| `apparent_magnitude_star` | função | `(abs_mag, distance_km)` | m = M + 5 log10(d / 10 pc). |
| `angular_diameter_deg` | função | `(radius_km, distance_km)` |  |
| `plan` | função | `(observer, target, t, standoff=None, n_profile=60, eph=None) -> dict` | The trip from ``observer`` to ``target`` (body name, 'star:<HR>', 'HR <n>' or a facts id) on date ``t``. |
| `scene` | função | `(t, observer=None, max_parallax_err=0.2, eph=None) -> dict` | Everything a 3-D journey draws, heliocentric ICRS km: the planets, Sun and Moon at t (with radii), and the bright stars with a usable parallax (Hipparcos 2007 for the brightest, BSC5 otherwise) with absolute magnitude and B-V; stars without a parallax are listed by direction only. |

#### `pinneapple_physics.astro.lensing`

Gravitational lensing demos on the real sky: a Schwarzschild black hole placed in front of the stars you are looking at, rendered with the geodesic ray tracer of :mod:`pinneapple_physics.blackhole.raytrace`, plus the closed forms (weak-field deflection, Einstein radius, shadow diameter) checked …

<small>`pinneapple_physics/astro/lensing.py` · 366 linhas · testes: 32</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `deflection_angle_rad` | função | `(mass_kg, impact_parameter_m)` | Weak-field light deflection 4GM / (c^2 b). |
| `einstein_radius_rad` | função | `(mass_kg, d_lens_m, d_source_m)` |  |
| `shadow_diameter_rad` | função | `(mass_kg, distance_m)` | Angular diameter of a Schwarzschild shadow, 2 sqrt(27) GM / (c^2 D). |
| `eht_comparison` | função | `() -> list[dict]` |  |
| `sun_deflection_arcsec` | função | `(elongation_deg)` | Deflection of starlight by the Sun at an angular distance from its centre (Eddington 1919: 1.75" at the limb), using b = 1 AU sin(elongation). |
| `star_splat` | função | `(v, mag_limit=6.5, sigma_deg=0.1, mag_white=1.5)` | Display footprint of a star of visual magnitude ``v``: Gaussian width (deg) and peak brightness. |
| `catalog_sky` | função | `(width=4096, height=2048, mag_limit=6.5, milky_way=0.32, star_sigma_deg=0.1) -> np.ndarray` | Equirectangular RGB image (x = RA / 360 * width, Dec up) of the bright-star catalogue over the Milky Way glow: stars drawn as Gaussian splats with black-body colours, sized and scaled with magnitude (``star_splat``). |
| `tone_map` | função | `(img, exposure=1.6, gamma=0.75) -> np.ndarray` | Linear sky radiance -> display values in [0, 1]: a soft shoulder 1 - exp(-exposure x) (bright cores of stars saturate smoothly instead of clipping) and a display gamma that lifts the faint band. |
| `bright_stars_in_field` | função | `(ra_deg, dec_deg, radius_deg, mag_limit=3.5) -> int` | Number of catalogue stars brighter than ``mag_limit`` within ``radius_deg`` of (ra, dec). |
| `lensed_sky` | função | `(ra_deg, dec_deg, fov_deg=30.0, width=480, height=270, r_cam=30.0, sky=None, supersample=…)` | Render the real sky around (ra, dec) as seen through a Schwarzschild black hole at that position. |
| `trace_thin_disc` | função | `(width=640, height=360, inclination_deg=80.0, r_cam=60.0, fov_deg=32.0, max_crossings=3, …)` | Exact backward ray tracing for an infinitely thin equatorial disc around a Schwarzschild black hole (G = M = c = 1). |
| `disc_image` | função | `(tr, r_in=6.0, r_out=20.0, doppler=True, t_peak=3900.0, exposure=2.4, gamma=0.6, opacity=…)` | Colour and intensity of the disc for traced rays. |
| `accretion_view` | função | `(ra_deg, dec_deg, width=640, height=360, inclination_deg=86.0, r_cam=60.0, fov_deg=42.0, …)` | A Schwarzschild black hole with a thin accretion disc placed in front of the real sky around (ra, dec): exact null geodesics (``trace_thin_disc``), Doppler beaming and gravitational redshift (``disc_image``), the lensed background from the bright-star catalogue and the Milky Way, then a light bloom. |
| `lensing_view` | função | `(ra_deg, dec_deg, fov_deg=60.0, width=480, height=270, r_cam=120.0, sky=None, disc='auto'…)` | Display-ready view of a black hole in front of the sky around (ra, dec). |

#### `pinneapple_physics.astro.nbody`

Classical N-body integrators for the Sun-planets(-Moon) problem, with conservation diagnostics.

<small>`pinneapple_physics/astro/nbody.py` · 116 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `initial_state` | função | `(jd_tdb, bodies=SOLAR_SYSTEM, eph=None)` |  |
| `accelerations` | função | `(r, mu)` |  |
| `energy` | função | `(r, v, mu)` | Total energy per unit G (km^2/s^2 * km^3/s^2 / ... consistent units: sum mu_i v_i^2/2 - sum mu_i mu_j/r_ij). |
| `angular_momentum` | função | `(r, v, mu)` |  |
| `integrate` | função | `(r0, v0, mu, t_end_s, dt_s, method='yoshida4', n_out=200)` | Integrate from t=0 to ``t_end_s``; returns dict with times (s), r (n_out, N, 3), v, and the relative energy error and angular-momentum drift histories. |

#### `pinneapple_physics.astro.neural_propagator`

A learned propagator for near-Earth asteroids: one network for many bodies.

<small>`pinneapple_physics/astro/neural_propagator.py` · 240 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralPropagator` | classe | `(net=None, xm=None, xs=None, ys=None, info=None)` |  |
| `sample_population` | função | `(n, seed=0)` | NEO-like osculating elements: a 0.7-2.6 AU, e 0.02-0.65 (perihelion > 0.2 AU), i 0-35 deg, random angles. |
| `kepler_positions` | função | `(body, jd)` |  |
| `features` | função | `(body, tau)` | Inputs: a, e, sin/cos i, angles as sin/cos, tau = t / span, and the Kepler mean anomaly at t (sin/cos). |
| `rtn_basis` | função | `(rk, vk)` | Radial / transverse (along-track) / normal unit vectors of the Kepler orbit, (n, 3, 3) rows R, T, N. |
| `make_dataset` | função | `(n_bodies=160, n_t=60, seed=0, cache=None)` | Features and targets: the deviation from the Kepler orbit in its R/T/N frame (milli-AU). |
| `tisserand` | função | `(r_km, v_km_s, a_jup_au=5.2026)` | Tisserand parameter w.r.t. Jupiter from heliocentric states (ICRF axes; inclination to the ecliptic). |
| `evaluate` | função | `(model, bodies, n_t=60)` | Errors of the learned propagator and of Kepler alone against the full integrator, per body, plus timing. |
| `trust_report` | função | `(folder=MODEL_DIR, real=None) -> dict` | Verdict of the shipped network from its held-out evaluation: to be useful it must beat Kepler by 5x on the median of the per-body maximum error over unseen bodies (and not be worse than Kepler at the 90th percentile). |

#### `pinneapple_physics.astro.observing`

An observing session, with honest physics: plan a night, set up a telescope, slew to a target and track it.

<small>`pinneapple_physics/astro/observing.py` · 356 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Telescope` | classe | `(name, aperture_mm, focal_mm, kind, central_obstruction)` |  |
| `Eyepiece` | classe | `(name, focal_mm, afov_deg)` |  |
| `Camera` | classe | `(name, width_mm, height_mm, pixel_um)` |  |
| `target_altaz` | função | `(target, t, observer, refraction=True, eph=None)` | Apparent (alt, az, ra_of_date, dec_of_date) in degrees of a body name or a J2000 (ra, dec) pair at the times ``t`` (scalar or array). |
| `rise_transit_set` | função | `(target, observer, jd_start, days=1.0, eph=None) -> dict` | Rise, upper transit and set (UTC Julian dates) in [jd_start, jd_start + days). Bodies: the almanac definition (centre 34' + semi-diameter below the horizon). Stars and deep-sky objects: centre 34' below the geometric horizon (standard refraction), transit when the hour angle is zero. |
| `extinction` | função | `(alt_apparent_deg, k=None) -> dict` | Extinction (mag) per band at an apparent altitude: k * X with X the Kasten-Young air mass. |
| `nl_to_mag` | função | `(b_nl)` | Surface brightness: nanolamberts -> V mag/arcsec^2 (Krisciunas & Schaefer 1991, eq. 1 inverted). |
| `mag_to_nl` | função | `(m)` |  |
| `moon_brightness_nl` | função | `(moon_phase_angle_deg, moon_sep_deg, moon_alt_deg, target_alt_deg, k=0.172)` | Sky brightness (nanolamberts, V) added by moonlight at the target, Krisciunas & Schaefer (1991, PASP 103, 1033): B = f(rho) I*(alpha) 10^(-0.4 k X(Z_moon)) (1 - 10^(-0.4 k X(Z))), with the Moon's illuminance I* = 10^(-0.4 (3.84 + 0.026\|alpha\| + 4e-9 alpha^4)), the scattering function f(rho) = … |
| `dark_sky_nl` | função | `(target_alt_deg, k=0.172, b_zenith_nl=79.0)` | The natural night sky at the target (KS91 eq. 2): B_zen 10^(-0.4 k (X - 1)) X; 79 nL ~ 21.6 V mag/arcsec^2. |
| `twilight_mag` | função | `(sun_alt_deg)` | Approximate zenith sky brightness in twilight, V mag/arcsec^2, from the Sun's altitude: about 11 at sunset's civil limit (-6 deg), 16.5 at -12 deg and the dark-sky level beyond -18 deg (a linear fit to typical twilight photometry; good to ~1 mag, enough to decide what can be seen). |
| `sky_brightness` | função | `(sun_alt_deg, moon_phase_angle_deg, moon_sep_deg, moon_alt_deg, target_alt_deg, k=0.172, …)` | Sky brightness at the target, V mag/arcsec^2: dark sky + moonlight (Krisciunas & Schaefer 1991) + twilight. |
| `visibility` | função | `(target, observer, t, hours=24.0, step_min=5.0, min_alt=30.0, eph=None) -> dict` | The night for a target from ``observer`` over ``hours`` after ``t``: samples of altitude, azimuth, air mass, Sun and Moon altitude, Moon separation and sky brightness; rise / transit / set; astronomical twilight; the Moon's phase; the best window (dark sky, Sun < -18 deg, target above ``min_alt``, … |
| `optics` | função | `(telescope, eyepiece=None, camera=None, naked_eye_limit=6.0, wavelength_nm=550.0) -> dict` | What a telescope shows. Standard formulas: magnification M = F / f_e; true field = AFOV / M; exit pupil D / M; focal ratio F / D; Dawes limit 116 / D(mm) arcsec; Rayleigh limit 1.22 lambda / D; limiting magnitude m = m_eye + 5 log10(D / 7 mm) (the light grasp over a 7 mm dark-adapted pupil); light … |
| `axis_time` | função | `(delta_deg, rate_deg_s, accel_deg_s2) -> float` | Time to move one axis by ``delta_deg`` with a trapezoidal velocity profile (accelerate, coast, decelerate); a triangular profile when the move is too short to reach the slew rate. |
| `hour_angle_dec` | função | `(alt_deg, az_deg, lat_deg)` | (hour angle, declination) in degrees from (alt, az). |
| `slew` | função | `(start_altaz, target_altaz, lat_deg, mount='altaz', rate_deg_s=4.0, accel_deg_s2=2.0, set…)` | A GoTo slew: both axes move at once, each with a trapezoidal profile; the slew takes the slower axis plus a settling time. Refused below ``min_alt``. An equatorial (German) mount slews in hour angle and declination and must do a meridian flip when the target is on the other side of the meridian … |
| `parallactic_angle` | função | `(lat_deg, ha_deg, dec_deg)` | Parallactic angle q (deg): the angle at the target between the directions to the zenith and to the pole, tan q = sin H / (tan phi cos delta - sin delta cos H). |
| `field_rotation_rate` | função | `(lat_deg, alt_deg, az_deg)` | Field rotation of an alt-azimuth mount (deg per second of time): the rate of change of the parallactic angle, dq/dt = -omega cos(phi) cos(A) / cos(h), with A the azimuth from north through east; its size is the classical omega cos(phi) cos(A) / cos(h), infinite at the zenith. The image in an alt-az … |
| `survey_url` | função | `(ra_deg, dec_deg, fov_deg, width=512, height=512, rotation_deg=0.0, survey=SURVEY) -> str` | URL of a JPEG cut-out of a HiPS survey around (ra, dec) from CDS hips2fits (gnomonic projection, north up). |
| `survey_cutout` | função | `(ra_deg, dec_deg, fov_deg, width=512, height=512, cache_dir=None, timeout=8.0, survey=SUR…)` | JPEG bytes of a survey cut-out, cached on disk (rounded position and field, so nearby requests share a file). Raises OSError when the service cannot be reached and nothing is cached. |

#### `pinneapple_physics.astro.orbits`

Two-body orbits: osculating elements from a state vector, Kepler's equation, propagation, and the heliocentric orbits of the planets from the JPL ephemeris.

<small>`pinneapple_physics/astro/orbits.py` · 121 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `solve_kepler` | função | `(M, e, tol=1e-14, maxiter=50)` | Eccentric anomaly E from mean anomaly M (rad) for 0 <= e < 1 (Newton, Danby starter). |
| `elements_from_state` | função | `(r, v, mu=GM_SUN) -> dict` | Osculating Keplerian elements (angles in degrees) of a bound orbit from position (km) / velocity (km/s). |
| `state_from_elements` | função | `(el, mu=GM_SUN, dt_s=0.0)` | Position/velocity (km, km/s) after ``dt_s`` seconds of pure Keplerian motion. |
| `heliocentric_state` | função | `(body, jd_tdb, frame='ecliptic', eph=None)` | Heliocentric position (km) and velocity (km/s) of a body, J2000 ecliptic (default) or ICRF axes. |
| `planet_elements` | função | `(body, jd_tdb, eph=None) -> dict` | Osculating heliocentric elements (J2000 ecliptic) of a planet; Moon: geocentric elements. |
| `orbit_polyline` | função | `(body, jd_tdb, n=256, eph=None)` | One revolution of the osculating orbit, heliocentric J2000 ecliptic, in AU (n, 3). |
| `light_time_and_doppler` | função | `(range_km, range_rate_km_s, rest_wavelength_nm=656.281)` | One-way light time and the relativistic Doppler shift of a spectral line (H-alpha by default). |

#### `pinneapple_physics.astro.phenomena`

Almanac phenomena: rise / transit / set, twilight, Moon phase, planet magnitudes and elongations, oppositions and conjunctions, equinoxes and solstices, lunar and solar eclipses, transits of Mercury and Venus.

<small>`pinneapple_physics/astro/phenomena.py` · 435 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `find_roots` | função | `(f, jd0, jd1, step, tol=0.5 * SECOND)` | All sign changes of ``f(jd_array)`` in [jd0, jd1], each refined to ``tol`` days. Returns a list of (jd, direction) with direction +1 for a rising crossing. |
| `find_minima` | função | `(f, jd0, jd1, step, tol=1.0 * SECOND)` | Local minima of ``f`` (golden-section refinement). Returns a list of (jd, value). |
| `horizon_altitude` | função | `(body, distance_km=None) -> float` | Geometric altitude of the body's centre at rise/set: -(34' refraction + semi-diameter). |
| `rise_transit_set` | função | `(body, observer, jd_start, days=1.0, eph=None)` | Rise, upper transit and set times (UTC Julian dates) of a solar-system body in [jd_start, +days). |
| `twilight_events` | função | `(observer, jd_start, days=1.0, eph=None)` | Sunrise/sunset and the start/end of civil, nautical and astronomical twilight. |
| `sky_state` | função | `(sun_alt_deg) -> str` |  |
| `phase_angle` | função | `(body, t, eph=None)` | Sun-body-Earth angle (deg) and the Sun-body distance (km), geocentric. |
| `illuminated_fraction` | função | `(body, t, eph=None)` |  |
| `moon_phase` | função | `(t, eph=None) -> dict` | Illuminated fraction, phase angle, elongation, age-like phase longitude (0 new, 90 first quarter, 180 full) and the position angle of the bright limb (deg, from north through east). |
| `magnitude` | função | `(body, t, eph=None)` | Visual magnitude of the Sun, Moon and planets (Meeus ch. 41 / Explanatory Supplement 1992 formulas; Saturn without the ring term, so up to ~0.8 mag too faint when the rings are open). |
| `planet_events` | função | `(body, jd0, jd1, eph=None)` | Oppositions and conjunctions (geocentric ecliptic longitude of date equal to the Sun's or opposite), and for Mercury / Venus the greatest elongations; with elongation and magnitude at each event. |
| `lunar_phase_events` | função | `(jd0, jd1, eph=None)` | New Moon, first quarter, full Moon, last quarter between two dates. |
| `seasons` | função | `(jd0, jd1, eph=None)` | Equinoxes and solstices: apparent geocentric longitude of the Sun = 0, 90, 180, 270 deg. |
| `lunar_eclipses` | função | `(jd0, jd1, eph=None)` | Lunar eclipses: shadow geometry at each full Moon with the Danjon 2% enlargement of the Earth's shadow. Returns greatest-eclipse time, type and umbral / penumbral magnitudes. |
| `solar_eclipses` | função | `(jd0, jd1, observer=None, eph=None)` | Solar eclipses anywhere on Earth (geocentric criterion), with an approximate type, and the local circumstances (maximum obscuration and its time) for ``observer`` if given. |
| `planetary_transits` | função | `(jd0, jd1, eph=None)` | Transits of Mercury and Venus across the Sun (geocentric): inferior conjunctions with the planet's minimum separation from the Sun's centre below the solar radius plus the planet's. |

#### `pinneapple_physics.astro.satellites`

Earth satellites from public two-line elements (CelesTrak GP data): SGP4 propagation (the ``sgp4`` package, MIT, Vallado et al. 2006 reference implementation), TEME -> terrestrial frame, topocentric look angles, Earth-shadow test, standard-magnitude brightness model and visible-pass prediction.

<small>`pinneapple_physics/astro/satellites.py` · 246 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Satellite` | classe | `(name, line1, line2)` |  |
| `gmst82` | função | `(t)` | GMST (rad), IAU 1982 expression used by the SGP4 TEME frame. |
| `parse_tle` | função | `(text) -> list[Satellite]` |  |
| `fetch_tles` | função | `(group='stations', cache_dir=None, max_age_hours=6.0, timeout=15.0) -> tuple[list[Satelli…)` | Current TLEs of a CelesTrak group, cached on disk for ``max_age_hours`` (CelesTrak asks clients not to poll more than once per update cycle). Returns (satellites, info) where info has ``source``, ``fetched_utc`` and ``error`` when the network is unavailable and no cache exists. |
| `sun_itrs` | função | `(t, eph=None)` | Geocentric Sun position (km) in the terrestrial frame. |
| `look` | função | `(sat, observer, t, eph=None) -> dict` | Topocentric altitude/azimuth/range, sub-satellite point, illumination and magnitude, vectorised over t. |
| `magnitude` | função | `(std_mag, range_km, phase_rad, sunlit=1.0)` | Diffuse-sphere brightness: m = m_std + 5 log10(d / 1000 km) - 2.5 log10(pi F(phase)), with F = (sin p + (pi - p) cos p) / pi (equals 1/pi at 90 deg, where m = m_std). |
| `find_passes` | função | `(sat, observer, t0, days=1.0, min_alt=10.0, step_s=20.0, eph=None) -> list[dict]` | Passes above the horizon in [t0, t0 + days): rise / culmination / set (UTC), max altitude, and visibility (satellite sunlit, observer's Sun below -6 deg, altitude >= ``min_alt``) with the brightest magnitude while visible. Times are refined to ~0.1 s. |

#### `pinneapple_physics.astro.smallbodies`

Asteroids and comets: JPL Small-Body Database elements, a high-accuracy propagator in the field of the Sun, the planets and the Moon from the DE421 ephemeris (with the Sun's relativistic term), close approaches to the Earth, and a learned (neural) propagator checked against it.

<small>`pinneapple_physics/astro/smallbodies.py` · 222 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SmallBody` | classe | `(name, epoch_jd, a_au, e, i_deg, node_deg, peri_deg, M_deg, kind, neo, pha, H, diameter_k…)` |  |
| `Trajectory` | classe | `(body, jd0, jd1, eph=None, rtol=1e-11, gr=True)` | One adaptive DOP853 integration from the epoch over [jd0, jd1] with dense output: positions and velocities anywhere in the span cost an interpolation, not a new integration. |
| `load_body` | função | `(key) -> SmallBody` |  |
| `available_bodies` | função | `()` | Keys of the bundled objects, the near-Earth asteroids first. |
| `propagate` | função | `(body, jd_out, eph=None, rtol=1e-11, gr=True)` | Heliocentric ICRF positions (km) and velocities (km/s) at TDB Julian dates ``jd_out``. |
| `close_approaches` | função | `(body, jd0, jd1, target='earth', max_dist_au=0.2, step_days=0.25, eph=None, traj=None)` | Minima of the distance to ``target`` (geocentre by default), refined to ~1 s by golden section on the dense trajectory. |
| `horizons_vectors` | função | `(path=None)` | Parse a JPL Horizons VECTORS table (CSV, AU-D, ecliptic J2000): jd, positions (AU), velocities (AU/d). |
| `cad_reference` | função | `(body, year)` | The JPL close-approach record (body = Earth) for a given year, with its 3-sigma distance range. |
| `validation_summary` | função | `(eph=None) -> dict` | The propagator against independent references: Ceres vs JPL Horizons vectors (2022, from 2018 elements) and the Apophis flyby of 2029-04-13 vs the JPL close-approach table (distance with its 3-sigma range, time, speed). |

#### `pinneapple_physics.astro.spaceweather`

Space weather for observers: a Kp forecast with calibrated uncertainty and the probability of seeing the aurora.

<small>`pinneapple_physics/astro/spaceweather.py` · 193 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_history` | função | `()` |  |
| `geomagnetic_latitude` | função | `(lat_deg, lon_deg)` | Dipole (centred, IGRF-13 2020) geomagnetic latitude. |
| `oval_boundary` | função | `(kp)` | Equatorward geomagnetic latitude of the visible aurora for a Kp (NOAA SWPC table: 66.5 deg at Kp 0 to 48.1 deg at Kp 9, ~2.04 deg per Kp). |
| `kp_needed` | função | `(maglat_deg, horizon_margin_deg=0.0)` |  |
| `build_xy` | função | `(kp, f107, horizon)` |  |
| `train_forecaster` | função | `(horizon=1, seed=0, epochs=300)` | Quantile network for Kp ``horizon`` 3-hour steps ahead, conformalised; evaluated on the last two months. |
| `forecast` | função | `(model, kp_hist, f107_now)` | Forecast (lo, median, hi) for Kp ``horizon`` steps after the last value of ``kp_hist`` (needs >= 27 days). |
| `aurora_probability` | função | `(lo, med, hi, maglat_deg)` | P(Kp >= needed) from the forecast quantiles (piecewise-linear CDF through q10 / q50 / q90, clipped 0..9), overhead and low on the poleward horizon (~6 deg more reach). |
| `fetch_live_kp` | função | `(timeout=10.0)` | Last ~7 days of 3-hourly planetary Kp from NOAA SWPC (JSON). Returns (iso times, kp) or raises. |

#### `pinneapple_physics.astro.spk`

A small reader for JPL SPK kernels (DAF files, segment types 2 and 3: Chebyshev position / position+velocity).

<small>`pinneapple_physics/astro/spk.py` · 124 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Segment` | classe | `(target, center, frame, kind, start_et, end_et, init, intlen, rsize, n, coeffs)` |  |
| `SPK` | classe | `(path)` | Memory-mapped SPK kernel. ``state(target, center, et)`` chains segments through common centres. |

#### `pinneapple_physics.astro.stars`

Bright-star catalogue (Yale BSC5) and stellar astrophysics: distances, absolute magnitudes, effective temperature from colour, bolometric corrections from a black-body V-band integral, luminosity, radius, Planck spectra and black-body colours.

<small>`pinneapple_physics/astro/stars.py` · 198 linhas · testes: 23</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `StarCatalog` | classe | `(hr, name, desig, con, ra, dec, pmra, pmdec, v, bv, sp, plx, plx_dyn, rv)` |  |
| `load_bright_stars` | função | `() -> StarCatalog` |  |
| `load_constellations` | função | `() -> list` | IAU constellation figures with a ``key`` (the IAU abbreviation; Serpens is split into ``Ser1`` = Caput and ``Ser2`` = Cauda), the Latin ``name``, the ``names`` per language ({"en": ..., "pt": ...}) and a label point. |
| `constellation_names` | função | `(lang=None) -> dict` | Constellation names by key (IAU abbreviation, ``Ser1``/``Ser2`` for the two halves of Serpens): the IAU Latin name (``en``) and the Brazilian Portuguese name (``pt``), e.g. ``Cru`` -> Crux / Cruzeiro do Sul. With ``lang``, a {key: name} map in that language. Sources: ``constellation_names.json``. |
| `planck_lambda` | função | `(wavelength_m, temperature_k)` | Spectral radiance B_lambda (W m^-2 sr^-1 m^-1). |
| `wien_peak_m` | função | `(temperature_k)` |  |
| `blackbody_rgb` | função | `(temperature_k)` | sRGB colour of a black body (reuses the black-hole renderer's fit, :mod:`pinneapple_physics.blackhole`). |
| `blackbody_spectrum` | função | `(temperature_k, n=300, lam_min_nm=100.0, lam_max_nm=3000.0)` |  |
| `teff_from_bv` | função | `(bv)` | Effective temperature (K) from B-V, Ballesteros (2012, EPL 97, 34008), black-body based; good to ~5% for main-sequence and giant stars between B-V = -0.3 and 1.8. |
| `bolometric_correction` | função | `(teff)` | BC_V for a black body of temperature ``teff``, zero-pointed so that BC_V(Sun, 5772 K) = M_bol,sun - M_V,sun = -0.07. A physically transparent approximation (real stellar atmospheres differ by ~0.1-0.3 mag for cool M stars and hot O stars). |
| `star_properties` | função | `(cat, i, extinction_av=0.0) -> dict` | Distance, absolute magnitude, effective temperature, luminosity and radius of catalogue star ``i``. |
| `hr_diagram` | função | `(cat, max_rel_parallax_error=None, min_parallax_mas=5.0)` | Points (B-V, M_V) for stars with a usable trigonometric parallax (>= ``min_parallax_mas``). |

#### `pinneapple_physics.astro.stars_ml`

Machine learning on the bright-star catalogue, checked against the catalogue's own spectral classification.

<small>`pinneapple_physics/astro/stars_ml.py` · 154 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `luminosity_group` | função | `(sp)` |  |
| `spectral_class` | função | `(sp)` |  |
| `dataset` | função | `(cat=None, min_parallax_mas=5.0)` |  |
| `gmm_fit` | função | `(X, k=3, iters=200, seed=0)` | Gaussian mixture by expectation-maximisation (full covariances), k-means++-style initialisation. |
| `cross_validate` | função | `(X, y, k=15, folds=5, seed=0)` |  |
| `analyse` | função | `(seed=0) -> dict` |  |

#### `pinneapple_physics.astro.surrogate`

A physics-informed neural surrogate for orbit propagation, checked against the classical integrator.

<small>`pinneapple_physics/astro/surrogate.py` · 268 linhas · testes: 32</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `reference_trajectory` | função | `(body, jd_tdb, days, n=400, eph=None)` | Heliocentric positions (AU) and velocities (AU/day) of every body from the Yoshida-4 N-body integration. |
| `train_orbit_pinn` | função | `(body='mars', jd_tdb=2461327.0, days=365.0, **kwargs) -> dict` | Train the PINN orbit surrogate (thread-safe; see ``_train_orbit_pinn``). |

#### `pinneapple_physics.astro.timescales`

Time scales for positional astronomy: UTC, TAI, TT, TDB, UT1 and sidereal time.

<small>`pinneapple_physics/astro/timescales.py` · 216 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Time` | classe | `(utc, dut1)` | An instant (or array of instants) on the UTC scale with the derived scales as Julian dates. |
| `tai_minus_utc` | função | `(jd_utc)` | TAI-UTC in seconds. Before 1972 returns ``nan`` (UTC then was rubber seconds; use :func:`delta_t`). |
| `delta_t_model` | função | `(year)` | TT - UT1 (s) from the Espenak & Meeus (2006) polynomials, 1900-2150. Used before 1972 and as a fallback. |
| `dut1` | função | `(jd_utc)` |  |
| `tdb_minus_tt` | função | `(jd_tt)` | TDB - TT (s), the 2-term periodic approximation (error < 30 us, Explanatory Supplement eq. 10.2). |
| `jd_from_datetime` | função | `(dt) -> float` |  |
| `datetime_from_jd` | função | `(jd) -> datetime` |  |
| `julian_centuries` | função | `(jd)` |  |
| `era` | função | `(jd_ut1)` | Earth rotation angle (rad), IAU 2000. |
| `gmst_rad` | função | `(t)` | Greenwich mean sidereal time (rad), IAU 2006 (ERA-based, Capitaine et al. 2005). |
| `gast_rad` | função | `(t)` | Greenwich apparent sidereal time (rad): GMST + equation of the equinoxes (Delta-psi cos eps). |
| `gast_hours` | função | `(t)` |  |

#### `pinneapple_physics.astro.validation`

Accuracy checks of the observatory's methods against independent references, as data.

<small>`pinneapple_physics/astro/validation.py` · 198 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `check_novas` | função | `()` |  |
| `check_horizons` | função | `()` |  |
| `check_usno_almanac` | função | `()` |  |
| `check_eclipses` | função | `()` |  |
| `check_iss_passes` | função | `()` |  |
| `check_skyfield_live` | função | `(n=24, seed=1)` | Random epochs 1972-2049 (the UTC era) and observers: topocentric apparent alt/az of all bodies vs skyfield (optional). |
| `check_integrators` | função | `()` |  |
| `run_all` | função | `(include_skyfield=True) -> list[dict]` |  |

#### `pinneapple_physics.blackhole`

Black-hole accretion: simulations and "black hole weather" forecasting with a trust horizon.

<small>`pinneapple_physics/blackhole/__init__.py` · 20 linhas · testes: 19</small>

#### `pinneapple_physics.blackhole.capture`

Record a Twin3D scene to PNG frames / GIF with headless Chromium (Playwright), for docs and posts.

<small>`pinneapple_physics/blackhole/capture.py` · 79 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `capture_twin` | função | `(folder, steps, out_dir=None, *, field='', cmap='turbo', view='', zoom=1.0, size=(1280, 7…)` | Screenshot the exported scene in ``folder`` at each time step; returns the PNG paths. |

#### `pinneapple_physics.blackhole.convergence`

Resolution study of the black-hole accretion simulations: is the forecaster's training data converged?

<small>`pinneapple_physics/blackhole/convergence.py` · 262 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `window_mask` | função | `(grid) -> np.ndarray` | Cells of any grid whose centre lies in the physical forecast window. |
| `cell_volumes` | função | `(grid) -> np.ndarray` | 2 pi r^2 dr d(cos theta) of every cell. |
| `remap` | função | `(a, faces_f, faces_c) -> np.ndarray` | Conservative (volume-weighted) remap of cell averages (..., nr, nth) between two spherical grids covering the same domain; the grids need not nest. The volume r^2 dr sin(theta) dtheta is separable, so the remap is one overlap matrix per axis. |
| `remap_r` | função | `(a, rf_f, rf_c) -> np.ndarray` | Remap of a radial profile (shell averages) by overlap length in log r. |
| `psd_slope` | função | `(x, dt, f_lo, f_hi) -> float` | Slope of the Welch power spectrum of x in log-log between f_lo and f_hi. |
| `run_statistics` | função | `(folder, t0, t1, log_rho_range=LOG_RHO_RANGE) -> dict` | Time-window statistics of one run on its own grid: scalars as time series, profiles and fields as time means. |
| `resolution_study` | função | `(runs, *, t0=2000.0, t1=10000.0, log=print) -> dict` | ``runs``: {nr: [run folders, one per seed]} with ntheta = nr / 2 and a constant refinement ratio. |

#### `pinneapple_physics.blackhole.dataset`

The black-hole weather dataset: density frames of whole simulated trajectories, packed for the model zoo.

<small>`pinneapple_physics/blackhole/dataset.py` · 170 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `encode` | função | `(log10_rho) -> np.ndarray` | log10 density -> uint16 (raises outside [Q_LO, Q_HI]). |
| `decode` | função | `(q) -> np.ndarray` | uint16 column ``log10_rho_q`` -> log10 density (float32). |
| `deterministic_npz` | função | `(arrays) -> bytes` | npz with fixed zip timestamps and order, so the SHA-256 does not change between identical builds. |
| `build_arrays` | função | `(runs, conv=None, *, splits=SPLITS, t0=T0, t1=T1) -> dict[str, np.ndarray]` | Arrays of the zoo dataset (``<split>__<column>``) from the simulate_run folders under ``runs`` and the resolution-study report ``conv`` (its scalars per grid and seed become the ``convergence`` split). |
| `write_npz` | função | `(arrays, path) -> Path` | All splits in one file (``<split>__<column>`` arrays). |
| `write_split_files` | função | `(arrays, folder) -> dict[str, Path]` | One ``<split>.npz`` per split in ``folder`` (the zoo layout for large datasets: a user downloads only the split they load, and every file stays far below the 100 MB a git hosting allows). Returns {file name: path}. |
| `load_trajectories` | função | `(split, path=None) -> dict[int, dict]` | Whole trajectories of a split, by seed, frames in time order. ``path``: a local data.npz (default: the zoo). |
| `window_volumes` | função | `(path=None) -> np.ndarray` | 2 pi r^2 dr d(cos theta) of the window cells. |

#### `pinneapple_physics.blackhole.figures`

Figures for the black-hole weather study: skill against lead time and the resolution study.

<small>`pinneapple_physics/blackhole/figures.py` · 121 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_skill` | função | `(report, path, model=None, title='') -> str` | Left: MAE against lead for the model and the baselines. Right: skill score against persistence with its 95 % interval and the skill horizon. |
| `plot_convergence` | função | `(report, path) -> str` | Time-mean shell density and specific angular momentum on every grid, and the mean accretion rate per grid with its 95 % interval. |

#### `pinneapple_physics.blackhole.forecast`

Black-hole weather forecasting: the U-Net of Duarte, Nemmen & Navarro (2022), a trust horizon and a physics check.

<small>`pinneapple_physics/blackhole/forecast.py` · 473 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DensityCodec` | classe | `(log_min, log_max, r_cells, theta_trim)` | Log-normalisation and crop shared by training and inference (stored with the checkpoint). |
| `DuarteUNet` | classe | `(filters=32, frames=5, residual=False, padding='zeros')` | The U-Net of Duarte et al. (2022) (``create_auto_encoder`` in the original ``src/models.py``). |
| `TrainConfig` | classe | `(filters, frames, residual, padding, epochs, batch_size, lr, alpha, clip, loss, rollout_b…)` |  |
| `MassEnvelope` | classe | `(rate_max, margin)` | Bound on how fast the mass in the forecast window can change, learned from the training simulations. |
| `MassProjection` | classe | `(codec, cell_volume, envelope, frame_dt)` | Project forecast frames onto the mass budget seen in training: the window mass may change between frames by at most ``envelope.rate_max`` (relative, per unit time), so a frame that gains or loses more is rescaled (uniformly in density, i.e. a shift in normalised log density) to the nearest allowed … |
| `make_blocks` | função | `(x, k=5, stride=1, out_blocks=1) -> tuple[np.ndarray, np.ndarray]` | Frames (T, H, W) -> inputs (N, k, H, W) = frames i..i+k-1 and targets = frames i+k..i+(1+out_blocks)k-1 (``out_blocks`` > 1 gives the targets of a multi-step rollout loss). |
| `duarte_loss` | função | `(pred, target, alpha=8.0)` | MAE over the window + alpha * MAE over the high-density pixels (target > 0.5). |
| `duarte_regional_loss` | função | `(pred, target, regions=None)` | MSE over the window + sum of weight * MAE over the paper's four regions (fractions of the crop). |
| `train_forecaster` | função | `(X, Y, Xv, Yv, cfg, codec, out_dir, *, log=print, cell_volume=None, init=None) -> dict[st…)` | Adam + the Duarte loss; keeps the weights with the best validation loss in ``out_dir/forecaster.pt``. |
| `train_on_trajectories` | função | `(train, val, cfg, out_dir, *, cell_volume=None, val_blocks=1, val_stride=1, init=None, lo…)` | Train on whole trajectories of log10 density frames (T, H, W), already cropped to the forecast window and ``frame_dt`` apart: blocks never cross trajectories, and validation trajectories are separate ones. |
| `evaluate_loss` | função | `(model, X, Y, cfg, batch=32) -> float` | Loss of the forecast unrolled over all the target blocks in ``Y`` (one block: the plain loss). |
| `save_forecaster` | função | `(path, model, cfg, codec, meta) -> None` |  |
| `load_forecaster` | função | `(path) -> tuple[DuarteUNet, DensityCodec, dict]` |  |
| `rollout` | função | `(model, x0, n_blocks, *, clip=False, mass_projection=None) -> np.ndarray` | Iterative forecast: feed each predicted block back in. x0 (k, H, W) -> (n_blocks * k, H, W). |
| `lead_time_scores` | função | `(pred, truth, last_input, mean_state, weights=None) -> dict[str, np.ndarray]` | Per lead (frame) on normalised log-density: MAE of the forecast, of persistence (the last input frame) and of the time-mean flow; the anomaly correlation with the truth, anomalies taken from the time-mean flow (the ACC of weather forecasting); and the **tendency correlation**, between the forecast … |
| `trust_horizon` | função | `(scores, frame_dt, acc_min=0.6, block=1) -> dict[str, float]` | Lead time up to which the forecast stays useful: before its anomaly correlation first drops below ``acc_min`` (the weather convention), and before its error first exceeds persistence / the time mean. |
| `window_mass` | função | `(rho, cell_volume) -> np.ndarray` | Mass in the (cropped) window for density frames (T, H, W) and cell volumes (H, W). |

#### `pinneapple_physics.blackhole.hydro`

Viscous hydrodynamics of accretion around a Schwarzschild black hole, axisymmetric (r, theta), PyTorch.

<small>`pinneapple_physics/blackhole/hydro.py` · 506 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RIAFConfig` | classe | `(nr, ntheta, r_in, r_out, gamma, alpha, viscosity, cfl, rho_floor, p_floor, v_max, t_cap,…)` |  |
| `AccretionFlow` | classe | `(cfg)` | 2.5-D (axisymmetric, with v_phi) viscous accretion flow; ``run`` saves snapshots of the primitives. |
| `torus_state` | função | `(flow) -> np.ndarray` | Equilibrium torus with l(R) = l_c (R/R_c)^a (pressure maximum at R_c, inner edge at R_edge) in an atmosphere. |
| `bondi_pw` | função | `(r, *, cs_inf, gamma=1.4, rho_inf=1.0)` | Transonic Bondi accretion in the Paczynski-Wiita potential (exact, by root finding). |

#### `pinneapple_physics.blackhole.hydro_numba`

Numba kernels for :class:`~pinneapple_physics.blackhole.hydro.AccretionFlow` (``RIAFConfig(backend="numba")``).

<small>`pinneapple_physics/blackhole/hydro_numba.py` · 321 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NumbaStepper` | classe | `(flow)` | Holds the numpy geometry of an AccretionFlow and advances its state with the numba kernels. |
| `rhs` | função | `(W, r, th, rf, thf, area_r, area_t, vol, inv_r, geo_t, Rc, gamma, alpha, visc, dU, fluxes)` | dU on interior cells (5, nr, nth) and boundary fluxes [mass_in, mass_out, angmom_in, angmom_out]. |
| `prim_to_cons` | função | `(W, Rc, gamma, U)` | Interior primitives of W (with ghosts) -> conserved U (5, nr, nth). |
| `cons_to_prim` | função | `(U, W, Rc, r, gamma, rho_floor, p_floor, v_max, t_cap)` | Conserved U -> interior primitives of W, with floors and caps; returns the number of capped cells. |
| `boundaries` | função | `(W, outer, has_outer)` |  |
| `time_step` | função | `(W, r, dr, dth, gamma, alpha, visc, cfl)` |  |

#### `pinneapple_physics.blackhole.raytrace`

General-relativistic ray tracing of an accretion flow around a Schwarzschild black hole ("Interstellar" view).

<small>`pinneapple_physics/blackhole/raytrace.py` · 316 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RayPaths` | classe | `(ray, ir, ith, ds, kdotv, lapse, captured, escape_dir, shape)` | Where each ray crosses the emitting region, in camera-to-far order (flattened over rays). |
| `Camera` | classe | `(width, height, r_cam, inclination_deg, fov_deg, roll_deg, dphi, ds_in, ds_out, max_phi)` |  |
| `blackbody_rgb` | função | `(T) -> np.ndarray` | Approximate sRGB colour (0..1) of a blackbody at temperature T [K] (Tanner Helland fit), (..., 3). |
| `star_field` | função | `(width=4096, height=2048, n=9000, seed=3) -> np.ndarray` | Equirectangular star map (H, W, 3) with a faint Milky-Way-like band, deterministic. |
| `bloom` | função | `(img, strength=0.35, sigma=6.0) -> np.ndarray` | Cinematic glow: add a blurred copy of the bright parts. |
| `render` | função | `(paths, rho, temperature, velocity=None, *, doppler=True, kappa=0.0, exposure=1.0, scale=…)` | Render one frame. |

#### `pinneapple_physics.blackhole.runs`

Generate and read the accretion-flow simulations behind the black-hole weather forecaster (issue #399).

<small>`pinneapple_physics/blackhole/runs.py` · 89 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `load_run` | função | `(folder) -> dict` | -> dict(t, frames (T, 5, nr, ntheta), mdot, grid, config). |
| `simulate_run` | função | `(cfg, folder, *, t_end=10000.0, every=10.0, chunk=50, name=None, log=print) -> str` | Run (or resume) one simulation into ``folder``; returns the folder. |

#### `pinneapple_physics.blackhole.skill`

Skill of the black-hole weather forecaster against persistence, climatology and DMD, with physics checks.

<small>`pinneapple_physics/blackhole/skill.py` · 178 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DMDBaseline` | classe | `(trajectories, rank)` | Rank-r DMD of frame sequences (``pinneapple_neural`` ``DynamicModeDecomposition``, mean-centred), iterated from one frame; the component of that frame outside the DMD subspace is kept frozen (persistence). |
| `window_mass` | função | `(x, codec, vol) -> np.ndarray` | Mass in the window for normalised frames (..., H, W). |
| `implied_inflow` | função | `(x, codec, vol, dt, shells) -> np.ndarray` | Net inward mass flow through the shells (radial indices) implied by the change of the enclosed mass, -dM(<r)/dt between consecutive frames: (frames - 1, len(shells)). Ignores flow through the polar edges. |
| `evaluate_skill` | função | `(models, train, val, test, *, frame_dt, cell_volume, blocks=20, start_every=5, block_M=10…)` |  |

#### `pinneapple_physics.blackhole.twin`

3D digital-twin view of an axisymmetric accretion flow with PINNeAPPle-Twin3D.

<small>`pinneapple_physics/blackhole/twin.py` · 120 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CutawayMesh` | classe | `(r_faces, theta_faces, *, r_view, phi_cut_deg=270.0, n_phi=72, offset=(0.0, 0.0, 0.0))` | Geometry of one cutaway panel and the map from (nr, ntheta) cell fields to its vertices. |
| `sphere` | função | `(radius, center=(0, 0, 0), n=24) -> tuple[np.ndarray, np.ndarray]` |  |
| `accretion_scene` | função | `(grid, panels, times, *, field='log10_density', unit='', r_view=60.0, gap=0.35, title='Bl…)` | One cutaway panel per entry of ``panels`` (name -> (T, nr, ntheta) cell values), laid out along ``layout`` (default: left to right on screen for the view that looks into the cut, ``view=1,0.75,-1`` in the viewer). |

#### `pinneapple_physics.closed_form`

Subpacote de modelos físicos em forma fechada (sem EDP), portados de ferramentas de produtos para serem compartilhados e testados; o `__init__` importa só parte dos módulos (não importa `eddy_coil`, `electromagnetics`, `heat_exchanger`, `pipe_insulation`, `pump_npsh`, `thin_airfoil`).

<small>`pinneapple_physics/closed_form/__init__.py` · 73 linhas · testes: 103</small>

#### `pinneapple_physics.closed_form.burgers`

Solução exata da equação de Burgers viscosa (Cole-Hopf) e invíscida (características) com condição inicial `-sin(pi x)` em [-1, 1].

<small>`pinneapple_physics/closed_form/burgers.py` · 73 linhas · testes: 80</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `burgers_sine_exact` | função | `(x, t, nu, n_nodes=120) -> np.ndarray` | Calcula `u(x, t)` da Burgers viscosa com condição inicial senoidal por quadratura (Cole-Hopf). |
| `burgers_sine_inviscid_exact` | função | `(x, t) -> np.ndarray` | Calcula a solução de entropia da Burgers invíscida com condição inicial senoidal. |

#### `pinneapple_physics.closed_form.cantilever_fatigue`

Tensão de flexão de viga em balanço e vida em fadiga pela relação de Basquin (equações de livro-texto, Shigley e Dowling).

<small>`pinneapple_physics/closed_form/cantilever_fatigue.py` · 73 linhas · testes: 32</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `stress_from_moment_pa` | função | `(width_m, thickness_m, moment_n_m) -> float` | Tensão `sigma = M c / I` numa seção retangular sólida; base comum das outras funções de tensão. |
| `cantilever_root_bending_stress_pa` | função | `(width_m, thickness_m, load_n, length_m) -> float` | Tensão de flexão na raiz engastada da viga em balanço, onde o momento é máximo (`M = F L`). |
| `cantilever_bending_stress_at_position_pa` | função | `(width_m, thickness_m, load_n, length_m, x_from_root_m) -> float` | Perfil de tensão de flexão ao longo da mesma viga, numa posição medida a partir da raiz. |
| `basquin_fatigue_life_cycles` | função | `(stress_amplitude_pa, fatigue_strength_coeff_pa, fatigue_strength_exponent) -> float` | Ciclos até a falha pela equação de Basquin, dada a amplitude de tensão e as constantes de fadiga do material. |
| `basquin_allowable_stress_pa` | função | `(target_life_cycles, fatigue_strength_coeff_pa, fatigue_strength_exponent) -> float` | Basquin invertida: maior amplitude de tensão que ainda atinge a vida alvo em ciclos. |

#### `pinneapple_physics.closed_form.compressible`

Escoamento compressível 1-D de gás caloricamente perfeito: razões isentrópicas, choque normal, pressão de Pitot e `Cp_max` newtoniano modificado.

<small>`pinneapple_physics/closed_form/compressible.py` · 78 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `isentropic_ratios` | função | `(M, gamma=1.4) -> dict[str, np.ndarray]` | Razões de estagnação/estática (T, p, rho) e razão de áreas `A/A*` num número de Mach (NACA 1135). |
| `normal_shock` | função | `(M, gamma=1.4) -> dict[str, np.ndarray]` | Saltos através de um choque normal para Mach a montante `M >= 1` (NACA 1135). |
| `pitot_pressure_ratio` | função | `(M, gamma=1.4)` | Razão entre a pressão de Pitot e a pressão estática do escoamento livre. |
| `cp_max_newtonian` | função | `(M, gamma=1.4)` | Coeficiente de pressão no ponto de estagnação da teoria newtoniana modificada (Lees 1955). |

#### `pinneapple_physics.closed_form.eddy_coil`

Impedância de uma bobina de correntes parasitas com núcleo de ar sobre uma placa condutora (Dodd & Deeds, 1968).

<small>`pinneapple_physics/closed_form/eddy_coil.py` · 69 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `skin_depth` | função | `(f, sigma, mu_r=1.0) -> float` | Profundidade de penetração `1 / sqrt(pi f mu sigma)`. |
| `coil_impedance` | função | `(f, *, r1, r2, l1, l2, turns, sigma, thickness=math.inf) -> dict` | Impedância da bobina no ar, sua variação sobre a placa, impedância normalizada e profundidade de penetração. |

#### `pinneapple_physics.closed_form.electromagnetics`

Soluções fechadas dos presets eletromagnéticos, registradas como solução `"analytic"` de `pp.solve`.

<small>`pinneapple_physics/closed_form/electromagnetics.py` · 125 linhas · testes: 41</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `two_dielectric_capacitor` | função | `(y, V, d, d1, eps1, eps2)` | Potencial num capacitor de placas paralelas com dois dielétricos (Griffiths, ex. 4.7). |
| `wire_az` | função | `(x, y, a, J, nu=1.0)` | Potencial vetor `A_z` de um condutor redondo com densidade de corrente uniforme (lei de Ampère). |
| `tm_cavity_mode` | função | `(x, y, t, m, n, Lx, Ly, eps=1.0, mu=1.0)` | Onda estacionária TM_mn numa cavidade retangular PEC (campos Ez, Hx, Hy). |
| `skin_effect_slab` | função | `(x, delta, A0=1.0)` | Fasor do campo que se difunde num semiespaço condutor (efeito pelicular). |
| `cylinder_tm_total` | função | `(x, y, k0, radius, eps_r)` | Campo `Ez` total de uma onda plana espalhada por um cilindro dielétrico (série de funções cilíndricas, via `pinneapple_simulation`). |

#### `pinneapple_physics.closed_form.fin_array_conduction`

Modelo fechado de condução e convecção em conjunto de aletas (Incropera), generalizado para qualquer arranjo de superfícies estendidas.

<small>`pinneapple_physics/closed_form/fin_array_conduction.py` · 125 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `fin_array_profile_params` | função | `(k_material_w_mk, h_conv_w_m2k, n_fins, fin_thickness_m, fin_length_m, transverse_extent_…)` | Calcula uma vez as grandezas intermediárias do modelo de aletas, compartilhadas pelo valor escalar e pelo perfil espacial. |
| `fin_array_base_temperature_c` | função | `(params, t_ambient_c, power_w) -> float` | Temperatura na base do conjunto de aletas (verdade de referência escalar). |
| `fin_array_temperature_at_position_c` | função | `(params, t_ambient_c, x_from_root_m, corrected_length_m=None) -> float` | Perfil de temperatura ao longo da aleta pela mesma equação 1-D, com correção de ponta adiabática. |

#### `pinneapple_physics.closed_form.heat`

Condução transiente 1-D numa placa (série de Fourier) e aleta de pino com ponta convectiva (Incropera).

<small>`pinneapple_physics/closed_form/heat.py` · 79 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `slab_transient` | função | `(x, t, L, alpha, T_left, T_right, T0, n_terms=None) -> np.ndarray` | Temperatura `T(x, t)` numa placa com temperaturas impostas nas faces e temperatura inicial uniforme. |
| `pin_fin_theta` | função | `(xi, h, k, D, L)` | Excesso de temperatura adimensional ao longo de uma aleta de pino com ponta convectiva. |
| `pin_fin_heat` | função | `(h, k, D, L, T_base, T_inf) -> float` | Calor dissipado [W] por uma aleta de pino com ponta convectiva. |

#### `pinneapple_physics.closed_form.heat_exchanger`

Dimensionamento de trocadores de calor: efetividade-NTU, fator de correção da LMTD e coeficientes do lado dos tubos e do casco.

<small>`pinneapple_physics/closed_form/heat_exchanger.py` · 155 linhas · testes: 41</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `effectiveness` | função | `(ntu, cr, arrangement='counterflow', n_shell=1) -> float` | Efetividade `epsilon(NTU, C_r)` para contracorrente, paralelo, casco-tubo ou fluxo cruzado. |
| `ntu_from_effectiveness` | função | `(eps, cr, arrangement='counterflow') -> float` | Inverte `effectiveness` por bisseção para obter o NTU. |
| `lmtd_correction_1_2` | função | `(t_hot_in, t_hot_out, t_cold_in, t_cold_out) -> float` | Fator F de correção da LMTD de um trocador 1-2 (Bowman et al., 1940). |
| `gnielinski` | função | `(re, pr) -> dict[str, float]` | Número de Nusselt do lado dos tubos (Gnielinski; laminar desenvolvido abaixo de Re = 2300). |
| `hausen_mean_nusselt` | função | `(re, pr, d_over_l) -> float` | Nusselt médio laminar com entrada térmica num tubo de parede isotérmica (Hausen, 1943). |
| `mean_nusselt_from_temperatures` | função | `(re, pr, d_over_l, t_wall, t_in, t_out) -> float` | Nusselt médio de um tubo isotérmico a partir das temperaturas de entrada e saída (balanço de energia). |
| `kern_shell` | função | `(m_dot, mu, k, cp, *, shell_id, baffle_spacing, tube_od, pitch, layout='triangular') -> d…)` | Coeficiente do lado do casco pelo método de Kern (área de fluxo cruzado, diâmetro equivalente, Re, Nu, h). |
| `one_two_exchanger_ode` | função | `(ntu, cr, n=4001, profiles=False)` | Efetividade de um trocador 1-2 integrando as três equações de energia por diferenças finitas (modelo de Nagle), sem a fórmula fechada. |

#### `pinneapple_physics.closed_form.helmholtz_resonator`

Equação de sintonia de ressonador de Helmholtz (porta de caixa acústica bass-reflex).

<small>`pinneapple_physics/closed_form/helmholtz_resonator.py` · 50 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `helmholtz_resonant_frequency_hz` | função | `(port_radius_m, port_length_m, cabinet_volume_l, end_correction_factor=1.7) -> float` | Frequência de ressonância para uma porta de raio e comprimento dados num volume de caixa, com fator de correção de extremidade. |
| `port_volume_m3` | função | `(port_radius_m, port_length_m) -> float` | Volume de ar da porta, usado como restrição ou objetivo de projeto. |

#### `pinneapple_physics.closed_form.membrane_diffusion`

Transporte difusivo controlado por membrana (primeira lei de Fick), como em adesivos transdérmicos de reservatório.

<small>`pinneapple_physics/closed_form/membrane_diffusion.py` · 148 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `steady_state_release_rate_mg_day` | função | `(diffusion_coeff_m2_s, partition_coeff, reservoir_conc_kg_m3, area_m2, membrane_thickness…)` | Taxa de liberação em regime permanente (mg/dia) através da membrana limitante. |
| `depletion_time_days` | função | `(diffusion_coeff_m2_s, partition_coeff, membrane_thickness_m, reservoir_thickness_m) -> f…)` | Tempo de esgotamento do reservatório; depende só das espessuras e da permeabilidade, não da área nem da concentração. |
| `concentration_at_membrane_depth` | função | `(reservoir_conc_kg_m3, partition_coeff, membrane_thickness_m, z_from_reservoir_m) -> float` | Perfil linear de concentração dentro da membrana em regime permanente. |
| `post_depletion_decay_rate_constant_per_day` | função | `(diffusion_coeff_m2_s, partition_coeff, membrane_thickness_m, reservoir_thickness_m) -> f…)` | Constante de decaimento de primeira ordem depois que o excesso de fármaco do reservatório acaba. |
| `simulate_release_timeseries` | função | `(diffusion_coeff_m2_s, partition_coeff, reservoir_conc_kg_m3, area_m2, membrane_thickness…)` | Série temporal de liberação e esgotamento combinando as fórmulas do módulo. |

#### `pinneapple_physics.closed_form.particle_drag`

Coeficiente de arrasto de uma esfera rígida em escoamento uniforme por correlações empíricas.

<small>`pinneapple_physics/closed_form/particle_drag.py` · 32 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `clift_sphere_cd` | função | `(re) -> float` | Arrasto da esfera por Clift, Grace & Weber (1978), faixa 20 < Re <= 260 (avisa fora dela). |
| `schiller_naumann_cd` | função | `(re) -> float` | Arrasto da esfera por Schiller & Naumann (1933), Re < 800 (avisa acima). |

#### `pinneapple_physics.closed_form.pcb_thermal`

Blocos térmicos de placas de circuito: condutividade efetiva do empilhamento, vias térmicas, convecção e radiação nas faces, e dados de encapsulamentos para o modelo de dois resistores JEDEC.

<small>`pinneapple_physics/closed_form/pcb_thermal.py` · 179 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Layer` | classe | `(name, kind, thickness_m, coverage)` | Camada do empilhamento (cobre ou dielétrico) com espessura e cobertura de cobre; dá `k_xy` e `k_z` pela regra da fração de área. |
| `Environment` | classe | `(t_ambient_c, air_velocity_m_s, orientation, emissivity, extra)` | Condições do ambiente da placa: temperatura, velocidade do ar (0 = convecção natural), orientação e emissividade. |
| `stackup` | função | `(n_copper=4, thickness_mm=1.6, outer_oz=1.0, inner_oz=1.0, outer_coverage=0.3, inner_cove…)` | Monta um empilhamento simétrico de camadas de sinal, planos e dielétrico. |
| `effective_conductivity` | função | `(layers) -> Dict[str, float]` | Condutividade efetiva da placa: regra paralela no plano e regra em série na espessura. |
| `via_copper_fraction` | função | `(n_vias, drill_mm, plating_um, area_m2) -> float` | Fração de área de cobre dos barris de vias metalizadas (não preenchidas) sob uma área. |
| `convection_coefficients` | função | `(width_m, depth_m, t_surface_c, env) -> Dict[str, float]` | Coeficientes h das faces superior e inferior (convecção natural ou forçada mais radiação linearizada). |

#### `pinneapple_physics.closed_form.pipe_flow`

Escoamento desenvolvido em tubos e canais: fatores de atrito de Darcy, perda de carga e altura da primeira célula para um y+ alvo.

<small>`pinneapple_physics/closed_form/pipe_flow.py` · 169 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `laminar` | função | `(Re) -> float` | Fator de atrito laminar `64 / Re` (avisa para Re >= 2300). |
| `blasius` | função | `(Re) -> float` | Correlação de Blasius para tubo liso (avisa fora de 4e3 a 1e5). |
| `colebrook` | função | `(Re, rel_rough=0.0, tol=1e-14, max_iter=100) -> float` | Fator de atrito de Darcy pela equação implícita de Colebrook, resolvida por iteração. |
| `haaland` | função | `(Re, rel_rough=0.0) -> float` | Aproximação explícita de Haaland para a equação de Colebrook. |
| `dean_number` | função | `(Re, D, Rc) -> float` | Número de Dean de um tubo curvo. |
| `curved_pipe_ratio` | função | `(De) -> float` | Razão entre o atrito laminar de tubo curvo e de tubo reto (White, 1929). |
| `friction_factor` | função | `(Re, rel_rough=0.0) -> float` | Escolhe `laminar` abaixo de Re = 2300 e `colebrook` acima (sem aviso na faixa de transição). |
| `darcy_pressure_drop` | função | `(f, L, D, rho, U) -> float` | Perda de pressão de Darcy-Weisbach num duto. |
| `parallel_plates_pressure_drop` | função | `(mu, L, h, U) -> float` | Perda de pressão laminar entre placas paralelas infinitas (Poiseuille plano). |
| `first_cell_height` | função | `(y_plus, Re, D, U, nu, f=None, rel_rough=0.0) -> float` | Altura da primeira célula junto à parede para atingir um y+ desejado. |

#### `pinneapple_physics.closed_form.pipe_insulation`

Perda de calor em tubo isolado e espessura econômica de isolamento (método ISO 12241 / VDI 2055).

<small>`pinneapple_physics/closed_form/pipe_insulation.py` · 151 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `air` | função | `(t_c) -> dict[str, float]` | Propriedades do ar (nu, k, Pr, beta) interpoladas de tabela numa temperatura. |
| `surface_coefficient` | função | `(ts, ta, d, eps, wind=0.0) -> dict[str, float]` | Coeficiente externo de um cilindro horizontal no ar: radiação mais convecção mista. |
| `heat_loss` | função | `(d_pipe, thickness, t_in, t_amb, *, k0, k1=0.0, eps=0.9, wind=0.0) -> dict[str, float]` | Perda de calor por metro, temperatura de superfície e coeficientes de um tubo isolado ou nu. |
| `radial_fd` | função | `(d_pipe, thickness, t_in, t_surface, *, k0, k1=0.0, n=400) -> dict[str, np.ndarray]` | Perfil de temperatura no isolamento por volumes finitos radiais com k(T) e iterações de Picard. |
| `crf` | função | `(rate, years) -> float` | Fator de recuperação de capital. |
| `economic_thickness` | função | `(d_pipe, t_in, t_amb, thicknesses, *, k0, k1, eps, wind, hours, energy_price, efficiency,…)` | Custo anual (energia + isolamento anualizado) por espessura e escolha da espessura econômica. |

#### `pinneapple_physics.closed_form.plate_fin_heatsink`

Modelo térmico e hidráulico de dissipador de aletas planas com correlações publicadas (Teertstra, Bar-Cohen, Lee, Muzychka); os resultados trazem avisos quando uma correlação sai da faixa.

<small>`pinneapple_physics/closed_form/plate_fin_heatsink.py` · 309 linhas · testes: 18</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HeatSinkGeometry` | classe | `(base_width, base_depth, base_thickness, n_fins, fin_thickness, fin_height, material)` | Geometria do dissipador em metros (base, número, espessura e altura das aletas, material), com `fin_gap` e `mass_kg`. |
| `OperatingPoint` | classe | `(power_w, t_ambient_c, air_velocity_m_s, source_width, source_depth, tim_resistance_k_m2_…)` | Ponto de operação: potência, temperatura ambiente, velocidade do ar, tamanho da fonte e resistência da interface térmica. |
| `air_properties` | função | `(t_c, p_pa=101325.0) -> Dict[str, float]` | Propriedades do ar a 1 atm interpoladas da tabela de Incropera, com densidade de gás ideal. |
| `geometry_errors` | função | `(g, op) -> List[str]` | Lista os erros de geometria ou de ponto de operação (material desconhecido, dimensões inválidas). |
| `forced_convection` | função | `(g, v_approach, air) -> Dict[str, float]` | Convecção forçada em canais de placas paralelas (Teertstra) e perda de pressão. |
| `natural_convection` | função | `(g, dt_k, air) -> Dict[str, float]` | Convecção natural entre placas verticais isotérmicas (Bar-Cohen & Rohsenow). |
| `spreading_resistance` | função | `(k, tb, source_area, plate_area, r_conv) -> Dict[str, float]` | Resistência de espalhamento da base (Lee et al., 1995), incluindo condução 1-D pela base. |
| `evaluate` | função | `(g, op, max_iter=60, h_scale=1.0, t_root_guess_c=None) -> Dict[str, object]` | Resolve a rede de resistências térmicas e devolve temperatura da fonte, resistências e avisos. |

#### `pinneapple_physics.closed_form.pump_npsh`

Bomba centrífuga num sistema de sucção e recalque: ponto de operação, NPSH disponível contra requerido e margem de cavitação.

<small>`pinneapple_physics/closed_form/pump_npsh.py` · 94 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `iapws97_psat` | função | `(t_c) -> float` | Pressão de saturação da água pela IAPWS-IF97. |
| `water_density` | função | `(t_c) -> dict[str, float]` | Densidade e viscosidade da água interpoladas de tabela numa temperatura. |
| `atmosphere` | função | `(z_m) -> float` | Pressão atmosférica ISO 2533 numa altitude (troposfera). |
| `pipe_loss` | função | `(q, d, length, k_minor, rough, rho, mu) -> dict[str, float]` | Perda de carga de um tubo com perdas localizadas numa vazão. |
| `fit_quadratic` | função | `(q, y) -> np.ndarray` | Ajuste por mínimos quadrados de uma curva quadrática na vazão a pontos de catálogo. |
| `duty_point` | função | `(head_coef, h_static, system_loss, q_max) -> float` | Vazão onde a altura da bomba iguala a do sistema (bisseção). |
| `npsh_available` | função | `(p_surface, t_c, z_suction, h_loss) -> dict[str, float]` | NPSH disponível no flange de sucção. |

#### `pinneapple_physics.closed_form.rocket_nozzle`

Bocal de foguete ideal: escoamento isentrópico 1-D de gás caloricamente perfeito num bocal convergente-divergente (Sutton & Biblarz).

<small>`pinneapple_physics/closed_form/rocket_nozzle.py` · 114 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `area_ratio` | função | `(M, gamma) -> float` | Razão de áreas `A/A*` num número de Mach. |
| `mach_from_area_ratio` | função | `(ratio, gamma, *, supersonic=True, tol=1e-12, max_iter=300) -> float` | Inverte `A/A*` por bisseção, no ramo supersônico (padrão) ou subsônico. |
| `pressure_ratio` | função | `(M, gamma) -> float` | Razão entre pressão estática e de estagnação num Mach. |
| `temperature_ratio` | função | `(M, gamma) -> float` | Razão entre temperatura estática e de estagnação num Mach. |
| `choked_mass_flow` | função | `(p0, throat_area, gamma, R_gas, T0) -> float` | Vazão mássica numa garganta bloqueada. |
| `characteristic_velocity` | função | `(gamma, R_gas, T0) -> float` | Velocidade característica c*, que só depende do gás e da temperatura da câmara. |
| `thrust_coefficient` | função | `(p_exit, p0, p_atm, ratio, gamma) -> float` | Coeficiente de empuxo (termo de quantidade de movimento mais termo de pressão). |
| `specific_impulse` | função | `(cf, c_star, g0=G0) -> float` | Impulso específico `Isp = C_F c* / g0`. |
| `nozzle_performance` | função | `(*, gamma, R_gas, T_chamber, p_chamber, area_ratio_exit, p_atm=101325.0) -> dict` | Estado de saída e desempenho de um bocal ideal expandido até uma razão de áreas. |

#### `pinneapple_physics.closed_form.thin_airfoil`

Aerodinâmica 2-D fechada: sustentação de linha média NACA de 4 dígitos pela teoria de aerofólio fino e arrasto de perfil por correlação.

<small>`pinneapple_physics/closed_form/thin_airfoil.py` · 61 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `naca4_camber_slope` | função | `(x, m, p) -> np.ndarray` | Inclinação `dz/dx` da linha média NACA de 4 dígitos. |
| `thin_airfoil_cl` | função | `(m, p, alpha_deg, n_quad=4001) -> float` | Coeficiente de sustentação pela teoria de aerofólio fino, por quadratura trapezoidal. |
| `turbulent_flat_plate_cf` | função | `(reynolds) -> float` | Coeficiente médio de atrito de placa plana totalmente turbulenta. |
| `hoerner_profile_cd` | função | `(t_c, reynolds) -> float` | Arrasto de perfil de uma seção aerodinâmica com fator de forma de Hoerner. |

#### `pinneapple_physics.codegen`

pinneapple_physics.codegen — automatic physical-code generation.

<small>`pinneapple_physics/codegen/__init__.py` · 51 linhas · **sem teste direto**</small>

#### `pinneapple_physics.codegen.fdm_script_generator`

Standalone finite-difference *script* generator for PINNeAPPle.

<small>`pinneapple_physics/codegen/fdm_script_generator.py` · 548 linhas · testes: 13 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PDEParser` | classe | `()` | Classify a ``pinneapple_physics.pde_environment.ProblemSpec`` (or any duck-typed object exposing ``.pde.kind``, ``.coords``, ``.fields``, ``.domain_bounds``, ``.pde.params``) into one of the three curated FDM templates ``FDMScriptGenerator`` knows how to render. |
| `FDMScriptGenerator` | classe | `(parser=None) -> None` | Renders a curated PDE description into a standalone, runnable FDM Python script (source text, not an in-memory callable). |

#### `pinneapple_physics.codegen.fenics_script_generator`

Standalone UFL/dolfinx *script* generator for PINNeAPPle.

<small>`pinneapple_physics/codegen/fenics_script_generator.py` · 614 linhas · testes: 13</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FEniCSScriptGenerator` | classe | `()` | Renders a curated set of UFL/dolfinx finite-element physics problems into standalone, runnable Python scripts, chosen by inspecting a ProblemSpec-like object's physics shape (see ``classify_physics_shape`` / the module docstring). |
| `classify_physics_shape` | função | `(spec) -> str` | Classify a ProblemSpec-like object into "diffusion" \| "stokes" \| "elasticity" — see module docstring for the exact heuristic. |

#### `pinneapple_physics.compile_api`

``pp.compile``: optimise a physics problem once, solve it faster, with the same results.

<small>`pinneapple_physics/compile_api.py` · 137 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Benchmark` | classe | `(problem, epochs, repeats, baseline_s, compiled_s, speedup, max_param_diff, max_param_abs…)` |  |
| `CompiledProblem` | classe | `(problem, optimize='physics') -> None` | A problem with its optimisations decided. ``solve`` and ``loss_fn`` use them. |
| `compile` | função | `(problem, optimize='physics', **preset_kwargs) -> CompiledProblem` | Prepare a problem (a ``PhysicalProblem``, a ``ProblemSpec`` or a preset name) for faster solving. |

#### `pinneapple_physics.conservation`

Conservation as a hard constraint on model outputs: project a predicted field so that its weighted integral (total mass, energy, charge, water) equals a prescribed value exactly.

<small>`pinneapple_physics/conservation.py` · 83 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConservationProjection` | classe | `(model, weights, mode='additive', target_fn=None)` | ``model`` followed by an exact projection. ``target_fn(x, y_pred)`` gives the conserved total for each batch entry (e.g. ``lambda x, y: integral(x_state, w)`` for a closed system); by default the integral of the input field ``x`` itself (same grid as the output). |
| `integral` | função | `(field, weights, dims=1) -> Tensor` | Weighted sum over the last ``dims`` dimensions (``weights`` has that trailing shape). |
| `project_integral` | função | `(field, weights, target, mode='additive', dims=None, eps=1e-30) -> Tensor` | Return ``field`` corrected so that ``integral(field, weights) == target`` (per leading batch entry). |

#### `pinneapple_physics.darcy_checks`

Verificações físicas de campos de Darcy 2-D (`-div(a grad u) = f`, `u = 0` no contorno) sem solução de referência, na mesma discretização do solver de referência; alimenta `pinneapple check` (`physics = "darcy"`) e o relatório de confiança.

<small>`pinneapple_physics/darcy_checks.py` · 857 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoefficientEnvelope` | classe | `(a_min, a_max, ranges, rel_tol)` | Envelope dos coeficientes de treino (faixa de `a` e de estatísticas do campo) para dizer se uma entrada está fora da distribuição. |
| `DarcyCheck` | classe | `(name, kind, values, value, limit, passed, critical, detail)` | Uma verificação sobre um lote: valores por amostra, agregado, limite, aprovado/reprovado e se é crítica. |
| `UseProfile` | classe | `(name, use, limits, reasons)` | Perfil de uso de uma previsão com os limites físicos que esse uso exige e o motivo de cada um. |
| `darcy_operator_apply` | função | `(u, a) -> np.ndarray` | Aplica o operador de volumes finitos de `-div(a grad u)` a `u`. |
| `darcy_residual` | função | `(u, a, f=1.0) -> np.ndarray` | Resíduo por célula `f - A_a u`; zero para a solução do próprio solver. |
| `residual_error_bound` | função | `(u, a, f=1.0) -> np.ndarray` | Limite certificado do erro relativo na norma de energia, só a partir do resíduo. |
| `residual_l2_bound` | função | `(u, a, f=1.0) -> np.ndarray` | Limite certificado do erro relativo em L2, só a partir do resíduo. |
| `boundary_flux_balance` | função | `(u, a, f=1.0) -> dict[str, np.ndarray]` | Fluxo líquido pelo contorno comparado com a integral da fonte (conservação discreta). |
| `boundary_values` | função | `(u) -> np.ndarray` | Valores de `u` nos quatro lados, extrapolados das duas células vizinhas. |
| `coefficient_stats` | função | `(a) -> dict[str, np.ndarray]` | Estatísticas por amostra de um campo de coeficientes, usadas na verificação de aplicabilidade. |
| `darcy_checks` | função | `(a, u, f=1.0, *, reference=None, envelope=None, tolerances=None, quantile=0.9, physics_ki…)` | Roda todas as verificações físicas (e, opcionalmente, envelope e comparação com referência) num lote de previsões. |
| `physics_metrics` | função | `(a, u, f=1.0) -> dict[str, np.ndarray]` | Valores por amostra das métricas físicas (`energy_bound`, `l2_bound`, `bc`, `flux`, `positivity`). |
| `calibrate_tolerances` | função | `(a, u, f=1.0, *, quantile=0.99, floor=None) -> dict[str, float]` | Calcula limites a partir das próprias previsões do modelo em entradas de validação dentro da distribuição. |
| `profile_verdicts` | função | `(a, u, f=1.0, *, profiles=None, quantile=0.9) -> dict[str, Any]` | Para cada perfil de uso, compara o quantil de cada métrica com o limite e decide APPROVED ou REJECT. |
| `sample_checks` | função | `(a, u, f=1.0, *, tolerances=None, envelope=None) -> list[dict[str, Any]]` | Verificações de uma única previsão no formato que `pinneapple check` usa. |
| `sample_verdict` | função | `(checks) -> str` | PASS se todas as verificações passam, senão FAIL (status de saída de `pinneapple check`). |
| `estimated_error` | função | `(u, a, f=1.0) -> np.ndarray` | Mapa do erro de `u` estimado só pelo resíduo. |
| `flux_magnitude` | função | `(u, a) -> np.ndarray` | Módulo do fluxo `\|a grad u\|` por célula a partir dos fluxos de face do solver. |
| `estimated_flux_error` | função | `(u, a, f=1.0) -> np.ndarray` | Mapa do erro estimado do fluxo, relativo ao maior fluxo. |
| `region_masks` | função | `(n, wall_cells=2, interior_from=8) -> dict[str, np.ndarray]` | Máscaras booleanas de região de parede e de interior numa grade n x n. |
| `regional_errors` | função | `(u, a, f=1.0, *, wall_cells=2, interior_from=8) -> dict[str, np.ndarray]` | Erro relativo estimado sem referência nas regiões de parede e interior. |
| `use_verdicts` | função | `(a, u, f=1.0, *, envelope=None, wall_cells=2, interior_from=8) -> dict[str, dict[str, Any…)` | Para uma previsão, quais usos ela atende e por quê, só pelo resíduo e pelo envelope. |
| `trust_report` | função | `(checks, *, model='', kinds=None)` | Decisão de implantação (`TrustReport`: APPROVED / REVIEW / REJECT) de uma lista de `DarcyCheck`. |
| `darcy_trust_report` | função | `(a, u, f=1.0, *, model='', kinds=…, **kwargs)` | Roda `darcy_checks` e devolve `(TrustReport, checks)`. |
| `save_darcy_npz` | função | `(path, a, u, **extra)` | Grava um campo de Darcy no formato de grade que `pinneapple check` lê. |
| `check_config_entry` | função | `(name, path, *, f=1.0, tolerances=None, envelope=None, coefficient='a', solution='u') -> …)` | Gera uma tabela `[[result]]` de `pinneapple-check.toml` para uma previsão de Darcy salva. |
| `nodal_darcy_residual` | função | `(u, k, f=1.0, dx=1.0, *, cropped=1, cross_denominator=None) -> np.ndarray` | Resíduo da forma nodal (diferenças finitas, layout `Darcy2D` do PhysicsNeMo) nos nós com vizinhos conhecidos. |
| `nodal_boundary_values` | função | `(u, *, cropped=1) -> np.ndarray` | Valores de contorno extrapolados até os nós fantasmas nulos, no layout nodal. |
| `nodal_flux_balance` | função | `(u, k, f=1.0, dx=1.0, *, cropped=1) -> np.ndarray` | Desequilíbrio relativo de um balanço de volume de controle em torno dos nós armazenados. |
| `solve_nodal_darcy` | função | `(k, f=1.0, dx=1.0, *, cross_denominator=None) -> np.ndarray` | Resolve diretamente a forma nodal; serve de referência para `nodal_darcy_residual`. |
| `check_nodal_darcy_batch` | função | `(k, u, f=1.0, dx=1.0, *, cropped=1, alt_cross_denominators=None, tolerances=None, floor_f…)` | Verifica um lote de pares de treino nodais `(k, u)` antes de alguém treinar com eles. |

#### `pinneapple_physics.deepxde_bridge`

pinneapple_physics.deepxde_bridge -- DeepXDE <-> PINNeAPPle integration glue.

<small>`pinneapple_physics/deepxde_bridge/__init__.py` · 60 linhas · testes: 9</small>

#### `pinneapple_physics.deepxde_bridge.adapter`

``ProblemSpec`` -> real DeepXDE ``dde.Model`` translation and training.

<small>`pinneapple_physics/deepxde_bridge/adapter.py` · 468 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DeepXDESolveResult` | classe | `(deepxde_model, spec, final_loss, train_time_s)` | Outcome of :func:`solve_with_deepxde`. |
| `require_deepxde` | função | `() -> Any` | Import ``deepxde``. |
| `is_deepxde_available` | função | `() -> bool` | Return ``True`` if ``deepxde`` can be imported. |
| `solve_with_deepxde` | função | `(spec, *, n_domain=2000, n_boundary=200, n_initial=100, layers=(50, 50, 50, 50), epochs=1…)` | Translate ``spec`` into a real DeepXDE problem and train it for real. |

#### `pinneapple_physics.ensemble`

Adaptive ensembles of physics models: PINNs, neural operators (FNO, DeepONet...), GNNs (MeshGraphNet...), numerical solvers and closed forms, selected and combined online from the errors they make on new cases.

<small>`pinneapple_physics/ensemble.py` · 636 linhas · testes: 68</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Expert` | classe | `(name, predict, cost, kind)` | ``predict(query) -> array`` of the common output shape; ``cost``: fixed cost per call (any unit), or None to use the measured wall time. |
| `EnsembleRun` | classe | `(names, errors, ensemble_errors, weights, active, coverage, costs, predictions, suspect)` |  |
| `PhysicsEnsemble` | classe | `(experts, mode='combine', etas=(0.5, 2.0, 8.0), alphas=(0.0, 0.01, 0.05), point_weights=N…)` | Online selection / combination of physics models. |
| `relative_l2` | função | `(pred, ref, weights=None) -> float` |  |
| `from_callable` | função | `(name, fn, cost=None, kind='numerical') -> Expert` |  |
| `from_solution` | função | `(name, solution, points, cost=None, shape=None) -> Expert` | A ``pp.solve`` result (PINN, FEM, analytic, external solver...): evaluated at ``points(query)``. |
| `from_torch` | função | `(name, model, to_input, from_output=None, cost=None, kind='neural') -> Expert` | Any ``torch.nn.Module`` (FNO, DeepONet, MeshGraphNet, a PINN network...): ``to_input(query)`` builds its input (tensor, GraphBatch...), the output's ``.y`` is used if present, and ``from_output(y, query)`` maps it to the common points (e.g. drop the batch and channel axes). |
| `residual_baseline` | função | `(experts, domain_queries, residual_fn) -> dict[str, float]` | Median physics residual of each expert on queries from its own domain (its training or validation cases), known before operation: the reference for ``PhysicsEnsemble(residual_baseline=...)``. Experts missing from ``domain_queries`` are left out (they fall back to their recent history). |
| `fit_static_weights` | função | `(predictions, references, n_folds=5, point_weights=None, seed=0) -> dict[str, Any]` | Offline ensemble: convex weights (>= 0, sum 1) minimising the summed squared error on a validation set. |

#### `pinneapple_physics.ensemble_viz`

Animated view of an adaptive physics ensemble: prediction against the reference field, case by case, with the model the ensemble chose and how its weights moved.

<small>`pinneapple_physics/ensemble_viz.py` · 131 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `animate_ensemble` | função | `(run, references, path='ensemble.gif', *, x=None, times=None, frames_per_case=4, fps=10, …)` | Write a GIF of the ensemble run. ``cases``: subset of case indices to animate (default: all). ``regimes``: ``[(first_case, label), ...]`` drawn as dashed lines on the timeline. Returns the path written. |

#### `pinneapple_physics.experiment`

``Experiment``: one reproducible run of a method on a problem, with its metrics and provenance.

<small>`pinneapple_physics/experiment.py` · 174 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ExperimentResult` | classe | `(config, problem_fingerprint, solution, metrics, wall_time_s, started_at, environment, n_…)` |  |
| `Experiment` | classe | `(problem, method, options, seed, reference, reference_options, n_eval, metric_names, name…)` | A method applied to a problem, with a seed, options and an optional reference to score against. |
| `run_all` | função | `(experiments) -> Dict[str, ExperimentResult]` | Run several experiments; keyed by experiment name. |

#### `pinneapple_physics.maxwell_checks`

Verificações das equações de Maxwell sobre campos numa grade (de PINN, FNO, solver ou medição), sem referência: resíduos relativos com teste em `2h`, balanço de Poynting, contorno PEC, reciprocidade e passividade; alimenta `pinneapple check` (`physics = "maxwell"`).

<small>`pinneapple_physics/maxwell_checks.py` · 549 linhas · testes: 51</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MaxwellCheck` | classe | `(name, key, value, limit, passed, critical, detail, kind)` | Resultado de uma verificação: nome, chave, valor, limite, aprovado, criticidade e detalhe. |
| `FieldSet` | classe | `(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega=None, interior=1)` | Campos numa grade tensorial com derivadas por diferenças centrais, máscara de interior e versão engrossada. |
| `relative_residual` | função | `(terms, mask=None, floor=0.0) -> float` | Resíduo relativo `\|\|soma dos termos\|\| / soma das normas`; NaN quando a lei não restringe o campo. |
| `gauss_electric` | função | `(grid, fields, *, eps=EPS0, tol=None, **kw) -> MaxwellCheck` | Verifica a lei de Gauss elétrica `div D - rho`. |
| `gauss_magnetic` | função | `(grid, fields, *, mu=MU0, tol=None, **kw) -> MaxwellCheck` | Verifica a lei de Gauss magnética `div B`. |
| `faraday` | função | `(grid, fields, *, mu=MU0, omega=None, tol=None, **kw) -> MaxwellCheck` | Verifica a lei de Faraday `curl E + dB/dt`. |
| `ampere_maxwell` | função | `(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega=None, tol=None, **kw) -> MaxwellCheck` | Verifica a lei de Ampère-Maxwell `curl H - J - dD/dt`. |
| `continuity` | função | `(grid, fields, *, omega=None, tol=None, **kw) -> MaxwellCheck` | Verifica a equação de continuidade da carga `d rho/dt + div J`. |
| `poynting_balance` | função | `(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega=None, tol=None) -> MaxwellCheck` | Verifica o teorema de Poynting sobre a caixa da grade. |
| `tangential_continuity` | função | `(F1, F2, normal, *, surface_current=None, tol=None, name='E') -> MaxwellCheck` | Verifica `n x (F1 - F2) = K` em pontos de interface. |
| `pec_boundary` | função | `(grid, fields, faces, *, tol=None) -> MaxwellCheck` | Verifica `n x E = 0` nas faces declaradas PEC. |
| `reciprocity` | função | `(S, *, tol=None) -> MaxwellCheck` | Verifica `S = S^T` de uma rede recíproca. |
| `passivity` | função | `(S, *, lossless=False, tol=None) -> MaxwellCheck` | Verifica passividade (maior valor singular de S no máximo 1) ou unitariedade se sem perdas. |
| `maxwell_checks` | função | `(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega=None, laws=None, pec_faces=(), refer…)` | Roda todas as verificações que os campos fornecidos permitem e, opcionalmente, compara com uma referência. |
| `maxwell_trust_report` | função | `(grid, fields, *, model='', **kwargs)` | Roda `maxwell_checks` e devolve `(TrustReport, checks)` com a decisão de implantação. |
| `sample_checks` | função | `(grid, fields, **kwargs) -> list[dict[str, Any]]` | Verificações no formato que `pinneapple check` usa. |
| `sample_verdict` | função | `(checks) -> str` | PASS ou FAIL a partir da lista de verificações. |
| `save_maxwell_npz` | função | `(path, grid, fields, **extra)` | Grava eixos e componentes de campo (fasores como pares `_re`/`_im`) no formato lido por `pinneapple check`. |
| `check_config_entry` | função | `(name, path, *, parameters=None, tolerances=None) -> str` | Gera um bloco `[[result]]` de `pinneapple-check.toml` para um campo salvo. |

#### `pinneapple_physics.metrics`

``pp.metrics``: the error metrics used everywhere in the library, with one convention.

<small>`pinneapple_physics/metrics.py` · 182 linhas · testes: 36</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `l2` | função | `(pred, true)` | Euclidean norm of the error, per field: ``\|\|pred - true\|\|_2``. |
| `relative_l2` | função | `(pred, true)` | ``\|\|pred - true\|\|_2 / \|\|true\|\|_2`` per field (``nan`` where ``\|\|true\|\| = 0``). |
| `rmse` | função | `(pred, true)` | Root mean square error per field. |
| `mae` | função | `(pred, true)` | Mean absolute error per field. |
| `max_abs` | função | `(pred, true)` | Largest absolute error per field (the L-infinity norm of the error). |
| `relative_linf` | função | `(pred, true)` | ``max\|pred - true\| / max\|true\|`` per field (``nan`` where ``max\|true\| = 0``). |
| `r2` | função | `(pred, true)` | Coefficient of determination per field (``nan`` when ``true`` is constant). |
| `per_field` | função | `(metric, pred, true, fields) -> dict[str, float]` | ``{field: value}`` for an ``(N, F)`` comparison. |
| `summary` | função | `(pred, true, fields, names=…) -> dict[str, dict[str, float]]` | ``{metric: {field: value}}`` for several metrics at once (the shape benchmark tables use). |
| `pooled` | função | `(pred, true) -> dict[str, float]` | One number per metric over every entry, all fields together: ``relative_l2``, ``rmse``, ``mse``, ``max_abs``, ``r2``. For leaderboards that rank models by a single value; use the per-field functions for anything reported per quantity. When the shapes differ, a 1-D side gets a trailing axis and the … |
| `rollout_relative_l2` | função | `(pred_seq, true_seq, *, aggregate='steps') -> float` | Relative L2 error of a time rollout: ``pred_seq`` and ``true_seq`` have shape ``(T, ...)``, axis 0 is time. |

#### `pinneapple_physics.multiscale`

Subpacote de física multiescala para ensembles adaptativos (filtros, fechamentos, resíduos, consistência, seleção por escala); numpy puro, exceto `learned` (PyTorch).

<small>`pinneapple_physics/multiscale/__init__.py` · 90 linhas · testes: 29</small>

#### `pinneapple_physics.multiscale.closures`

Fechamentos de subfiltro para LES 2-D em forma de vorticidade (Smagorinsky, dinâmico, gradiente) e testes a priori, a posteriori e de saúde sem rótulo; a classe base interna levanta `NotImplementedError` em `nu_t`.

<small>`pinneapple_physics/multiscale/closures.py` · 381 linhas · testes: 29 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Smagorinsky` | classe | `(cs, delta, name)` | Fechamento de Smagorinsky `nu_t = (cs delta)^2 \|S\|`. |
| `DynamicSmagorinsky` | classe | `(delta, test_ratio, test_kind, name, last_c)` | Smagorinsky dinâmico com coeficiente calculado por filtro de teste (identidade de Germano, média no domínio). |
| `GradientModel` | classe | `(delta, coef, name)` | Modelo de gradiente (Clark) para a tensão de subfiltro. |
| `NoModel` | classe | `(name)` | Sem modelo de subfiltro (DNS sub-resolvida). |
| `AprioriReport` | classe | `(closure, filter, stress_correlation, tendency_correlation, transfer_true, transfer_model…)` | Resultado do teste a priori: correlações de tensão e tendência, transferência de energia e retroespalhamento. |
| `AposterioriReport` | classe | `(closure, times, spectrum_error, field_error, dissipation_les, dissipation_ref, stable, t…)` | Resultado do teste a posteriori: erros de espectro e campo, dissipação, estabilidade e crescimento de enstrofia. |
| `ClosureHealth` | classe | `(score, ok, reasons, enstrophy_ratio, tail_ratio, germano)` | Pontuação de saúde de um fechamento sem rótulos, com motivos e indicadores. |
| `strain` | função | `(w, solver) -> dict[str, np.ndarray]` | Gradientes de velocidade, componentes de deformação e `\|S\|` a partir da vorticidade. |
| `stress_tendency` | função | `(a, b, solver) -> np.ndarray` | Tendência de vorticidade `-curl(div tau)` de uma tensão desviadora. |
| `true_subfilter` | função | `(w_fine, filt, n_coarse=None, length=2 * np.pi) -> dict[str, np.ndarray]` | Termos exatos de subfiltro de um campo resolvido para um filtro dado. |
| `apriori_test` | função | `(w_fine, filt, closure, n_coarse, length=2 * np.pi) -> AprioriReport` | Teste a priori: compara tensão e tendência do fechamento com os termos exatos de subfiltro. |
| `spectrum_error` | função | `(w, w_ref, kmax, length=2 * np.pi, kmin=1) -> float` | Diferença média de `log10 E(k)` entre dois espectros numa faixa de números de onda. |
| `aposteriori_test` | função | `(les, w0_bar, ref_bar, times, dt=0.005, t_long=None) -> AposterioriReport` | Teste a posteriori: roda a LES com o fechamento e compara com a referência filtrada ao longo do tempo. |
| `closure_health` | função | `(w, w0, kc, length=2 * np.pi, *, forced=False, max_enstrophy_growth=1.05, tail_band=(0.05…)` | Saúde de uma previsão LES sem rótulos (crescimento de enstrofia, acúmulo espectral, erro de Germano). |
| `germano_error` | função | `(w, closure, solver, test_ratio=2.0, test_kind='sharp', delta=None) -> float` | Erro do fechamento na identidade de Germano entre o filtro da grade e um filtro de teste. |

#### `pinneapple_physics.multiscale.comparison`

Compara ensembles num fluxo de consultas multiescala: seleção cega à escala contra sensível à escala, com e sem verificações e modelos que ligam escalas; gera tabelas e figuras.

<small>`pinneapple_physics/multiscale/comparison.py` · 439 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EnsembleConfig` | classe | `(aware, router, checks, experts, error_tolerance, cost_weight)` | Configuração de um ensemble: sensível à escala ou não, roteador, verificações, especialistas, tolerância e peso de custo. |
| `evaluate_experts` | função | `(path, queries, initial_states, experts, checks=None, n_out=None) -> dict[str, np.ndarray]` | Roda cada especialista em cada consulta uma vez e guarda saídas e verificações (falhas viram NaN). |
| `adapt_output` | função | `(raw, source, target, aware, length=2 * np.pi) -> np.ndarray` | Adapta uma saída filtrada num filtro de origem ao filtro pedido, de forma cega ou sensível à escala. |
| `raw_residual` | função | `(pred, nu, dt, length=2 * np.pi) -> float` | Resíduo relativo da equação de vorticidade na escala da grade. |
| `closed_multiscale_residual` | função | `(pred, nu, dt, filt, h, length=2 * np.pi) -> float` | Resíduo da equação filtrada em duas larguras, com a tendência de subfiltro estimada pelo modelo de gradiente. |
| `run_ensembles` | função | `(queries, references, table, profiles, configs, *, dt, h, length=2 * np.pi, cost_weight=0…)` | Roda cada configuração no mesmo fluxo de consultas, reaproveitando as saídas já avaliadas. |
| `summarise_runs` | função | `(runs, profiles, configs, widths, *, wrong_factor=1.5, wrong_margin=0.03, solver=None, or…)` | Resume por configuração o erro médio, erro por largura, escolhas erradas e custo. |
| `closure_flag_table` | função | `(tables, targets, names, kc, length=2 * np.pi) -> list[dict[str, Any]]` | Tabela de como as verificações sem rótulo marcaram cada especialista e o erro quando marcado ou não. |
| `markdown_tables` | função | `(rows, widths, flags=(), sc_rows=(), bench_rows=()) -> str` | Gera as tabelas em Markdown dos resumos e estudos. |
| `summary_figure` | função | `(rows, widths, path, sc_rows=(), colors=…, unit='h') -> Path` | Figura de quatro painéis com erro por largura, erro contra custo, escolhas erradas e estudo de condicionamento. |
| `frontier_figure` | função | `(points, path, baselines=(), colors=…, title=…) -> Path` | Figura de fronteira custo-erro por configuração. |

#### `pinneapple_physics.multiscale.consistency`

Consistência entre escalas: uma previsão fina e uma grossa do mesmo estado devem concordar depois de levadas à mesma escala. [sem teste direto]

<small>`pinneapple_physics/multiscale/consistency.py` · 114 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConsistencyReport` | classe | `(passed, field_error, spectrum_error, integrals, failed, tolerance, reasons)` | Resultado da verificação de consistência: aprovado, erros de campo e espectro, integrais e motivos. |
| `cross_scale_consistency` | função | `(fine, coarse, coarse_filter, fine_filter=None, *, length=2 * np.pi, tol=0.05, integrals=…)` | Filtra e restringe a previsão fina para a escala da grossa e compara campo, espectro e integrais. |

#### `pinneapple_physics.multiscale.filters`

Filtros espaciais (caixa, gaussiano, espectral) em grades periódicas com funções de transferência exatas, restrição, prolongamento, refiltragem e erro de comutação.

<small>`pinneapple_physics/multiscale/filters.py` · 271 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Filter` | classe | `(kind, delta)` | Filtro de largura `delta` e tipo dado; `delta = 0` é a identidade. |
| `transfer` | função | `(kind, k, delta) -> np.ndarray` | Função de transferência 1-D de um filtro. |
| `wavenumbers` | função | `(n, length=2 * np.pi) -> np.ndarray` | Números de onda angulares de uma grade periódica, na ordem da FFT. |
| `apply_filter` | função | `(u, filt, length=2 * np.pi, axes=None) -> np.ndarray` | Filtra um campo periódico nos eixos dados. |
| `restrict` | função | `(u, n, axes=None) -> np.ndarray` | Restrição de Fourier para uma grade periódica mais grossa, preservando a média. |
| `prolong` | função | `(u, n, axes=None) -> np.ndarray` | Interpolação de Fourier (preenchimento com zeros) para uma grade mais fina. |
| `post_filter_transfer` | função | `(source, target, k, max_gain=2.0) -> np.ndarray` | Transferência do pós-filtro que leva de um filtro de origem a um alvo, com ganho limitado. |
| `realizability_mismatch` | função | `(source, target, kmax, *, slope=-2.0, n=4096, max_gain=2.0, kmin=1.0) -> float` | Fração do filtro alvo que não pode ser obtida por pós-filtragem de um campo já filtrado. |
| `refilter` | função | `(u, source, target, length=2 * np.pi, axes=None, max_gain=2.0) -> np.ndarray` | Converte um campo filtrado num filtro para outro filtro, até onde for possível. |
| `filter_variable_width` | função | `(u, kind, delta, length=2 * np.pi, axes=None, decimals=10) -> np.ndarray` | Filtro cuja largura varia no espaço. |
| `spectral_derivative` | função | `(u, axis, length=2 * np.pi, order=1) -> np.ndarray` | Derivada espectral exata do interpolante trigonométrico. |
| `commutation_error` | função | `(u, kind, delta, axis=-1, length=2 * np.pi, axes=None) -> np.ndarray` | Erro de comutação entre filtrar e derivar (não nulo para largura variável). |
| `low_modes` | função | `(kmax) -> list[tuple[int, int]]` | Lista dos modos 2-D de baixa frequência num corte quadrado, um de cada par `k, -k`. |
| `fourier_basis` | função | `(modes, n, length=2 * np.pi) -> np.ndarray` | Base real de cossenos e senos para os modos dados numa grade n x n. |
| `fourier_coefficients` | função | `(w, modes, length=2 * np.pi) -> np.ndarray` | Coeficientes de um campo periódico na base de Fourier. |

#### `pinneapple_physics.multiscale.learned`

Modelos aprendidos que ligam escalas, em PyTorch: FNO condicionado pela largura do filtro, LES diferenciável, fechamento CNN treinado a posteriori e PINN de Fourier-Galerkin.

<small>`pinneapple_physics/multiscale/learned.py` · 501 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScaleConditionedFNO2d` | classe | `(in_channels=1, out_channels=1, width=32, modes=12, layers=4, conditioned=True, delta_ref…)` | FNO periódico cujas camadas são moduladas pela largura do filtro; `conditioned=False` dá a versão cega à escala. |
| `TorchVorticityLES2D` | classe | `(n, length=2 * math.pi, dtype=torch.float64)` | Solver pseudo-espectral 2-D de vorticidade diferenciável (RK4 com fator integrante) com fechamento opcional. |
| `LearnedClosure` | classe | `(hidden=16, layers=3, kernel=5)` | Fechamento CNN periódico que prevê o fluxo de vorticidade de subfiltro. |
| `FourierGalerkinPINN` | classe | `(kmax=4, hidden=256, scale=5.0, t_ref=0.3, length=2 * math.pi)` | Rede informada pela física para as grandes escalas da vorticidade 2-D, treinada só no resíduo de Galerkin truncado, sem dados. |
| `torch_transfer` | função | `(kind, k, delta) -> torch.Tensor` | Função de transferência de filtro em torch. |
| `torch_filter` | função | `(x, kind, delta, length=2 * math.pi) -> torch.Tensor` | Filtra um tensor (B, C, n, n) com largura escalar ou por amostra. |
| `torch_refilter` | função | `(x, kind, d_from, d_to, length=2 * math.pi, max_gain=2.0) -> torch.Tensor` | Leva um tensor filtrado numa largura para outra, por amostra. |
| `filtering_consistency_loss` | função | `(model, x, d_small, d_large, kind='gaussian', length=2 * math.pi) -> torch.Tensor` | Perda de consistência: filtrar a saída na largura pequena deve dar a saída na largura grande. |
| `fit_scale_conditioned` | função | `(model, inputs, targets, kind='gaussian', steps=2000, batch=16, lr=0.002, consistency_wei…)` | Treina um modelo condicionado por escala em várias larguras, com perda relativa mais consistência. |
| `train_closure_aposteriori` | função | `(closure, solver, w0, nu, reference, dt, save_steps, steps=300, batch=8, lr=0.001, seed=0…)` | Treina um fechamento a posteriori, rolando a LES e comparando com a referência filtrada e seus espectros. |
| `as_numpy_closure` | função | `(closure, n, length=2 * math.pi)` | Embrulha um fechamento aprendido para uso em `VorticityNS2D` (numpy, sem gradiente). |
| `periodic_cnn2d` | função | `(in_channels=2, out_channels=2, width=32, dilations=(1, 2, 4, 8, 4, 2, 1)) -> nn.Sequenti…)` | CNN 2-D dilatada com preenchimento circular e GELU. |
| `torus_graph` | função | `(n, offsets=(1, 2, 4)) -> tuple[torch.Tensor, torch.Tensor]` | Arestas de um grafo de grade periódica com vizinhos em várias distâncias. |
| `fit_supervised` | função | `(forward, params, X, Y, steps, seed=0, lr=0.002, batch=16, log='') -> list[float]` | Treino supervisionado com Adam e agenda cosseno no erro quadrático relativo. |

#### `pinneapple_physics.multiscale.ns2d`

Navier-Stokes 2-D incompressível em forma de vorticidade no quadrado periódico, para DNS e LES, com espectro de energia e resíduo da equação de vorticidade.

<small>`pinneapple_physics/multiscale/ns2d.py` · 218 linhas · testes: 45</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `VorticityNS2D` | classe | `(n, nu, length, forcing, drag, closure, dealias)` | Solver pseudo-espectral 2-D de vorticidade com forçamento, arrasto linear e fechamento opcional. |
| `velocity_from_vorticity` | função | `(w, length=2 * np.pi) -> tuple[np.ndarray, np.ndarray]` | Velocidade `(u, v)` de um campo de vorticidade periódico. |
| `energy_spectrum` | função | `(w, length=2 * np.pi, kmax=None) -> tuple[np.ndarray, np.ndarray]` | Espectro de energia cinética por cascas a partir da vorticidade. |
| `random_vorticity` | função | `(rng, n, k0=4.0, energy=0.5, length=2 * np.pi, batch=None) -> np.ndarray` | Vorticidade aleatória com espectro de pico em `k0` e energia total dada. |
| `exact_advection` | função | `(w, length=2 * np.pi) -> np.ndarray` | Termo `div(u w)` sem aliasing, calculado numa grade 3/2 mais fina. |
| `vorticity_residual` | função | `(w_prev, w_next, dt, nu, length=2 * np.pi, forcing=None, drag=0.0, closure_tendency=None)…)` | Resíduo da equação de vorticidade entre dois instantâneos (trapezoidal no tempo). |

#### `pinneapple_physics.multiscale.residuals`

Resíduos físicos avaliados em várias larguras de filtro, separados em partes de subfiltro, comutação e modelo. [sem teste direto]

<small>`pinneapple_physics/multiscale/residuals.py` · 120 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MultiscaleResidual` | classe | `(filters, raw, subfilter, commutation, model)` | Curva de resíduo por largura de filtro, com exportação para dicionário e Markdown. |
| `multiscale_residuals` | função | `(state, residual_fn, filters, *, length=2 * np.pi, subfilter_fn=None, commutation_fn=None…)` | Calcula a curva de resíduo de qualquer equação a partir de funções de resíduo, subfiltro e comutação. |
| `ns2d_multiscale_residuals` | função | `(w_prev, w_next, dt, nu, filters, *, length=2 * np.pi, forcing=None, drag=0.0, subfilter=…)` | Resíduo multiescala da equação de vorticidade 2-D entre dois instantâneos. |

#### `pinneapple_physics.multiscale.scales`

Seleção de modelos sensível à escala: cada modelo declara as escalas que resolve e cada consulta as que precisa. [sem teste direto]

<small>`pinneapple_physics/multiscale/scales.py` · 245 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScaleProfile` | classe | `(name, filter, grid_spacing, closure, re_range, time_step, fidelity, cost, notes)` | Escalas que um modelo representa: filtro da saída, espaçamento da grade, fechamento, faixa de Reynolds, passo de tempo, fidelidade e custo. |
| `ScaleQuery` | classe | `(filter, re, time_scale, kmax)` | Escalas que uma consulta pede: filtro da saída, Reynolds, escala de tempo e número de onda máximo. |
| `ScaleCheck` | classe | `(name, score, covered, mismatch, reasons)` | Resultado da compatibilidade de escala de um modelo com uma consulta. |
| `ScaleRoute` | classe | `(chosen, weights, checks, awake, rung, escalated, reasons)` | Decisão de roteamento: modelo escolhido, pesos, verificações, especialistas ativos e escalonamento. |
| `ScaleAwareRouter` | classe | `(profiles, query_fn=None, *, strength=1.0, floor=0.0001, abstain_below=0.05, ladder=True,…)` | Roteador que pondera especialistas pela compatibilidade de escala, com escada de fidelidade (surrogate, LES, solver resolvido). |
| `scale_compatibility` | função | `(profile, query, *, mismatch_tol=0.25, re_band=0.15, slope=-2.0) -> ScaleCheck` | Diz se um perfil pode responder a uma consulta e quão bem suas premissas de filtragem combinam. |

#### `pinneapple_physics.multiscale.turbulence`

Turbulência 2-D em decaimento como benchmark multiescala: dados de referência, conjuntos filtrados, fluxos de consultas e benchmark de fechamentos.

<small>`pinneapple_physics/multiscale/turbulence.py` · 295 linhas · testes: 29</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DecayingTurbulence2D` | classe | `(length, n_out, t_out, spinup, k0, energy, re_bands, widths, width_probabilities, kinds, …)` | Configuração do benchmark (padrões do experimento em `docs/core_concepts/multiscale_ensembles.md`) com geração de dados e consultas. |
| `benchmark_closures` | função | `(bench, data, closures, cases_per_band=8, times=(0.1, 0.2, 0.3), t_long=2.0, dt=0.005) ->…)` | Avalia fechamentos em casos separados de cada faixa de Reynolds na grade LES. |
| `scale_conditioning_study` | função | `(bench, data, idx, conditioned, fixed, blind=None, widths=(1, 2, 4, 6, 8), kind='gaussian…)` | Erros em cada largura de filtro para modelos condicionados, fixos e cegos. |

#### `pinneapple_physics.multiscale.viz`

Animação de uma rodada de ensemble sensível à escala (escala pedida, modelo escolhido, espectros, pesos). [sem teste direto]

<small>`pinneapple_physics/multiscale/viz.py` · 111 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `animate_multiscale_ensemble` | função | `(references, predictions, active, names, weights, widths, labels, cutoffs, path='multisca…)` | Gera um GIF com um quadro por consulta da rodada do ensemble. |

#### `pinneapple_physics.non_hermitian`

Física não hermitiana: dímeros e cadeias PT-simétricos, pontos excepcionais (EPs) e se eles podem ser sintonizados; numpy, sympy para a parte simbólica e matplotlib para as figuras.

<small>`pinneapple_physics/non_hermitian/__init__.py` · 29 linhas · testes: 8</small>

#### `pinneapple_physics.non_hermitian.ep_tuning`

Verifica simbolicamente se um parâmetro de controle move o EP de um dímero PT-simétrico segundo uma lei desejada, contra o teorema de impossibilidade (registro Zenodo citado no código), e mostra três rotas para contorná-lo.

<small>`pinneapple_physics/non_hermitian/ep_tuning.py` · 226 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EPTuningReport` | classe | `(ep_law, target, residual, exact, degree, perturbation_class, assumptions, theorem_applie…)` | Relatório da análise: lei do EP, alvo, resíduo, se é exata, classe da perturbação, premissas e rotas. |
| `classify_coupling` | função | `(b_ab, b_ba=None, lam=None) -> str` | Classifica o acoplamento como constante, só de fase, recíproco polinomial, recíproco não polinomial ou não recíproco. |
| `analyze_ep_tuning` | função | `(b_ab, b_ba=None, target=None, lam=None, dimension=2) -> EPTuningReport` | Checagem simbólica de uma lei de sintonia do EP contra o alvo e o teorema. |
| `coupling_classes` | função | `() -> list[tuple[str, str, Any, Any]]` | As quatro classes de acoplamento do artigo como leis sympy prontas para `analyze_ep_tuning`. |
| `route_a_sqrt_coupling` | função | `(J, G) -> Callable[[float], complex]` | Rota A: acoplamento `sqrt(J^2 + lam J G)`, que dá exatamente a lei linear em `gamma_EP^2`. |
| `route_c_asymmetric` | função | `(J, G)` | Rota C: acoplamento não recíproco que dá a lei linear, mas quebra `H12 = conj(H21)`. |
| `feedback_coupling` | função | `(J, G, lam, b0=None, iters=60, tol=1e-14) -> tuple[float, int]` | Rota B: acoplamento atualizado por realimentação cujo ponto fixo é o acoplamento da rota A. |
| `theorem_profile` | função | `()` | O teorema como `ModelProfile` de `physics_audit`, para auditar se ele restringe um projeto. |
| `design_features` | função | `(H, polynomial) -> dict[str, float]` | Características de um Hamiltoniano de dispositivo para a auditoria de `theorem_profile`. |

#### `pinneapple_physics.non_hermitian.pt_dimer`

Dímeros e cadeias PT-simétricos: Hamiltoniano, espectro, EPs (fórmula fechada e numérico) e coalescência de autovetores. [sem teste direto]

<small>`pinneapple_physics/non_hermitian/pt_dimer.py` · 185 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PTDimer` | classe | `(coupling, coupling_ba, onsite)` | Sistema de dois sítios com ganho e perda, acoplamento constante ou função do parâmetro de controle, recíproco ou não. |
| `PTChain` | classe | `(n, b, b_back)` | Cadeia de N sítios com ganho no primeiro, perda no último e acoplamento entre vizinhos. |
| `pt_violation` | função | `(H) -> float` | Medida de violação da simetria PT de um Hamiltoniano (0 se simétrico). |
| `eigenvector_overlap` | função | `(H) -> float` | Maior sobreposição normalizada entre autovetores à direita (1 num EP). |
| `petermann_factor` | função | `(H) -> np.ndarray` | Fator de Petermann de cada modo (1 para matriz hermitiana, diverge num EP). |

#### `pinneapple_physics.non_hermitian.viz`

Figuras matplotlib para dímeros PT e sintonia de EP.

<small>`pinneapple_physics/non_hermitian/viz.py` · 121 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plot_spectrum` | função | `(ax, dimer, lam=0.0, gamma_max=None, n=400)` | Partes real e imaginária dos autovalores contra gamma, com o EP e a sobreposição de autovetores. |
| `plot_ep_laws` | função | `(ax, J=1.0, G=0.5, lam=None)` | `gamma_EP^2(lam)` de cada classe de acoplamento contra o alvo linear. |
| `plot_coupling_plane` | função | `(ax, J=1.0, G=0.5, lam=None)` | Trajetória do acoplamento no plano complexo; `gamma_EP` é a distância à origem. |
| `plot_chain_threshold` | função | `(ax, sizes=(2, 3, 4, 5), b_values=None)` | Limiar de quebra PT de cadeias de N sítios contra `\|b\|^2`. |
| `ep_tuning_figure` | função | `(path, J=1.0, G=0.5) -> str` | Monta e grava a figura 2x2 da sintonia de EP e devolve o caminho. |

#### `pinneapple_physics.online_learning`

Online learning with expert advice, shared by every adaptive method in the library (time-series forecasting in ``pinneapple_systems.time_series.adaptive`` and physics-model ensembles in ``pinneapple_physics.ensemble``).

<small>`pinneapple_physics/online_learning.py` · 126 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FixedShare` | classe | `(n, eta, alpha)` | Exponentially weighted average with fixed share (one per horizon). |
| `AdaHedge` | classe | `(k)` | Parameter-free Hedge: follow the leader while it works, hedge with eta = ln K / Delta otherwise. |
| `AdaptiveConformal` | classe | `(level=0.9, gamma=0.01, window=500)` | Adaptive conformal inference on absolute residuals: alpha_{t+1} = alpha_t + gamma (alpha - err_t). |
| `FixedShareGrid` | classe | `(n, etas=(0.5, 2.0, 8.0), alphas=(0.0, 0.01, 0.05))` | Fixed-Share aggregators over ``n`` experts for every (eta, alpha) in the grid, weighted by AdaHedge. |

#### `pinneapple_physics.paper_problems`

Test problems of published physics-ML papers, as the papers (or, where the paper is silent, their official code) define them: domain, equation, initial / boundary data, training and test grids, and a reference solution.

<small>`pinneapple_physics/paper_problems.py` · 114 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GridCase` | classe | `(name, equation, x_range, t_range, exact, n_train, n_test, bc, lbfgs_steps, coef, source)` | One test problem on a tensor (x, t) grid with an analytic reference. |
| `tensor_grid` | função | `(x_range, t_range, nx, nt) -> dict[str, np.ndarray]` | The official ``get_data`` grid: ``meshgrid(x, t)`` ('xy' indexing, rows are times), flattened row-major. |
| `time_sequence` | função | `(points, k=5, dt=0.0001) -> np.ndarray` | PINNsFormer's pseudo-sequence: each (x, t) repeated k times with t + i*dt, i = 0..k-1 -> (N, k, 2). |
| `convection_exact` | função | `(x, t, beta=50.0)` | u_t + beta u_x = 0, u(x, 0) = sin x, periodic on [0, 2 pi]: u = sin(x - beta t). |
| `reaction_exact` | função | `(x, t, rho=5.0)` | u_t = rho u (1 - u), u(x, 0) = h(x) = exp(-(x - pi)^2 / (2 (pi/4)^2)): logistic solution u = h e^{rho t} / (h e^{rho t} + 1 - h). |
| `wave_exact` | função | `(x, t, c2=4.0)` | u_tt = c2 u_xx on [0, 1], u(0) = u(1) = 0, u(x, 0) = sin(pi x) + 0.5 sin(3 pi x), u_t(x, 0) = 0: u = sin(pi x) cos(c pi t) + 0.5 sin(3 pi x) cos(3 c pi t), c = sqrt(c2). |
| `pinnsformer_case` | função | `(name) -> GridCase` |  |

#### `pinneapple_physics.paper_problems_xtfc`

The test problems of the X-TFC paper (Schiassi et al., arXiv:2005.10632v1), as the paper states them: equation, domain, constraints, exact solution and the run settings it reports (neurons, activation, weight range, training grid). No result of the paper is in this module; the targets live in …

<small>`pinneapple_physics/paper_problems_xtfc.py` · 214 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `XTFCCase` | classe | `(name, problem, exact, n_train, n_neurons, activation, weight_range, paper_iterations, pa…)` |  |
| `p6_rhs` | função | `(X)` |  |
| `p7_rhs` | função | `(X)` |  |
| `xtfc_case` | função | `(name) -> XTFCCase` |  |

#### `pinneapple_physics.pcb`

Populated-PCB thermal model (the engine of the PCB Hotspot app).

<small>`pinneapple_physics/pcb/__init__.py` · 16 linhas · testes: 10</small>

#### `pinneapple_physics.pcb.engine`

PCB Hotspot engine: evaluation, what-if studies and calibration to measurements.

<small>`pinneapple_physics/pcb/engine.py` · 392 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `model_scope` | função | `(comps, env) -> Dict[str, Any]` | What the model covers, which way each simplification errs, what to do about it today and what is planned -- shown in every report. |
| `evaluate` | função | `(board, comps, env, *, quick=False, n_cells=60) -> Dict[str, Any]` | ``quick``: one solve on a coarser grid (interactive dragging). Full: band + checks. |
| `field_3d` | função | `(board, out) -> Dict[str, Any]` |  |
| `what_if` | função | `(board, comps, env, n_cells=48) -> Dict[str, Any]` | Re-solve the board with common fixes and report the effect on the critical component (and on every junction). |
| `calibrate` | função | `(board, comps, env, measurements, *, noise_std_c=1.0, n_ensemble=16, n_iterations=8, seed…)` | Fit surface-cooling and in-plane-spreading scale factors to measured temperatures with PINNeAPPle's Ensemble Kalman Inversion. |

#### `pinneapple_physics.pcb.solver`

3D finite-volume thermal model of a populated PCB.

<small>`pinneapple_physics/pcb/solver.py` · 341 linhas · testes: 16</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Component` | classe | `(name, x_mm, y_mm, power_w, package, side, tj_max_c, theta_jb, theta_jc, w_mm, d_mm, h_mm…)` |  |
| `Board` | classe | `(width_mm, depth_mm, n_copper, thickness_mm, outer_oz, inner_oz, outer_coverage, inner_co…)` |  |
| `validate` | função | `(board, comps) -> List[str]` |  |
| `overlaps` | função | `(comps) -> List[str]` | Same-side component pairs whose footprints overlap (a layout error the heat model would otherwise silently superpose). |
| `solve` | função | `(board, comps, env, *, n_cells=60, h_override=None, k_xy_scale=1.0, h_scale=1.0, max_iter…)` | Steady temperatures. ``h_override`` fixes {'top','bottom'} coefficients (tests / calibration); ``k_xy_scale`` / ``h_scale`` scale the in-plane conductivity and the surface coefficients (calibration / uncertainty). |

#### `pinneapple_physics.pde_environment`

_Módulo sem docstring e sem texto._

<small>`pinneapple_physics/pde_environment/__init__.py` · 105 linhas · **sem teste direto**</small>

#### `pinneapple_physics.pde_environment.builder`

ProblemBuilder — fluent API for constructing ProblemSpec objects.

<small>`pinneapple_physics/pde_environment/builder.py` · 638 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ProblemBuilder` | classe | `(name) -> None` | Fluent builder for ``ProblemSpec``. |

#### `pinneapple_physics.pde_environment.capabilities`

PDE family knowledge base — part of pinneapple_environment.

<small>`pinneapple_physics/pde_environment/capabilities.py` · 314 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PDEFamily` | classe | `(kind, aliases, description, default_params, coords_hint, fields_hint, tags)` | Descriptor for a known PDE family. |
| `list_pde_families` | função | `() -> List[str]` | Return sorted list of canonical PDE kind strings. |
| `get_pde_family` | função | `(kind_or_alias) -> Optional[PDEFamily]` | Look up a PDEFamily by kind string or alias (case-insensitive). |
| `identify_pde` | função | `(description) -> List[Tuple[str, float]]` | Score known PDE families against a free-text description. |
| `suggest_problem_spec` | função | `(description) -> Dict[str, Any]` | Given a free-text description, suggest ProblemSpec constructor arguments. |

#### `pinneapple_physics.pde_environment.condition_sampling`

Sampling points for conditions whose region is given by a selector function.

<small>`pinneapple_physics/pde_environment/condition_sampling.py` · 68 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `sample_condition_points` | função | `(select, domain_bounds, coords, n, rng) -> np.ndarray` | Up to ``n`` points (float32, shape ``(m, len(coords))``) accepted by ``select(X) -> bool mask``. |
| `box_face_normals` | função | `(points, domain_bounds, coords, time_coord='t') -> Tuple[np.ndarray, np.ndarray]` | Outward unit normals of the bounding box at ``points`` (one column per coordinate, 0 for time). |

#### `pinneapple_physics.pde_environment.conditions`

_Módulo sem docstring e sem texto._

<small>`pinneapple_physics/pde_environment/conditions.py` · 495 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ConditionSpec` | classe | `(name, kind, fields, selector_type, selector, value_fn, weight, order, deriv_coord, inter…)` | Generic constraint spec. |
| `DirichletBC` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0) -> Con…)` | Construct a Dirichlet boundary condition. |
| `NeumannBC` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0, order=…)` | Construct a Neumann boundary condition. |
| `RobinBC` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0) -> Con…)` | Construct a Robin boundary condition. |
| `InterfaceBC` | função | `(name, fields, selector_type='all', selector=None, value_fn=None, weight=1.0, k_a=1.0, k_…)` | Construct an interface / coupling boundary condition between two subdomains. |
| `InitialCondition` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0) -> Con…)` | Construct an initial condition. |
| `DataConstraint` | função | `(name, fields=None, selector_type='all', selector=None, value_fn=None, weight=1.0) -> Con…)` | Construct a data constraint (supervised loss at specific points). |

#### `pinneapple_physics.pde_environment.environment_typing`

_Módulo sem docstring e sem texto._

<small>`pinneapple_physics/pde_environment/environment_typing.py` · 6 linhas · testes: 26</small>

#### `pinneapple_physics.pde_environment.presets`

_Módulo sem docstring e sem texto._

<small>`pinneapple_physics/pde_environment/presets/__init__.py` · 14 linhas · testes: 21</small>

#### `pinneapple_physics.pde_environment.presets._curve_geometry`

Parametric curve/polygon geometry for the two tag-based presets whose domain is NOT a box or a circular cylinder: ``axial_compressor_cascade_2d`` (a circular-arc cascade blade) and ``car_external_aero`` (an Ahmed-body- inspired car silhouette). See ``tag_geometry.py`` for where these are used and …

<small>`pinneapple_physics/pde_environment/presets/_curve_geometry.py` · 227 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `circular_arc_cascade_blade_points` | função | `(n, rng, *, flow_angle_in_deg, flow_angle_out_deg, chord, x_le, y_le) -> np.ndarray` |  |
| `ahmed_body_polygon` | função | `(car_length, car_height, *, slant_angle_deg=_AHMED_DEFAULT_SLANT_ANGLE_DEG, n_arc=24, x0=…)` | Closed polygon (M, 2) tracing the Ahmed-body-inspired silhouette, nose at x=x0, tail at x=x0+car_length, underbody at y=ground_clearance, roof at y=car_height. Vertices are ordered counter-clockwise starting at the bottom of the front fillet. |
| `naca4_symmetric_polygon` | função | `(chord, thickness_ratio, *, n=80, x0=0.0, y0=0.0) -> np.ndarray` | Closed polygon (M, 2) tracing a symmetric (zero-camber) NACA 4-digit airfoil of the given ``chord`` and ``thickness_ratio`` t (e.g. t=0.12 for NACA 0012), leading edge at (x0, y0), trailing edge at (x0+chord, y0). Vertices run leading edge -> upper surface -> trailing edge -> lower surface -> back … |
| `polygon_perimeter_sample` | função | `(vertices, n, rng) -> np.ndarray` | Sample n points uniformly (by arc length) along the CLOSED polygon whose vertices are given in order (the closing edge back to vertices[0] is included). |
| `polygon_contains` | função | `(vertices, X) -> np.ndarray` | Vectorized ray-casting point-in-polygon test (closed polygon, vertices in order). Returns a bool array, one per row of X (N, 2). |

#### `pinneapple_physics.pde_environment.presets._utils`

Shared helpers for preset modules (not part of the public API).

<small>`pinneapple_physics/pde_environment/presets/_utils.py` · 10 linhas · **sem teste direto**</small>

#### `pinneapple_physics.pde_environment.presets.academics`

_Módulo sem docstring e sem texto._

<small>`pinneapple_physics/pde_environment/presets/academics.py` · 113 linhas · testes: 37</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `laplace_2d_default` | função | `() -> ProblemSpec` |  |
| `poisson_2d_default` | função | `() -> ProblemSpec` |  |
| `burgers_1d_default` | função | `(nu=0.01) -> ProblemSpec` |  |

#### `pinneapple_physics.pde_environment.presets.astroml_bridge`

astroML-backed statistics for astronomical data: periodograms, deconvolved densities, fits with errors.

<small>`pinneapple_physics/pde_environment/presets/astroml_bridge.py` · 576 linhas · testes: 12 · palavras-chave no código (heurística; o texto acima diz o status real): shim ou obsoleto</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Periodogram` | classe | `(freq, power, method, n_terms, n_obs, false_alarm_levels, meta)` | A periodogram on a frequency grid. ``freq`` in cycles per unit of ``t``; ``power`` dimensionless. |
| `XDResult` | classe | `(weights, means, covariances, log_likelihood, model)` | Extreme-deconvolution Gaussian mixture of the underlying (error-free) distribution. |
| `LineFit` | classe | `(slope, intercept, covariance, chi2, dof, method)` | ``y = slope * x + intercept`` with a 2x2 covariance of ``(intercept, slope)``. |
| `astroml_available` | função | `() -> bool` | True if ``astroML`` can be imported (does not import the heavy sub-packages). |
| `frequency_grid` | função | `(t, period_range=None, f_min=None, f_max=None, oversample=10.0, n_max=200000) -> np.ndarr…)` | Uniform frequency grid [1 / unit of t] for a periodogram of samples at times ``t``. |
| `lomb_scargle_periodogram` | função | `(t, y, dy, freq=None, *, period_range=None, oversample=10.0, generalized=True, significan…)` | Generalised (floating-mean) Lomb-Scargle periodogram, ``astroML.time_series.lomb_scargle``. |
| `multiterm_periodogram` | função | `(t, y, dy, freq=None, *, n_terms=3, period_range=None, oversample=10.0, time_unit='d') ->…)` | Multi-harmonic periodogram, ``astroML.time_series.multiterm_periodogram``. |
| `bootstrap_peak_distribution` | função | `(t, y, dy, freq, n_bootstraps=100, random_state=0, generalized=True) -> np.ndarray` | Highest periodogram peak of ``n_bootstraps`` resamplings of the data (``lomb_scargle_bootstrap``). |
| `best_period` | função | `(t, y, dy, *, period_range=None, n_terms=1, oversample=10.0, refine=True) -> float` | Period [unit of t] of the highest periodogram peak. |
| `light_curve_sample` | função | `(t, y, dy, *, band='', time_unit='d', value_unit='mag', **provenance)` | An irregularly sampled light curve as a point-domain :class:`PhysicalSample` (no astroML needed). |
| `xd_fit` | função | `(X, Xerr, n_components, *, max_iter=200, tol=1e-05, random_state=0) -> XDResult` | Extreme deconvolution, ``astroML.density_estimation.XDGMM``. |
| `bayesian_block_edges` | função | `(t, x=None, sigma=None, *, fitness='events', p0=0.05, gamma=None) -> np.ndarray` | Bin edges from Bayesian Blocks (Scargle et al. 2013), ``astroML.density_estimation.bayesian_blocks``. |
| `knuth_histogram` | função | `(data) -> tuple[np.ndarray, np.ndarray, float]` | Histogram with Knuth's (2006) Bayesian optimal bin width: ``(counts, edges, width)``. |
| `fit_line_with_errors` | função | `(x, y, dy) -> LineFit` | Weighted least squares with errors on ``y`` only, ``astroML.linear_model.LinearRegression(y_error=dy)``. |
| `fit_line_tls` | função | `(x, y, dx, dy, initial=None) -> LineFit` | Line fit with errors on both axes: maximises ``astroML.linear_model.TLS_logL`` (Hogg, Bovy & Lang 2010). |
| `load_dataset` | função | `(name, *, data_home=None, download_if_missing=False, **kwargs)` | Call an ``astroML.datasets`` fetcher by short name (see ``DATASETS``) and return what it returns. |
| `check_period_recovery` | função | `(seed=0, period=0.613, n=150, noise=0.15) -> dict` | Irregular sinusoid (amplitude 0.3, Gaussian noise) over 200 d: recovered period vs the true one. |
| `check_xd_means` | função | `(seed=0, n=600) -> dict` | Two 2-D Gaussians, per-point errors drawn in [0.2, 1.0]: worst error of the XD means / component spread. |
| `check_block_step` | função | `(seed=0, t_step=6.3, p0=0.001) -> dict` | Noisy series (300 points over 10 s, step 1 -> 2, noise 0.2): the interior Bayesian-Blocks edge vs the step. |
| `check_line_fits` | função | `(seed=0) -> dict` | y = 2.5 x - 1 with errors on both axes: worst relative slope error of the WLS and TLS fits. |
| `validate` | função | `(cases=None, log=print, **kw) -> list[dict]` | Run the synthetic checks; returns one row per case and logs a table. Needs astroML, no network. |

#### `pinneapple_physics.pde_environment.presets.astrophysics`

Astrophysics and space-systems problem presets.

<small>`pinneapple_physics/pde_environment/presets/astrophysics.py` · 1.486 linhas · testes: 72</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `kepler_exact_state` | função | `(t, mu, a, e) -> Dict[str, np.ndarray]` | Exact Kepler two-body trajectory at times `t` (perigee at t=0). |
| `kepler_two_body_orbit` | função | `(mu=398600.4418, a=8000.0, e=0.15) -> ProblemSpec` | Restricted two-body (Kepler) orbit -- planar Cartesian formulation. |
| `cw_exact_velocity` | função | `(t, n, x0, y0, z0, vx0, vy0, vz0) -> Dict[str, np.ndarray]` | Time derivative of :func:`_cw_exact_state` (Clohessy-Wiltshire relative velocity, closed form). |
| `space_debris_cw_relative_motion` | função | `(n=0.0011, x0=1.0, y0=0.0, z0=0.2, vx0=0.0, vy0=-0.0015, vz0=0.0008) -> ProblemSpec` | Clohessy-Wiltshire (Hill's) equations -- space-debris close-approach / conjunction-assessment and proximity-operations relative motion. |
| `satellite_j2_perturbation` | função | `(mu=398600.4418, J2=0.00108262668, Re=6378.137, a=7000.0, e=0.001, inclination_deg=98.7) …)` | LEO satellite orbit with the J2 (Earth-oblateness) perturbation. |
| `spacecraft_attitude_euler_rotation` | função | `(I1=100.0, I3=150.0, w1_0=0.05, w3_0=0.5) -> ProblemSpec` | Torque-free rigid-body spacecraft attitude dynamics (axisymmetric). |
| `lane_emden_exact_theta` | função | `(xi, n) -> np.ndarray` | Closed-form Lane-Emden solution for n in {0, 1, 5} (Chandrasekhar, 1939). All three verified this session with `sympy`: substituted into theta'' + (2/xi)theta' + theta^n = 0 and confirmed exact zero residual. Raises for any other n (no closed form exists in general; must be solved numerically, e.g. … |
| `lane_emden_polytrope` | função | `(n=1.0, xi_min=0.001, xi_max=3.0) -> ProblemSpec` | Lane-Emden equation for a self-gravitating polytropic star. |
| `nfw_potential_exact` | função | `(r, G, rho_s, rs) -> np.ndarray` | Closed-form NFW gravitational potential (Navarro, Frenk & White, 1996/1997). Verified this session with `sympy`: the spherical Laplacian of this expression equals 4*pi*G*rho_NFW(r) exactly (checked symbolically, zero residual) for the NFW density profile rho(r) = rho_s / [(r/rs)(1+r/rs)^2]. |
| `nfw_source_fn` | função | `(G, rho_s, rs)` | Return a ``ctx["source_fn"]`` callable for the "poisson" PDE kind: f(X, ctx) = 4*pi*G*rho_NFW(\|X\|), matching the density profile whose potential is `nfw_potential_exact` (both verified together, see `nfw_potential_exact`'s docstring). |
| `nfw_dark_matter_potential` | função | `(G=1.0, rho_s=1.0, rs=1.0, r_max=10.0) -> ProblemSpec` | Gravitational potential of a Navarro-Frenk-White (NFW) dark-matter halo -- Poisson's equation with the NFW density profile as source. |
| `sod_exact_solution` | função | `(x, t, gamma=1.4, rho_l=1.0, u_l=0.0, p_l=1.0, rho_r=0.125, u_r=0.0, p_r=0.1, x0=0.5) -> …)` | Exact Riemann solution for the Sod shock tube (Sod, 1978; algorithm per Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics", 3rd ed., Ch. 4). Returns rho, u, p at positions `x` and time `t`. |
| `sod_shock_tube_astro` | função | `(gamma=1.4, x0=0.5, t_end=0.2) -> ProblemSpec` | 1D compressible Euler shock tube (Sod, 1978) -- the standard validation case for every astrophysical hydrodynamics code (used to verify e.g. FLASH, Athena++, RAMSES, Enzo on release). Represents an idealized discontinuity (e.g. a contact between two interstellar- medium phases, or the initial … |
| `cr3bp_omega_gradient` | função | `(x, y, mu)` | Gradient of the CR3BP effective (synodic-frame) potential Omega(x,y) = 0.5*(x^2+y^2) + (1-mu)/r1 + mu/r2, i.e. the RHS of the equations of motion x'' - 2y' = dOmega/dx, y'' + 2x' = dOmega/dy (Szebehely, "Theory of Orbits", 1967, Ch. 1-2). A pure-numpy reference implementation, independent of the … |
| `cr3bp_collinear_lagrange_point` | função | `(mu, x0, tol=1e-14, max_iter=100) -> float` | Solve for a collinear (L1/L2/L3) Lagrange point via 1D Newton- Raphson on dOmega/dx=0 restricted to y=0 (same algorithm style as this module's `_kepler_solve_E`: a plain, dependency-free Newton-Raphson, not scipy). `x0` must be a starting guess in the correct branch (see `cr3bp_lagrange_points` for … |
| `cr3bp_lagrange_points` | função | `(mu) -> Dict[str, tuple]` | All 5 Lagrange-point equilibria of the planar CR3BP for a given mass ratio `mu`. L4/L5 are the EXACT equilateral-triangle points (x=1/2-mu, y=+-sqrt(3)/2) -- true for any mu, verified symbolically with `sympy` this session (dOmega/dx and dOmega/dy both reduce to exactly 0 there, since r1=r2=1 by … |
| `cr3bp_planar_synodic` | função | `(mu=0.012150585609624, dx0=0.0001, dy0=0.0, n_periods=4.0) -> ProblemSpec` | Planar circular restricted three-body problem (CR3BP), Earth-Moon system, synodic (co-rotating) frame -- Szebehely's standard dimensionless normalization: unit distance = Earth-Moon separation (~384,400 km), unit time such that G(m_Earth+m_Moon)=1 and the frame's angular velocity = 1 (so 1 time … |
| `schwarzschild_light_bending_u_exact` | função | `(phi, m, b) -> np.ndarray` | First-order (weak-field, m/b << 1) perturbative solution of the EXACT Schwarzschild null-geodesic equation d^2u/dphi^2 + u = 3 m u^2 (u := 1/r, m := GM/c^2), expressed in terms of the asymptotic impact parameter b: u(phi) = cos(phi)/b + (m/b^2)*(1 + sin(phi)^2) |
| `schwarzschild_deflection_angle_weak_field` | função | `(GM, c, b) -> float` | Einstein's weak-field light-deflection formula, delta_phi = 4GM/(c^2 b) (Einstein, 1916; famously confirmed observationally for starlight grazing the Sun by Dyson, Eddington & Davidson's 1919 eclipse expedition -- twice the Newtonian/light-corpuscle deflection of 2GM/(c^2 b)). |
| `schwarzschild_light_bending_weak_field` | função | `(GM=1.32712440018e+20, c=299792458.0, b=695700000.0) -> ProblemSpec` | Light deflection by a spherical mass in the Schwarzschild weak-field limit -- the null-geodesic ("photon orbit") equation, u := 1/r as a function of the orbital angle phi: |
| `shakura_sunyaev_flux_exact` | função | `(r, GM, Mdot, R_in) -> np.ndarray` | Closed-form radial flux profile of a steady, geometrically-thin, optically-thick alpha-disk (Shakura & Sunyaev, 1973, A&A, 24, 337), F(r) := sigma_SB * T_eff(r)^4 (so T_eff(r) = (F(r)/sigma_SB)^(1/4)): |
| `shakura_sunyaev_teff_exact` | função | `(r, GM, Mdot, R_in, sigma_SB=5.670374419e-08) -> np.ndarray` | Effective temperature profile T_eff(r) = (F(r)/sigma_SB)^(1/4), the classic Shakura-Sunyaev (1973) alpha-disk result. Clamps F to >= 0 before the 1/4 power (r <= R_in gives F=0, i.e. no disk material interior to the zero-torque truncation radius). |
| `shakura_sunyaev_accretion_disk` | função | `(M_bh_solar=10.0, Mdot_edd_frac=0.1, eta=0.1, r_out_factor=10000.0) -> ProblemSpec` | Steady, geometrically-thin, optically-thick alpha-disk (Shakura & Sunyaev, 1973, A&A, 24, 337) around a compact object -- the canonical accretion-disk model underlying essentially every X-ray-binary and AGN accretion-disk spectral calculation since the 1970s. |

#### `pinneapple_physics.pde_environment.presets.astropy_bridge`

Astropy-backed unit handling and coordinate transforms for astrophysics.

<small>`pinneapple_physics/pde_environment/presets/astropy_bridge.py` · 296 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `astro_constant_si` | função | `(name) -> float` | Return the SI-unit float value of ``ASTRO_CONSTANTS[name]``. |
| `constant_provenance` | função | `(name) -> str` | Return the real astropy literature reference string for a constant. |
| `with_units` | função | `(value, unit_str=None) -> 'u.Quantity'` | Attach real astropy units to a bare number, or parse a unit string. |
| `to_si` | função | `(quantity, target_unit=None) -> float` | Convert an astropy Quantity (or a parsable string like "5 Msun") to a plain float, in SI units by default or in ``target_unit`` if given. |
| `transform_coordinates` | função | `(x, y, z, obstime, frame_in='gcrs', frame_out='itrs', vx=None, vy=None, vz=None, unit='km…)` | Transform a Cartesian position (and optional velocity) between GCRS and ITRS using real ``astropy.coordinates`` machinery. |

#### `pinneapple_physics.pde_environment.presets.cfd`

_Módulo sem docstring e sem texto._

<small>`pinneapple_physics/pde_environment/presets/cfd.py` · 459 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ns_incompressible_2d_default` | função | `(Re=100.0, Umax=1.0) -> ProblemSpec` |  |
| `ns_incompressible_3d_default` | função | `(Re=100.0, Umax=1.0) -> ProblemSpec` |  |
| `lid_driven_cavity_3d` | função | `(Re=100.0, size=1.0, lid_velocity=1.0) -> ProblemSpec` | 3D Lid-Driven Cavity benchmark. |
| `channel_flow_3d` | função | `(Re=100.0, length=2.0, height=1.0, width=1.0, Umax=1.0) -> ProblemSpec` | 3D pressure-driven rectangular channel (Poiseuille) flow. |
| `pipe_flow_3d` | função | `(Re=100.0, radius=0.5, length=2.0, Umax=1.0) -> ProblemSpec` | 3D Hagen-Poiseuille pipe flow in a circular cylinder. |

#### `pinneapple_physics.pde_environment.presets.electromagnetics`

Electromagnetic presets: each has a closed-form solution (``pp.solve(problem, "analytic")``, registered from :mod:`pinneapple_physics.closed_form.electromagnetics`), so any method can be scored against the exact field.

<small>`pinneapple_physics/pde_environment/presets/electromagnetics.py` · 184 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `em_two_dielectric_capacitor` | função | `(V=1.0, eps1=1.0, eps2=4.0, d1=0.5, h=0.02) -> ProblemSpec` |  |
| `em_two_dielectric_capacitor_differential` | função | `(V=1.0, eps1=1.0, eps2=4.0, d1=0.5) -> ProblemSpec` |  |
| `em_wire_az` | função | `(a=0.25, J=1.0, nu=1.0) -> ProblemSpec` |  |
| `em_tm_cavity` | função | `(m=1, n=1, periods=0.5) -> ProblemSpec` |  |
| `em_tm_cavity_integral` | função | `(m=1, n=1, periods=0.5, h=0.02) -> ProblemSpec` |  |
| `em_skin_effect` | função | `(delta=0.2, omega=1.0, nu=1.0, length=1.0) -> ProblemSpec` |  |
| `em_cylinder_scattering` | função | `(k0=2.0 * math.pi, radius=0.25, eps_cyl=4.0, eps_cyl_i=0.0, half_width=1.0) -> ProblemSpec` |  |

#### `pinneapple_physics.pde_environment.presets.engineering`

Engineering problem presets.

<small>`pinneapple_physics/pde_environment/presets/engineering.py` · 1.169 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `rocket_nozzle_cfd` | função | `(gamma=1.4, R_gas=287.0, T_inlet=3500.0, p_inlet=10000000.0, throat_radius=0.1, exit_radi…)` | 2D axisymmetric compressible flow through a convergent-divergent rocket nozzle. |
| `rocket_structural` | função | `(E=200000000000.0, nu=0.3, p_internal=10000000.0, T_inner=800.0, T_outer=293.0, alpha_T=1…)` | Rocket motor casing under combined internal pressure + thermal gradient. |
| `aircraft_wing_aerodynamics` | função | `(Re=5000000.0, Ma=0.3, alpha_deg=5.0, chord=1.0, rho_inf=1.225, U_inf=102.0, nu_air=1.5e-…)` | 2D incompressible/low-Ma airfoil aerodynamics (NACA-like profile). |
| `aircraft_wing_structural` | função | `(E=70000000000.0, nu=0.33, lift_load=50000.0, span=5.0, thickness=0.1) -> ProblemSpec` | 2D plane stress model of an aircraft wing spar under lift bending load. |
| `car_external_aero` | função | `(U_inf=33.3, rho=1.225, nu_air=1.5e-05, car_length=4.5, car_height=1.4) -> ProblemSpec` | 2D external aerodynamics around a simplified car body. |
| `car_brake_thermal` | função | `(k_disc=55.0, rho_disc=7100.0, cp_disc=500.0, q_friction=2000000.0, T_ambient=293.0, h_co…)` | Transient heat conduction in a brake disc during emergency braking. |
| `car_suspension_fatigue` | função | `(E=210000000000.0, nu=0.3, F_vertical=8000.0, F_lateral=3000.0, arm_length=0.35, arm_widt…)` | Plane stress fatigue analysis of a suspension wishbone arm. |
| `cpu_heatsink_thermal` | função | `(k_aluminium=205.0, k_copper=400.0, q_cpu=200.0, T_ambient=293.0, h_fin=50.0, die_area=0.…)` | 3D steady-state heat conduction/convection in a CPU heatsink. |
| `pcb_thermal` | função | `(k_pcb=0.3, k_z_pcb=0.25, q_components=None, T_ambient=308.0, h_natural=10.0, board_lengt…)` | 2D PCB heat spreading model with multiple component heat sources. |
| `fan_cooler_cfd` | função | `(rpm=2000.0, blade_radius=0.05, hub_radius=0.01, rho=1.225, nu_air=1.5e-05, n_blades=7) -…)` | 2D radial fan flow in a PC cooler. |
| `industrial_furnace_thermal` | função | `(T_hot_gas=1600.0, T_ambient=300.0, k_refractory=1.5, k_insulation=0.1, h_hot_gas=50.0, h…)` | 3D steady-state heat transfer in an industrial furnace wall. |
| `refractory_lining` | função | `(layers=None, T_hot=1700.0, T_cold=350.0, total_thickness=0.5) -> ProblemSpec` | Multi-layer refractory wall with different conductivities. |
| `furnace_combustion_zone` | função | `(T_flame=2200.0, T_wall=1600.0, rho_flue=0.24, nu_flue=0.0002, U_inlet=5.0, furnace_lengt…)` | Simplified combustion zone: hot gas flow with volumetric heat release. |
| `datacenter_airflow_2d` | função | `(U_cold_aisle=2.5, T_cold_air=291.0, T_hot_max=318.0, Q_rack=20000.0, rack_height=2.0, ra…)` | 2D server rack row cooling (cold-aisle/hot-aisle containment). |
| `datacenter_server_thermal` | função | `(Q_cpu=250.0, Q_gpu=400.0, Q_ram=20.0, T_inlet_air=295.0, U_fan=3.0, h_forced=120.0, k_bo…)` | 2D server board conduction + forced convection from fans. |
| `datacenter_cfd_3d` | função | `(room_length=20.0, room_width=10.0, room_height=3.0, n_racks=20, Q_per_rack=15000.0, T_su…)` | Simplified 3D datacenter room CFD for hot-spot prediction. |

#### `pinneapple_physics.pde_environment.presets.industry`

_Módulo sem docstring e sem texto._

<small>`pinneapple_physics/pde_environment/presets/industry.py` · 347 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `steady_heat_conduction_3d_default` | função | `() -> ProblemSpec` |  |
| `transient_heat_3d_default` | função | `() -> ProblemSpec` |  |
| `linear_elasticity_3d_default` | função | `() -> ProblemSpec` |  |
| `darcy_pressure_only_3d_default` | função | `() -> ProblemSpec` |  |
| `helmholtz_acoustics_3d_default` | função | `() -> ProblemSpec` |  |
| `wave_ultrasound_3d_default` | função | `() -> ProblemSpec` |  |
| `reaction_diffusion_2d_default` | função | `() -> ProblemSpec` |  |

#### `pinneapple_physics.pde_environment.presets.multidisciplinary`

Multidisciplinary problem presets.

<small>`pinneapple_physics/pde_environment/presets/multidisciplinary.py` · 1.100 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `climate_atmosphere_2d` | função | `(g=9.81, f=0.0001, H0=1000.0, U_ref=10.0, lon_min=0.0, lon_max=360.0, lat_min=-90.0, lat_…)` | 2D shallow water equations for atmospheric dynamics. |
| `climate_ocean_gyre` | função | `(beta=2e-11, tau0=0.1, r=1e-07, rho0=1025.0, H=500.0, L=1000000.0, W=500000.0) -> Problem…)` | Ocean gyre circulation — Stommel beta-plane model. |
| `crystal_phonon` | função | `(k=150.0, tau=1e-12, vg=3000.0, T_hot=320.0, T_cold=280.0, L_domain=1e-06) -> ProblemSpec` | Phonon transport / lattice dynamics in 1D — simplified Boltzmann transport. |
| `material_fracture_2d` | função | `(E=210000000000.0, nu=0.3, Gc=2700.0, l0=0.005, Lx=1.0, Ly=1.0) -> ProblemSpec` | Phase-field fracture model 2D — Bourdin-Francfort-Marigo formulation. |
| `black_scholes_1d` | função | `(sigma=0.2, r=0.05, K=100.0, T=1.0) -> ProblemSpec` | Black-Scholes PDE for European call option pricing. |
| `heston_pde_2d` | função | `(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.05, K=100.0, T=1.0, S_max=300.0, v_max…)` | Heston stochastic volatility PDE for European option pricing (2D). |
| `pk_two_compartment` | função | `(k12=0.5, k21=0.3, kel=0.2, V1=10.0, dose=500.0, t_end=24.0) -> ProblemSpec` | Two-compartment pharmacokinetic (PK) model — ODE system. |
| `drug_diffusion_tissue` | função | `(D=1e-10, lam=5e-05, C0=1.0, Lx=0.01, Ly=0.005, t_end=3600.0) -> ProblemSpec` | Drug diffusion and first-order metabolism in 2D tissue. |
| `sir_epidemic` | função | `(beta=0.3, gamma=0.1, N=1000000.0, I0=100.0, t_end=200.0) -> ProblemSpec` | SIR epidemic model — ODE system. |
| `opinion_dynamics_2d` | função | `(D=0.01, alpha=1.0, Lx=1.0, Ly=1.0, t_end=10.0) -> ProblemSpec` | 2D spatial opinion dynamics PDE — continuum Hegselmann-Krause model. |

#### `pinneapple_physics.pde_environment.presets.registry`

Registry mapping problem preset names to factory functions.

<small>`pinneapple_physics/pde_environment/presets/registry.py` · 106 linhas · testes: 119</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register_preset` | função | `(name)` | Decorator to register a preset factory function by name. |
| `get_preset` | função | `(name, **kwargs) -> ProblemSpec` | Instantiate a problem preset by name. |
| `list_presets` | função | `() -> List[str]` | Return sorted list of all registered preset names. |

#### `pinneapple_physics.pde_environment.presets.solid_mechanics`

Solid mechanics presets — elasticity, axisymmetric, contact problems.

<small>`pinneapple_physics/pde_environment/presets/solid_mechanics.py` · 669 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `axisymmetric_linear_elasticity_2d_default` | função | `(r_min=10.0, r_max=50.0, z_min=0.0, z_max=100.0, E=210000000000.0, nu=0.3, p_inner=0.0, p…)` | 2D axisymmetric linear elasticity in (r, z) coordinates. |
| `thick_walled_cylinder_lame_default` | função | `(a=20.0, b=60.0, L=100.0, p_a=100000000.0, p_b=0.0, E=210000000000.0, nu=0.3) -> ProblemS…)` | Thick-walled hollow cylinder under internal/external pressure (Lamé problem). |
| `lame_analytical` | função | `(r, a, b, p_a, p_b, E, nu) -> dict` | Lamé analytical solution for thick-walled cylinder. |
| `threaded_coupling_tc50_box_default` | função | `(clearance=0.1, thread_height=0.8, offset=1.0, E=210000000000.0, nu=0.3, traction_top=100…)` | 2D axisymmetric FEM preset for TC50 threaded coupling BOX body. |
| `threaded_coupling_tc50_rotating_default` | função | `(body='BOX', E=210000000000.0, nu=0.3, p_inner=20000000.0, F_axial=500000.0, T_torque=400…)` | TC50 threaded coupling under combined loading. |
| `threaded_coupling_tc50_pin_default` | função | `(thread_height=0.8, E=210000000000.0, nu=0.3, traction_top=100000000.0) -> ProblemSpec` | TC50 threaded coupling PIN body — same physics as BOX, different geometry bounds. |

#### `pinneapple_physics.pde_environment.presets.structural`

Structural mechanics problem presets.

<small>`pinneapple_physics/pde_environment/presets/structural.py` · 429 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `plane_stress_2d_default` | função | `(E=210000000000.0, nu=0.3, load_x=0.0, load_y=-1000.0) -> ProblemSpec` | 2D plane stress linear elasticity. |
| `plane_strain_2d_default` | função | `(E=210000000000.0, nu=0.3) -> ProblemSpec` | 2D plane strain linear elasticity (deep structures, dams, tunnels). |
| `von_mises_2d_default` | função | `(E=210000000000.0, nu=0.3) -> ProblemSpec` | Plane stress with von Mises yield criterion as extra output field. |
| `linear_elasticity_3d` | função | `(E=210000000000.0, nu=0.3) -> ProblemSpec` | 3D linear elasticity for brackets, frames, and general structures. |
| `rotary_coupling_torsion_default` | função | `(E=210000000000.0, nu=0.3, r_inner=0.05, r_outer=0.08, torque=10000.0, axial_force=500000…)` | Rotary coupling under combined torsion and axial tension. |
| `thermoelasticity_2d_default` | função | `(E=70000000000.0, nu=0.33, alpha_T=2.3e-05, dT=100.0) -> ProblemSpec` | 2D thermoelastic coupling: thermal expansion due to temperature change. |

#### `pinneapple_physics.pde_environment.presets.tag_geometry`

Real-geometry fixtures for ``selector_type="tag"`` presets.

<small>`pinneapple_physics/pde_environment/presets/tag_geometry.py` · 816 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_tag_batch` | função | `(name, spec, *, n_col=20000, n_bc_per_face=3000, seed=7, user_ctx=None)` | Build a real-geometry batch for preset ``name`` using its :data:`TAG_GEOMETRY_FIXTURES` entry. Raises ``KeyError`` if ``name`` has no fixture (i.e. it is one of the ``NOT_FIXABLE_WITHOUT_REAL_ GEOMETRY`` presets, or an untagged preset that needs no fixture at all). |
| `tag_solve_kwargs` | função | `(name, spec, *, n_bc_per_face=1000, seed=7) -> Dict[str, Any]` | The extra ``solve_pde`` keyword arguments preset ``name`` needs: ``{}`` when ``spec`` has no ``selector_type="tag"`` condition, the :func:`build_tag_batch` geometry when it has a fixture here, and ``KeyError`` otherwise (the honest failure: no geometry is invented):: |
| `solve_pde_kwargs_from_batch` | função | `(batch) -> Dict[str, Any]` | Strip the ``x_col`` key (``solve_pde`` samples its own collocation points from ``spec.domain_bounds`` and doesn't accept one) and convert numpy arrays to torch tensors, ready to splat into ``solve_pde(spec, model, **solve_pde_kwargs_from_batch(batch))``. |
| `per_tag_losses` | função | `(spec, model, batch, *, w_pde=1.0, w_bc=10.0) -> dict[str, float]` | Every loss term of ``compile_problem(spec)`` for ``model`` on a :func:`build_tag_batch` batch, as floats: ``total``, ``pde`` and one entry per boundary condition (``bc_<name>``), so the per-tag masking can be seen to do distinct work. Numpy arrays of the batch are converted to torch tensors … |
| `preset_name` | função | `(spec) -> str` | The registry key this spec came from: ``problem_id``, the ``meta["preset"]`` stamp :func:`~pinneapple_physics.pde_environment.presets.registry.get_preset` writes, or ``""``. |
| `has_default_geometry` | função | `(spec) -> bool` | True when this spec's tag conditions can be satisfied by its own fixture. |
| `default_geometry` | função | `(spec, *, n_bc_per_face=1000, seed=7) -> Dict[str, Any]` | The ``solve_pde`` keyword arguments that satisfy this spec's tag conditions with the preset's own box or cylinder geometry, or ``{}`` when the spec has no fixture. |
| `default_geometry_report` | função | `(spec) -> str` | One line for a card or a log: which geometry a preset uses for its tagged boundaries. |

#### `pinneapple_physics.pde_environment.presets.terramechanics`

Terramechanics physics residuals and preset for Bekker-Wong PINN surrogate.

<small>`pinneapple_physics/pde_environment/presets/terramechanics.py` · 200 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TerramechanicsResiduals` | classe | `(c_Pa=1400.0, phi_deg=30.0, R_m=0.125, b_m=0.06, n_phys=256, R_factor=0.06, slip_range=(0…)` | Physics residuals for the Bekker-Wong PINN surrogate. |
| `bekker_wong_surrogate_2d` | função | `(c_Pa=1400.0, phi_deg=30.0, R_m=0.125, b_m=0.06) -> ProblemSpec` | Return a ProblemSpec for the 2-D Bekker-Wong PINN surrogate. |

#### `pinneapple_physics.pde_environment.presets.turbomachinery`

Turbomachinery problem presets.

<small>`pinneapple_physics/pde_environment/presets/turbomachinery.py` · 480 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `axial_compressor_meanline` | função | `(num_stages=5, pressure_ratio=3.0, mass_flow_rate=4.37, rpm=10000.0, inlet_total_pressure…)` | 1D mean-line thermodynamic analysis of a multi-stage axial compressor. |
| `axial_compressor_cascade_2d` | função | `(inlet_mach=0.5, flow_angle_in_deg=50.0, flow_angle_out_deg=20.0, chord=0.1, pitch_to_cho…)` | 2D compressible Euler flow through an axial compressor blade cascade. |
| `axial_compressor_stage_3d` | função | `(rpm=10000.0, pressure_ratio_stage=1.4, mass_flow_rate=4.37, hub_radius=0.1, tip_radius=0…)` | 3D single compressor stage in cylindrical coordinates (r, theta, z). |

#### `pinneapple_physics.pde_environment.scales`

_Módulo sem docstring e sem texto._

<small>`pinneapple_physics/pde_environment/scales.py` · 18 linhas · testes: 69</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ScaleSpec` | classe | `(L, U, alpha)` | Optional nondimensionalization / scaling parameters. Keep minimal; advanced scaling can be built later. |

#### `pinneapple_physics.pde_environment.spec`

_Módulo sem docstring e sem texto._

<small>`pinneapple_physics/pde_environment/spec.py` · 51 linhas · testes: 154</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PDETermSpec` | classe | `(kind, fields, coords, params, meta)` | PDE descriptor used by the compiler. |
| `ProblemSpec` | classe | `(name, dim, coords, fields, pde, conditions, sample_defaults, scales, field_ranges, refer…)` | Complete problem specification. |

#### `pinneapple_physics.pde_environment.turbulence_presets`

RANS/LES turbulence closure presets for PINN training.

<small>`pinneapple_physics/pde_environment/turbulence_presets.py` · 992 linhas · testes: 49</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `KOmegaSSTResiduals` | classe | `(nu=1e-05, rho=1.0, consts=None) -> None` | k-omega SST two-equation turbulence model residuals for PINN. |
| `SpalartAllmarasResiduals` | classe | `(nu=1e-05) -> None` | Spalart-Allmaras one-equation turbulence model residuals. |
| `WALEResiduals` | classe | `(nu=1e-05, cw=0.325, delta=1.0, spatial_indices=(0, 1, 2), t_index=3) -> None` | WALE (Wall-Adapting Local Eddy-viscosity) subgrid closure for filtered incompressible Navier-Stokes -- Nicoud & Ducros (1999) -- as an algebraic (no transport equation) 3D LES residual. |
| `get_rans_preset` | função | `(model_name='k-omega-sst', **kwargs)` | Get a RANS/LES turbulence closure residual object by name. |

#### `pinneapple_physics.pde_environment.turbulence_selector`

Unified turbulence-model selector.

<small>`pinneapple_physics/pde_environment/turbulence_selector.py` · 171 linhas · testes: 26</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TurbulenceModel` | classe | `()` | Turbulence closure selector. |
| `get_turbulence_closure` | função | `(model, dim=3, solver_family='pinn', **kwargs) -> Optional[Any]` | Return the right turbulence closure object/parameter for a solver family. |

#### `pinneapple_physics.physical_problem`

``PhysicalProblem``: one description of a physics problem for every part of the library.

<small>`pinneapple_physics/physical_problem.py` · 534 linhas · testes: 12</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Parameter` | classe | `(name, value, unit, role, bounds, distribution, description)` | A named physical parameter. |
| `Quantity` | classe | `(name, unit, field, description)` | A quantity of interest: what the user actually wants out of the solution (a field, a coefficient, a peak value). ``field`` names the model output it derives from, when there is one. |
| `PhysicalProblem` | classe | `(name, coords, fields, pde, conditions, domain_bounds, geometry, geometry_kind, parameter…)` | A complete physics problem: geometry, physics, conditions, parameters, solver choice and intent. |

#### `pinneapple_physics.physics_audit`

Auditoria física: qual modelo usar, quando confiar nele e quando sua formulação física deve ser revista; `diagnose` junta avaliação e diagnóstico num caso. Numpy puro, sem torch.

<small>`pinneapple_physics/physics_audit/__init__.py` · 171 linhas · testes: 37</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Diagnosis` | classe | `(model, features, audit, validity, attribution, recommendations, unrepresented)` | Resultado de `diagnose`: auditoria de premissas, validade, atribuição, recomendações e mecanismos não representados, com relatório legível. |
| `diagnose` | função | `(profile, features, prediction=None, query=None, monitor=None, error=None, error_toleranc…)` | Audita as premissas do perfil para o caso, roda o monitor na previsão, atribui a causa da falha e recomenda ações. |

#### `pinneapple_physics.physics_audit.assumptions`

Premissas físicas de um modelo, as condições em que cada uma vale e um auditor que as checa caso a caso. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/assumptions.py` · 420 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Condition` | classe | `(feature, lo, hi, band, log)` | Condição `lo <= feature <= hi` com borda suave; classifica como válida, marginal ou violada e dá uma pontuação. |
| `Assumption` | classe | `(name, description, conditions, breaks_into, missing_mechanism)` | Simplificação em que um modelo se apoia, com o fenômeno que aparece quando ela falha e o mecanismo que falta. |
| `AssumptionCheck` | classe | `(name, status, score, evidence, breaks_into, missing_mechanism, features)` | Resultado da verificação de uma premissa num caso. |
| `Envelope` | classe | `(ranges, samples, sample_features, band)` | Onde um modelo foi treinado ou validado: faixas por característica e, opcionalmente, amostras para distância ao vizinho mais próximo. |
| `ModelProfile` | classe | `(name, family, equations, assumptions, mechanisms, envelope, cost, notes)` | O que a auditoria sabe de um modelo: família, equações, premissas, mecanismos, envelope e custo. |
| `AssumptionAudit` | classe | `(model, status, compatibility, checks, envelope, triggered, missing_mechanisms)` | Resultado da auditoria de um modelo num caso: status, compatibilidade, verificações e mecanismos ausentes. |
| `laminar` | função | `(re_crit=2300.0, feature='Re', geometry='pipe') -> Assumption` | Premissa de escoamento laminar abaixo do Reynolds de transição. |
| `turbulent` | função | `(re_min=4000.0, re_max=math.inf, feature='Re') -> Assumption` | Premissa de escoamento totalmente turbulento acima de um Reynolds mínimo. |
| `hydraulically_smooth` | função | `(eps_plus_max=5.0) -> Assumption` | Premissa de parede hidraulicamente lisa (Reynolds de rugosidade baixo). |
| `incompressible` | função | `(ma_max=0.3) -> Assumption` | Premissa de incompressibilidade por limite no número de Mach. |
| `creeping_flow` | função | `(re_max=1.0) -> Assumption` | Premissa de escoamento de Stokes (Reynolds muito baixo). |
| `continuum` | função | `(kn_max=0.01) -> Assumption` | Premissa de meio contínuo por limite no número de Knudsen. |
| `boussinesq` | função | `(max_beta_dt=0.1) -> Assumption` | Premissa de Boussinesq por limite em `beta dT`. |
| `constant_properties` | função | `(max_dt_over_t=0.1) -> Assumption` | Premissa de propriedades constantes por limite na variação relativa de temperatura. |
| `small_strain` | função | `(max_strain=0.01) -> Assumption` | Premissa de pequenas deformações. |
| `linear_elastic` | função | `(max_stress_ratio=1.0) -> Assumption` | Premissa de material linear elástico por limite na razão de tensões. |
| `lumped_capacitance` | função | `(bi_max=0.1) -> Assumption` | Premissa de capacitância concentrada por limite no número de Biot. |
| `forced_convection` | função | `(ri_max=0.1) -> Assumption` | Premissa de convecção forçada por limite no número de Richardson. |
| `diffusion_dominated` | função | `(pe_max=1.0) -> Assumption` | Premissa de transporte dominado por difusão (Péclet baixo). |
| `advection_dominated` | função | `(pe_min=10.0) -> Assumption` | Premissa de transporte dominado por advecção (Péclet alto). |
| `attached_flow` | função | `(alpha_stall_deg=12.0) -> Assumption` | Premissa de escoamento colado abaixo do ângulo de estol. |
| `shallow_water` | função | `(max_aspect=0.05) -> Assumption` | Premissa de águas rasas por limite na razão de aspecto. |
| `get_assumption` | função | `(a) -> Assumption` | Devolve uma `Assumption` a partir de um nome do catálogo ou do próprio objeto. |
| `audit_assumptions` | função | `(profile, features) -> AssumptionAudit` | Checa todas as premissas e o envelope de um perfil para as características de um caso. |

#### `pinneapple_physics.physics_audit.attribution`

Atribuição de falhas físicas: a causa provável de uma previsão errada ou suspeita, com evidências e confiança. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/attribution.py` · 246 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Cause` | classe | `(code, title, probability, score, evidence)` | Uma causa candidata com código, título, probabilidade, pontuação e evidências. |
| `Attribution` | classe | `(failed, causes, confidence, evidence_available, error, notes)` | Resultado da atribuição: se falhou, causas ordenadas, confiança e evidências disponíveis. |
| `attribute_failure` | função | `(*, validity=None, audit=None, error=None, error_tolerance=0.05, residual_baseline=None, …)` | Pondera as evidências de um caso (validade, auditoria, erro, resolução, medição, dispersão) e estima a causa. |

#### `pinneapple_physics.physics_audit.improvement`

Melhoria guiada pela física: quando nenhum modelo basta, qual experimento ou mudança vale mais. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/improvement.py` · 179 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Recommendation` | classe | `(action_id, action, cause, priority, gain, cost, rationale, details, cases)` | Ação recomendada com causa, prioridade, ganho, custo e justificativa. |
| `design_points` | função | `(features, spread=0.15, n=5, seed=0, keys=None) -> list[dict[str, float]]` | Pontos de projeto de alta fidelidade em torno de um caso, por hipercubo latino. |
| `recommend` | função | `(attribution, audit=None, graph=None, features=None, model=None, min_probability=0.1) -> …)` | Lista ações ordenadas para um caso, nomeando o mecanismo que falta quando há grafo de conhecimento. |
| `improvement_plan` | função | `(cases, top=None) -> list[Recommendation]` | Agrega recomendações de muitos casos, somando prioridades da mesma ação. |

#### `pinneapple_physics.physics_audit.knowledge`

Grafo de conhecimento físico: fenômenos, equações, premissas, parâmetros, mecanismos e famílias de modelos e suas relações. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/knowledge.py` · 328 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Node` | classe | `(id, kind, label, data)` | Nó do grafo com id, tipo, rótulo e dados. |
| `Edge` | classe | `(src, rel, dst, data)` | Aresta do grafo com origem, relação, destino e dados. |
| `PhysicsKnowledgeGraph` | classe | `()` | Grafo de conhecimento com consultas sobre mecanismos, premissas, fenômenos disparados e mecanismos não representados. |
| `default_graph` | função | `(models=()) -> PhysicsKnowledgeGraph` | Catálogo padrão de premissas, fenômenos e equações, mais os modelos dados. |

#### `pinneapple_physics.physics_audit.monitor`

Monitor de validade física: se uma previsão ainda é plausível sem verdade de referência (resíduo, conservação, limites, contorno, invariantes, extrapolação). [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/monitor.py` · 260 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CheckResult` | classe | `(name, kind, value, tolerance, passed, critical, detail)` | Resultado de uma verificação de validade: valor, tolerância, aprovado e criticidade. |
| `ValidityCheck` | classe | `(name, kind, fn, tolerance, critical, describe)` | Verificação `fn(prediction, query) -> valor` que passa quando o valor não excede a tolerância. |
| `ValidityReport` | classe | `(status, results, alarms, index)` | Relatório de uma previsão: status, resultados, alarmes e índice. |
| `PhysicsValidityMonitor` | classe | `(checks, persistence=3, drift_ratio=3.0, drift_alpha=0.3, baseline_cases=10, baselines=No…)` | Roda um conjunto de verificações em cada previsão e guarda o histórico, com persistência e detecção de deriva. |
| `residual_check` | função | `(residual_fn, tolerance, name='pde_residual', critical=False) -> ValidityCheck` | Cria uma verificação sobre a norma do resíduo da EDP. |
| `conservation_check` | função | `(total_fn, target_fn, rel_tol=0.01, name='conservation', critical=False, abs_tol=0.0) -> …)` | Cria uma verificação de desequilíbrio relativo de uma grandeza conservada. |
| `bounds_check` | função | `(lo=None, hi=None, name='bounds', select=None, max_fraction=0.0, critical=True) -> Validi…)` | Cria uma verificação da fração de valores fora de `[lo, hi]` (crítica por padrão). |
| `boundary_check` | função | `(predict_boundary, boundary_value, rel_tol=0.01, name='boundary_condition', critical=Fals…)` | Cria uma verificação do desvio RMS relativo entre a previsão no contorno e a condição de contorno. |
| `constitutive_check` | função | `(fn, tolerance, name='constitutive', critical=False) -> ValidityCheck` | Cria uma verificação de qualquer relação constitutiva que a previsão deve satisfazer. |
| `invariant_check` | função | `(invariant_fn, expected, rel_tol=0.01, name='invariant', critical=False, abs_tol=0.0) -> …)` | Cria uma verificação da variação relativa de um invariante. |
| `envelope_check` | função | `(envelope, features_fn, min_score=0.5, name='training_envelope') -> ValidityCheck` | Cria uma verificação de extrapolação a partir da pontuação do envelope de treino. |

#### `pinneapple_physics.physics_audit.regime`

Números adimensionais que dizem em que física um caso está (Re, Ma, Pe, Kn, Gr, Pr, Ra, Ri, Bi, Fr, St, Da, rugosidade).

<small>`pinneapple_physics/physics_audit/regime.py` · 178 linhas · testes: 31</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `reynolds` | função | `(velocity, length, nu=None, *, rho=None, mu=None) -> float` | Número de Reynolds. |
| `mach` | função | `(velocity, sound_speed=None, *, gamma=1.4, R=287.05, T=None) -> float` | Número de Mach, com velocidade do som dada ou de gás ideal. |
| `peclet` | função | `(velocity, length, diffusivity) -> float` | Número de Péclet. |
| `knudsen` | função | `(mean_free_path, length) -> float` | Número de Knudsen. |
| `grashof` | função | `(g, beta, dT, length, nu) -> float` | Número de Grashof. |
| `prandtl` | função | `(nu, alpha) -> float` | Número de Prandtl. |
| `rayleigh` | função | `(g, beta, dT, length, nu, alpha) -> float` | Número de Rayleigh. |
| `richardson` | função | `(g, beta, dT, length, velocity) -> float` | Número de Richardson. |
| `biot` | função | `(h, length, k) -> float` | Número de Biot. |
| `froude` | função | `(velocity, length, g=9.81) -> float` | Número de Froude. |
| `strouhal` | função | `(frequency, length, velocity) -> float` | Número de Strouhal. |
| `dominant_frequency` | função | `(signal, dt=1.0, discard=0.5, window=True, refine=True, pad=1) -> float` | Frequência do pico espectral mais forte de um sinal amostrado (por exemplo desprendimento de vórtices). |
| `damkohler` | função | `(reaction_rate, length, velocity) -> float` | Número de Damköhler. |
| `roughness_reynolds` | função | `(re, eps_d, f=None) -> float` | Reynolds de rugosidade `eps+`, com atrito estimado por Haaland se não for dado. |
| `regime_features` | função | `(*, velocity=None, length=None, nu=None, sound_speed=None, diffusivity=None, alpha=None, …)` | Calcula todos os números adimensionais cujas entradas foram dadas, num dicionário de características. |

#### `pinneapple_physics.physics_audit.routing`

Roteamento de modelos ciente da física: pondera cada modelo pelo quanto o caso atual cabe nas suas premissas, não só pelos erros passados. [sem teste direto; usado via `physics_audit`]

<small>`pinneapple_physics/physics_audit/routing.py` · 142 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `RouteDecision` | classe | `(chosen, weights, compatibility, awake, escalate, audits, reasons)` | Decisão de roteamento: modelo escolhido, pesos, compatibilidades, modelos ativos, escalonamento e motivos. |
| `PhysicsAwareRouter` | classe | `(profiles, features_fn=None, strength=1.0, floor=0.0001, abstain_below=0.05, escalate_bel…)` | Roteador que usa a compatibilidade física como prior dos pesos do ensemble, com especialistas adormecidos e escalonamento ao solver. |

#### `pinneapple_physics.pinn_solver`

Subpacote do compilador de PINN; reexporta `compile_problem`, pesos de perda, DoMINO e utilidades de PINN estocástico.

<small>`pinneapple_physics/pinn_solver/__init__.py` · 28 linhas · testes: 1</small>

#### `pinneapple_physics.pinn_solver.compiler`

Reexporta `compile_problem`, `LossWeights`, `AdaptiveWeights`, `SingleBatchDataset` e `dict_collate`.

<small>`pinneapple_physics/pinn_solver/compiler/__init__.py` · 11 linhas · testes: 5</small>

#### `pinneapple_physics.pinn_solver.compiler.autograd_ops`

Operadores diferenciais por autograd (gradiente, jacobiano, divergente, laplaciano, derivada temporal, derivada normal) com cache opcional de derivadas.

<small>`pinneapple_physics/pinn_solver/compiler/autograd_ops.py` · 153 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `derivative_cache` | função | `() -> Iterator[Dict[str, int]]` | Gerenciador de contexto que memoiza primeiras derivadas dentro de uma avaliação de perda e conta acertos e falhas. |
| `ensure_tensor` | função | `(y)` | Extrai um tensor da saída do modelo (tensor puro ou objeto com atributo de saída). |
| `grad` | função | `(y, x) -> torch.Tensor` | Gradiente de um campo escalar em relação às entradas, com grafo para derivadas de ordem maior. |
| `jacobian` | função | `(Y, x) -> torch.Tensor` | Jacobiano de um campo vetorial em relação às entradas. |
| `divergence` | função | `(V, x, coord_indices=None) -> torch.Tensor` | Divergente de um campo vetorial. |
| `laplacian` | função | `(y, x, coord_indices=None) -> torch.Tensor` | Laplaciano (soma das segundas derivadas não mistas). |
| `time_derivative` | função | `(y, x, t_index) -> torch.Tensor` | Derivada em relação à coordenada de tempo. |
| `norm_dot_grad` | função | `(y, x, normals) -> torch.Tensor` | Derivada normal `n . grad y`. |
| `mse` | função | `(a, b) -> torch.Tensor` | Erro quadrático médio entre dois tensores. |

#### `pinneapple_physics.pinn_solver.compiler.compile`

O compilador central: transforma um `ProblemSpec` numa função de perda por termo, com resíduos para dezenas de tipos de EDP (Laplace, Poisson, Navier-Stokes, elasticidade, Maxwell, Black-Scholes, órbitas, etc.) e perdas de condições Dirichlet, Neumann, Robin, interface, inicial e dados; tipo desconhecido levanta `ValueError`.

<small>`pinneapple_physics/pinn_solver/compiler/compile.py` · 2.604 linhas · testes: 121</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `compile_problem` | função | `(spec, *, weights=None, cache_derivatives=False)` | Compila o `spec` em `loss_fn(model, y_hat, batch) -> {nome: perda}`; `cache_derivatives=True` compartilha derivadas entre termos. |

#### `pinneapple_physics.pinn_solver.compiler.dataset`

Dataset de lote único e função de collate para lotes PINN em dicionário. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/compiler/dataset.py` · 89 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SingleBatchDataset` | classe | `(batch)` | Embrulha um lote pronto (por exemplo de `STLDomainBatchBuilder`) num `Dataset`, ciclando pontos de contorno, inicial e dados. |
| `dict_collate` | função | `(batch) -> Dict[str, Any]` | Junta uma lista de lotes PINN em dicionário (usa `pinneapple_data.collate` se disponível). |

#### `pinneapple_physics.pinn_solver.compiler.em_kinds`

Tipos eletromagnéticos do compilador: equações de Maxwell em forma diferencial (eletrostática, magnetostática, correntes parasitas, TM, 3-D, frequência) e integral (volumes de controle). [sem teste direto]

<small>`pinneapple_physics/pinn_solver/compiler/em_kinds.py` · 261 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `em_residuals` | função | `(kind, *, fields, xcol, coords, params, ctx, model, field_names, split_fields) -> list[to…)` | Devolve a lista de resíduos nos pontos de colocação para um tipo eletromagnético. |

#### `pinneapple_physics.pinn_solver.compiler.loss`

Pesos de perda: `LossWeights` fixos lidos por `compile_problem` e `AdaptiveWeights` autonormalizados para laços de treino próprios.

<small>`pinneapple_physics/pinn_solver/compiler/loss.py` · 107 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LossWeights` | classe | `(w_pde, w_bc, w_ic, w_data)` | Pesos fixos dos termos de EDP, contorno, inicial e dados. |
| `AdaptiveWeights` | classe | `(base_weights, momentum, eps, max_ratio, enabled, _ema)` | Pesos adaptativos por nome de termo, a partir de média móvel exponencial defasada da perda bruta, limitados por `max_ratio`. |

#### `pinneapple_physics.pinn_solver.domino`

DoMINO: decomposição de domínio para PINN, com um PINN por subdomínio retangular sobreposto e perda de continuidade na interface. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/domino.py` · 385 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Subdomain` | classe | `(bounds, overlap=0.1)` | Subdomínio retangular com sobreposição e testes de pertinência e interface. |
| `SubdomainPINN` | classe | `(subdomain, model)` | PINN de um único subdomínio. |
| `DoMINO` | classe | `(subdomains, model_factory, interface_weight=10.0)` | Treinador de PINN por decomposição de domínio: particiona, combina as saídas e penaliza descontinuidade na interface. |

#### `pinneapple_physics.pinn_solver.factory`

Fábrica PINN legada baseada em SymPy; reexporta `NeuralNetwork`, `PINN`, `PINNFactory`, `PINNProblemSpec` e indica `VanillaPINN` de `pinneapple_neural` como modelo preferido.

<small>`pinneapple_physics/pinn_solver/factory/__init__.py` · 23 linhas · testes: 7</small>

#### `pinneapple_physics.pinn_solver.factory.autodiff`

Cálculo e cache de derivadas por autograd exigidas pelos átomos `Derivative` do SymPy. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/factory/autodiff.py` · 166 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DerivativeKey` | classe | `(base, wrt)` | Chave estável para guardar derivadas em cache. |
| `DerivativeComputer` | classe | `(independent_vars, dependent_vars, device, coord_scales=None) -> None` | Calcula e guarda as derivadas por autograd pedidas pelas expressões SymPy. |
| `ensure_requires_grad` | função | `(inputs) -> None` | Liga `requires_grad=True` em todos os tensores de entrada. |

#### `pinneapple_physics.pinn_solver.factory.pinn_factory`

Fábrica de PINN: compila equações simbólicas e monta uma perda unificada; `NeuralNetwork` e `PINN` são shims de compatibilidade sobre `VanillaPINN`. [shim parcial]

<small>`pinneapple_physics/pinn_solver/factory/pinn_factory.py` · 412 linhas · testes: 1</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `NeuralNetwork` | classe | `(num_inputs, num_outputs, num_layers, num_neurons, activation=nn.Tanh())` | [shim] MLP compatível com a API antiga `(num_inputs, num_outputs, num_layers, num_neurons)`, sobre `VanillaPINN`. |
| `PINN` | classe | `(neural_network, inverse_params_names=None, initial_guesses=None, dtype=torch.float32)` | [shim] Rede mais parâmetros inversos treináveis, mantida só por compatibilidade de API. |
| `PINNProblemSpec` | classe | `(pde_residuals, independent_vars, dependent_vars, conditions, inverse_params, loss_weight…)` | Especificação PINN independente de modelo: resíduos em string, variáveis, condições, parâmetros inversos, pesos e sinais exógenos. |
| `PINNFactory` | classe | `(spec)` | Compila as equações SymPy em resíduos torch e gera uma função de perda unificada. |

#### `pinneapple_physics.pinn_solver.factory.signals`

Provedores de sinais exógenos (dados de sensor, forçamento) para `PINNFactory`. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/factory/signals.py` · 62 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TabulatedSignal` | classe | `(x, values, *, input_index=0, extrapolate='clamp')` | Interpolação linear por partes de um sinal amostrado, avaliada em torch, com extrapolação constante ou linear. |

#### `pinneapple_physics.pinn_solver.factory.sympy_backend`

Compilador de equações SymPy para lambdas torch via `lambdify`; o gradiente não atravessa a avaliação simbólica (para isso o código indica `SymbolicPDE`). [sem teste direto]

<small>`pinneapple_physics/pinn_solver/factory/sympy_backend.py` · 243 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CompiledEquation` | classe | `(expr, derivatives, call, args_order, exogenous, runtime_constants)` | Equação compilada: expressão, derivadas necessárias, função chamável e ordem dos argumentos. |
| `SympyTorchCompiler` | classe | `(independent_vars, dependent_vars, inverse_params=None, extra_namespace=None)` | Compila strings de equações em lambdas torch usando SymPy. |
| `SympyBackend` | classe | `(*, extra_namespace=None) -> None` | [shim] Embrulho de compatibilidade que só devolve um `SympyTorchCompiler`. |

#### `pinneapple_physics.pinn_solver.io`

Reexporta mapeamentos e o leitor de dados UPD; `UPDItem`, `ConditionSpec`, `SamplingSpec` e `Batch` vêm de `pinneapple_data.upd_types`. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/io/__init__.py` · 28 linhas · **sem teste direto**</small>

#### `pinneapple_physics.pinn_solver.io.mappings`

Mapeamentos entre coordenadas e variáveis UPD e as variáveis independentes e dependentes do PINN, com transformações (segundos, graus para radianos, normalização). [sem teste direto]

<small>`pinneapple_physics/pinn_solver/io/mappings.py` · 197 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoordMapping` | classe | `(ind_vars, coord_sources, coord_transform, normalize_to_unit, _deriv_scale)` | Mapeia coordenadas UPD para variáveis independentes do PINN, com transformação e escala de derivadas. |
| `VarMapping` | classe | `(dep_vars, upd_to_dep, enforce_present)` | Mapeia nomes de variáveis UPD para variáveis dependentes do PINN. |
| `PINNMapping` | classe | `(coord, var)` | Mapeamento completo de coordenadas e variáveis. |
| `build_default_mapping_atmosphere` | função | `(dep_vars, ind_vars=None, use_lev=False, time_transform='seconds_since_start', angle_tran…)` | Mapeamento padrão para fragmentos de reanálise atmosférica UPD (tempo, lat, lon e, opcionalmente, nível). |

#### `pinneapple_physics.pinn_solver.io.upd_dataset`

Leitor de dataset UPD (Zarr + JSON, via xarray) que produz amostras prontas para PINN. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/io/upd_dataset.py` · 253 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UPDDataset` | classe | `(item, mapping, device='cpu', dtype=torch.float32)` | Lê UPD e gera pontos de colocação, pontos de condição e pares supervisionados opcionais. |

#### `pinneapple_physics.pinn_solver.schemas`

Pacote vazio que contém o adaptador de schema. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/schemas/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_physics.pinn_solver.schemas.schema_adapter`

Adaptador de Physical Schema (UPD) para `PINNProblemSpec` e especificações de amostragem; o próprio docstring descreve premissas de MVP. [sem teste direto]

<small>`pinneapple_physics/pinn_solver/schemas/schema_adapter.py` · 325 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SchemaAdapterOptions` | classe | `(default_loss_weights, default_condition_weight, build_sampling_from_conditions)` | Opções da conversão de schema: pesos padrão de perda, peso de condição e se cria amostragem a partir das condições. |
| `SchemaAdapter` | classe | `(mapping, options=None) -> None` | Converte um schema físico em `PINNProblemSpec` (para `PINNFactory`) e, opcionalmente, em especificações de amostragem para `UPDDataset`. |

#### `pinneapple_physics.pinn_solver.stochastic`

Treino de PINN estocástico condicionado por código latente: a rede aprende uma distribuição de saídas físicas e casa média e covariância do ensemble com estatísticas reais.

<small>`pinneapple_physics/pinn_solver/stochastic.py` · 378 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LatentConditionedModel` | classe | `(base, latent_dim) -> None` | Embrulha um modelo para aceitar também um código latente `xi`, concatenado à entrada. |
| `CorrelatedLatentField` | classe | `(bounds, coord_order, spatial_axes, periodic_axes, latent_dim, grid_res=(64, 24, 32), len…)` | Campo latente espacialmente correlacionado, alternativa ao sorteio independente por ponto. |
| `sample_latent` | função | `(n, latent_dim, device=None, dtype=torch.float32) -> torch.Tensor` | Sorteia códigos latentes `N(0, I)`. |
| `ensemble_forward` | função | `(model, x, latent_dim, n_samples, latent_sampler=None) -> torch.Tensor` | Avalia várias amostras latentes em cada ponto numa única passagem em lote. |
| `mean_covariance_loss` | função | `(model, x, latent_dim, n_samples, mean_target, cov_target=None, cov_index_pairs=None, fie…)` | Perda que casa a média e, opcionalmente, a covariância Monte Carlo do ensemble latente com estatísticas fornecidas. |

#### `pinneapple_physics.solid`

Solid mechanics post-processing (strains, stresses, equilibrium residuals, von Mises) by autograd.

<small>`pinneapple_physics/solid/__init__.py` · 13 linhas · **sem teste direto**</small>

#### `pinneapple_physics.solid.axisymmetric`

Axisymmetric linear elasticity in cylindrical coordinates ``(r, z)``, by autograd.

<small>`pinneapple_physics/solid/axisymmetric.py` · 122 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `lame_parameters` | função | `(E, nu) -> tuple[float, float]` | ``(lambda, mu)`` from Young's modulus ``E`` and Poisson's ratio ``nu`` (isotropic). |
| `strains_rz` | função | `(u, x, *, r_min=None, create_graph=True) -> dict[str, torch.Tensor]` | Small strains ``e_rr, e_tt, e_zz, e_rz`` (and ``e_rt, e_tz`` when ``u`` has ``u_theta``), each ``(N, 1)``. |
| `stresses_rz` | função | `(u, x, E, nu, *, r_min=None, create_graph=True) -> dict[str, torch.Tensor]` | Stresses ``s_rr, s_tt, s_zz, s_rz`` (and ``s_rt, s_tz`` with torsion), each ``(N, 1)``, by Hooke's law. |
| `equilibrium_residual_rz` | função | `(u, x, E, nu, *, r_min=None) -> torch.Tensor` | Pointwise equilibrium residual ``[R_r, R_z]`` (``(N, 2)``), plus ``R_theta`` (``(N, 3)``) with torsion. |
| `von_mises_from_components` | função | `(s_rr, s_zz, s_tt, s_rz=0.0, s_rt=0.0, s_tz=0.0) -> Any` | Von Mises stress ``sqrt(((s_rr-s_zz)^2 + (s_zz-s_tt)^2 + (s_tt-s_rr)^2 + 6 (s_rz^2 + s_rt^2 + s_tz^2)) / 2)``. |
| `von_mises_rz` | função | `(u, x, E, nu, *, r_min=None, create_graph=False) -> torch.Tensor` | Von Mises stress ``(N, 1)`` of ``u`` (all six stress components when ``u`` has ``u_theta``). |

#### `pinneapple_physics.solving`

``pp.solve`` and ``pp.compare``: one entry point to solve a problem by any method and compare methods.

<small>`pinneapple_physics/solving.py` · 594 linhas · testes: 71</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MethodNotAvailable` | classe | `()` | The requested method cannot solve this problem (missing solver output, missing option, ...). |
| `Solution` | classe | `(problem, method, predict_fn, wall_time_s, history, info, model, grid, kind)` |  |
| `Comparison` | classe | `(problem, reference, n_points, rows)` |  |
| `as_problem` | função | `(problem, **preset_kwargs) -> PhysicalProblem` | ``PhysicalProblem`` from a ``PhysicalProblem``, a PDE ``ProblemSpec`` or a preset name. |
| `register_method` | função | `(name, kind='custom')` | Decorator registering ``fn(problem, **options) -> Solution`` under ``name`` for ``solve``/``compare``. ``kind`` labels the family of the method (``classical``, ``neural``, ...) in every Solution it returns. |
| `list_methods` | função | `(kind=None) -> List[str]` | Registered method names, optionally only those of one ``kind``. |
| `list_kinds` | função | `() -> Dict[str, List[str]]` | Methods grouped by kind. |
| `register_analytic` | função | `(name, matcher)` | Decorator: ``factory(problem) -> fn(X)`` is the exact solution for every problem where ``matcher`` is True. |
| `list_analytic` | função | `() -> List[str]` |  |
| `analytic_for` | função | `(problem) -> Optional[str]` | Name of the closed-form solution this library knows for ``problem``, or ``None``. |
| `solve` | função | `(problem, method='pinn', *, preset_kwargs=None, **options) -> Solution` | Solve ``problem`` with ``method`` (a registered name or a callable ``(problem, **options) -> Solution``). |
| `compare` | função | `(problem, methods, *, reference='reference', options=None, points=None, n_points=4096, se…)` | Solve or reuse each method and score it against ``reference`` at the same points. |

#### `pinneapple_physics.symbolic_pde`

pinneapple_symbolic — Symbolic PDE compiler for PINNeAPPle.

<small>`pinneapple_physics/symbolic_pde/__init__.py` · 52 linhas · **sem teste direto**</small>

#### `pinneapple_physics.symbolic_pde.bc`

Boundary condition enforcement for PINNeAPPle symbolic PDE module.

<small>`pinneapple_physics/symbolic_pde/bc.py` · 447 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `HardBC` | classe | `(distance_fn, bc_value_fn) -> None` | Hard boundary condition enforcement via a distance-function ansatz. |
| `PeriodicBC` | classe | `(period, dim=0) -> None` | Periodic boundary condition enforcement via coordinate transformation. |
| `MultiPeriodicBC` | classe | `(periods) -> None` | Exact periodic-BC embedding for two or more coordinate axes at once. |
| `DirichletBC` | classe | `(value_fn, boundary_fn=None, weight=1.0) -> None` | Soft Dirichlet boundary condition via an MSE penalty term. |
| `NeumannBC` | classe | `(flux_fn, normal_fn, boundary_fn=None, weight=1.0) -> None` | Soft Neumann boundary condition via autograd normal derivative. |

#### `pinneapple_physics.symbolic_pde.compiler`

Symbolic PDE compiler for PINNeAPPle.

<small>`pinneapple_physics/symbolic_pde/compiler.py` · 771 linhas · testes: 5 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SymbolicPDE` | classe | `(expr, coord_syms, field_syms, param_syms=None, grad_method='autograd', grid=None) -> None` | Symbolic PDE residual — compile to a PyTorch autograd callable. |
| `pde_from_sympy` | função | `(expr, coords, fields, params=None, grad_method='autograd', grid=None) -> SymbolicPDE` | Create a SymbolicPDE from a SymPy residual expression. |
| `auto_residual` | função | `(model, coords_tensor, derivative_fns) -> Dict[str, torch.Tensor]` | Compute field values and named derivative quantities via autograd. |

#### `pinneapple_physics.symbolic_pde.gradient_backends`

Pluggable derivative backends for :class:`SymbolicPDE`.

<small>`pinneapple_physics/symbolic_pde/gradient_backends.py` · 130 linhas · testes: 5</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `GridAxisSpec` | classe | `(coord_names, dims, L)` | Maps SymPy coordinate names to axes of a structured-grid field tensor. |
| `grid_derivative` | função | `(field, wrt, *, grid, method) -> torch.Tensor` | Compute an arbitrary-order partial derivative of a grid-shaped field. |

#### `pinneapple_physics.symbolic_pde.pinn`

A parsed equation system (:mod:`.text`) as a :class:`~pinneapple_analysis.validation.pinn.PINNProblem`.

<small>`pinneapple_physics/symbolic_pde/pinn.py` · 194 linhas · testes: 10</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PINNWithParameters` | classe | `()` |  |
| `pinn_problem` | função | `(system, bounds, *, coefficients=None, unknowns=None, width=32, depth=4, exact=None, name…)` | ``(PINNProblem, model)`` for ``system`` on the box ``bounds``. ``unknowns``: coefficient -> initial guess, trained with the network (read back with ``model.estimates()``). ``exact``: optional ``X -> (N, n_fields)``. |
| `has_field` | função | `(expr) -> bool` |  |

#### `pinneapple_physics.symbolic_pde.text`

Equations as text, parsed safely into SymPy: ``"u_t = alpha*lap(u) + sin(pi*x)"``.

<small>`pinneapple_physics/symbolic_pde/text.py` · 486 linhas · testes: 17</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EquationSyntaxError` | classe | `()` | The text is not an equation this parser accepts (the message says what and where). |
| `Context` | classe | `(coords, fields, time)` |  |
| `Equation` | classe | `(text, lhs, rhs, coefficients, kind, on)` |  |
| `EquationSystem` | classe | `(ctx, pdes, conditions, coefficients, unknowns)` |  |
| `parse_expression` | função | `(text, ctx) -> tuple[sp.Expr, set[str]]` | ``text`` (no ``=``) as a SymPy expression, and the coefficient names it uses. |
| `parse_equation` | função | `(text, ctx, *, kind='pde', on='') -> Equation` |  |
| `parse_system` | função | `(pdes, conditions=None, *, coords, fields, coefficients=None, unknowns=None, time='t') ->…)` | Parse PDE texts and conditions (``{"kind", "on", "equation"}``) into an :class:`EquationSystem`. EquationSyntaxError names the first text that fails. |
| `classify` | função | `(system) -> dict[str, Any]` | The PDE family of a single-field system (heat, wave, poisson, laplace, helmholtz, burgers, advection_diffusion, advection, reaction_diffusion, or ``custom``), its orders and linearity. |
| `derivative_key` | função | `(dv) -> str` | Environment key of a field derivative: ``u__xx`` for ``u_xx`` (letters sorted). |
| `derivative_keys` | função | `(expr) -> dict[str, tuple[str, str]]` | ``{key: (field, wrt letters)}`` for every field derivative in ``expr``. |
| `compile_expr` | função | `(expr, backend='numpy')` | A function ``env -> value`` for ``expr`` built as a closure tree (no ``eval``, no ``lambdify``). |

#### `pinneapple_physics.tracking`

Experiment tracking adapter: send an ``ExperimentResult`` to MLflow or Weights & Biases.

<small>`pinneapple_physics/tracking.py` · 82 linhas · testes: 3</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `flat_metrics` | função | `(result) -> Dict[str, float]` |  |
| `flat_params` | função | `(result) -> Dict[str, str]` |  |
| `log_result` | função | `(result, tracker, **kw) -> Any` | Send ``result`` to ``tracker``; returns the tracker's run id (or what the callable returns). |

#### `pinneapple_physics.tribology`

Tribology: sliding wear of a bar against a counterface (Archard's law with contact-pressure redistribution).

<small>`pinneapple_physics/tribology/__init__.py` · 5 linhas · testes: 20</small>

#### `pinneapple_physics.tribology.archard`

Sliding wear of a bar end pressed against a rigid, flat counterface.

<small>`pinneapple_physics/tribology/archard.py` · 136 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WearMaterial` | classe | `(name, k, hardness_MPa, E_GPa)` |  |
| `BarWear` | classe | `(material, load_N, length_mm, width_mm, crown_radius_mm, layer_mm, nx)` | A bar of section length (along sliding, crowned) x width, end radius R, pressed with load F. |
| `winkler_parabolic_contact` | função | `(F_N, width_mm, R_mm, k_w) -> dict[str, float]` | Closed form for a parabolic gap g = x² / (2R) on a Winkler layer of modulus k_w (MPa/mm), load F over a width b: p = k_w (δ - x²/2R) on \|x\| < a, a = sqrt(2 R δ), F = (4/3) b k_w a δ. |

#### `pinneapple_physics.weather`

Subpacote de previsão global do tempo; o `__init__` está vazio (os módulos são importados diretamente). [sem teste direto]

<small>`pinneapple_physics/weather/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_physics.weather.data`

Baixa ERA5 do WeatherBench2 (bucket público, grade 64 x 32) para um array local normalizado e calcula a climatologia para escores de anomalia. [sem teste direto]

<small>`pinneapple_physics/weather/data.py` · 176 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Era5Store` | classe | `(path)` | Leitura preguiçosa (memory map) do ERA5 normalizado gravado por `download`. |
| `channel_names` | função | `(levels=LEVELS) -> list[str]` | Nomes dos canais: variáveis de altitude em cada nível mais as de superfície. |
| `download` | função | `(dest, years=(1990, 2022), levels=LEVELS, url=ERA5_64x32, log=print, normalization=None) …)` | Baixa e normaliza o ERA5 para `state.npy`, `constants.npy`, `times.npy` e `meta.json`. |
| `climatology` | função | `(store, url=CLIMATOLOGY_64x32) -> np.ndarray` | Climatologia normalizada do WeatherBench2, guardada em cache ao lado do store. |
| `climatology_at` | função | `(clim, time) -> np.ndarray` | Seleciona a climatologia do dia do ano e da hora de um instante. |

#### `pinneapple_physics.weather.evaluate`

Habilidade de previsões globais por antecedência (RMSE e ACC ponderados por área) e horizonte útil de cada previsão.

<small>`pinneapple_physics/weather/evaluate.py` · 154 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PublishedForecasts` | classe | `(store, names=None)` | Previsões publicadas no WeatherBench2 na grade 64 x 32, no layout de canais do store (só inicializações 00 e 12 UTC). |
| `rmse` | função | `(f, o, w) -> np.ndarray` | RMSE ponderado por área em latitude e longitude. |
| `acc` | função | `(f, o, clim, w) -> np.ndarray` | Coeficiente de correlação de anomalias ponderado por área. |
| `first_crossing` | função | `(leads_h, values, threshold, below=True) -> float` | Antecedência interpolada em que uma curva cruza um limiar pela primeira vez. |
| `horizons` | função | `(leads_h, acc_curve, rmse_curve, clim_rmse) -> dict` | Horizonte útil (ACC acima de 0,6) e horizonte sem habilidade (RMSE acima do da climatologia). |
| `score_forecasts` | função | `(store, clim, forecasts, init_idx, steps, variables=SCORED) -> dict` | Pontua várias previsões (RMSE e ACC por variável e antecedência) a partir de inicializações dadas. |
| `persistence` | função | `(store)` | Previsão de referência por persistência do estado inicial. |
| `climatology_forecast` | função | `(store, clim)` | Previsão de referência pela climatologia. |

#### `pinneapple_physics.weather.events`

Estudos de caso de tempo extremo e os estados ERA5 em torno deles, do store de treino ou do ERA5 ARCO em resolução completa médio na grade do modelo.

<small>`pinneapple_physics/weather/events.py` · 156 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Event` | classe | `(key, title, init, peak, var, region, center, what, score, vmin, vmax, extra_inits)` | Descrição de um evento extremo: inicialização, pico, variável, região, centro do globo e forma de resumir. |
| `by_key` | função | `(key) -> Event` | Busca um evento do catálogo pela chave. |
| `coarsen` | função | `(field, src_lat, src_lon, lat, lon) -> np.ndarray` | Média ponderada por área de um campo fino em cada célula do modelo. |
| `arco_states` | função | `(store, times, log=print) -> np.ndarray` | Estados normalizados na grade do store lidos do ERA5 ARCO, com precipitação de 6 h somada. |
| `event_states` | função | `(store, event, leads, cache_dir=None, log=print) -> tuple[np.ndarray, np.ndarray]` | Instantes e estados normalizados em torno de um evento, do store ou do ARCO. |
| `region_mask` | função | `(lat, lon, box) -> np.ndarray` | Máscara booleana de uma caixa lat/lon, tratando a passagem por 360 graus. |

#### `pinneapple_physics.weather.model`

Rede de previsão global numa grade lat-lon: U-Net com convoluções esféricas que prevê a variação de 6 horas do estado.

<small>`pinneapple_physics/weather/model.py` · 122 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SConv` | classe | `(cin, cout, k=3)` | Convolução esférica: preenchimento periódico em longitude e através dos polos antes da convolução. |
| `Block` | classe | `(cin, cout)` | Bloco residual com duas convoluções esféricas 3x3, group norm e GELU. |
| `SphereUNet` | classe | `(n_state, n_extra, widths=(128, 192, 256, 384), blocks=2, patch=1, refine=64)` | U-Net esférica, com opção de dobrar células em canais (`patch`) e camada de suavização em resolução total. |
| `sphere_pad` | função | `(x, p) -> torch.Tensor` | Preenche um tensor (B, C, lat, lon): circular em longitude e através dos polos em latitude. |
| `toa_insolation` | função | `(times, lat, lon) -> np.ndarray` | Cosseno do ângulo zenital solar cortado em 0 (insolação normalizada no topo da atmosfera). |
| `static_fields` | função | `(lat, lon, constants) -> np.ndarray` | Campos fixos de entrada: constantes (máscara terra-mar, orografia) mais seno e cosseno de latitude e longitude. |

#### `pinneapple_physics.weather.train`

Treino da rede de previsão: um passo de 6 horas primeiro, depois rollouts mais longos, com MSE ponderado por área e por canal. [sem teste direto]

<small>`pinneapple_physics/weather/train.py` · 198 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Forecaster` | classe | `(store, model)` | Rede treinada com o necessário para fazer rollout a partir de um instante ERA5, salvar e carregar. |
| `train` | função | `(store_path, out, *, train_years=(1990, 2017), val_years=(2018, 2019), widths=(192, 256, …)` | Treina por um tempo de parede dividido por uma agenda de comprimentos de rollout e guarda o melhor checkpoint na validação. |

#### `pinneapple_physics.weather.viz`

Visualizações de previsão contra verdade: globos lado a lado, antecedência correndo e faixa de habilidade (ACC), em GIF e figuras.

<small>`pinneapple_physics/weather/viz.py` · 399 linhas · testes: 8</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `display_units` | função | `(var, x) -> np.ndarray` | Converte variáveis para unidades de exibição (°C, decâmetros, hPa, mm). |
| `coastlines` | função | `() -> list[np.ndarray]` | Carrega as linhas de costa do arquivo `coastline_110m.npz` do pacote. |
| `orthographic` | função | `(field, lat, lon, center=(-15.0, -50.0), size=420)` | Imagem do campo no hemisfério visível centrado num ponto (projeção ortográfica). |
| `project_lines` | função | `(lines, center, size=420)` | Projeta linhas (costas) na mesma projeção ortográfica. |
| `regional` | função | `(field, lat, lon, box, size=(360, 260))` | Imagem equirretangular interpolada de uma região. |
| `regional_lines` | função | `(lines, box, size=(360, 260))` | Projeta linhas na imagem regional. |
| `forecast_gif` | função | `(path, var, truth, forecast, lat, lon, times, *, skill_leads_h, skill_acc, horizon, cente…)` | GIF animado com ERA5 e previsão lado a lado por antecedência e a faixa de habilidade. |
| `fill_gaps` | função | `(x, y) -> np.ndarray` | Interpolação linear sobre NaNs (modelos publicados a cada 12 h num eixo de 6 h). |
| `skill_figure` | função | `(path, leads_h, curves, horizons_by_model, var, metric='acc', dpi=110)` | Figura de habilidade por antecedência de cada modelo com o horizonte útil marcado. |
| `upsample_sphere` | função | `(field, lat, lon, factor=4)` | Reamostragem cúbica de um campo lat/lon, periódica em longitude, só para exibição. |
| `earth2_gif` | função | `(path, var, truth, forecast, lat, lon, times, *, lead_hours, horizon, title, center=(20.0…)` | Animação em estilo de globos iluminados com rotação, quadros interpolados e barra de antecedência colorida pelo horizonte. |

### Exemplos de uso reais do repositório

**Treinar um preset com `solve_pde` (pontos de tag vindos de geometria)** — `examples/pde_environment/07_tagged_presets_real_geometry.py`

```python
    batch = build_tag_batch(name, spec, n_col=4000, n_bc_per_face=500, seed=1)
    print("tag_masks sizes:", {k: int(v.sum()) for k, v in batch["ctx"]["tag_masks"].items()})

    before = per_tag_losses(spec, model, batch)
    print("per-tag losses BEFORE training:", before)

    kwargs = solve_pde_kwargs_from_batch(batch)
    result = pp.solve_pde(spec, model, epochs=epochs, n_collocation=4000, seed=1, lr=1e-3, **kwargs)
    losses = result["history"]["loss"]
```

**Sintonia de ponto excepcional e auditoria das premissas do teorema** — `examples/non_hermitian/ep_tuning.py`

```python
for _key, title, b, b_ba in coupling_classes():
    print(f"\n== {title}")
    print(analyze_ep_tuning(b, b_ba))

print("\n== numerical check (J = 1, G = 0.5)")
d_a = PTDimer(route_a_sqrt_coupling(1.0, 0.5))
ab, ba = route_c_asymmetric(1.0, 0.5)
d_c = PTDimer(ab, ba)
d_l = PTDimer(lambda x: 1.0 + 0.25 * x)
```

**Verificações de Darcy sem referência sobre previsões de outro framework** — `examples/vs_physicsnemo/07_physicsnemo_trains_pinneapple_checks/run.py`

```python
            ref = dc.darcy_checks(data["a"], u, DARCY["f"], reference=data["u"], envelope=envelope)
            drift = dc.darcy_checks(data["a"], u, DARCY["f"], tolerances=calibrated[m], physics_kind="drift")
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 114 de 166.
- Funções de teste em arquivos que importam algum módulo do pacote: 1.075.

Itens do catálogo de métodos (`pinneapple_catalog/method_status.json`) cujo código está neste pacote:

| id | método | status | testes contra referência | outros testes |
|---|---|---|---:|---:|
| E55 | S-N curve + Goodman + Miner | validated | 1 | 14 |
| E58 | Fin array conduction | tested | 0 | 4 |
| E63 | Helmholtz resonator | tested | 0 | 4 |
| E64 | Membrane diffusion | tested | 0 | 5 |
| E65 | Cantilever fatigue | validated | 1 | 4 |
| S38 | Maxwell physics checks of predicted fields (Gauss, Faraday, Ampere-Maxwell, continuity, Poynting, interfaces, reciprocity, passivity) | validated | 4 | 2 |
| T10 | Domain decomposition: XPINN, DoMINO/D3M | tested | 0 | 2 |

### Limitações conhecidas

- `solve_pde` não deriva regiões de condições `selector_type="tag"` a partir de `domain_bounds`: sem pontos e máscaras vindos de geometria real, levanta `TagConditionsUnresolved`.
- `compile_problem` só aceita os tipos de EDP implementados em `compile.py`/`em_kinds.py`; um tipo desconhecido levanta `ValueError("Unsupported PDE kind")`. Nos tipos eletromagnéticos diferenciais, saltos dos mapas de material não são derivados (interfaces só são impostas pelos tipos integrais).
- A fábrica SymPy (`pinn_solver.factory`) é caminho legado: `NeuralNetwork`, `PINN` e `SympyBackend` são shims de compatibilidade, e `SympyTorchCompiler` usa `lambdify`, opaco ao autograd.
- Vários módulos não têm teste direto: `pinn_solver.domino`, `pinn_solver.io`, `pinn_solver.schemas`, `factory.autodiff/signals/sympy_backend`, `compiler.dataset/em_kinds`, `multiscale.consistency/residuals/scales/viz`, `non_hermitian.pt_dimer`, `weather.data/train` e os submódulos de `physics_audit` (testados só via o `__init__`).
- Dependências opcionais: `multiscale.learned`, `pinn_solver` e `weather` precisam de PyTorch; `weather` e `pinn_solver.io` usam xarray/Zarr e acesso aos buckets públicos do Google Cloud; as figuras usam matplotlib (e scipy em `weather.viz`); `non_hermitian.ep_tuning` usa sympy.
- `weather` trabalha na grade grossa 64 x 32 do WeatherBench2; as previsões publicadas para comparação existem só nas inicializações 00 e 12 UTC.
- As correlações de `closed_form` valem nas faixas citadas nos docstrings e emitem avisos fora delas (por exemplo `particle_drag`, `pipe_flow`, `plate_fin_heatsink`); o `__init__` de `closed_form` não importa todos os módulos.
- As classes de fechamento de `multiscale.closures` herdam de uma base interna cujo `nu_t` levanta `NotImplementedError`; `schema_adapter` declara premissas de MVP no próprio docstring.

## 2. `pinneapple_presets`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 21 | 5.247 | 22 | 31 | 50 (3 arquivos) | 16 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` |  | 1 | 1 | 27 |
| `cases` |  | 14 | 20 | 4 |
| `core` |  | 1 | 10 | 0 |
| `registry` |  | 1 | 6 | 0 |
| `report` |  | 1 | 4 | 27 |
| `surrogate` |  | 1 | 6 | 46 |
| `viz3d` |  | 1 | 6 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_presets`

Industrial presets: engineering problems already set up, solved and validated with PINNeAPPle.

<small>`pinneapple_presets/__init__.py` · 19 linhas · testes: 27</small>

#### `pinneapple_presets.__main__`

python -m pinneapple_presets list | card NAME | run NAME [k=v ...] [--surrogate] | validate NAME | surrogate NAME | report NAME|all [--build-surrogate] [--no-3d] | index

<small>`pinneapple_presets/__main__.py` · 83 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` |  |

#### `pinneapple_presets.cases`

_Módulo sem docstring e sem texto._

<small>`pinneapple_presets/cases/__init__.py` · 1 linhas · **sem teste direto**</small>

#### `pinneapple_presets.cases.airfoil`

Wing / blade section at Re 4e6: lift, drag and moment from the verified zoo surrogate of OpenFOAM RANS.

<small>`pinneapple_presets/cases/airfoil.py` · 290 linhas · testes: 4</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AirfoilSection` | classe | `()` |  |
| `PresetCache` | classe | `()` |  |
| `thin_airfoil_zero_lift` | função | `(code) -> float` | alpha_L0 (deg) of a NACA 4-digit camber line by thin-airfoil theory (numerical quadrature). |

#### `pinneapple_presets.cases.eddy_current`

Eddy-current inspection of aircraft skin: probe impedance, depth of penetration and thinning sensitivity.

<small>`pinneapple_presets/cases/eddy_current.py` · 308 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `EddyCurrentInspection` | classe | `()` |  |

#### `pinneapple_presets.cases.footbridge`

Steel footbridge: crowd strength, deflection, lateral-torsional buckling and pedestrian vibration (EN 1990 A2).

<small>`pinneapple_presets/cases/footbridge.py` · 375 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SteelFootbridge` | classe | `()` |  |
| `section_class_ok` | função | `(h, b, tf, tw, fy) -> tuple` | EN 1993-1-1 Table 5.2, class 3 limits: web in bending c/t <= 124 eps, outstand flange c/t <= 14 eps (welded: c ignores the weld throat, conservative). Returns (ok, web c/t, web limit, flange c/t, flange limit). |

#### `pinneapple_presets.cases.heatsink`

Plate-fin heat sink in ducted forced air for a power module (resistance network + 3-D finite-volume base).

<small>`pinneapple_presets/cases/heatsink.py` · 361 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PlateFinHeatsink` | classe | `()` |  |

#### `pinneapple_presets.cases.insulation`

Insulation of a steam line: heat loss, surface temperature and the economic thickness (ISO 12241, VDI 2055).

<small>`pinneapple_presets/cases/insulation.py` · 279 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SteamLineInsulation` | classe | `()` |  |

#### `pinneapple_presets.cases.ipm_traction`

Interior-PM traction motor (2004 Toyota Prius geometry): torque, ripple, MTPA and voltage at speed by FEA.

<small>`pinneapple_presets/cases/ipm_traction.py` · 252 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `IPMTraction` | classe | `()` |  |
| `machine` | função | `(n_gap) -> IPMMachine` |  |

#### `pinneapple_presets.cases.pcb`

Junction temperatures on a power PCB (DC-DC / motor-drive stage) by 3-D finite volumes of the stack-up.

<small>`pinneapple_presets/cases/pcb.py` · 266 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PowerPCB` | classe | `()` |  |

#### `pinneapple_presets.cases.pipe_network`

Water distribution network: pressures, flows and least-cost pipe sizing (global gradient algorithm).

<small>`pinneapple_presets/cases/pipe_network.py` · 289 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WaterNetwork` | classe | `()` |  |

#### `pinneapple_presets.cases.pressure_vessel`

Cylindrical pressure vessel shell and heads to ASME VIII Div. 1 (UG-27, UG-32), checked by Lamé and 3-D FEM.

<small>`pinneapple_presets/cases/pressure_vessel.py` · 346 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PressureVessel` | classe | `()` |  |
| `ring_mesh` | função | `(ri, ro, h, nr, nt, nz) -> HexMesh` | Quarter ring r in [ri, ro], theta in [0, 90 deg], z in [0, h] as an 8-node hexahedral mesh. |
| `lame` | função | `(p, ri, ro, r, E, nu) -> dict` | Lamé stresses at radius r and the plane-strain radial displacement (closed ends ignored). |

#### `pinneapple_presets.cases.pump_npsh`

Centrifugal pump lifting from a sump: duty point, NPSH available vs required and the cavitation margin.

<small>`pinneapple_presets/cases/pump_npsh.py` · 276 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PumpNPSH` | classe | `()` |  |

#### `pinneapple_presets.cases.shell_tube`

Shell-and-tube closed-loop water cooler (TEMA E, one shell pass, two tube passes): rating by epsilon-NTU.

<small>`pinneapple_presets/cases/shell_tube.py` · 304 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ShellTubeCooler` | classe | `()` |  |

#### `pinneapple_presets.cases.spm_motor`

Surface-PM servo motor: torque, ripple, cogging and back-EMF by 2-D nonlinear magnetostatic FEM.

<small>`pinneapple_presets/cases/spm_motor.py` · 292 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `SPMMotor` | classe | `()` |  |

#### `pinneapple_presets.cases.water_hammer`

Water hammer from a valve closure at the end of a steel water main (method of characteristics).

<small>`pinneapple_presets/cases/water_hammer.py` · 369 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `WaterHammer` | classe | `()` |  |
| `rigid_column_max_head` | função | `(*, length, diameter, friction_factor, q0, h_reservoir, h_downstream, t_close, closure_ex…)` | Peak valve head of the same closure by rigid-column theory (incompressible liquid, rigid pipe; Wylie & Streeter 1993 sec. 1.3): ``(L / g A) dQ/dt = H_res - h_f(Q) - H_valve``, valve ``Q = tau Q0 sqrt(dH/dH0)``. Valid when the closure is much slower than 2L/a; an independent check of the MOC there. |

#### `pinneapple_presets.core`

Industrial preset: an engineering problem already set up and solved with PINNeAPPle.

<small>`pinneapple_presets/core.py` · 342 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Param` | classe | `(name, unit, default, source, lo, hi, help, choices)` | One input. ``lo``..``hi`` is the validity domain of the preset (solver and correlations), not a soft hint: a run outside it is flagged and the applicability check contradicts. |
| `Output` | classe | `(name, unit, help)` |  |
| `Reference` | classe | `(name, params, output, value, unit, source, kind, tol)` | A citable reference value for one output at given parameters. ``kind``: "analytic", "standard", "benchmark", "experiment" or "independent_solver"; ``tol`` is the relative error accepted for a pass. |
| `SurrogateSpec` | classe | `(inputs, outputs, n_train, n_test, bounds, log, log_outputs)` | Which inputs a surrogate spans (each over its Param domain unless ``bounds`` narrows it), which outputs it predicts, and the DOE size (Latin hypercube, solver runs) with its hold-out share. |
| `Result` | classe | `(preset, params, outputs, trust, decision, fidelity, warnings, meta, fields)` | Outputs of one run with its checks, decision and provenance. |
| `Preset` | classe | `(name, title, sector, problem, question, params, outputs, validity, pipeline, library, su…)` | Base class; a preset sets the class attributes and implements :meth:`solve`, :meth:`checks`, :meth:`decide` and :meth:`references` (and optionally :meth:`scene`, :meth:`convergence`, :meth:`figures`). |
| `check` | função | `(name, ok, value=None, limit=None, source='', reason='', critical=False) -> Check` | A trust-report check from a pass/fail (None = not run). |
| `environment` | função | `() -> dict` | Versions and hardware of this run (recorded with every result). |
| `lhs` | função | `(n, bounds, seed=0, log=()) -> np.ndarray` | Latin hypercube sample of ``n`` points in the box (``log[i]``: that axis in log space). |
| `call_quiet` | função | `(fn, *a, **kw)` |  |

#### `pinneapple_presets.registry`

Registry of the industrial presets and the index the Hub reads (``pinneapple_presets/index.json``).

<small>`pinneapple_presets/registry.py` · 97 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register` | função | `(cls) -> type` | Class decorator: add a user preset (a :class:`Preset` subclass with a unique ``name``). |
| `names` | função | `() -> list[str]` | Names of every industrial preset (built in and registered), sorted. |
| `load` | função | `(name) -> Preset` | The preset ``name`` (see :func:`names`). |
| `list_presets` | função | `(sector=None) -> list[dict]` | Name, title, sector and capabilities of every preset (``sector``: substring filter). |
| `index` | função | `() -> dict` | Everything the Hub shows: per preset the data sheet, the measured results and the figure paths (relative to the repository root). |
| `write_index` | função | `(path=INDEX_PATH) -> str` |  |

#### `pinneapple_presets.report`

Solve, validate and document one preset: results, figures, 3-D stills, surrogate, inversion, optimisation, design variants, README (data sheet + measured results) and the Lab manifest, in ``examples/industrial_presets/<name>/``.

<small>`pinneapple_presets/report.py` · 322 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `write_report` | função | `(preset, out_dir=None, *, surrogate='load', three_d=True, lab=True) -> dict` | ``surrogate``: "build" (run the DOE and retrain), "load" (use the stored model; build if missing) or "skip". Returns the summary written to ``results/summary.json``. |
| `compare_variants` | função | `(p, variants, res, root, three_d=True) -> dict` | Run every variant with the same pipeline; table of the key outputs, a bar chart and a 3-D gallery. |
| `readme` | função | `(p, s) -> str` |  |
| `write_catalogue` | função | `(root=None) -> str` | ``examples/industrial_presets/README.md``: one row per preset from its stored summary (measured values only) and the folder's Lab manifest. |

#### `pinneapple_presets.surrogate`

Surrogate of a preset: Latin-hypercube DOE over its parameters run with the real solver, an anisotropic (ARD) Gaussian process per output, error measured on held-out solver runs, the training box as validity domain.

<small>`pinneapple_presets/surrogate.py` · 232 linhas · testes: 46</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `ARDGP` | classe | `()` | Gaussian process, squared-exponential kernel with one length scale per input, constant variance and a nugget, fitted by maximum marginal likelihood (L-BFGS on log hyper-parameters, a few restarts). Inputs are scaled to [0, 1] and the target standardised before fitting. |
| `PresetSurrogate` | classe | `(meta, models, test=None)` | Trained surrogate of one preset: one ARD GP per output on unit-scaled inputs. |
| `surrogate_path` | função | `(name) -> str` |  |
| `load_surrogate` | função | `(name) -> PresetSurrogate \| None` |  |
| `build_surrogate` | função | `(preset, *, seed=0, save=True, tol=0.05, n_train=None, n_test=None) -> PresetSurrogate` | Run the DOE with the real solver, fit, measure the hold-out error and (``save``) store the model. |
| `serve_spec` | função | `(name) -> dict \| None` | What a model server needs to expose the surrogate: inputs with bounds and units, outputs, hold-out error and the Python call. |

#### `pinneapple_presets.viz3d`

3-D pictures of preset results on the library's studio scene.

<small>`pinneapple_presets/viz3d.py` · 144 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `box` | função | `(lo, hi, n=(1, 1, 1))` | Closed axis-aligned box from ``lo`` to ``hi`` with ``n`` cells per side on each face; returns (V, F). |
| `tube` | função | `(path, radius, n_around=16)` | Open tube of ``radius`` (scalar or one per path point) along the polyline ``path`` (N x 3); returns (V, F, s) with ``s`` the index of the path point of each vertex (to put per-point values on the surface). |
| `extrude` | função | `(section, x)` | Prism of the closed 2-D polygon ``section`` (M x 2, in the y-z plane, counter-clockwise) along the stations ``x``; returns (V, F, i) with ``i`` the station index of each vertex. Ends are capped with a fan (convex or star-shaped sections). |
| `merge` | função | `(parts)` | Concatenate (V, F[, values...]) meshes into one; returns the same tuple shape. |
| `stills` | função | `(scene, out_dir, name, modes=(), size=(1100, 700), views=None, frame=None) -> dict[str, s…)` | Write ``out_dir/<name>_viewer.html`` (self-contained, regenerated by the scripts, not committed) and one PNG per field in ``modes`` ("real" for the materials). Returns {mode: png path or None if no browser}. |
| `field_on` | função | `(idx, values) -> np.ndarray` | Per-vertex values from per-station values (``idx`` from :func:`tube` / :func:`extrude`). |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 5 de 21.
- Funções de teste em arquivos que importam algum módulo do pacote: 50.

### Limitações conhecidas

- _Nenhuma registrada._

## 3. `pinneapple_problemdesign`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 28 | 5.642 | 40 | 21 | 123 (9 arquivos) | 19 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `agent` |  | 1 | 1 | 0 |
| `autonomy` |  | 1 | 5 | 41 |
| `cli` |  | 1 | 1 | 0 |
| `codegen` |  | 1 | 1 | 0 |
| `elicitation` |  | 4 | 3 | 0 |
| `extractor` |  | 1 | 1 | 0 |
| `knowledge` |  | 4 | 7 | 22 |
| `merge` |  | 1 | 1 | 0 |
| `method_selection` |  | 1 | 3 | 11 |
| `policy` |  | 1 | 1 | 0 |
| `preset_authoring` |  | 1 | 5 | 15 |
| `protocol` |  | 1 | 4 | 20 |
| `providers` |  | 2 | 3 | 0 |
| `renderers` |  | 3 | 2 | 0 |
| `research_loop` |  | 1 | 3 | 9 |
| `schema` |  | 1 | 16 | 70 |
| `state` |  | 1 | 2 | 0 |
| `unified_agent` |  | 1 | 2 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_problemdesign`

_Módulo sem docstring e sem texto._

<small>`pinneapple_problemdesign/__init__.py` · 62 linhas · testes: 58</small>

#### `pinneapple_problemdesign.agent`

Design agent for interactive problem elicitation via LLM.

<small>`pinneapple_problemdesign/agent.py` · 194 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DesignAgent` | classe | `(llm, use_orchestrator_bridge=True)` | High-level flow: - ingest user message - structured extraction (json_mode=True) - safe merge into ProblemSpec - non-invention policy => unknown -> Gap, assumptions -> Assumption - stage-based gap generation - once sufficient => build plan (FNO-first) and render report |

#### `pinneapple_problemdesign.autonomy`

Autonomy safety-boundary gate for the autonomous physics research loop.

<small>`pinneapple_problemdesign/autonomy.py` · 481 linhas · testes: 41</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `AutonomyLevel` | classe | `()` | How much of the research loop's step-by-step decision-making runs without a human in the loop. |
| `ConsequentialActionClassifier` | classe | `(*, name_patterns=None, category_patterns=None, tag_patterns=None, exceptions=None) -> No…)` | Explicit, inspectable (no ML/fuzzy matching) classifier for whether a tool call is "consequential": crosses outside pure sandboxed simulation/computation/validation because it costs real money, invokes a licensed external tool per-run, or would affect something outside this sandbox (an external … |
| `ApprovalRequest` | classe | `(tool_name, tool_args, reason, autonomy_level, is_consequential)` | Everything a human approver needs to decide whether a gated action should proceed. Passed to ``approver_fn`` by :class:`ApprovalGate`. |
| `ApprovalDecision` | classe | `(approved, reason, requires_human)` | The outcome of :meth:`ApprovalGate.check`. |
| `ApprovalGate` | classe | `(autonomy_level, approver_fn=None, iteration_report_every=5, confidence_pause_threshold=N…)` | The gate itself: decides, per tool call, whether it may proceed autonomously or needs a human's explicit sign-off. |

#### `pinneapple_problemdesign.cli`

CLI for interactive Pinneapple Design agent.

<small>`pinneapple_problemdesign/cli.py` · 39 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `()` |  |

#### `pinneapple_problemdesign.codegen`

Code generation: translate elicited ProblemSpec → concrete Pinneapple API objects.

<small>`pinneapple_problemdesign/codegen.py` · 374 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_pinneapple_spec` | função | `(design_spec) -> PinneappleSpec` | Translate an elicited :class:`~pinneapple_problemdesign.schema.ProblemSpec` into a :class:`~pinneapple_problemdesign.schema.PinneappleSpec`. |

#### `pinneapple_problemdesign.elicitation`

_Módulo sem docstring e sem texto._

<small>`pinneapple_problemdesign/elicitation/__init__.py` · 6 linhas · **sem teste direto**</small>

#### `pinneapple_problemdesign.elicitation.questions`

Stage-specific gap questions for problem elicitation.

<small>`pinneapple_problemdesign/elicitation/questions.py` · 101 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `build_stage_gaps` | função | `(stage, spec_dict) -> List[Gap]` |  |

#### `pinneapple_problemdesign.elicitation.stages`

Elicitation stage definitions and STAGES_ORDER.

<small>`pinneapple_problemdesign/elicitation/stages.py` · 76 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Stage` | classe | `(name, description, required_fields)` |  |

#### `pinneapple_problemdesign.elicitation.validators`

Validation and suggestion for ProblemSpec (warnings, assumptions).

<small>`pinneapple_problemdesign/elicitation/validators.py` · 42 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `validate_and_suggest` | função | `(spec) -> Tuple[List[str], List[Assumption]]` |  |

#### `pinneapple_problemdesign.extractor`

JSON extraction from user text for requirements analysis.

<small>`pinneapple_problemdesign/extractor.py` · 140 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `extract_from_text` | função | `(llm, user_text) -> Tuple[Dict[str, Any], Dict[str, Any]]` |  |

#### `pinneapple_problemdesign.knowledge`

_Módulo sem docstring e sem texto._

<small>`pinneapple_problemdesign/knowledge/__init__.py` · 10 linhas · **sem teste direto**</small>

#### `pinneapple_problemdesign.knowledge.mapping`

Plan builder: dispatches to FNO-first or PINN-first based on task_type.

<small>`pinneapple_problemdesign/knowledge/mapping.py` · 483 linhas · testes: 22</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `available_orchestrator_tools` | função | `(spec) -> List[Dict[str, object]]` | Read-only lookup of real, currently-registered tools from ``pinneapple_worldmodel.orchestrator.PhysicsOrchestrator``'s ``PhysicsToolRegistry`` that are relevant to *spec*. |
| `build_plan` | função | `(spec, gaps, use_orchestrator_bridge=True) -> Plan` | Dispatch to the appropriate plan builder based on spec.task_type. |
| `build_plan_pinn_first` | função | `(spec, gaps) -> Plan` | Plan for PDE-solution and inverse-problem tasks via PINNFactory. |
| `build_plan_fno_first` | função | `(spec, gaps) -> Plan` |  |
| `build_plan_cfd_first` | função | `(spec, gaps) -> Plan` | Plan for external-aerodynamics/CFD/fluid-dynamics problems, using PINNeAPPle's Lattice-Boltzmann solver, RANS/LES turbulence-closure presets, vortex-identification post-processing, and (optionally) the OpenFOAM bridge, instead of defaulting to generic PINN/FNO advice. |

#### `pinneapple_problemdesign.knowledge.physics_knowledge`

Phenomenon -> governing-equation -> typical-parameters -> assumptions knowledge base for problem design.

<small>`pinneapple_problemdesign/knowledge/physics_knowledge.py` · 615 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhenomenonEntry` | classe | `(phenomenon, governing_equation, typical_parameters, assumptions, preset_module, preset_f…)` | One physics-knowledge record, traceable back to a real preset. |
| `lookup_phenomenon` | função | `(name_or_keyword) -> List[PhenomenonEntry]` | Look up knowledge-base entries by a case-insensitive keyword. |

#### `pinneapple_problemdesign.knowledge.pinneapple_capabilities`

Mapping of Pinneapple module capabilities for plan generation.

<small>`pinneapple_problemdesign/knowledge/pinneapple_capabilities.py` · 41 linhas · **sem teste direto**</small>

#### `pinneapple_problemdesign.merge`

Deep merge of extracted spec into existing ProblemSpec.

<small>`pinneapple_problemdesign/merge.py` · 33 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `merge_into_spec` | função | `(spec, partial) -> None` |  |

#### `pinneapple_problemdesign.method_selection`

Recommend a numerical method + turbulence closure from a computed flow regime.

<small>`pinneapple_problemdesign/method_selection.py` · 346 linhas · testes: 11 · palavras-chave no código (heurística; o texto acima diz o status real): placeholder/stub citado no docstring</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `MethodRecommendation` | classe | `(numerical_method, turbulence_model, flow_regime, rationale, confidence)` | A recommended (numerical_method, turbulence_model) pair for a flow. |
| `recommend_method` | função | `(velocity, length_scale, kinematic_viscosity, *, geometry='internal_pipe', compressible=F…)` | Recommend a numerical solver + turbulence closure for a flow. |
| `recommend_method_from_spec` | função | `(spec) -> Optional[MethodRecommendation]` | Convenience wrapper: pull physical parameters out of a ``ProblemSpec`` and call :func:`recommend_method`, or return ``None`` if they aren't present. |

#### `pinneapple_problemdesign.policy`

Non-invention policy: apply assumptions and gaps to design state.

<small>`pinneapple_problemdesign/policy.py` · 67 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `apply_non_invention_policy` | função | `(state, *, unknown_fields, assumptions_suggested, gaps_suggested) -> None` |  |

#### `pinneapple_problemdesign.preset_authoring`

LLM-assisted preset *drafting*, gated by a real, automatically-generated self-consistency check -- not a preset *authoring pipeline* in the sense of producing something trusted for production use.

<small>`pinneapple_problemdesign/preset_authoring.py` · 707 linhas · testes: 15</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PresetDraftError` | classe | `(stage, message, *, raw='') -> None` | Raised when the LLM's proposal cannot be turned into a real, checkable PDE. |
| `ManufacturedSolutionCheck` | classe | `(exact_solution_used, residual_at_exact_solution, tolerance, passed, n_points, notes)` | One method-of-manufactured-solutions self-consistency check. |
| `DraftedPreset` | classe | `(phenomenon_description, proposed_equation_latex_or_sympy, sympy_pde, proposed_bcs, propo…)` | An LLM-proposed PDE preset, together with the real verification that was run against it. See module docstring for the safety model. |
| `run_manufactured_solution_check` | função | `(sympy_pde, coord_syms, field_syms, *, param_syms=None, params=None, tolerance=0.0001, n_…)` | Run a real method-of-manufactured-solutions self-consistency check against `sympy_pde`, the same technique as ``tests/test_manufactured_solutions.py``: compile `sympy_pde` with the real `pinneapple_physics.symbolic_pde` machinery, substitute a simple closed-form exact solution for every field … |
| `draft_preset` | função | `(phenomenon_description, *, llm_provider='anthropic', model=None, use_knowledge_base=True…)` | Ask an LLM to draft a candidate governing PDE for `phenomenon_description`, then run a real manufactured-solution self-consistency check against it. |

#### `pinneapple_problemdesign.protocol`

LLMProvider protocol, LLMMessage, LLMResponse, GeminiProvider.

<small>`pinneapple_problemdesign/protocol.py` · 100 linhas · testes: 20</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LLMMessage` | classe | `(role, content)` |  |
| `LLMResponse` | classe | `(text, raw)` |  |
| `LLMProvider` | classe | `()` | Single interface for the module. |
| `GeminiProvider` | classe | `(model='gemini-1.5-pro')` | Gemini provider using Google Generative AI SDK. |

#### `pinneapple_problemdesign.providers`

_Módulo sem docstring e sem texto._

<small>`pinneapple_problemdesign/providers/__init__.py` · 47 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LLMProvider` | classe | `()` | Provider interface used by pinneapple_problemdesign. |

#### `pinneapple_problemdesign.providers.cosmos_provider`

Cosmos LLM provider for HTTP gateway access.

<small>`pinneapple_problemdesign/providers/cosmos_provider.py` · 187 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CosmosProviderConfig` | classe | `(base_url, api_key, timeout_s, route)` | Cosmos provider configuration. |
| `CosmosProvider` | classe | `(base_url=None, *, api_key=None, timeout_s=None, route=None, session=None)` | Cosmos-backed LLM provider (via your gateway). |

#### `pinneapple_problemdesign.renderers`

_Módulo sem docstring e sem texto._

<small>`pinneapple_problemdesign/renderers/__init__.py` · 5 linhas · **sem teste direto**</small>

#### `pinneapple_problemdesign.renderers.report_json`

JSON renderer for DesignReport.

<small>`pinneapple_problemdesign/renderers/report_json.py` · 35 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_json_report` | função | `(report, *, indent=2) -> str` |  |

#### `pinneapple_problemdesign.renderers.report_md`

Markdown renderer for DesignReport.

<small>`pinneapple_problemdesign/renderers/report_md.py` · 133 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `render_markdown_report` | função | `(report, warnings) -> str` |  |

#### `pinneapple_problemdesign.research_loop`

``AutonomousResearchAgent``: the autonomous research cycle on top of ``UnifiedPhysicsAgent``.

<small>`pinneapple_problemdesign/research_loop.py` · 423 linhas · testes: 9</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `IterationRecord` | classe | `(iteration, problem_description, approval, agent_result, guardrail_report, confidence, no…)` | What happened during one iteration of the loop -- kept in full so a human reviewing a run afterwards can see every decision, not just the final outcome. |
| `ResearchLoopResult` | classe | `(status, iterations, final_confidence, final_result, memory_record_ids, prior_lessons)` | The full, inspectable outcome of an :meth:`AutonomousResearchAgent.run` call. |
| `AutonomousResearchAgent` | classe | `(llm, *, autonomy_level=AutonomyLevel.SUPERVISED, approver_fn=None, memory=None, physics_…)` | The Phase B research loop: hypothesize (revise description) -> simulate (``UnifiedPhysicsAgent``) -> validate (``PhysicsGuardrail`` / ``PhysicsConfidenceScore``) -> decide (stop or revise) -> repeat, behind ``ApprovalGate``'s non-negotiable safety floor, with ``ExperimentMemory`` learning across … |

#### `pinneapple_problemdesign.schema`

Problem design schema: Assumption, Risk, DataSpec, Gap, ProblemSpec.

<small>`pinneapple_problemdesign/schema.py` · 290 linhas · testes: 70</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Assumption` | classe | `(text, confidence, needs_confirmation)` |  |
| `Risk` | classe | `(text, level, mitigation)` |  |
| `DataSpec` | classe | `(sources, format, sampling, variables_observed, target_variables, known_quality_issues, m…)` |  |
| `PhysicsSpec` | classe | `(governing_equations, boundary_conditions, initial_conditions, constraints, parameters_kn…)` |  |
| `GeometrySpec` | classe | `(domain, representation, sensors, coordinate_system, cad_format, voxel_resolution, aoa_sw…)` |  |
| `ValidationSpec` | classe | `(primary_metrics, acceptance_criteria, robustness_tests, ood_scenarios)` |  |
| `DeploymentSpec` | classe | `(environment, latency_budget_ms, update_policy, monitoring)` |  |
| `ConstraintsSpec` | classe | `(hardware, max_training_time, interpretability, compliance)` |  |
| `ProblemSpec` | classe | `(title, goal, task_type, inputs, outputs, horizon, input_window, frequency, domain_contex…)` |  |
| `Gap` | classe | `(id, question, severity, rationale, how_to_obtain, resolved)` |  |
| `PlanStep` | classe | `(title, why, actions, pinneapple_modules, exit_criteria)` |  |
| `Plan` | classe | `(recommended_approach, alternatives, steps, go_no_go)` |  |
| `PinneappleSpec` | classe | `(pde_kind, pde_confidence, coords, fields, pde_params, environment_kwargs, model_name, mo…)` | Concrete Pinneapple API objects derived from the elicited design spec. |
| `DesignReport` | classe | `(spec, gaps, plan, pinneapple_spec, created_at)` |  |
| `uses_pinn_approach` | função | `(spec) -> bool` | Whether ``spec`` should be solved with a PDE-residual PINN pipeline. |
| `uses_cfd_approach` | função | `(spec) -> bool` | Whether ``spec`` describes an external-aerodynamics/CFD problem that should get the LBM/turbulence-closure/vortex-postprocessing plan (``knowledge.mapping.build_plan_cfd_first``) instead of generic PINN/FNO advice. |

#### `pinneapple_problemdesign.state`

DesignState, Turn for problem design conversation state.

<small>`pinneapple_problemdesign/state.py` · 33 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Turn` | classe | `(role, content)` |  |
| `DesignState` | classe | `(spec, gaps, plan, history, stage, done)` |  |

#### `pinneapple_problemdesign.unified_agent`

Compose ``DesignAgent`` and ``PhysicsOrchestrator`` into one working loop.

<small>`pinneapple_problemdesign/unified_agent.py` · 572 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `UnifiedAgentResult` | classe | `(status, design_report, problem_statement, execution_result, design_state, gaps, question…)` | Bundled result of :meth:`UnifiedPhysicsAgent.run`. |
| `UnifiedPhysicsAgent` | classe | `(llm, *, use_orchestrator_bridge=True, orchestrator=None, orchestrator_verbose=True) -> N…)` | Compose a real ``DesignAgent`` with a real ``PhysicsOrchestrator``. |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 9 de 28.
- Funções de teste em arquivos que importam algum módulo do pacote: 123.

### Limitações conhecidas

- _Nenhuma registrada._

## 4. `pinneapple_science`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.253 | 33 | 49 | 19 (1 arquivos) | 10 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `__main__` |  | 1 | 1 | 0 |
| `atlas` |  | 1 | 4 | 0 |
| `core` |  | 1 | 7 | 19 |
| `em` |  | 7 | 34 | 19 |
| `instruments` |  | 1 | 10 | 19 |
| `numerics` |  | 1 | 8 | 19 |
| `plots` |  | 1 | 7 | 0 |
| `registry` |  | 1 | 5 | 0 |
| `report` |  | 1 | 6 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_science`

PINNeAPPle science (``pp.science``): experiments by area of science, at academic and research level, theory and practice (a virtual bench), in 0-D (lumped), 1-D, 2-D, 3-D and 3-D + time; and the atlas of every area.

<small>`pinneapple_science/__init__.py` · 25 linhas · testes: 19</small>

#### `pinneapple_science.__main__`

``python -m pinneapple_science atlas [--html PATH] | list [--area A] [--dim D] | run NAME [--dim D] [--quick] | report NAME|all [--quick] [--no-register] | table``.

<small>`pinneapple_science/__main__.py` · 73 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `main` | função | `(argv=None) -> int` |  |

#### `pinneapple_science.atlas`

The atlas of science areas: what :mod:`pinneapple_science` covers, what the rest of the library already covers, and what is planned.

<small>`pinneapple_science/atlas.py` · 273 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Area` | classe | `(id, title, branch, status, description, subareas)` |  |
| `Atlas` | classe | `(schema, version, statuses, dims, areas)` |  |
| `load` | função | `(path=ATLAS_PATH) -> Atlas` | The atlas (``pp.science.atlas()``): every area and subarea with its status and pointers, from the versioned data file ``pinneapple_science/data/atlas.json``. |
| `validate` | função | `(atlas=None, *, check_presets=True, check_lab=True) -> list[str]` | Problems with the atlas (empty when it is consistent): unknown statuses, pointers to modules that do not exist, presets or Lab experiments that are not registered, ``pp.science`` experiments that are not registered or not listed, and ``available`` subareas without an experiment. |

#### `pinneapple_science.core`

The common shape of a science experiment: what it declares, what every run returns.

<small>`pinneapple_science/core.py` · 348 linhas · testes: 19 · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Reference` | classe | `(value, source, kind, unit)` | A reference value and where it comes from. ``kind`` is one of ``closed_form`` (a formula, cited), ``published`` (a tabulated value from a paper or book), ``codata`` (a fundamental constant), ``limit`` (an asymptotic limit), ``conservation`` (a quantity that must be conserved), ``convergence`` (the … |
| `Comparison` | classe | `(quantity, computed, reference, rel_error, tol, passed, kind, source, unit, how, abs_erro…)` |  |
| `Result` | classe | `(experiment, title, area, topic, level, kind, dim, params, values, comparisons, checks, f…)` | What a run returns. Build it through the recording methods (``value``, ``compare``, ``check``, ``figure``, ``data``); ``trust`` is assembled from the comparisons and checks when the run ends. |
| `Experiment` | classe | `(name, title, area, topic, level, kind, dims, description, references, params, quick)` | Base class. Subclasses set the class attributes and implement ``_run(res, dim, p)``, which fills the :class:`Result` ``res`` for dimension ``dim`` with parameters ``p`` (defaults merged with the caller's). |
| `jsonable` | função | `(v) -> Any` | Plain JSON types (numpy scalars and arrays to Python numbers and lists; NaN and inf to None). |
| `observed_order` | função | `(errors, ratio=2.0) -> list[float]` | Observed orders ``log(e_i / e_{i+1}) / log(ratio)`` of a refinement ladder. |
| `central_derivative` | função | `(f, x, h) -> float` |  |

#### `pinneapple_science.em`

Electromagnetism experiments of :mod:`pinneapple_science`, built on ``pp.em`` (:mod:`pinneapple_simulation.electromagnetics`) and ``pp.maxwell`` (:mod:`pinneapple_physics.maxwell_checks`).

<small>`pinneapple_science/em/__init__.py` · 3 linhas · **sem teste direto**</small>

#### `pinneapple_science.em.bench`

Bench experiments (practice): a virtual laboratory with instruments that add noise, calibration error and resolution, a fit with its uncertainty, and the comparison with the value the bench was built with.

<small>`pinneapple_science/em/bench.py` · 220 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Mu0Solenoid` | classe | `()` |  |
| `RCDischarge` | classe | `()` |  |

#### `pinneapple_science.em.fdtd_study`

A configurable 3-D + time FDTD study on top of :mod:`pinneapple_simulation.electromagnetics.fdtd`: a domain in wavelengths, sources (Hertzian dipoles, soft E sources) and materials (spheres, boxes, cylinders with any :class:`~pinneapple_simulation.electromagnetics.media.Medium`) from a plain dict, …

<small>`pinneapple_science/em/fdtd_study.py` · 199 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FDTDConfig` | classe | `(f0, size, cells_per_wavelength, npml, bandwidth, periods, sources, materials)` |  |
| `FDTDRun` | classe | `(config, dx, n, power, current_moment, ntff, frames, frame_times, volume, volume_grid, sl…)` |  |
| `run_fdtd` | função | `(cfg, *, snapshots=24, volume_half=3, volume_steps=24, maxwell_block=True, snap_window=(2…)` | Run the configured study: returns power at f0 (near-to-far-field box two cells inside the CPML), the snapshots of E_z on the three central planes during ``snap_window`` (periods of f0), a small block of all six components around a point 0.6 wavelength from the first source on ``volume_steps`` … |
| `ladder_values` | função | `(values) -> list[float]` |  |

#### `pinneapple_science.em.induction`

Induction and circuits, academic level: Faraday's law with a magnet falling through a loop (3-D + time) and the RC, RL and series RLC circuits (lumped, 0-D in space).

<small>`pinneapple_science/em/induction.py` · 271 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FaradayInduction` | classe | `()` |  |
| `Circuits` | classe | `()` |  |
| `magnet_rings` | função | `(a, length, Br, n_rings, n_seg, z_center=0.0)` | Segments and per-segment currents of a cylindrical magnet (radius ``a``, ``length``, remanence ``Br``) as ``n_rings`` polygonal rings of the bound surface current ``K = Br / mu0`` (A/m). |
| `loop_flux` | função | `(segs, I_ring, R, n_path=64) -> float` | Flux through a circular loop of radius ``R`` in the plane z = 0 by Stokes' theorem, ``oint A . dl`` (:func:`pinneapple_simulation.electromagnetics.statics.flux_through_path`). A surface integral of B_z would be singular when the magnet's current sheet crosses the loop's disc; the line integral is … |

#### `pinneapple_science.em.research`

Electromagnetism, research level: a configurable FDTD study in 3-D + time with mesh convergence and the observed order, Monte Carlo uncertainty propagation, an inverse problem (permittivity from a measured reflection spectrum) and a surrogate with its hold-out error.

<small>`pinneapple_science/em/research.py` · 548 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `FDTDStudy3D` | classe | `()` |  |
| `WaveguideUQ` | classe | `()` |  |
| `InversePermittivity` | classe | `()` |  |
| `HallSurrogate` | classe | `()` |  |
| `slice_viewer` | função | `(run, path, title='FDTD: E_z in 3-D + time') -> str` | A single-file three.js page (Studio viewer): the three central planes as surfaces, E_z r on them animated in time (play, pause, time slider). |

#### `pinneapple_science.em.statics`

Electro- and magnetostatics, academic level: Coulomb and superposition, Gauss's law, the parallel-plate capacitor with its edge effect, Biot-Savart (wire, loop, solenoid), Ampere's law and the Hall effect.

<small>`pinneapple_science/em/statics.py` · 901 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CoulombSuperposition` | classe | `()` |  |
| `GaussLaw` | classe | `()` |  |
| `ParallelPlateCapacitor` | classe | `()` |  |
| `BiotSavart` | classe | `()` |  |
| `AmpereLaw` | classe | `()` |  |
| `HallEffect` | classe | `()` |  |
| `ring_potential` | função | `(points, Q, R) -> np.ndarray` | Potential of a uniformly charged ring (radius ``R``, charge ``Q``, plane z = 0, centred on the z axis): ``phi = Q / (4 pi eps0) (2 / pi) K(m) / sqrt((R + s)^2 + z^2)``, ``m = 4 R s / ((R + s)^2 + z^2)`` (Jackson, *Classical Electrodynamics*, 3rd ed., sec. 3.3 / problem 3.3; Smythe, *Static and … |
| `palmer_strip_capacitance` | função | `(w, d, eps_r=1.0) -> float` | Capacitance per unit length of two parallel thin strips (width ``w``, gap ``d``) with the edge correction of Palmer (1937): ``eps (w/d) [1 + d/(pi w) (1 + ln(2 pi w / d))]`` (F/m). Asymptotic for ``w >> d``. H. B. Palmer, "The capacitance of a parallel-plate capacitor by the Schwartz-Christoffel … |
| `kirchhoff_disc_capacitance` | função | `(a, d, eps_r=1.0) -> float` | Capacitance of two thin coaxial discs (radius ``a``, gap ``d``), Kirchhoff (1877): ``eps [pi a^2 / d + a (ln(16 pi a / d) - 1)]`` (F), asymptotic for ``d << a``. G. Kirchhoff, Monatsber. Akad. Wiss. Berlin (1877) 144; see also Jackson, Am. J. Phys. 68, 789 (2000). |
| `strip_capacitor_2d` | função | `(w, d, h, L=40.0, growth=1.08)` | ``(C per unit length, grid, phi)`` of two thin strips at +-1/2 V by finite volumes on a graded quarter domain (mirror at x = 0, antisymmetry phi = 0 at y = 0, grounded box at distance ``L``). |
| `disc_capacitor` | função | `(a, d, h, *, cartesian, L=12.0)` | Capacitance of two coaxial thin discs at +-1/2 V: an axisymmetric (r, z) solution, or a 3-D Cartesian one on one eighth of the domain (mirror planes x = 0, y = 0, antisymmetry at z = 0, staircase disc edge). |
| `hall_bar` | função | `(L, W, beta, n, V=1.0, nx=None)` | Potential in a Hall bar ``[0, L] x [0, W]`` (contacts at x = 0 and x = L held at 0 and ``V``; insulating edges y = 0, W) for Hall parameter ``beta = mu B``: Laplace inside, ``d phi/dy = beta d phi/dx`` on the edges (``J_y = 0`` with the Drude conductivity tensor ``sigma0/(1+beta^2) [[1, beta], … |

#### `pinneapple_science.em.waves`

Waves and optics, academic level: radiation of an oscillating dipole, a plane wave in a lossy medium, reflection and refraction at an interface (Fresnel and Snell), waveguide modes, cavity resonances and Mie scattering.

<small>`pinneapple_science/em/waves.py` · 599 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `DipoleRadiation` | classe | `()` |  |
| `LossyPlaneWave` | classe | `()` |  |
| `FresnelSnell` | classe | `()` |  |
| `WaveguideModes` | classe | `()` |  |
| `CavityResonance` | classe | `()` |  |
| `MieScattering` | classe | `()` |  |
| `dipole_cartesian` | função | `(X, Y, Z, Il, omega)` | Phasor E and H (Cartesian components) of a z-directed Hertzian dipole at the origin (:func:`pinneapple_simulation.electromagnetics.waves.hertzian_dipole_fields`). |

#### `pinneapple_science.instruments`

Virtual laboratory instruments and the fits a bench experiment ends with.

<small>`pinneapple_science/instruments.py` · 194 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Instrument` | classe | `(name, unit, noise, gain_u, offset_u, resolution, seed, _rng, _gain, _offset)` | A reading instrument. ``noise``: standard deviation of the random noise per reading (in the unit read); ``gain_u``: relative standard uncertainty of the gain (calibration); ``offset_u``: standard uncertainty of the zero offset; ``resolution``: the quantisation step (0 for none). The gain and offset … |
| `Oscilloscope` | classe | `(rate, duration, v_range, bits, noise, gain_u, timebase_u, seed)` | A digitising oscilloscope: samples ``signal(t)`` at ``rate`` over ``duration`` after ``t0`` with vertical noise, a vertical gain error, an ADC of ``bits`` over the full scale ``±v_range`` and a time-base error ``timebase_u`` (relative). |
| `LinearFit` | classe | `(slope, intercept, slope_u, intercept_u, cov, chi2, dof, residuals)` |  |
| `CurveFit` | classe | `(params, errors, cov, chi2, dof, residuals)` |  |
| `FieldProbe` | função | `(**kw) -> Instrument` | A Hall magnetic-field probe (tesla). |
| `Ammeter` | função | `(**kw) -> Instrument` |  |
| `Voltmeter` | função | `(**kw) -> Instrument` |  |
| `linear_fit` | função | `(x, y, sigma_y=None, *, through_origin=False, scale_by_chi2=True) -> LinearFit` | Weighted least squares ``y = a x + b`` (``b = 0`` with ``through_origin``). Uncertainties from the covariance ``(A^T W A)^{-1}``, scaled by the reduced chi-square when it exceeds one (``scale_by_chi2``), so a scatter larger than the stated sigma is not hidden. |
| `curve_fit` | função | `(model, x, y, p0, sigma_y=None) -> CurveFit` | Non-linear least squares (scipy ``curve_fit``) with the same uncertainty rule as :func:`linear_fit`. |
| `combine` | função | `(*relative_uncertainties) -> float` | Root-sum-square of independent relative standard uncertainties (GUM eq. 10 for a product of powers one). |

#### `pinneapple_science.numerics`

Small numerical building blocks shared by the experiments: graded tensor grids, a finite-volume solver for ``div(eps grad phi) = -rho`` in 1-D, 2-D and 3-D with fixed-potential nodes and mirror (symmetry) faces, the discrete field energy, and a scalar Helmholtz eigen-solver for waveguide …

<small>`pinneapple_science/numerics.py` · 244 linhas · testes: 19</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `TensorGrid` | classe | `(axes)` | Node coordinates per axis (1, 2 or 3 axes). |
| `graded_axis` | função | `(core, h, length, growth=1.12, start=0.0) -> np.ndarray` | Nodes from ``start``: spacing exactly ``h`` over at least ``core`` (rounded up to whole cells), then growing geometrically by ``growth`` until ``start + length``. |
| `solve_potential` | função | `(grid, fixed, values, *, eps=None, rho=None, tol=1e-10, direct=None, mean='harmonic') -> …)` | Solve ``-div(eps grad phi) = rho`` with ``phi = values`` on the ``fixed`` nodes (boolean array; put the outer box there for a grounded enclosure). Faces without fixed nodes are zero-flux, which is the mirror condition of a symmetry plane. ``eps`` (relative or absolute, per node) defaults to 1, … |
| `field_energy` | função | `(grid, phi, eps=None, mean='harmonic') -> float` | Discrete ``1/2 sum eps (grad phi)^2 dV`` (per unit length in 2-D), with the face fluxes of the scheme. |
| `helmholtz_eigs` | função | `(nx, ny, a, b, kind, n=6, inside=None, modes=False)` | The ``n`` smallest cut-off wavenumbers ``k_c`` of a waveguide cross-section from the scalar Helmholtz eigenproblem ``-lap psi = k_c^2 psi`` on an ``nx`` x ``ny`` cell-centred grid of the box ``a`` x ``b``: ``kind="TM"`` (``psi = E_z``, Dirichlet on the wall) or ``"TE"`` (``psi = H_z``, Neumann). … |
| `spectral_peaks` | função | `(signal, dt, f_max, *, n_peaks=8, min_rel=0.001, pad=8) -> np.ndarray` | Frequencies of the strongest local maxima of the spectrum of ``signal`` below ``f_max`` (Hann window, ``pad``-fold zero padding, parabolic interpolation of the log spectrum), ascending. Peaks below ``min_rel`` times the largest are dropped. |
| `refine_ladder` | função | `(base, levels, ratio=2) -> Sequence[int]` |  |
| `gp_loo_scale` | função | `(gp) -> float` | Factor for the predictive standard deviation of a fitted Gaussian process (an object with ``L``, the Cholesky factor of its training covariance, and ``alpha = K^-1 z``, as :class:`pinneapple_presets.surrogate.ARDGP`) from its leave-one-out residuals on the training data: ``sqrt(mean(z_i^2))`` with … |

#### `pinneapple_science.plots`

Figure conventions of :mod:`pinneapple_science`: one palette (computed = slot 1, reference = slot 2, in that fixed order), a single-hue sequential map for magnitudes, a two-hue diverging map with a neutral middle for signed fields, thin marks, recessive axes. Every function returns a matplotlib …

<small>`pinneapple_science/plots.py` · 83 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `style` | função | `(ax) -> None` |  |
| `figure` | função | `(ncols=1, nrows=1, *, width=4.2, height=3.3, three_d=(), title='')` | ``(fig, axes)``: ``axes`` is a flat list; indices in ``three_d`` get 3-D axes. |
| `badge` | função | `(ax, text, ok=True, loc='lower right') -> None` | A small text box with a measured result (the ink is the text colour; the frame carries pass / fail). |
| `close` | função | `(fig) -> None` |  |
| `log_ticks_ok` | função | `(values) -> bool` |  |
| `to_rgb` | função | `(fig) -> np.ndarray` | The figure as an RGB array (for animation frames). |
| `integer_ticks` | função | `(ax, values) -> None` | Plain integer labels at ``values`` on a log x axis (no minor labels like 3x10^1). |

#### `pinneapple_science.registry`

The experiments of :mod:`pinneapple_science`, by name (modules are imported on first use).

<small>`pinneapple_science/registry.py` · 68 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `register` | função | `(cls) -> type[Experiment]` | Class decorator: add an experiment to the catalogue. |
| `get` | função | `(name) -> Experiment` |  |
| `names` | função | `() -> list[str]` |  |
| `experiments` | função | `(area=None, *, level=None, kind=None, dim=None) -> list[dict[str, Any]]` | Cards of the experiments matching every filter given (``area`` matches the area id or its prefix). |
| `run` | função | `(name, dim=None, *, quick=False, **params) -> Result` | Run one experiment in one dimension (its first by default) with parameters over its defaults. |

#### `pinneapple_science.report`

Committed results of :mod:`pinneapple_science`: one folder per experiment with every dimension it runs in, its Lab manifest (``experiment.json``), its Lab seal and a README; and the catalogue with the atlas.

<small>`pinneapple_science/report.py` · 204 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `folder_of` | função | `(name, root=None) -> str` |  |
| `write` | função | `(name, *, root=None, quick=False, register=True, seal=True) -> dict[str, Any]` | Run ``name`` in every dimension, save the results, write the README and the Lab manifest, register it in the Lab and stamp the figures with its seal. Returns ``{"folder", "results", "item", "sealed"}``. |
| `readme` | função | `(name, results, sealed, item) -> str` |  |
| `table` | função | `(root=None) -> list[dict[str, Any]]` | Experiment x dimension x quantity x reference x measured error, from the committed ``result.json`` files. |
| `write_catalogue` | função | `(root=None, register=True) -> str` | ``examples/science/README.md`` (one row per comparison from the committed results), ``ATLAS.md``, ``atlas.html`` and the folder's manifest. |
| `write_all` | função | `(*, root=None, quick=False, register=True) -> list[dict[str, Any]]` |  |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 6 de 16.
- Funções de teste em arquivos que importam algum módulo do pacote: 19.

### Limitações conhecidas

- _Nenhuma registrada._

## 5. `pinneapple_condense`

### Objetivo

_Sem texto de objetivo._

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 18 | 6.006 | 29 | 115 | 27 (3 arquivos) | 3 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `algebraic` |  | 1 | 6 | 6 |
| `benchmark` |  | 1 | 14 | 0 |
| `exceptions` |  | 1 | 6 | 14 |
| `falsify` |  | 1 | 21 | 6 |
| `figures` |  | 1 | 8 | 0 |
| `flow` |  | 1 | 9 | 6 |
| `fv` |  | 1 | 8 | 21 |
| `hierarchy` |  | 1 | 2 | 14 |
| `homogenize` |  | 1 | 7 | 14 |
| `learned` |  | 1 | 10 | 14 |
| `meshless` |  | 1 | 9 | 14 |
| `meshless2d` |  | 1 | 9 | 6 |
| `package` |  | 1 | 7 | 27 |
| `query` |  | 1 | 8 | 7 |
| `query_benchmark` |  | 1 | 1 | 0 |
| `substructure` |  | 1 | 15 | 21 |
| `transient` |  | 1 | 4 | 14 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_condense`

Adaptive Physics Condensation (work in progress).

<small>`pinneapple_condense/__init__.py` · 2 linhas · testes: 27</small>

#### `pinneapple_condense.algebraic`

Algebraic condensation: the method of :mod:`.substructure` for *any* assembled sparse operator.

<small>`pinneapple_condense/algebraic.py` · 299 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Partition` | classe | `(block, sep, interiors, adjacent, shape, size, origin, coords, fieldid)` | Blocks of the unknowns: ``block`` per unknown (flat id), ``sep`` mask, and per block its interior and adjacent separator unknowns in canonical local order. |
| `Condensation` | classe | `(A, part, timings)` | Exact algebraic condensation of ``A x = b`` over a :class:`Partition`. |
| `Reduced` | classe | `(cond, fc, modes)` | Galerkin-reduced separator problem of a :class:`Condensation` (exact when every face keeps all its modes). |
| `partition` | função | `(A, coords, block_size, origin=(0.0, 0.0), fieldid=None, primal=None, h=None) -> Partition` | Blocks of side ``block_size`` from ``origin``; separators from the coupling graph (see the module docstring). ``h``: a length below the smallest spacing (for the canonical rounding; default block size / 1e4). |
| `faces` | função | `(part, A) -> dict` | Separator unknowns grouped by the set of blocks they couple to. Two blocks: a face, keyed by ``(orientation, field, length)`` (orientation 0: the two blocks are x-neighbours) with canonical positions along it; three or more blocks, or primal unknowns: kept full. |
| `face_modes` | função | `(snapshots, fc, m) -> dict` | POD modes per face type from separator snapshots (full-length solution vectors of the snapshot problem with its face description ``fc``): the constant plus the ``m - 1`` leading left singular vectors of the constant-free face traces. |

#### `pinneapple_condense.benchmark`

The measured studies of the condensation benchmark (each returns plain JSON-able numbers).

<small>`pinneapple_condense/benchmark.py` · 396 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `peak_rss_gb` | função | `() -> float` |  |
| `isolated` | função | `(func, timeout=3600, module=…, **kwargs) -> dict` | Run ``<module>.<func>(**kwargs)`` (default: this module) in a fresh Python process; its result plus the peak RSS of that process (``peak_rss_gb``). |
| `mesh_convergence` | função | `(shape=(8, 8), levels=((4, 1), (8, 2), (16, 4)), seed=1) -> dict` | Reference solutions at increasing resolution (cells per block side, vertical multiplier), the observed order from the last three levels, the Richardson extrapolation and the error of the default resolution (8, 1). |
| `accuracy_vs_modes` | função | `(shape=(12, 12), ms=(1, 2, 4, 6, 8, 12, 16, 24), seed=1, modes='pod') -> dict` |  |
| `refinement_study` | função | `(shape=(12, 12), m0=2, steps=6, seed=1, hot=True) -> dict` | Doerfler face refinement from ``m0`` modes (``p``: double the modes of marked faces; ``full``: un-reduce them) vs uniform ``m``; a power map with a sub-pitch hotspot (``hot``) makes the error local. |
| `correction_study` | função | `(nb=32, ms=(6, 12), seed=0, power_scale=4.0, rtol=1e-09, every=10) -> dict` | The exact correction (:meth:`.substructure.Condensed.correct`) on a large hotspot case: iterations to the fine solution from each reduced model, and the Tmax error along the way (every ``every`` iterations). |
| `homogenization_study` | função | `(shape=(12, 12), seed=1) -> dict` |  |
| `scaling_point` | função | `(nb, m=12, reference=True, flat=True, hier=True, seed=2) -> dict` | One ``nb x nb`` package: wall time and result of each method that is asked for (call through :func:`isolated` per method for its own peak memory). |
| `configurations` | função | `(nb=64, m=12, n_configs=20, seed=0, check=2) -> dict` | Build once, then solve ``n_configs`` new power maps; the first ``check`` are also solved by the reference. |
| `learned_study` | função | `(nb=32, m=12, n_train=48, seed=1, defects=4, exact_reference=True, tol_T=0.01) -> dict` |  |
| `transient_study` | função | `(shape=(6, 6), dt=0.0002, steps=150, mk=((8, 8), (16, 16), (32, 32)), seed=1) -> dict` |  |
| `root_compressibility` | função | `(nb=64, m=12, tol=1e-08) -> dict` | Numerical rank (relative ``tol``) of the off-diagonal blocks between the four sides of the root Schur complement: how much an H-matrix representation of the top operator could save. |
| `meshless_study` | função | `(nb=2, hfs=(6, 8, 10, 12), methods=('gfdm', 'rbf_fd'), interfaces=('pairs', 'naive'), ls_…)` | Track 1b: RBF-FD and GFDM point clouds from the implicit geometry against the fine FV reference, for a range of spacings ``pitch / hf``, interface treatments (node pairs with flux continuity and the contact law, or the naive single-node collocation), and least-squares collocation (``ls_hfs``). … |
| `exceptions_study` | função | `(shape=(12, 12), seed=1, budgets=…) -> dict` | Track 1c: the Keff prior plus residual-selected exception patches against random patches at the same budget, on a smooth power map and on one with a sub-pitch hotspot; and the 1-D column prior for comparison. |

#### `pinneapple_condense.exceptions`

Prior field plus exception patches (track 1c): solve exactly only where a cheap prior does not hold.

<small>`pinneapple_condense/exceptions.py` · 190 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `prior` | função | `(pkg, kind='keff') -> tuple[np.ndarray, float]` | The prior on the fine cells ``(nx, ny, nz)`` and its cost (s). |
| `block_indicator` | função | `(pkg, sysm, T, f) -> np.ndarray` | ``\|\|K T - f\|\|`` over the cells of each block (W), ``(nbx, nby)``. |
| `correct` | função | `(pkg, T0, blocks, *, halo=1, blend='core', sysm=None, f=None) -> tuple[np.ndarray, dict]` | Solve the fine problem on the patches around ``blocks`` (boolean ``(nbx, nby)``) with ``T0`` outside. |
| `energy_correct` | função | `(pkg, T, sysm) -> tuple[np.ndarray, float]` | Add ``alpha * phi`` (``phi``: the response to a uniform map, i.e. one block with adiabatic sides, by symmetry) so that the Robin outflow equals the power. |
| `run` | função | `(pkg, *, budget=0.1, select='residual', halo=1, blend='core', prior_kind='keff', energy=T…)` | The whole method at one budget (fraction of blocks selected); returns the field and its costs. |
| `pareto` | função | `(pkg, budgets=…, *, halos=(0, 1), selects=('residual', 'random'), prior_kind='keff', seed…)` | Error against the fraction of cells solved exactly, for each selection rule and halo; with the reference. |

#### `pinneapple_condense.falsify`

Falsification studies of the condensation method (``docs/research/condensation_falsification.md``): each function is one pre-registered test, returns plain JSON-able numbers and is meant to run in a fresh process (:func:`.benchmark.isolated` with ``module="pinneapple_condense.falsify"``) so that …

<small>`pinneapple_condense/falsify.py` · 989 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `t1_reference` | função | `(nb, seed=2, m=12, rtol=1e-10, fine=True, maxiter=2000) -> dict` | Reference peak junction temperature of an ``nb x nb`` package: the fine AMG solve (``fine``) and the exact condensation (PCG on the unreduced interfaces, preconditioned by the m-mode model) with its convergence trail (T1b: iterations and time to 1 mK). |
| `t1_point` | função | `(nb, m, seed=2) -> dict` | The memoised hierarchy with ``m`` interface modes: peak junction temperature, build and solve time. |
| `f0_validation` | função | `(poiseuille=(16, 32, 64), cavity=(32, 64, 128), nusselt=(16, 32, 64)) -> dict` | The flow solver against plane Poiseuille flow, the Re = 100 cavity and the one-sided-flux parallel-plate Nusselt number (5.385). |
| `f2_stokes` | função | `(n_pins=16, cells=16, pin=8, ms=(1, 2, 4, 8, 16), n_snap=24, seed=0) -> dict` | F2 / F2r: Stokes through an ``n_pins x n_pins`` pin array; exact algebraic condensation against the global direct solve, then reduced faces from snapshots of a 4 x 4 array with random smooth inlet profiles. |
| `f1_advection` | função | `(n_pins=8, cells=16, pin=8, pes=(0.1, 1, 10, 100, 1000), ms=(1, 2, 4, 8, 16), n_snap=24, …)` | F1 / F1r: steady advection-diffusion through the pin array (velocity: the Stokes solution), block Peclet sweep; exact condensation of the non-symmetric operator, then reduced faces from a 4 x 4 snapshot problem with random pin sources and inlet profiles at the same Peclet number. |
| `f3_cht` | função | `(n_pins=(8, 4), cells=16, pin=8, res=(1, 10, 50, 100), ms=(2, 4, 8, 16), n_snap=16, tol=1…)` | F3 / F3r: cold plate (fin array in a free-slip strip), Navier-Stokes by Picard, full solve against condensed Picard (every Oseen iterate condensed by content), then the conjugate energy equation; reduced faces for the flow at the largest Re; the fraction of reversed face flux on block faces. |
| `thermal_case` | função | `(pkg, m=12, rtol=1e-10) -> dict` | One package: fine FV peak junction temperature; m-mode condensation (Tmax error, energy-norm effectivity of the residual estimate); exact condensation (PCG to ``rtol``) against the fine solve. |
| `t2_contrast` | função | `(nb=32, ratios=(1000.0, 10000.0, 100000.0), m=12, seed=2) -> dict` | T2: the underfill conductivity lowered so that k_copper / k_underfill = ratio (baseline 401 / 0.5 = 802). |
| `t3_anisotropy` | função | `(nb=32, factors=(100.0, 0.01), m=12, seed=2) -> dict` | T3: TIM and interposer with in-plane / through-thickness conductivity ratio ``factor``. |
| `t8_adversarial` | função | `(nb=16, m=12, seed=2) -> dict` | T8: power maps unlike the snapshot set (same total power as the baseline hotspot map). |
| `t9_thin` | função | `(nb=16, m=12, seed=2) -> dict` | T9: thin layers -- TIM 30 -> 3 -> 1 um (1 um with one cell row), die 100 -> 20 um. |
| `t4_no_repetition` | função | `(nb=32, m=12, seed=1, n_train=48) -> dict` | T4: every block distinct (continuous random variants, four defects). Against one fine FV solve: the exact path (every block condensed, m-mode interfaces, then the PCG correction to the fine solution), and the learned path (:func:`.learned.solve_adaptive`, training timed separately). |
| `t5_variants` | função | `(nb=64, Vs=(1, 4, 16, 64, 256), m=12, seed=0) -> dict` | T5: ``V`` discrete variants at random positions: hierarchy build (memoised subtrees) against the flat condensation build, both including the V block condensations. |
| `t6_t7_nonlinear` | função | `(nb=16, case='k', seed=2, power_scale=2.0) -> dict` | T6 (``case="k"``: silicon and copper k(T)) or T7 (``case="R"``: R_c(T) = R_0 (1 + beta (T - T_0)) on the die/TIM contact, beta doubling R_c over the temperature range of the linear solution). |
| `t10_pitch` | função | `(nb=16, pitches=(4e-05, 2e-05), ms=(2, 4, 8, 12, 16, 24), n_snap=24, seed=2) -> dict` | T10: columns of blocks alternating in pitch, through algebraic condensation (each pitch its own block operator, found by content); baseline: the uniform pitch through the same pipeline. Reduced faces from snapshots of a 4 x 4 package with the same column pattern and random power maps. |
| `m1_advection` | função | `(pins=4, densities=(8, 16, 32), truth_cells=64, pes=(1, 10, 100), methods=('rbf_fd', 'gfd…)` | M1: meshless advection-diffusion on point clouds (velocity interpolated from the finest FV Stokes solution) against FV at equal unknowns; truth: FV at ``truth_cells`` per unit cell. |
| `m2_stokes` | função | `(pins=4, densities=(8, 16, 32), truth_cells=64, methods=('rbf_fd', 'gfdm'), seed=0) -> di…)` | M2: collocated meshless Stokes on fluid points with pin-wall points; pressure drop against FV. |
| `m3_hybrid` | função | `(pins=2, densities=(8, 16, 32), truth_cells=64, k_pin=400.0, k_mat=1.0, R_c=0.001, stenci…)` | M3: FV-at-interfaces hybrid against FV everywhere at equal unknowns (truth: FV at ``truth_cells``), with the error split between the FV band and the meshless points, and the stencil sensitivity. |
| `m4_meshless_condensed` | função | `(pins=4, cells=16, seed=0) -> dict` | M4: meshless matrices condensed by :mod:`.algebraic`: the hybrid conduction (replicated point clouds and FV bands: identical blocks expected) and the meshless advection-diffusion (velocity differs per block). |
| `exception_patches` | função | `(A, b, x0, x_true, block, nbshape, budgets, select='residual', halo=1, seed=0)` | Generic prior + exception patches: blocks ranked by the norm of ``b - A x0`` over their unknowns (or random), the patch (selected blocks + ``halo`` rings) solved with the prior as Dirichlet data outside. |
| `e1_cfd` | função | `(n_pins=8, fine=16, coarse=4, budgets=…, Pe_b=100.0, Re=50.0, seed=0) -> dict` | E1: residual-selected against random exception patches on the advection-diffusion case (Pe_b = 100) and on the conjugate energy field of the cold plate (Re_fin = 50); prior: the same case on a 4x coarser grid. |

#### `pinneapple_condense.figures`

Figures of the condensation benchmark (matplotlib, from the JSON of :mod:`.benchmark`).

<small>`pinneapple_condense/figures.py` · 214 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `scaling` | função | `(points, path) -> str` | Wall time and peak memory against the number of blocks (two panels, one quantity each). |
| `accuracy` | função | `(res, path) -> str` |  |
| `refinement` | função | `(res, path) -> str` |  |
| `keff_maps` | função | `(maps, path) -> str` | Fine junction temperatures, the homogenised block averages, and their difference. |
| `transient` | função | `(res, path) -> str` |  |
| `meshless_convergence` | função | `(res, path, mesh_error=None) -> str` | \|Tmax error\| and \|energy balance\| against the number of nodes, per method and interface treatment. |
| `meshless_fields` | função | `(res, method, path) -> str` | The point cloud on a vertical slice coloured by temperature, and the junction-temperature error map. |
| `pareto` | função | `(res, path) -> str` | Error against the fraction of cells solved exactly (exception patches), per selection rule and halo. |

#### `pinneapple_condense.flow`

Steady 2-D incompressible flow and heat on a staggered (MAC) finite-volume grid, assembled as one sparse system: the reference and the operator of the CFD falsification tests (``docs/research/condensation_falsification.md``).

<small>`pinneapple_condense/flow.py` · 330 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Grid` | classe | `(nx, ny, h, solid, walls, closed, lid, index)` |  |
| `ScalarIndex` | classe | `(nx, ny)` |  |
| `assemble_oseen` | função | `(g, nu, inlet=None, adv=None, force=(0.0, 0.0)) -> tuple[sp.csr_matrix, np.ndarray]` | The Oseen (Stokes when ``adv`` is None) system ``A x = b``; ``inlet``: u on the inlet faces (ny,); ``adv``: advecting face velocities ``(au (nx+1, ny), av (nx, ny+1))``; ``force``: body force per unit mass. |
| `split` | função | `(g, x) -> tuple[np.ndarray, np.ndarray, np.ndarray]` |  |
| `solve` | função | `(A, b) -> np.ndarray` |  |
| `picard` | função | `(g, nu, inlet=None, tol=1e-08, maxiter=200, relax=1.0, linear_solver=None, force=(0.0, 0.…)` | Navier-Stokes by Picard (Oseen) iteration from Stokes: ``linear_solver(A, b) -> x`` (default: direct). Returns the solution, the iteration history of the relative velocity change and the per-iteration systems' solve times. |
| `assemble_scalar` | função | `(g, u, v, k, rhoc, q, T_in=0.0, scheme='hybrid') -> tuple[sp.csr_matrix, np.ndarray]` | ``rho c (u . grad T) - div(k grad T) = q`` on every cell; ``k``, ``rhoc``, ``q`` are (nx, ny) cell fields (``rhoc`` zero in solids: no advection there; face velocities are zero on solid faces anyway). ``scheme``: "hybrid" or "upwind". |
| `scalar_coords` | função | `(g) -> np.ndarray` |  |
| `pin_array` | função | `(n_pins, cells, pin, h, inlet_cells=0, outlet_cells=0) -> np.ndarray` | Solid mask of an array of square pins: ``cells`` per unit cell side, a ``pin x pin`` pin centred in each, with ``inlet_cells`` / ``outlet_cells`` empty columns before and after. |

#### `pinneapple_condense.fv`

Cell-centred finite volumes for steady and transient conduction on a voxel box, and the reference solver.

<small>`pinneapple_condense/fv.py` · 174 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `System` | classe | `(K, shape, ports, g_top, g_bottom, Gz, C)` | An assembled conduction operator: ``K`` (W/K), the cell count, the port blocks, the Robin conductances of the top and bottom cell rows (``(nx, ny)``) and the vertical face conductances ``Gz`` (``(nx, ny, nz - 1)``). |
| `SolveInfo` | classe | `(method, seconds, iterations, residual)` |  |
| `assemble` | função | `(k, dz, R, dx, h_top, h_bottom, *, ports=False, rho_c=None, dy=None) -> System` | Assemble the conduction operator of a voxel box (see the module docstring). ``k``: ``(nx, ny, nz)`` array or a ``(kx, ky, kz)`` tuple; ``R``: contact resistance of each horizontal face, ``(nz - 1,)`` or ``(nx, ny, nz - 1)``; ``rho_c``: volumetric heat capacity, returned as the diagonal ``C`` (J/K) … |
| `assemble_package` | função | `(pkg, *, transient=False) -> System` | The whole package (:class:`.package.Package`) as one box, adiabatic on the sides: the reference model. |
| `source_vector` | função | `(pkg, power=None) -> np.ndarray` | Heat source of every cell (W): the power map on the active cell row. |
| `factor` | função | `(K)` | Sparse LU of a symmetric matrix (minimum degree on ``K + K^T``, symmetric pivoting). |
| `solve` | função | `(K, f, *, method='auto', rtol=1e-11, maxiter=4000) -> tuple[np.ndarray, SolveInfo]` | Solve ``K T = f``. ``auto``: sparse direct (SuperLU, minimum-degree ordering on ``K + K^T``) up to 6e4 unknowns, else conjugate gradients with an algebraic multigrid preconditioner (``pyamg``, if installed; Jacobi otherwise) to relative residual ``rtol``. |
| `solve_package` | função | `(pkg, power=None, **kw) -> tuple[np.ndarray, SolveInfo]` | Reference steady temperature rise of every cell, shape ``(nx, ny, nz)``. |

#### `pinneapple_condense.hierarchy`

Hierarchical (nested) condensation with reuse of identical subtrees.

<small>`pinneapple_condense/hierarchy.py` · 356 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Hierarchy` | classe | `(pkg, m=6, *, modes='pod', keep_root=True)` | Nested condensation of a package (uniform ``m`` interface modes on every face). |
| `estimate` | função | `(h, sol, chunk=4096) -> np.ndarray` | Per-face error indicator of a hierarchical solution, as :meth:`.substructure.Condensed.estimate`: the fine residual on the port nodes (interiors are exact) measured in the inverse two-sided face Schur complement. |

#### `pinneapple_condense.homogenize`

Effective-conductivity (Keff) homogenisation: the baseline every condensation must beat.

<small>`pinneapple_condense/homogenize.py` · 121 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `effective_layer` | função | `(k_m, k_i, phi) -> tuple[float, float]` | (k_xy, k_z) of a layer with a volume fraction ``phi`` of aligned prisms of conductivity ``k_i``. |
| `effective_properties` | função | `(stack, variant=BASE) -> dict[str, np.ndarray]` | Per z row of one block: ``kxy``, ``kz``, ``rho_c`` (volume average), ``dz`` and contact ``R``. |
| `homogenized_system` | função | `(pkg, cells=1, transient=False)` | The homogenised package on ``cells x cells`` lateral cells per block (``n`` must be divisible by it). |
| `coarse_power` | função | `(pkg, cells=1, power=None) -> np.ndarray` |  |
| `solve_homogenized` | função | `(pkg, cells=1, power=None)` | Coarse temperature field ``(nbx * cells, nby * cells, nz)`` and the solve info. |
| `homogenized_observables` | função | `(pkg, T, sysm, cells=1) -> dict[str, np.ndarray]` | Per block: mean junction temperature ``Tj_mean``, the die/TIM heat ``q_tim``, the heat down the microbump layer ``q_ubump_layer`` and the share a bump would take by the parallel rule (``q_ubump`` estimate). |
| `compare` | função | `(pkg, ref, hom) -> dict[str, float]` | What homogenisation erases, against fine-scale reference observables (:func:`.substructure. reference_observables`): the peak junction temperature, block mean and block peak junction temperatures, the intra-block junction spread the coarse model cannot represent, and the interface heats. |

#### `pinneapple_condense.learned`

Condensation with a learned block operator, error indicators and an error-driven fallback to exact condensation.

<small>`pinneapple_condense/learned.py` · 310 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `LearnedBlocks` | classe | `(stack, m, bounds, to_variant, n_train, regressor, seed, timings)` | A learned map ``theta -> reduced block operator``. ``fit()`` condenses ``n_train`` Latin-hypercube samples of ``bounds`` exactly and trains the regression. |
| `AdaptiveResult` | classe | `(obs, tb, exact_blocks, checked, failed, rounds, timings, flagged_std, flagged_ood, estim…)` |  |
| `default_variant` | função | `(theta) -> Variant` |  |
| `reduced_block` | função | `(stack, variant, Phi) -> dict[str, np.ndarray]` | Exact reduced operator of one block on the interface modes ``Phi`` (no cache: one factorisation). |
| `assemble_blocks` | função | `(Sr, dofs, n) -> sp.csc_matrix` | Global interface matrix from per-block reduced operators ``(N, d, d)`` and their dofs ``(N, d)``. |
| `solve_blocks` | função | `(ops, pb, dofs, n)` | Assemble, factorise, solve; return the per-block reduced port values ``(N, d)`` and the timings. |
| `block_observables` | função | `(ops, tb, pb, n2) -> dict[str, np.ndarray]` |  |
| `solve_adaptive` | função | `(learned, theta, power, *, tol_std=0.05, tol_T=0.01, n_hot=8, n_audit=16, max_rounds=5, s…)` | Learned condensation with the verification loop of the module docstring. ``theta``: ``(nbx, nby, p)`` block parameters; ``power``: ``(nbx * n, nby * n)`` W per active cell. |
| `solve_exact` | função | `(learned, theta, power) -> tuple[dict, float]` | Every block condensed exactly (one factorisation per block): the reference of the learned path. |
| `variant_field` | função | `(shape, seed=0, *, defects=0, defect_k=0.05, bounds=None) -> np.ndarray` | Smooth random block parameters inside ``bounds`` (a degraded-TIM gradient, scattered bump voids, TSV quality) and ``defects`` nearly open microbumps (``ubump_k = defect_k``, outside the default training box). |

#### `pinneapple_condense.meshless`

Meshless solution of the chip-package benchmark (track 1b): points straight from the implicit geometry, no volume mesh.

<small>`pinneapple_condense/meshless.py` · 637 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Cloud` | classe | `(X, region, kind, normal, partner, area, k, R, seconds, info)` |  |
| `Geometry` | classe | `(pkg)` | Implicit description of a package: what material is at any point, and its surfaces. |
| `MeshlessSolution` | classe | `(cloud, T, A, b, timings, stencils)` |  |
| `point_cloud` | função | `(pkg, *, h_xy=None, per_layer=3, h_max=2.5e-05, per_feature=4, band_h=2.0, gap=0.3, jitte…)` | Nodes from the implicit geometry (see the module docstring); ``h_xy``: spacing cap (default: the pitch / 6); ``per_layer``: nodes across each layer; ``per_feature``: nodes across each inclusion. ``density``: an optional callable ``X -> factor in (0, 1]`` that thins the interior lattice (adaptive … |
| `region_stencils` | função | `(X, region, Q, q_region, n, box, reach, normal=None)` | Stencils of the ``n`` nearest nodes of the same region for every query point. With ``box = (W0, W1)`` the package sides are mirror planes (adiabatic): nodes within ``reach`` of a side (or two) are reflected and a reflected copy stands for its original node. Returns ``(Xe, idx_e, col)``: the … |
| `assemble` | função | `(cloud, pkg, *, method='rbf_fd', degree=2, n=None, interface='pairs', oversample=False)` | The collocation system of the module docstring. Returns ``(A, b, stencils)``. |
| `solve` | função | `(pkg, cloud=None, *, method='rbf_fd', degree=2, n=None, interface='pairs', solver='auto',…)` | Point cloud (unless given), assembly, solve. ``solver``: sparse LU, or GMRES with an ILU preconditioner above 2e5 nodes. ``condition``: also a 1-norm condition-number estimate (an extra factorisation). |
| `evaluate` | função | `(sol, pkg, P, *, degree=None) -> np.ndarray` | Temperature at points ``P`` (interpolated with stencils of the region each point lies in). |
| `observables` | função | `(sol, pkg) -> dict` | Junction temperatures at the centres of the reference's active cells (``Tj`` as in :func:`.substructure.reference_observables`), the die/TIM heat, the outflows and the energy balance. |

#### `pinneapple_condense.meshless2d`

2-D meshless (RBF-FD / GFDM) discretisations of the CFD falsification cases (``docs/research/ condensation_falsification.md``, section C), on point clouds built from the implicit geometry of a square-pin array.

<small>`pinneapple_condense/meshless2d.py` · 339 linhas · testes: 6</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `pin_sdf` | função | `(X, pins, pin_frac, offset=(1.0, 0.0))` | Signed distance to the nearest square pin (negative inside); unit cells of side 1 from ``offset``. |
| `cloud` | função | `(pins, h, pin_frac=0.5, jitter=0.25, inlet_blocks=1, outlet_blocks=1, seed=0, fluid_only=…)` | Points over ``(inlet + pins_x + outlet) x pins_y`` unit cells: one jittered lattice of spacing ``h`` per unit cell, the same in every cell (shifted by integers). ``fluid_only`` drops the points inside (and within 0.3 h of) the pins; ``wall_points`` adds points on the pin walls (spacing ``h``). … |
| `bilinear_mac` | função | `(g, u, v, X)` | Velocity of a MAC solution (grid ``g`` of :mod:`.flow`, lengths in unit cells) at points ``X``. |
| `snap_to_box` | função | `(X, box, h)` | Points within h / 2 of the box moved onto it (boundary nodes: the conditions are collocated where they hold). |
| `knn` | função | `(X, Q, n, tree=None)` | The ``n`` nearest nodes of each query point, ties broken by the offset (not by the k-d tree's traversal order): a block replicated by an integer shift then gets the same stencils, so the same matrix. |
| `advection_diffusion` | função | `(X, box, h, vel, alpha, q, *, method='rbf_fd', n=15, degree=2)` | Collocated ``vel . grad T - alpha lap T = q``. Points within 3 h / 8 of the box carry the boundary condition: inlet ``T = 0``; outlet, top and bottom zero normal gradient. |
| `stokes` | função | `(X, wall, box, h, nu=1.0, *, method='rbf_fd', n=20, degree=2)` | Collocated Stokes (u, v, p at every fluid point). Rows: momentum and continuity at interior points; no-slip on pin-wall points (continuity replaced by the normal momentum balance there is not used: pressure at wall points is left to the continuity equation); inlet u = 1, v = 0; outlet p = 0, du/dx … |
| `hybrid_conduction` | função | `(pins, cells, pin_frac, k_pin, k_mat, R_c, *, band=2, point_h=None, jitter=0.25, n=12, de…)` | Conduction ``-div(k grad T) = q`` (source ``q_pin`` in the pins) in a pins_x x pins_y array of unit cells, adiabatic sides, cooled on top (Robin ``h_top``, T_inf = 0), with a contact resistance ``R_c`` on every pin wall. FV cells (``cells`` per unit cell side) within ``band`` cells of a pin wall; … |
| `fv_conduction` | função | `(pins, cells, pin_frac, k_pin, k_mat, R_c, q_pin=1.0, h_top=1.0)` | The same conduction problem on finite volumes everywhere (the reference and the equal-unknowns baseline). |

#### `pinneapple_condense.package`

A synthetic multilayer chip package built from one repeated unit cell (a *block*).

<small>`pinneapple_condense/package.py` · 253 linhas · testes: 27</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Layer` | classe | `(name, thickness, nz, matrix, inclusion, side)` | One layer of the stack. ``inclusion``: material of the square inclusion centred in each block (bump, TSV), ``side``: its side as a fraction of the block pitch. |
| `Variant` | classe | `(inclusion_k, removed, contact)` | What makes a block differ from the base block. ``inclusion_k``: ((layer, factor), ...) multiplies the inclusion conductivity of a layer (a void fraction lowers it); ``removed``: layers whose inclusion is missing (replaced by the matrix); ``contact``: ((lower, upper, factor), ...) multiplies a … |
| `Stack` | classe | `(layers, pitch, n, contact, h_top, h_bottom, active, materials, anisotropy)` | The layers (bottom to top), the block pitch and lateral cells per block, contact resistances between layers (m^2 K / W), Robin coefficients on the top of the lid and the bottom of the substrate (W/m^2/K), and the layer whose bottom cell row is the active (heat-generating, junction) surface. |
| `Package` | classe | `(stack, layout, variants, power)` | A lateral ``(nbx, nby)`` array of blocks over one stack. ``layout[I, J]`` indexes ``variants``; ``power`` is the heat (W) of every active cell, shape ``(nbx * n, nby * n)``. |
| `reference_stack` | função | `(n=8, vertical=1) -> Stack` | The benchmark stack: 40 um block pitch (one microbump, one TSV and one C4 per block; a simplification: C4 pitch is usually larger than microbump pitch), ~1 mm total height, 8 x 8 cells per block side by default. |
| `hotspot_power` | função | `(stack, shape, total, *, hotspots=3, background=0.4, width=0.08, seed=0, centres=None) ->…)` | A non-uniform power map on the active cells (W per cell): ``background`` of ``total`` spread uniformly, the rest in Gaussian hotspots (``width`` as a fraction of the die side) at random or given ``centres``. |
| `uniform_package` | função | `(shape, stack=None, *, total_power=None, power_density=1000000.0, seed=0, hotspots=3) -> …)` | A package of identical blocks with a hotspot power map. ``power_density``: mean heat flux over the die (W/m^2; 1e6 W/m^2 = 100 W/cm^2) unless ``total_power`` is given. Change ``layout`` / ``variants`` afterwards for heterogeneous blocks. |

#### `pinneapple_condense.query`

Physics queries on a chip package (track 2): the methods behind :class:`pinneapple_query.Engine` for the package of :mod:`.package`.

<small>`pinneapple_condense/query.py` · 411 linhas · testes: 7</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `State` | classe | `(power, obs, method, uncertainty, seconds, solver=None, sol=None, field=None, obs_unc=Non…)` | A solved configuration: per-block observables, its source, its own uncertainty (K, the largest over the junction), the uncertainty of each observable (``obs_unc``: same keys as ``obs``, absolute; ``None`` for an exact state), and what is needed to read interior points (the hierarchy and its … |
| `PackageModel` | classe | `(pkg, m=8, reference_cells=2000000)` | Shared machinery of the methods: the package, the stored states, the hierarchies, extraction of a quantity from a state. |
| `Cached` | classe | `(model)` |  |
| `Bound` | classe | `(model)` | Rigorous interval from the M-matrix monotonicity (see the module docstring). |
| `Superposition` | classe | `(model, max_states=64)` | Linear combination of solved states (least squares on the power maps); the residual map bounded by monotonicity, plus the combined uncertainty of the states. |
| `Condensed` | classe | `(model, expected_s=1.0)` | A new solve with the memoised hierarchy at ``m``; uncertainty \|T_m - T_2m\| (the 2m value is returned). |
| `Reference` | classe | `(model)` | The fine finite-volume solve: the last resort. |
| `package_engine` | função | `(pkg, m=12, reference_cells=2000000, calibrate=True) -> tuple[Engine, PackageModel]` | A query engine over a package with the methods of this module. ``calibrate``: solve the package's own power map once with the condensed method (part of the build cost), which gives the condensed method its uncertainty prior and the store its first state. |

#### `pinneapple_condense.query_benchmark`

Measured demonstration of the physics query engine on the chip package (track 2).

<small>`pinneapple_condense/query_benchmark.py` · 125 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `query_demo` | função | `(nb=48, m=12, n_points=2000, n_interior=100, seed=0, reference=True) -> dict` |  |

#### `pinneapple_condense.substructure`

Static condensation of repeated blocks, reduced interfaces, on-demand recovery and an error-driven refinement.

<small>`pinneapple_condense/substructure.py` · 661 linhas · testes: 21</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `BlockModel` | classe | `(stack, variant, S, W, Y, Z, ports, Om, obs_names, seconds, O_p, O_t)` | The condensed form of one distinct block (see the module docstring for the symbols). |
| `CondensedState` | classe | `(t, pb, seconds)` | A solved condensed configuration: the interface dofs and the block powers it was solved for. |
| `Condensed` | classe | `(pkg, m=6, *, refined_x=None, refined_y=None, modes='pod', shift=0.0, face_m=None)` | The condensed package: block operators reused across identical blocks, ``m`` interface modes per face (``refined_x`` / ``refined_y``: faces kept exact; ``face_m``: modes of every face, x faces first), a global sparse interface system factorised once. |
| `block_functionals` | função | `(stack, variant, sysb) -> tuple[sp.csr_matrix, list[str]]` | Rows of the linear observables of one block over its cells: the ``n * n`` junction temperatures, then the vertical heat (W, positive upwards) through the die/TIM interface, the microbump, the TSV and the C4 bump (each at the middle face of its layer, over the inclusion cells), and the Robin outflow … |
| `block_model` | função | `(stack, variant=BASE, *, cache=True, shift=0.0) -> BlockModel` | Condense one block (cached per ``(stack, variant, shift)``). ``shift`` ``s`` condenses ``K + s C`` instead of ``K`` (the Laplace-domain operator at rate ``s`` [1/s]; one implicit-Euler step for ``s = 1/dt``). |
| `clear_cache` | função | `() -> None` |  |
| `face_weights` | função | `(stack) -> np.ndarray` | Area (m^2) of each port node of a face, order ``(lateral, k)``. |
| `trace_snapshots` | função | `(stack, patch=4, shifts=(0.0,), *, global_size=16, global_maps=24) -> dict[str, np.ndarra…)` | Interface traces the face modes are built from (cached per stack), two groups of equal weight: |
| `layered_modes` | função | `(stack, orient, m, shifts=(0.0,), *, groups='layer') -> np.ndarray` | ``m`` face modes that never tie two layers together: each layer gets its own constant, and the remaining ``m - n_layers`` modes are the leading per-layer POD modes of the traces (:func:`trace_snapshots`), allotted greedily by singular value across layers. A mode spanning several layers would fix … |
| `interface_modes` | função | `(bm, orient, m, method='pod', shifts=(0.0,)) -> np.ndarray` | ``m`` modes of a shared face (``orient`` ``"x"`` or ``"y"``), columns orthonormal in the area-weighted inner product, the first one constant. ``method="pod"`` (default): the rest are the leading POD modes of the traces of :func:`trace_snapshots` (load-informed Ritz-type modes); ``"cc"``: the … |
| `reference_observables` | função | `(pkg, T) -> dict[str, np.ndarray]` | The observables of :meth:`Condensed.observables` from a full cell field ``(nx, ny, nz)`` (same rows). |
| `energy_norm` | função | `(K, e) -> float` |  |
| `errors` | função | `(obs, ref, total_power) -> dict[str, float]` | Error of observables against a reference: Tmax and junction temperatures (K), interface heats relative to the reference total of that interface, and the energy balance (outflow - power, relative). |
| `tmax_cells` | função | `(T, active_k) -> tuple[float, float]` | (max over every cell, max over the active row): equal when all heat is generated on the active row with non-negative power (discrete maximum principle: every other cell is a weighted mean of its neighbours). |
| `iterate_refinement` | função | `(pkg, m, tol, *, fraction=0.5, max_steps=6, power=None)` | Solve, estimate, refine until ``sqrt(sum eta^2) <= tol * sqrt(t^T A t)`` (relative energy estimate). |

#### `pinneapple_condense.transient`

Transient conduction: Craig-Bampton condensation of the blocks, implicit Euler, and the fine reference.

<small>`pinneapple_condense/transient.py` · 199 linhas · testes: 14</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `CBBlock` | classe | `(K, M, W, G, Ot, Op, lam)` |  |
| `CBTransient` | classe | `(pkg, m=8, k=8, shifts=TRANSIENT_SHIFTS)` | Craig-Bampton model of a package (``m`` interface modes per face, ``k`` interior modes per block). |
| `cb_block` | função | `(pkg_stack, variant, Phi_f, k) -> CBBlock` | Craig-Bampton reduction of one block onto ``Phi_f`` interface modes and ``k`` fixed-interface modes. |
| `reference_transient` | função | `(pkg, powers, dt, every=1) -> dict` | Fine-grid implicit Euler with the same step (``C / dt + K`` factorised once, or AMG-CG for big grids). |

### Exemplos de uso reais do repositório

_Nenhum exemplo encontrado no repositório para este pacote._

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 15 de 18.
- Funções de teste em arquivos que importam algum módulo do pacote: 27.

### Limitações conhecidas

- _Nenhuma registrada._

## 6. `pinneapple_query`

### Objetivo

`pinneapple_query` é o motor genérico de consultas físicas (Physics Query Engine). A ideia: pedir uma grandeza física (por exemplo `T`) num lugar físico estável (por exemplo `die@0.3,0.7`), sob certas condições (por exemplo um mapa de potência), com uma precisão-alvo. A resposta traz o valor, a unidade, a incerteza, o domínio de validade, a proveniência e o método que a produziu, e diz se o valor foi *lido* de um estado já resolvido (`read`), *previsto* para condições não resolvidas (`predicted`) ou obtido por uma nova solução (`solved`).

O problema que resolve é escolher, a cada pergunta, o caminho mais barato que ainda cumpre a precisão pedida. Um `Planner` pergunta a cada `Method` registrado por uma `Offer` (se pode responder, a que custo esperado, com que limite de erro, dentro de que validade) e escolhe a oferta válida mais barata cujo limite atende o alvo; se nenhuma atende, fica com a de menor limite de erro; um método de último recurso (`fallback`, tipicamente uma solução completa) só entra quando ninguém oferece. O `Engine` guarda um log de cada resposta (método, tipo, latência) para auditoria e aceita ganchos (`hooks`).

O pacote é pequeno e só tem a parte genérica: `core` (consulta, ofertas, planejador, respostas) e `refs` (referências físicas independentes de malha, como `ubump[12,40]`, `interface:die/tim` ou `lid:top`). Os métodos de uma física concreta ficam junto dela: para o encapsulamento de chips, `pinneapple_condense.query.package_engine` registra os métodos `cached`, `superposition`, `bound`, `condensed` e `reference`. No resto da plataforma, o nó `query.package` de `pinneapple_flow` expõe isso como um passo de fluxo, e `examples/physics_query/run_demo.py` é a demonstração medida (resultado gravado em `examples/physics_query/results/query_demo.json`). É uma linha de pesquisa ("track 2" no CHANGELOG), não uma API estável de `pp.*`.

### Números do inventário

| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | módulos sem teste direto |
|---:|---:|---:|---:|---:|---:|
| 3 | 252 | 7 | 2 | 7 (1 arquivos) | 2 |

### Submódulos

| submódulo | o que faz | módulos | classes + funções | testes* |
|---|---|---:|---:|---:|
| `core` | Consulta (`Query`), oferta (`Offer`), resposta (`Answer`), interface de método (`Method`), planejador (`Planner`) e motor (`Engine`). | 1 | 7 | 0 |
| `refs` | Referências físicas estáveis (`PhysicalRef`) e o parser das suas formas em texto (`parse_ref`). | 1 | 2 | 0 |

\* maior número de testes que importam um módulo do submódulo.

### Módulos e API pública

#### `pinneapple_query`

Fachada do pacote: reexporta `Engine`, `Planner`, `Method`, `Offer`, `Answer`, `Query`, `condition_key`, `PhysicalRef` e `parse_ref`; o docstring mostra o uso com `pinneapple_condense.query.package_engine`.

<small>`pinneapple_query/__init__.py` · 17 linhas · testes: 7</small>

#### `pinneapple_query.core`

Motor de consultas: o `Planner` coleta ofertas dos métodos e escolhe a mais barata que cumpre a precisão; o `Engine` executa, mede o tempo, registra as ofertas e grava um log de auditoria. `Method` é só interface (levanta NotImplementedError); [sem teste direto] — coberto por `tests/test_query.py` via `pinneapple_condense.query`.

<small>`pinneapple_query/core.py` · 155 linhas · **sem teste direto** · palavras-chave no código (heurística; o texto acima diz o status real): levanta `NotImplementedError` em algum caminho</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `Query` | classe | `(quantity, where, time, conditions, accuracy, rigorous)` | Uma consulta: grandeza, referência física (`where`), tempo, condições, precisão absoluta desejada e se só valem métodos com limite rigoroso; `key` é o hash das condições. |
| `Offer` | classe | `(method, cost_s, bound, valid, rigorous, why)` | Oferta de um método para uma consulta: custo esperado em segundos, limite de erro esperado, se é válida, se o limite é rigoroso e o porquê. |
| `Answer` | classe | `(value, unit, uncertainty, kind, method, validity, provenance, seconds, offers, rigorous,…)` | Resposta: valor, unidade, incerteza, tipo (`read`, `predicted` ou `solved`), método, validade, proveniência, tempo gasto, ofertas consideradas e, se rigorosa, o intervalo. |
| `Method` | classe | `()` | Interface de um modo de responder: `offer` decide barato se pode responder e `answer` calcula; ambos levantam NotImplementedError na classe base. |
| `Planner` | classe | `(methods, fallback=None)` | Escolhe o método: a oferta válida mais barata que atende a precisão; senão a de menor limite de erro; o `fallback` só quando nenhum método oferece (sem fallback, levanta LookupError). |
| `Engine` | classe | `(planner, units=None, default_accuracy=None)` | Ponto de entrada `query(quantity, where, ...)`: aplica a precisão padrão por grandeza, consulta o `Planner`, mede a latência, anexa as ofertas, chama os ganchos e grava o log de auditoria. |
| `condition_key` | função | `(conditions) -> str` | Hash SHA-256 estável (16 caracteres) das condições de uma consulta; arrays entram pelos seus bytes. |

#### `pinneapple_query.refs`

Referências físicas independentes de malha (componente + coordenada fracionária local, índice de bloco, faixa de blocos, interface ou face) e o parser das formas de texto. [sem teste direto] — `tests/test_query.py` testa `parse_ref` pela fachada do pacote.

<small>`pinneapple_query/refs.py` · 80 linhas · **sem teste direto**</small>

| nome | tipo | assinatura | o que faz |
|---|---|---|---|
| `PhysicalRef` | classe | `(component, local, index, span, kind)` | Referência física imutável: componente, coordenadas locais fracionárias, índice de feature repetida, faixa de blocos e tipo (`point`, `feature`, `region`, `interface` ou `face:...`); `str()` volta à forma de texto. |
| `parse_ref` | função | `(where) -> PhysicalRef` | Converte textos como `die@0.31,0.72`, `ubump[12,40]`, `interface:die/tim[3:9,0:4]`, `region:die[...]` ou `lid:top` em `PhysicalRef`; rejeita coordenadas locais fora de [0, 1]. |

### Exemplos de uso reais do repositório

**Referências físicas e seus tipos** — `tests/test_query.py`

```python
def test_physical_references():
    r = parse_ref("die@0.25,0.5")
    assert (r.component, r.local, r.kind) == ("die", (0.25, 0.5), "point")
    assert parse_ref("ubump[3,4]").index == (3, 4) and parse_ref("ubump[3,4]").kind == "feature"
    i = parse_ref("interface:die/tim[0:2,1:3]")
    assert i.kind == "interface" and i.span == ((0, 2), (1, 3)) and i.component == "die/tim"
    assert parse_ref("lid:top").kind == "face:top"
    assert str(parse_ref("die@0.25,0.5,1")) == "die@0.25,0.5,1"
    with pytest.raises(ValueError):
        parse_ref("die@1.5,0.2")                                   # local coordinates are fractions
```

**Resolver uma vez, depois ler do estado resolvido** — `tests/test_query.py`

```python
def test_solved_then_read_with_physical_points(engine):
    pkg, eng, model = engine
    a = eng.query("Tj_max", "region:die", accuracy=1e-9)          # nothing solved: the reference solves
    assert a.kind == "solved" and a.method == "reference"
    b = eng.query("T", "die@0.4,0.6")
    c = eng.query("T", "die@0.4,0.6,0.5")
    assert b.kind == c.kind == "read" and b.method == "cached"
```

### Testes e status de validação

- Módulos com pelo menos um teste que os importa: 1 de 3.
- Funções de teste em arquivos que importam algum módulo do pacote: 7.

### Limitações conhecidas

- O pacote só traz a parte genérica: sem métodos concretos ele não responde nada. Hoje os únicos métodos estão em `pinneapple_condense.query` (encapsulamento de chips).
- `Method.offer` e `Method.answer` levantam NotImplementedError: cada física precisa implementar sua própria subclasse.
- Os módulos `core` e `refs` não têm teste que os importe diretamente; a cobertura vem de `tests/test_query.py`, que passa pelos métodos de `pinneapple_condense`.
- As formas de referência (`die@...`, `ubump[...]`, `tsv[...]`, `c4[...]`, `lid:top`) foram pensadas para encapsulamento de chips; outra geometria precisa definir seus próprios nomes de componente.
- A escolha do `Planner` confia nos custos e limites de erro que cada método declara na oferta; o motor não verifica se esses limites são verdadeiros.
- Sem precisão-alvo (nem na consulta nem em `default_accuracy`), o planejador escolhe a oferta de menor limite de erro, que pode ser a mais cara.
- É linha de pesquisa ("track 2" no CHANGELOG), sem entrada em `pp.*`.

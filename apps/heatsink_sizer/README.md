# HeatSink Sizer

**Plate-fin heat sinks, sized and verified by physics — in seconds.**

> Not the same tool as [HeatSink Design AI](../heatsink_inverse_ai/README.md): this app sizes plate-fin heat sinks from published forced/natural-convection correlations, pressure drop and a 3-D base-plate solve; that one demonstrates gradient-based inverse design through a neural surrogate of the 1-D fin-array formula with a fixed convection coefficient.

Day 1 of the PINNeAPPle 30-app program. A working SaaS prototype for thermal/mechanical
engineers at electronics, power-electronics, LED and industrial-drive companies who today
size heat sinks with spreadsheets, vendor calculators or a CFD run per candidate.

| You give it | You get |
|---|---|
| Heat load, temperature limit, ambient, heat-source footprint, fan speed (or natural convection), space envelope, process and materials | The lightest (or smallest, or coolest) **manufacturable** designs that meet the limit, each **verified by physics** with an uncertainty band, pressure drop, mass, a base temperature map and a physics-check report you can print |

## Run it

```bash
# from the repository root
pip install -e . fastapi "uvicorn[standard]" sympy
cd apps/heatsink_sizer
uvicorn heatsink_sizer.api:app --port 8080        # open http://localhost:8080
```

Docker (from the repository root):

```bash
docker build -f apps/heatsink_sizer/Dockerfile -t heatsink-sizer .
docker run -p 8080:8080 heatsink-sizer
```

### Put it on a server (HTTPS + login)

On any Linux server with Docker (2 vCPU / 2 GB RAM is enough; each worker holds
~600 MB), with ports 80/443 open and a DNS record for your domain:

```bash
cd apps/heatsink_sizer/deploy
cp .env.example .env          # set DOMAIN, HSS_USER, HSS_PASSWORD
docker compose up -d --build  # Caddy obtains the HTTPS certificate automatically
```

- `HSS_USER` / `HSS_PASSWORD`: HTTP Basic login for everything except `/health`
  (leave empty for open access).
- `HSS_MAX_SIZING`: concurrent sizing runs per worker (default 2); extra requests
  get HTTP 429 instead of piling up. Measured on 4 CPU cores: 8 simultaneous
  sizings finish in ≤ 13 s, 16 simultaneous evaluations in ≤ 2.7 s.

The trained surrogates ship in `artifacts/`. To retrain (≈4 min on 4 CPU cores:
11 000 physics solves + training):

```bash
python -m heatsink_sizer.surrogate                  # regenerate dataset + retrain
python -m heatsink_sizer.surrogate --reuse-dataset  # retrain on artifacts/dataset_*.json.gz
```

## How it works

```
requirements ──► sample 40 000 manufacturable candidates (envelope + process rules)
             ──► screen with the PINNeAPPle neural surrogate (conformal 95% upper bound,
                 pessimistic convection)                               ~0.1 s
             ──► round the best 24 to manufacturing precision
             ──► re-verify EACH with the physics engine                ~0.1 s / design
             ──► recommend only verified designs
```

**Physics engine** (`pinneapple_physics/closed_form/plate_fin_heatsink.py` +
`heatsink_sizer/field.py`), all published models:

| Piece | Model |
|---|---|
| Forced convection (ducted) | Teertstra, Yovanovich & Culham (2000) composite parallel-plate model |
| Natural convection (vertical fins) | Bar-Cohen & Rohsenow (1984), iterated on the fin temperature |
| Fin efficiency | Incropera & DeWitt, adiabatic-tip corrected length (`fin_array_conduction`) |
| Pressure drop | Muzychka & Yovanovich apparent friction + Shah & London f·Re + Kays & London losses |
| Base conduction / hotspot | 3D finite-volume solve of the real rectangular base and source (any position) |
| Cross-check | Lee et al. (1995) spreading resistance (independent closed form) |
| Air properties | Incropera Table A.4, film temperature |

**Surrogates** (`heatsink_sizer/surrogate.py`): one PINNeAPPle `ModelRegistry`
MLP per cooling mode, trained with the PINNeAPPle `Trainer` (best-validation
checkpoint) and an L-BFGS polish on physics-solved designs sampled with
`pinneapple_data.parameter_sampling` (Latin hypercube). They are used only to
*screen*; nothing a customer sees comes from them unverified.

## Validation (measured, not claimed)

| Check | Result |
|---|---|
| FVM energy balance (heat out / heat in − 1) | ~1e-13 |
| FVM vs Lee et al., square bases (aspect ≤ 2) | 1–5 % apart |
| FVM vs Lee et al., elongated bases (e.g. 200×30 mm) | Lee under-predicts up to 2× → FVM used, flagged in the report |
| Forced fin resistance vs. an independent method (Stephan developing-flow Nu + ε-NTU air heating), 0.5–6 m/s | 0–5 % apart |
| Forced pressure drop vs. fully developed friction + entry/exit losses | 2–4 % apart |
| Natural h vs. Elenbaas (1942) | 2 % apart |
| Natural optimum fin gap vs. Bar-Cohen & Rohsenow S_opt | 6.9 vs 6.5 mm |
| FVM grid convergence (coarse vs fine grid) | < 2 % (typically 0.1–0.5 %) |
| Spreading formula, source = base | reduces exactly to 1D conduction t/(kA) (test) |
| Teertstra limits | fully developed Nu = Re*·Pr/2 and developing-plate limit recovered (tests) |
| Natural convection | optimum fin count exists (too few: little area; too many: choked flow) (test) |
| **Forced surrogate** vs physics, 900 held-out designs | **mean 0.71 %, p95 2.1 %**; ΔP mean 1.0 % · 95 % bound covers 94.7 % · all physics laws pass → *trustworthy* |
| **Natural surrogate** vs physics, 750 held-out designs | **mean 0.72 %, p95 2.1 %** · 95 % bound covers 94.1 % · conductivity effect has the right sign in 98.9 % of designs beyond noise (tiny effect in still air) → *use with caution* |
| Screen vs physics on the designs actually recommended | 0.2–0.4 °C mean |

Every run of the app shows the same kind of evidence per design (energy balance, method
agreement, grid convergence, correlation validity) and per surrogate (error, convergence,
generalization, sensitivities, weights, uncertainty coverage, physics laws) in the
**Model trust** tab — see `artifacts/surrogate_*_report.json`.

## API

| Method | Path | Body |
|---|---|---|
| `POST` | `/api/size` | `{"power_w":100,"t_limit_c":70,"air_velocity_m_s":2,"source_width_mm":30,"source_depth_mm":30,"max_width_mm":100,"max_depth_mm":100,"max_height_mm":50,"materials":["al6063"],"process":"extruded","objective":"mass"}` |
| `POST` | `/api/evaluate` | `{"design":{"base_width_mm":80,"base_depth_mm":80,"base_thickness_mm":6,"n_fins":30,"fin_thickness_mm":1.2,"fin_height_mm":35,"material":"al6063"},"operating":{"power_w":100,"air_velocity_m_s":2,"t_limit_c":70}}` |
| `GET` | `/api/surrogate/{forced\|natural}/report` | – |
| `GET` | `/api/meta`, `/health` | – |

OpenAPI docs at `/docs`.

## Limitations (shown to the user, not hidden)

Every report ends with a **Model scope & validation** panel (collapsed by default, and included when printed). It lists the independent checks the model passed, then each assumption that applies to *this* case with four things: what the assumption is, which way it errs (conservative, can read low, or to confirm), what to do about it today, and the roadmap item that removes it. The text comes from `engine.model_scope()`.


- **Forced flow is assumed ducted** (shroud). Without a shroud, bypass flow can cut
  performance by 20–50 %; a bypass model is on the roadmap.
- **Laminar channel flow**; beyond Re ≈ 2300 the result is conservative.
- **Radiation is neglected**: conservative in natural convection (real parts run
  ~10–25 % cooler, more if anodised).
- **Natural convection assumes vertical fins**.
- The ±15 % band on the convection coefficient is a typical correlation accuracy — an
  engineering assumption, not a measurement of this design. Confirm critical designs by test.
- Plate fins only (no pin fins, heat pipes or vapour chambers yet).

## Roadmap to a paid product

1. Accounts, saved projects and PDF export with company branding.
2. Fan-curve matching (operating point from the fan's P–Q curve instead of a fixed speed).
3. Bypass model for unshrouded heat sinks; radiation with emissivity.
4. Vendor extrusion catalogue matching (closest off-the-shelf profile + cut length).
5. Multiple heat sources and STEP export of the recommended design.

## Author

The "About the author" button and the footer credit read `apps/_shared/static/shared/author.json` (name, role, bio, email, LinkedIn URL), shared by every app. Edit that one file to change them; the LinkedIn button appears once `linkedin` is set.

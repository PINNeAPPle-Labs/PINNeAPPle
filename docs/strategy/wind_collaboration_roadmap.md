# Wind energy with PINNeAPPle: what this POC shows and what a collaboration would build

**For:** Sridhar I. · **From:** Yan Barros · **Date:** 2026-11-27
**Status of the work behind it:** reproducible, runnable today, published in the
repository. Not a validated product, and deliberately explicit about the gaps.

---

## 1. What was built

A working vertical for wind energy on **public, licensed data**: the Penmanshiel
wind farm (14 × Senvion MM82, 28.7 MW, CC-BY-4.0), 10-minute SCADA for 2016-2017,
1.12 million turbine-samples.

| Component | What it does |
|---|---|
| Dataset layer | Parses the vendor export, uses the operator's **recorded** availability, downtime and curtailment (not heuristics), and reproduces the vendor's own density-adjusted wind speed to machine precision |
| Power-curve models | Analytic reference, IEC 61400-12-1 binned, 5-parameter logistic, boosted trees, and a **physics-informed monotone model** whose constraints hold by construction |
| Wake models | Jensen/PARK and Gaussian (Bastankhah & Porté-Agel), RSS superposition, farm model, direction polar |
| Farm ↔ data | Farm efficiency inferred from metered power, sector by sector, compared with the wake models; the wake decay constant fitted on a validation window and scored on a later test window |
| Operational | A farm-efficiency surrogate (MLP on the analytical model) and a curtailment/wake-steering study |
| Trust layer | VeriPhysics applicability map: what was verified, what is inferred, where not to extrapolate |

## 2. What it measures

**One turbine, 4-month temporal holdout, 15,964 samples:**

| Model | RMSE [kW] | Share of rated | Physical constraints satisfied |
|---|---|---|---|
| Boosted trees (met features) | 45.3 | 2.2 % | 1 of 4 |
| IEC binned curve | 46.6 | 2.3 % | 1 of 4 |
| Logistic (5 parameters) | 48.9 | 2.4 % | 4 of 4 |
| **Physics-informed (ours)** | **54.3** | **2.6 %** | **4 of 4** |
| Operator's own potential power | 57.1 | 2.8 % | — |

The four constraints: non-negative, non-decreasing in wind speed, capped at
rated, zero outside cut-in/cut-out. The industry-standard binned curve fails
three of them; so do both boosted-tree models. That is why the constrained model
exists: it costs 0.4 % of rated accuracy and it is the one you can hand to an
optimiser or an AEP model without shaping it first.

**Wakes, 96 sector × speed cells on the test window:**

| Model | Modelled farm efficiency | Observed | Bias | RMSE |
|---|---|---|---|---|
| Gaussian, textbook constant | 0.818 | 0.936 | −11.8 pp | 0.185 |
| Gaussian, constant fitted on validation | 0.906 | 0.936 | −3.0 pp | **0.126** |
| Jensen/PARK, textbook constant | 0.830 | 0.936 | −10.7 pp | 0.169 |

Three things this establishes and one it does not:

- The textbook wake-decay constants over-predict this farm's loss by ~11
  percentage points. On a 28.7 MW farm that is roughly a 10 % error in AEP —
  worth measuring rather than importing.
- One constant fitted on a validation window transfers to a later test window and
  cuts the RMSE by a third. That is the honest test of a calibrated parameter.
- Both model families tell the same story, so it is not an artefact of one
  formulation.
- **It does not reproduce the loss's direction dependence** (correlation +0.17).
  Whatever dominates the sector-to-sector variation here — terrain, calibration,
  sector management, real per-turbine performance — is not in the model. That is
  the gap the next phase has to attack, and it is the interesting one.

## 3. What this POC deliberately does not claim

- The wake models are analytical engineering models, **not CFD**. They are wrong
  in detail and are never presented as high-fidelity.
- The SCADA-observed efficiency is an **upper bound** on wake loss, not an
  independent measurement: it absorbs curtailment, icing and real per-turbine
  performance differences.
- The curtailment study is a **model estimate** under an explicit assumption
  (a thrust-coefficient multiplier stands in for yaw or derating, and the wake is
  not deflected). At this spacing it finds no gain, and an earlier version that
  found a 16 % gain was wrong — the fix is documented in the changelog.
- No result here has been validated against a met mast, a lidar, or a
  high-fidelity simulation.

## 4. Why this is the right starting point for a collaboration

The interesting work in wind + physics AI is not "fit a power curve" — that is
solved to 2 % by a binned table. It is:

1. **Turning measured fleet behaviour into validated physical parameters** (the
   wake decay constant here moved by 3× from its textbook value).
2. **Carrying the uncertainty honestly** into an AEP or a set-point decision.
3. **Knowing where the surrogate is not allowed to speak** — the applicability
   map, not a single R².

This POC does all three on a small, reproducible case, and it names what is
missing instead of smoothing it over.

## 5. Proposed roadmap

**Phase 0 — shared footing (2 weeks).**
Pin the data scope with you (which farms, which years, which signals), agree on
the reference turbines and calibration procedure, and turn this POC into the
shared baseline everyone measures against. Deliverable: a reproducible benchmark
record with the numbers in §2 as the reference row.

**Phase 1 — calibration that is a measurement, not a fit (4-6 weeks).**
Per-turbine calibration offsets estimated from the sectors where each machine is
free-stream, which turns the farm efficiency from an upper bound into an
estimate with an error bar. This is the single change that most improves the
wake comparison, and it needs no new simulation.

**Phase 2 — a physics-informed wake surrogate (6-10 weeks).**
The analytical wake field as the physics prior, a correction network on top
trained against the calibrated SCADA efficiency, and a VeriPhysics gate that
refuses to answer outside the measured envelope. This is the PINN-shaped part of
the problem and where your interest in wake interaction and optimisation fits
directly.

**Phase 3 — wake steering with a deflection model (8-12 weeks).**
The current study cannot represent wake deflection, which is the whole point of
steering. That needs a yaw-aware wake model (the FLORIS family) and a measured
steering campaign — a field-test design, not a software task.

**Phase 4 — from farm to fleet (ongoing).**
Multi-farm transfer: does a calibrated decay constant move between sites, and
what carries across? This is where a shared dataset with several farms becomes
worth much more than any single one.

**Cross-cutting, from day one.** Reference data for at least one case (a met mast
or a lidar campaign) so that "validated" can mean something; and a joint decision
on which results may be published, because the value of this work is that it can
be shown.

## 6. What we would need from the collaboration

- **Data**: the farms and variables above, and — critically — one case with a
  reference measurement that is not a nacelle anemometer.
- **Domain review**: your judgement on which of the residual effects (terrain,
  sector management, per-turbine performance) is likely to dominate, so the next
  phase attacks the right one.
- **A test case with a known answer**: a farm where the wake loss has already
  been estimated by another method, so the surrogate can be judged against
  something other than itself.

## 7. Where the work lives

| | |
|---|---|
| Code | `pinneapple_systems/wind` |
| Guide and full results | `docs/guides/wind_power.md` |
| Runnable case | `examples/wind_farm_penmanshiel` |
| Test suite | `tests/test_wind.py` |
| Data | Zenodo, doi:10.5281/zenodo.16807304 (CC-BY-4.0) |

# Latent-space methods and cybersecurity applications: evaluation and roadmap

Status: October 2026. Two directions were evaluated against what PINNeAPPle already ships, ranked by usefulness and
fit, and the top item of each was implemented with an experiment against an independent reference.

* **A. Latent space** — latent representations used inside the pipeline (problem → geometry → data → model → trust →
  decision), not as a stand-alone model zoo.
* **B. Cybersecurity** — applications where physics gives something a purely data-driven security tool does not
  have. Detection and defence only.

Scoring: *usefulness* (how often a user of the library would need it, and what it changes in a decision) and *fit*
(how much existing code it reuses, how naturally it sits in the pipeline) are rated 1–5; *effort* is in
person-weeks for a first version with tests, an example and a guide; *risk* is the chance that the acceptance test
fails for reasons other than engineering effort (method immaturity, missing data, compute).

## What exists today

Latent and reduced-order:

* `pinneapple_neural.architectures.rom`: `POD` (randomised, incremental), `DynamicModeDecomposition`, `HAVOK`,
  `OperatorInference` (quadratic, discrete and continuous), `SINDy`, `KoopmanAutoencoder`, `NeuralROM`, `ROMHybrid`,
  `DeepUQROM`, `ParametricPOD` (RBF, GP, linear, nearest; amplitude normalisation), and now
  `TrustedLatentSurrogate`.
* `pinneapple_neural.architectures.autoencoders` (dense, 2-D convolutional, VAE, Koopman and physics-informed
  Koopman autoencoders, AE-ROM hybrid), `continuous.latent_ode`, `continuous.deep_state_space`, neural operators
  (FNO, DeepONet, MS-DeepONet, GNO), `pinns.physics_diffusion` (diffusion model with PDE-residual guidance).
* Trust and uncertainty: `pinneapple_analysis.trust` (`TrustGate`: coordinate Mahalanobis OOD, PDE residual,
  ensemble variance; `TrustReport`: APPROVED / REVIEW / REJECT), `pinneapple_analysis.uncertainty` (split conformal
  with the exact finite-sample quantile, ensembles, MC dropout, calibration), `verification.geometry_ood_guardrail`
  (hand-crafted shape features, diagonal Mahalanobis).
* Assimilation: `pinneapple_analysis.state_estimation` (EKF, EnKF), `data_assimilation.var4d`,
  `inverse_problems.ensemble_kalman` (EKI).
* Lab: `pinneapple_lab` experiment `rom_study` (POD/DMD/OpInf against the library's solvers).

Security (`pinneapple_security`): integrity manifests, Ed25519/HMAC signing and DSSE, in-toto/SLSA attestation and
SBOM, hash-chained audit log, AES-GCM at rest, PII detection and pseudonymisation, differential privacy (mechanisms,
RDP accountant, DP-SGD), checkpoint scanning and `adversarial_sensitivity`, `anomaly` (residual CUSUM, replay
score), secrets scanning, data classification, compliance map; and now `cps` + `cps_bench`. Related:
`pinneapple_systems.digital_twin.monitoring.anomaly` (Mahalanobis on Kalman innovations).

What was missing: no latent model carried a calibrated per-query error statement into the trust card, and no
security module compared physics against data-driven detection on a reproducible benchmark.

## A. Latent space — ranked

| # | direction | usefulness | fit | effort (pw) | risk | verdict |
|---|---|---|---|---|---|---|
| 1 | **Latent-space trust: conformal error bounds, envelope OOD and decoded-field physics residual for latent surrogates** | 5 | 5 | 1 | low | **implemented** |
| 2 | Latent data assimilation (EnKF / 3D-Var in POD or autoencoder coordinates) for digital twins | 5 | 4 | 2 | low | next |
| 3 | Latent dynamics surrogates with stability and rollout trust (OpInf / Koopman / latent ODE, energy-preserving variants, horizon-dependent bounds) | 4 | 5 | 2–3 | medium | next |
| 4 | Physics-constrained latent spaces (conservation in latent coordinates, latent residual training, structure-preserving projections) | 4 | 3 | 3 | medium | later |
| 5 | Geometry latent spaces (shape autoencoders, implicit neural fields, point-cloud encoders) for design optimisation and geometry OOD | 4 | 3 | 4–6 | medium-high | later |
| 6 | Latent generative surrogates (latent diffusion / flow matching) with physics checks | 3 | 3 | 4–6 | high | research track |

### A1. Latent-space trust (implemented)

* **State of the art.** Normalised split-conformal regression (Lei et al., JASA 2018; Angelopoulos & Bates,
  arXiv:2107.07511) gives distribution-free, finite-sample error bounds from any spread estimate; conformal
  prediction has been applied to neural operators and PDE surrogates (e.g. Gopakumar et al., "Uncertainty
  quantification of surrogate models using conformal prediction", arXiv:2408.09881; Moya et al., conformalised
  DeepONets, 2024). Feature-space distance as an OOD score is standard (Lee et al., NeurIPS 2018, Mahalanobis in
  feature space). Residual-based a-posteriori estimators are the classic certification tool of reduced bases
  (Hesthaven, Rozza & Stamm, 2016).
* **What existed.** `ParametricPOD` with GP standard deviation; `TrustGate` (coordinate OOD, residual, ensemble) and
  `TrustReport`; split conformal in `uncertainty.conformal`. None of these calibrated a field-level error bound per
  query for a latent model, nor turned it into a decision.
* **Integration point.** model → **trust** → decision. `TrustedLatentSurrogate.trust_report` returns a
  `TrustReport`; REVIEW/REJECT means "run the solver for this query".
* **Acceptance test (met).** On a 2-D advection-diffusion-reaction problem with every answer checked against a full
  solve: in-envelope coverage of the 90 % bound within [0.85, 1]; error-bound AUC for "error > tolerance" higher than
  a box check; fewer approved-but-wrong answers than a box check. Measured (three seeds): coverage 0.92; AUC 0.980
  (bound), 0.976 (physics residual) vs 0.891 (box); approved-but-wrong 3.3 % vs 8.8 %; median error 0.84 %
  (nearest-design baseline 17 %); 65× faster than the solver in batch, 18× with all checks. Details:
  `docs/guides/trusted_latent_surrogate.md`.

### A2. Latent data assimilation

* **State of the art.** Assimilation in reduced coordinates: Amendola et al., "Data assimilation in the latent space
  of a convolutional autoencoder", ICCS 2021; Peyron et al., "Latent space data assimilation by using deep learning",
  QJRMS 147 (2021); reduced-order EnKF / POD-4D-Var (Cao et al., IJNMF 2007); review by Cheng et al., IEEE/CAA J.
  Autom. Sinica 10 (2023).
* **What exists.** `EnsembleKalmanFilter`, `var4d`, POD/AE encoders, latent dynamics (OpInf, Koopman), digital-twin
  monitoring.
* **Integration point.** data → model (twin state update). Encode the ensemble, forecast with the latent dynamics,
  update with sparse sensors through the decoder (observation operator = sensor sampling of the decoded field).
* **Effort / risk.** 2 pw, low: all parts exist; the work is the observation operator through the decoder and
  inflation/localisation in latent coordinates.
* **Acceptance test.** Twin of a 2-D advection-diffusion field observed by 10–20 noisy point sensors: latent EnKF
  RMSE of the full field ≤ 1.1 × full-space EnKF RMSE at ≥ 20× lower cost per cycle; both against the true
  simulated field (independent reference); spread–error ratio in [0.7, 1.3].

### A3. Latent dynamics surrogates with stability and rollout trust

* **State of the art.** Operator inference (Peherstorfer & Willcox, CMAME 2016) and its energy-preserving / stable
  variants (Sawant, Kramer, Peherstorfer, CMAME 2023; Goyal, Pontes Duff, Benner, 2023); Koopman autoencoders (Lusch,
  Kutz, Brunton, Nat. Commun. 2018); latent ODEs (Rubanova et al., NeurIPS 2019); LaSDI (Fries, He, Choi, CMAME
  2022); latent evolution of PDEs (Wu, Maruyama, Leskovec, NeurIPS 2022); encode-process-decode (Sanchez-Gonzalez et
  al., ICML 2020).
* **What exists.** OpInf, DMD, HAVOK, SINDy, Koopman AE, latent ODE, `rom_study` lab experiment.
* **Integration point.** model + trust: a horizon-dependent conformal bound (calibrated per forecast step) and a
  stability check (spectral radius / energy growth of the learned latent operator) in the trust card.
* **Effort / risk.** 2–3 pw, medium (long-horizon calibration is not exchangeable across horizons; calibrate per
  horizon).
* **Acceptance test.** Cylinder-wake and Burgers datasets from the library's solvers: 200-step rollout error of
  stable OpInf ≤ unconstrained OpInf and ≤ DMD; per-horizon 90 % bound covers ≥ 85 % of held-out trajectories at
  every horizon; ≥ 100× speed-up over the solver.

### A4. Physics-constrained latent spaces

* **State of the art.** Conservation-enforcing nonlinear-manifold ROMs (Lee & Carlberg, "Deep conservation", JCP
  2021), structure-preserving (port-Hamiltonian, GENERIC) latent dynamics (Hernández et al., JCP 2021), physics-
  informed latent training (Kim et al., JCP 2022, "A fast and accurate physics-informed neural network reduced order
  model"). `koopman_pi_ae` is a start.
* **Integration point.** model training; conserved quantities checked in the trust card.
* **Effort / risk.** 3 pw, medium.
* **Acceptance test.** On a conservative problem (advection of a scalar, heat in an insulated domain) the decoded
  total mass drifts < 1e-6 relative over 500 steps (vs ≥ 1e-3 for an unconstrained AE-ROM), with field error no
  worse than 1.2 × the unconstrained model, against the full solver.

### A5. Geometry latent spaces

* **State of the art.** DeepSDF (Park et al., CVPR 2019), occupancy networks (Mescheder et al., CVPR 2019), PointNet
  (Qi et al., CVPR 2017), geometry-informed neural operators (GINO, Li et al., NeurIPS 2023), Transolver (Wu et al.,
  ICML 2024).
* **What exists.** `geometry_ood_guardrail` (hand features), `pinneapple_design` parametric geometry and optimiser,
  car2d generator, voxelisation.
* **Integration point.** geometry → model (shape code as surrogate input) and geometry → trust (latent density as a
  geometry OOD score replacing hand features).
* **Effort / risk.** 4–6 pw, medium-high (needs a shape dataset and GPU training for 3-D).
* **Acceptance test.** On the car2d family: shape-code surrogate drag error ≤ the parametric surrogate's on held-out
  shapes; latent OOD AUC ≥ 0.9 for shapes from a different family (vs the current guardrail's AUC on the same split).

### A6. Latent generative surrogates

* **State of the art.** Latent diffusion (Rombach et al., CVPR 2022), flow matching (Lipman et al., ICLR 2023),
  autoregressive conditional diffusion for turbulence (Kohl, Chen, Thuerey, arXiv:2309.01745), physics-informed
  diffusion (Shu, Li, Barati Farimani, JCP 2023; Bastek, Sun, Kochmann, ICLR 2025), GenCast (Price et al., Nature
  2025).
* **What exists.** `PhysicsInformedDiffusion` with PDE-residual guidance, POD/AE encoders.
* **Integration point.** model (ensembles of plausible fields for uncertainty), trust (residual of every sample).
* **Effort / risk.** 4–6 pw, high (GPU, evaluation of distributional skill is itself research).
* **Acceptance test.** On a turbulent 2-D flow: CRPS of the generated ensemble ≤ that of a deep ensemble of the same
  cost; every released sample passes the residual check; energy spectrum within 10 % of the solver up to the
  resolved wavenumber.

## B. Cybersecurity — ranked

| # | application | usefulness | fit | effort (pw) | risk | verdict |
|---|---|---|---|---|---|---|
| 1 | **Physics-residual false-data-injection detection in cyber-physical systems (water, process, HVAC), with a reproducible benchmark** | 5 | 5 | 1.5 | low | **implemented** |
| 2 | Power-grid state-estimation FDI detection beyond chi-squared bad-data detection (IEEE test cases, dynamic residuals) | 5 | 4 | 2–3 | medium | next |
| 3 | Physics-residual screening of training data against poisoning, tied to dataset provenance | 4 | 5 | 1.5 | low-medium | next |
| 4 | Sensor spoofing detection for vehicles and robots (GNSS/IMU/wheel-odometry consistency through vehicle dynamics) | 4 | 3 | 3 | medium | later |
| 5 | Adversarial robustness and certification of physics surrogates (Lipschitz bounds, randomised smoothing, physics-consistent attacks) | 3 | 4 | 2–3 | medium | later |
| 6 | Federated, differentially private training on engineering data across sites | 3 | 4 | 3 | medium | later |
| 7 | Signed, re-verifiable trust reports and simulation results (verify the physics residual on load) | 3 | 5 | 1 | low | quick win |
| 8 | Side-channel and process-tampering detection from physical signals (acoustic, power) in manufacturing | 2 | 2 | 4+ | high | not now |

### B1. Physics-residual FDI detection (implemented)

* **State of the art.** False data injection against state estimation (Liu, Ning, Reiter, CCS 2009); physics-based
  detection in process control (Cárdenas et al., ASIACCS 2011); CUSUM on physical residuals and the stealthy-attack
  bound (Urbina et al., CCS 2016); survey: Giraldo et al., ACM CSUR 51(4) 2018; water-network competition BATADAL
  (Taormina et al., J. Water Resour. Plann. Manage. 2018).
* **What existed.** `anomaly.ResidualCUSUM` (one residual, no identification), `anomaly.replay_score`, digital-twin
  innovation monitoring.
* **What was built.** `cps.MassBalance` and `cps.FlowLaw` (constants identified from trusted data),
  `cps.PhysicsResidualDetector` (per-channel CUSUM, threshold calibrated to a false-alarm budget, blamed channel),
  `cps.CusumAlarm` (the same alarm rule for every detector), `cps.roc_auc`, `cps.window_auc`,
  `cps.episode_metrics`; `cps_bench`: seeded three-tank water network, six attacks, three data-driven baselines,
  nominal and shifted operating regimes.
* **Data.** Simulated. SWaT and WADI require a signed request; BATADAL is public but its host was not reachable from
  the build environment. A BATADAL adapter (`pinneapple_security.batadal`, #645) now runs the detectors on its
  training and attack datasets; its test set has no public labels, so the competition score is not reproduced.
  Results and limits: `docs/guides/physics_residual_attack_detection.md`.
* **Acceptance test (met).** At a calibrated budget of one false alarm per day: physics detector episode AUC ≥ 0.95
  on every single-sensor attack in both regimes; false alarms per day in the shifted regime ≤ 3 × budget; better than
  every data-driven baseline in the shifted regime. Measured (three seeds): AUC 1.000 on all five single-sensor
  attacks in both regimes; shifted-regime false alarms 0.47/day vs 84.4 (AR twin), 32.0 (PCA); in the nominal regime
  the AR twin matches the AUC (1.0) but localises the attacked sensor far less often (bias: 0.03 vs 0.98). Limits
  stated: a coordinated, physically consistent replay is invisible to any residual test except at a splice. Details:
  `docs/guides/physics_residual_attack_detection.md`.

### B2. Power-grid state-estimation FDI detection

* **State of the art.** Liu et al. 2009; Kosut et al., IEEE Trans. Smart Grid 2011 (malicious data attacks, GLRT);
  Manandhar et al., IEEE Systems Journal 2014 (Kalman filter + chi-squared/Euclidean detectors); survey Musleh, Chen,
  Dong, IEEE Trans. Smart Grid 11(3) 2020.
* **What exists.** EKF/EnKF, the `cps` residual framework, CUSUM; no power-flow solver of its own (pandapower,
  BSD-3, or MATPOWER cases, BSD-3, would supply IEEE 14/30/118-bus data).
* **Integration point.** data (SCADA/PMU) → physics (AC power-flow residual) → `PhysicsResidualDetector`.
* **Acceptance test.** IEEE 14- and 118-bus cases with load profiles: random FDI detected with AUC ≥ 0.95 at 1
  false alarm/day; stealthy FDI `a = H c` shown undetectable by static bad-data detection (chi-squared, as theory
  predicts) and detected by the dynamic residual with AUC ≥ 0.8 when load dynamics are modelled; compared with an
  isolation forest and an AR twin on the same measurements.

### B3. Physics-residual screening against data poisoning

* **State of the art.** Certified defences for data poisoning (Steinhardt, Koh, Liang, NeurIPS 2017); poisoning of
  scientific ML surrogates is barely studied; physics residuals give a label-free consistency test of each training
  sample.
* **What exists.** Residual functions, `TrustGate`, integrity manifests and attestation, `secrets_scan`.
* **Integration point.** data → model: before training, every simulated or measured sample is scored by its physics
  residual; outliers are quarantined and the decision is written to the signed dataset manifest.
* **Acceptance test.** Darcy/heat datasets with 1–10 % poisoned samples (scaled fields, swapped labels, smooth
  perturbations): detection AUC ≥ 0.95 for physics residual vs ≤ 0.8 for a data-only outlier detector; surrogate test
  error after screening within 10 % of the clean-data model.

### B4. Sensor spoofing detection for vehicles and robots

* **State of the art.** GPS spoofing of UAVs (Kerns et al., J. Field Robotics 2014); control-invariant attack
  detection (Choi et al., CCS 2018); SAVIOR (Quinonez et al., USENIX Security 2020).
* **What exists.** Vehicle bicycle-model runs, EKF/UKF, the `cps` framework.
* **Acceptance test.** Simulated vehicle with GNSS, IMU and wheel odometry: slow GNSS drift spoofing detected with
  AUC ≥ 0.95 and median delay ≤ 10 s at 1 false alarm per hour, vs an innovation-chi-squared EKF baseline.

### B5. Adversarial robustness and certification of physics surrogates

* **State of the art.** Interval bound propagation (Gowal et al., 2018), randomised smoothing (Cohen, Rosenfeld,
  Kolter, ICML 2019), Lipschitz-constrained networks.
* **What exists.** `model_security.adversarial_sensitivity` (empirical worst case), `TrustGate`.
* **Acceptance test.** For an MLP surrogate of a parametric PDE: certified output radius from IBP or smoothing that
  upper-bounds the empirical worst case (PGD) on 100 % of 200 test points; a physics-consistent attack (inputs that
  keep the PDE residual small) found or bounded.

### B6. Federated, differentially private training

* **State of the art.** FedAvg (McMahan et al., AISTATS 2017), secure aggregation (Bonawitz et al., CCS 2017), DP-SGD
  with RDP accounting (Abadi et al., CCS 2016; Mironov 2017).
* **What exists.** DP-SGD and RDP accountant; no federated loop.
* **Acceptance test.** Three sites with different operating regimes training a surrogate: federated DP model at
  ε ≤ 8 within 1.3 × the error of centralised non-private training and better than every site's local model on the
  pooled test set.

### B7. Signed, re-verifiable trust reports

* **What exists.** DSSE signing, attestation, `TrustReport.to_dict`.
* **Acceptance test.** A signed trust report whose recorded physics residual is recomputed on load from the signed
  model and inputs; any tampering with the report, the model or the inputs fails verification (property test over
  100 random byte flips).

### B8. Side-channel / physical-signal tampering detection

Acoustic or power side channels of additive manufacturing (Al Faruque et al., ICCPS 2016) need sensor data and
process models PINNeAPPle does not have. Not recommended now.

## Recommended roadmap

1. Done in this change: A1 (`TrustedLatentSurrogate`) and B1 (`cps`, `cps_bench`).
2. Next quarter: A2 latent data assimilation and B3 poisoning screening (both reuse what exists, low risk, and
   strengthen the twin and the data stage of the pipeline); B7 as a quick win; the BATADAL adapter for B1.
3. Then: A3 latent dynamics with rollout trust, B2 power-grid FDI (with pandapower or MATPOWER cases), A4
   physics-constrained latent spaces.
4. Research track, when a GPU budget and datasets exist: A5 geometry latent spaces, A6 latent generative
   surrogates, B4, B5, B6. B8 is not planned.

Each remaining item is tracked as a GitHub issue labelled `enhancement` with its acceptance test.

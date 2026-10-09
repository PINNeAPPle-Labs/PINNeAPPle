# Security for scientific and industrial Physics AI (`pinneapple_security`)

Simulation datasets, plant data and trained surrogates are valuable, often confidential, sometimes personal or
export-controlled, and increasingly used for decisions. `pinneapple_security` gives each step of a Physics AI workflow
a technical control and machine-checkable evidence. It is available as `pp.security`.

```bash
pip install pinneapple[security]     # adds `cryptography` (Ed25519, AES-GCM); everything else needs only NumPy
```

| Need | Module | Main calls |
|---|---|---|
| Prove a dataset or model folder did not change | `integrity` | `build_manifest(dir)`, `Manifest.verify(dir)`, `digest_array(a)` |
| Show who released it | `signing` | `generate_key()`, `sign_json` / `verify_json` (DSSE), `sign_file` / `verify_file` |
| Record how it was produced | `attestation` | `attest_experiment(result)`, `attest_files(...)` (in-toto + SLSA), `sbom()` (CycloneDX) |
| Keep a trustworthy record of actions | `audit` | `AuditLog(path, key).append(action, actor, target, data, reason)`, `.verify()` |
| Keep data confidential at rest | `encryption` | `encrypt_file`, `encrypt_array` (AES-256-GCM, scrypt passphrases) |
| Remove personal data | `privacy` | `find_pii`, `redact`, `pseudonymize`, `k_anonymity`, `sanitize_log` |
| Train on sensitive data without leaking records | `dp` | `DPSGD`, `RDPAccountant`, `dp_mean`, `dp_histogram` |
| Load third-party checkpoints safely | `model_security` | `scan_checkpoint`, `safe_load`, `adversarial_sensitivity` |
| Detect manipulated sensor data | `anomaly` | `ResidualCUSUM` on physics residuals, `balance_residual`, `replay_score` |
| Keep credentials out of repositories | `secrets_scan` | `scan_path` |
| Decide what may be shared, trained or exported | `classification` | `DataLabel`, `combine`, `HandlingPolicy.check` |
| Map evidence to frameworks | `compliance` | `assess(evidence)`, `report_markdown` |
| Do all of it with the rest of the library | `integrate` | `secure_save` / `verify_version` (ModelStore), `sign_model_card`, `attest_and_sign_experiment` |

## Physics as a security signal

An attacker who falsifies one sensor keeps the reading plausible on its own. What breaks is its consistency with the
physics that links it to other measurements: a mass or energy balance, a pump curve, a digital-twin surrogate.
`anomaly.ResidualCUSUM` learns the normal spread of that residual from trusted data and accumulates small persistent
deviations, so slow ramp attacks are caught, not just jumps. `replay_score` flags data that repeats an old recording
sample for sample.

## Signed model versions

```python
from pinneapple_registry.model_store import ModelStore
from pinneapple_security import integrate, signing
from pinneapple_security.audit import AuditLog

key = signing.generate_key()                      # keep private; publish key.public().to_dict()
store, log = ModelStore("models"), AuditLog("audit.jsonl", key=b"...32 bytes...")
v = integrate.secure_save(store, "burgers", model, key, actor="ana", audit=log, reason="baseline")
integrate.verify_version(store, "burgers", v, key.public())   # signature + manifest + checkpoint scan
```

`ModelStore.promote` rewrites `metadata.json`, so a promoted version no longer matches its signature until
`integrate.resign_version` signs it again (the audit trail keeps both roots).

## Command line

```bash
python -m pinneapple_security manifest data/ -o data.manifest.json
python -m pinneapple_security scan-model downloaded.pt
python -m pinneapple_security scan-secrets .
python -m pinneapple_security sanitize-log log.simpleFoam --hide ClientName -o log.public
python -m pinneapple_security sbom -o sbom.json
```

## What it is not

Evidence supports a control; it does not make an organisation compliant with NIST CSF, IEC 62443, ISO/IEC 27001,
21 CFR Part 11 or LGPD/GDPR. Regex PII detection misses free-text names and addresses. Differential privacy protects
records only for the (ε, δ) reported and only if the clipping bounds were not derived from the private data.
The HMAC options (signatures, audit chain) need the key to stay secret; Ed25519 signatures do not.

Full example: `examples/security/01_secure_physics_ai_pipeline.py`.

"""pinneapple_security — data and process security for scientific and industrial Physics AI.

What it covers, and where:

=====================  ==============================================================================================
``integrity``          SHA-256 of files, arrays and records; Merkle manifests of datasets and model folders
``signing``            Ed25519 / HMAC signatures, DSSE envelopes, detached file signatures
``attestation``        in-toto + SLSA provenance of runs and experiments; CycloneDX SBOM of the environment
``audit``              hash-chained, tamper-evident audit trail (ALCOA+ attributes)
``encryption``         AES-256-GCM at rest for bytes, files and arrays (scrypt key derivation)
``privacy``            PII detection (incl. CPF/CNPJ), redaction, keyed pseudonymisation, k-anonymity, log sanitising
``dp``                 differential privacy: Laplace/Gaussian mechanisms, DP statistics, RDP accountant, DP-SGD
``model_security``     pickle/checkpoint scanning, safe loading, adversarial sensitivity of surrogates
``anomaly``            physics-residual CUSUM and replay detection of manipulated sensor data (OT security)
``cps``                grey-box physics residuals (mass balances, flow laws) + CUSUM: false-data-injection detection
``cps_bench``          three-tank water-network benchmark: attacks, data-driven baselines, ROC/delay/false alarms
``secrets_scan``       credentials in code, notebooks, configs and solver cases
``classification``     data labels (level, personal data, export control) and handling policy
``compliance``         capability → control map (NIST CSF 2.0, IEC 62443, ISO 27001, 21 CFR 11, LGPD/GDPR, SSDF, SLSA)
``integrate``          signed ModelStore versions, signed model cards, attested experiments
``web``                HTTP hardening for the apps: security headers/CSP, body limits, rate limits, login lockout,
                       trusted-proxy client IP, CSRF origin check, CORS allow-list, auth audit events
``passwords``          salted password hashes (scrypt; argon2id / bcrypt when installed) and verification
``archives``           zip extraction without zip-slip, links or zip bombs (counted sizes, ratio, member limits)
=====================  ==============================================================================================

Dependencies: the standard library and NumPy; PyTorch for ``dp.DPSGD``, ``model_security.safe_load`` and
``adversarial_sensitivity``; ``cryptography`` (``pip install pinneapple[security]``) for Ed25519 and AES-GCM.
Command line: ``python -m pinneapple_security --help``.
"""
from __future__ import annotations

import importlib as _importlib

__all__ = [
    "integrity", "signing", "attestation", "audit", "encryption", "privacy", "dp", "model_security", "anomaly",
    "secrets_scan", "classification", "compliance", "integrate", "cps", "cps_bench", "web", "passwords",
    "archives",
    "build_manifest", "Manifest", "sha256_file", "digest_array", "generate_key", "hmac_key", "sign_json",
    "verify_json", "AuditLog", "find_pii", "redact", "sanitize_log", "scan_checkpoint", "safe_load", "scan_path",
    "DataLabel", "HandlingPolicy", "ResidualCUSUM", "sbom", "attest_experiment", "PhysicsResidualDetector",
    "hash_password", "verify_password", "safe_extract_zip",
]

_ATTRS = {
    "build_manifest": "integrity", "Manifest": "integrity", "sha256_file": "integrity", "digest_array": "integrity",
    "generate_key": "signing", "hmac_key": "signing", "sign_json": "signing", "verify_json": "signing",
    "AuditLog": "audit", "find_pii": "privacy", "redact": "privacy", "sanitize_log": "privacy",
    "scan_checkpoint": "model_security", "safe_load": "model_security", "scan_path": "secrets_scan",
    "DataLabel": "classification", "HandlingPolicy": "classification", "ResidualCUSUM": "anomaly",
    "sbom": "attestation", "attest_experiment": "attestation", "PhysicsResidualDetector": "cps",
    "hash_password": "passwords", "verify_password": "passwords", "safe_extract_zip": "archives",
}
_MODULES = set(__all__[:18])


def __getattr__(name: str):
    if name in _MODULES:
        mod = _importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _ATTRS:
        obj = getattr(_importlib.import_module(f".{_ATTRS[name]}", __name__), name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'pinneapple_security' has no attribute '{name}'")

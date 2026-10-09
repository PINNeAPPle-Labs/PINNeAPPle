"""Which pinneapple_security capability produces evidence for which control of the frameworks industry and research
are audited against — and a self-assessment report from the evidence a project actually has.

This maps; it does not certify. A framework control is met by an organisation's processes; the library only provides
technical means and machine-checkable evidence (manifests, signatures, attestations, audit-log verification, scans).

Frameworks (identifiers as published): NIST Cybersecurity Framework 2.0 (functions GV, ID, PR, DE, RS, RC);
IEC 62443-3-3 foundational requirements FR1–FR7 (industrial automation and control systems); ISO/IEC 27001:2022
Annex A (selected controls); 21 CFR Part 11 (US FDA electronic records and signatures); LGPD (Lei 13.709/2018) and
GDPR (Regulation (EU) 2016/679); NIST SP 800-218 (SSDF) and SLSA v1.0 for the software and model supply chain.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["CAPABILITIES", "CONTROLS", "controls_for", "assess", "report_markdown"]

# capability id -> what it is (module.function)
CAPABILITIES: dict[str, str] = {
    "manifest": "integrity.build_manifest / Manifest.verify — dataset and model digests, Merkle root",
    "signature": "signing.sign_json / sign_file — Ed25519 or HMAC signatures (DSSE envelope)",
    "attestation": "attestation.attest_experiment / attest_files — in-toto + SLSA provenance",
    "sbom": "attestation.sbom — CycloneDX 1.5 software bill of materials",
    "audit_log": "audit.AuditLog — hash-chained, tamper-evident audit trail",
    "encryption": "encryption.encrypt_file — AES-256-GCM at rest",
    "pii_scan": "privacy.find_pii / redact / sanitize_log",
    "pseudonymization": "privacy.pseudonymize — keyed HMAC tokens",
    "k_anonymity": "privacy.k_anonymity",
    "differential_privacy": "dp.DPSGD / RDPAccountant / dp_mean",
    "model_scan": "model_security.scan_checkpoint / safe_load",
    "adversarial_test": "model_security.adversarial_sensitivity",
    "physics_anomaly": "anomaly.ResidualCUSUM / replay_score — sensor manipulation detection",
    "secret_scan": "secrets_scan.scan_path",
    "classification": "classification.DataLabel / HandlingPolicy",
}

# (framework, control id, title, capabilities that give evidence)
CONTROLS: list[dict[str, Any]] = [
    {"framework": "NIST CSF 2.0", "id": "ID.AM", "title": "Asset management", "evidence": ["manifest", "sbom", "classification"]},
    {"framework": "NIST CSF 2.0", "id": "PR.DS", "title": "Data security", "evidence": ["encryption", "manifest", "signature", "pii_scan"]},
    {"framework": "NIST CSF 2.0", "id": "PR.PS", "title": "Platform security", "evidence": ["model_scan", "sbom", "secret_scan"]},
    {"framework": "NIST CSF 2.0", "id": "DE.CM", "title": "Continuous monitoring", "evidence": ["physics_anomaly", "audit_log"]},
    {"framework": "NIST CSF 2.0", "id": "DE.AE", "title": "Adverse event analysis", "evidence": ["physics_anomaly", "audit_log"]},
    {"framework": "NIST CSF 2.0", "id": "GV.SC", "title": "Cybersecurity supply chain risk management", "evidence": ["sbom", "attestation", "model_scan"]},
    {"framework": "IEC 62443-3-3", "id": "FR3", "title": "System integrity", "evidence": ["manifest", "signature", "physics_anomaly"]},
    {"framework": "IEC 62443-3-3", "id": "FR4", "title": "Data confidentiality", "evidence": ["encryption", "classification"]},
    {"framework": "IEC 62443-3-3", "id": "FR6", "title": "Timely response to events", "evidence": ["audit_log", "physics_anomaly"]},
    {"framework": "ISO/IEC 27001:2022", "id": "A.5.12", "title": "Classification of information", "evidence": ["classification"]},
    {"framework": "ISO/IEC 27001:2022", "id": "A.5.34", "title": "Privacy and protection of PII", "evidence": ["pii_scan", "pseudonymization", "k_anonymity", "differential_privacy"]},
    {"framework": "ISO/IEC 27001:2022", "id": "A.8.15", "title": "Logging", "evidence": ["audit_log"]},
    {"framework": "ISO/IEC 27001:2022", "id": "A.8.24", "title": "Use of cryptography", "evidence": ["encryption", "signature"]},
    {"framework": "ISO/IEC 27001:2022", "id": "A.8.28", "title": "Secure coding", "evidence": ["secret_scan", "model_scan", "sbom"]},
    {"framework": "21 CFR Part 11", "id": "11.10(c)", "title": "Protection of records", "evidence": ["manifest", "encryption", "signature"]},
    {"framework": "21 CFR Part 11", "id": "11.10(e)", "title": "Secure, time-stamped audit trails", "evidence": ["audit_log"]},
    {"framework": "21 CFR Part 11", "id": "11.70", "title": "Signature/record linking", "evidence": ["signature"]},
    {"framework": "LGPD", "id": "Art. 46", "title": "Security measures for personal data", "evidence": ["encryption", "pseudonymization", "pii_scan"]},
    {"framework": "GDPR", "id": "Art. 25", "title": "Data protection by design and by default", "evidence": ["pseudonymization", "differential_privacy", "k_anonymity"]},
    {"framework": "GDPR", "id": "Art. 32", "title": "Security of processing", "evidence": ["encryption", "pseudonymization", "audit_log"]},
    {"framework": "NIST SP 800-218 (SSDF)", "id": "PS.3", "title": "Archive and protect each software release", "evidence": ["manifest", "signature", "sbom"]},
    {"framework": "SLSA v1.0", "id": "Build L1/L2", "title": "Provenance exists / is signed", "evidence": ["attestation", "signature"]},
]


def controls_for(capability: str) -> list[dict[str, Any]]:
    if capability not in CAPABILITIES:
        raise KeyError(f"unknown capability {capability!r}")
    return [c for c in CONTROLS if capability in c["evidence"]]


def assess(evidence: dict[str, Any], frameworks: Iterable[str] = ()) -> dict[str, Any]:
    """``evidence``: capability → anything truthy (a path, a verification result with ``ok``, True). A control is
    ``covered`` when at least one of its capabilities has evidence, ``partial`` when some do but a verification failed."""
    fw = set(frameworks)
    unknown = sorted(set(evidence) - set(CAPABILITIES))
    if unknown:
        raise KeyError(f"unknown capabilities: {unknown}")

    def ok(v):
        return bool(v.get("ok", True)) if isinstance(v, dict) else bool(v)

    rows = []
    for c in CONTROLS:
        if fw and c["framework"] not in fw:
            continue
        have = [e for e in c["evidence"] if e in evidence and evidence[e]]
        good = [e for e in have if ok(evidence[e])]
        status = "covered" if good and len(good) == len(have) else ("partial" if have else "no evidence")
        rows.append({**c, "status": status, "with_evidence": have, "failing": sorted(set(have) - set(good))})
    n = {s: sum(r["status"] == s for r in rows) for s in ("covered", "partial", "no evidence")}
    return {"controls": rows, "summary": n}


def report_markdown(result: dict[str, Any], title: str = "Security evidence self-assessment") -> str:
    s = result["summary"]
    lines = [f"# {title}", "", "Technical evidence produced with `pinneapple_security`. Evidence supports a control; "
             "it does not by itself make an organisation compliant.", "",
             f"Covered: **{s['covered']}** · partial: **{s['partial']}** · no evidence: **{s['no evidence']}**", "",
             "| Framework | Control | Title | Status | Evidence |", "|---|---|---|---|---|"]
    for r in result["controls"]:
        ev = ", ".join(r["with_evidence"]) or "—"
        if r["failing"]:
            ev += f" (failing: {', '.join(r['failing'])})"
        lines.append(f"| {r['framework']} | {r['id']} | {r['title']} | {r['status']} | {ev} |")
    return "\n".join(lines) + "\n"

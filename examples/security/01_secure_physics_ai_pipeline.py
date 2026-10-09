"""Secure Physics AI pipeline, end to end, with pinneapple_security.

A plant shares 30 days of operating data (flow in, flow out, pump speed) to train a surrogate of the outlet flow.
The data is confidential and holds operator e-mails in a notes column. The pipeline:

1. labels the dataset, redacts personal data, writes a signed manifest of it and an encrypted copy (AES-GCM);
2. trains the surrogate with DP-SGD (differential privacy) and reports the (epsilon, delta) spent;
3. stores the model as a signed, scanned ModelStore version, with an audit-trail record;
4. writes a signed in-toto/SLSA provenance statement of the run;
5. uses the surrogate's physics residual to detect a spoofed outlet meter (CUSUM);
6. prints the evidence self-assessment against 21 CFR Part 11 and IEC 62443.

Runs in about 20 s on a laptop CPU (needs torch and cryptography: pip install pinneapple[security]).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from pinneapple_registry.model_store import ModelStore
from pinneapple_security import (anomaly, attestation, audit, classification, compliance, dp, encryption, integrate,
                                 integrity, privacy, signing)


def plant_data(n=720, seed=0):
    """Hourly data: outlet flow = inflow x pump efficiency curve + noise (the 'physics' the surrogate learns)."""
    rng = np.random.default_rng(seed)
    q_in = 50 + 8 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 1.0, n)
    speed = rng.uniform(0.6, 1.0, n)
    q_out = q_in * (0.92 + 0.06 * speed - 0.04 * speed ** 2) + rng.normal(0, 0.3, n)
    notes = ["ok" if i % 50 else f"checked by op{i}@plant-a.com" for i in range(n)]
    return q_in, speed, q_out, notes


def main(out_dir: str) -> dict:
    torch.manual_seed(0)
    key = signing.generate_key()
    log = audit.AuditLog(os.path.join(out_dir, "audit.jsonl"), key=b"plant-a-audit-key-32-bytes-long!")
    q_in, speed, q_out, notes = plant_data()

    # 1. label, redact, manifest
    label = classification.DataLabel(level="confidential", owner="Plant A", personal_data=True)
    print("publish raw data?", classification.HandlingPolicy().check("publish", label).reasons)
    clean_notes = [privacy.redact(t) for t in notes]
    data_dir = os.path.join(out_dir, "dataset")
    os.makedirs(data_dir)
    np.savez(os.path.join(data_dir, "ops.npz"), q_in=q_in, speed=speed, q_out=q_out)
    with open(os.path.join(data_dir, "notes.txt"), "w") as f:
        f.write("\n".join(clean_notes))
    man = integrity.build_manifest(data_dir)
    man.save(os.path.join(out_dir, "dataset.manifest.json"))
    signing.sign_file(os.path.join(out_dir, "dataset.manifest.json"), key)
    data_key = encryption.new_key()                       # held by the plant's key manager, not stored with the data
    encryption.encrypt_file(os.path.join(data_dir, "ops.npz"), data_key, out=os.path.join(out_dir, "ops.npz.enc"))
    log.append("dataset.ingest", "data-steward", target="plant-a/ops", data={"merkle_root": man.root},
               reason="surrogate training")
    print("dataset merkle root", man.root[:16], "| personal data left:", sum(bool(privacy.find_pii(t)) for t in clean_notes))

    # 2. DP-SGD surrogate q_out = f(q_in, speed) on standardised data
    X = torch.tensor(np.c_[(q_in - 50) / 8, (speed - 0.8) / 0.12], dtype=torch.float32)
    y = torch.tensor(((q_out - 47) / 8)[:, None], dtype=torch.float32)
    n_train = 600
    model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch = 60
    eng = dp.DPSGD(model, opt, lambda o, t: ((o - t) ** 2).mean(dim=1), noise_multiplier=2.0, max_grad_norm=1.0,
                   sample_rate=batch / n_train, seed=0)
    for _ in range(15):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            eng.step(X[idx], y[idx])
    with torch.no_grad():
        pred = model(X)[:, 0].numpy() * 8 + 47
    rmse = float(np.sqrt(np.mean((pred[n_train:] - q_out[n_train:]) ** 2)))
    eps = eng.epsilon(1e-5)
    print(f"DP surrogate: held-out RMSE {rmse:.2f} m3/h, privacy spent epsilon={eps['epsilon']:.2f} at delta=1e-5")

    # 3. signed, scanned model version + audit
    store = ModelStore(os.path.join(out_dir, "models"))
    v = integrate.secure_save(store, "plant-a-outlet", model, key, actor="ml-engineer", audit=log,
                              metadata={"metrics": {"rmse_m3h": rmse}, "dp": eps, "label": label.to_dict()},
                              reason="DP surrogate v1")
    check = integrate.verify_version(store, "plant-a-outlet", v, key.public())
    print("model version", v, "verified:", check["ok"])

    # 4. provenance of the run, signed
    st = attestation.attest_files({"model.pt": store.checkpoint_path("plant-a-outlet", v)},
                                  {"epochs": 15, "noise_multiplier": 2.0, "max_grad_norm": 1.0, "batch": batch},
                                  inputs={"ops.npz": os.path.join(data_dir, "ops.npz")}, repo=None)
    signing.sign_json(st, key, payload_type="application/vnd.in-toto+json")

    # 5. spoofed outlet meter: the physics residual (measured - surrogate) drifts, CUSUM alarms
    resid = q_out - pred
    det = anomaly.ResidualCUSUM().fit(resid[n_train:n_train + 60])
    spoofed = q_out[n_train:].copy()
    spoofed[60:] -= 0.6                          # meter reads ~1.2 % low from sample 60: hides a small leak
    r = det.run(spoofed - pred[n_train:])
    after = [a["t"] for a in r["alarms"] if a["t"] >= 60]
    print("spoofing detected", (after[0] - 60) if after else None, "samples after it started")
    log.append("sensor.alarm", "twin-monitor", target="plant-a/q_out", data={"first_alarm": after[0] if after else None})

    # 6. evidence report
    ev = {"encryption": True, "manifest": True, "signature": check, "audit_log": log.verify(), "attestation": True,
          "differential_privacy": True, "pii_scan": True, "physics_anomaly": bool(after), "model_scan": check["scan"]}
    rep = compliance.assess(ev, frameworks=["21 CFR Part 11", "IEC 62443-3-3"])
    print(compliance.report_markdown(rep, "Plant A surrogate: evidence"))
    return {"rmse": rmse, "epsilon": eps["epsilon"], "verified": check["ok"], "detect_delay": (after[0] - 60) if after else None,
            "audit_ok": log.verify()["ok"], "summary": rep["summary"]}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        main(d)

"""Secret scanning, data classification policy, compliance map, attestations, SBOM, ModelStore integration, CLI."""
import json
import os

import pytest

from pinneapple_security import attestation, classification, compliance, secrets_scan, signing
from pinneapple_security.__main__ import main as cli


def test_secret_scan_finds_tokens_and_ignores_placeholders(tmp_path):
    (tmp_path / "cfg.yaml").write_text(
        "aws_key: AKIAIOSFODNN7EXAMPLQ\n"
        "db: postgresql://admin:S3cr3tPw@db.internal:5432/runs\n"
        'password = "hunter2hunter2"\n'
        'api_key = "your_api_key_here"\n'
        "sha: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08\n")
    (tmp_path / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("ghp_" + "a" * 36)
    found = secrets_scan.scan_path(str(tmp_path))
    rules = sorted({f.rule for f in found})
    assert rules == ["aws_access_key_id", "password_assignment", "private_key_block", "url_with_password"], rules
    assert all("S3cr3tPw" not in f.excerpt and "hunter2hunter2" not in f.excerpt for f in found)


def test_handling_policy_and_label_inheritance():
    pol = classification.HandlingPolicy()
    plant = classification.DataLabel(level="confidential", owner="client A", personal_data=False)
    ops = classification.DataLabel(level="internal", owner="ops", personal_data=True)
    turbine = classification.DataLabel(level="internal", export_control="9E003", allowed_countries=["BR", "US"])
    derived = classification.combine([plant, ops, turbine])
    assert derived.level == "confidential" and derived.personal_data and derived.export_control == "9E003"
    d = pol.check("publish", derived)
    assert not d.allowed and set(d.requires) == {"dpo", "export_officer"}
    assert pol.check("train", plant).allowed
    assert not pol.check("share_external", plant).allowed
    assert pol.check("share_external", plant, approvals=["owner"]).allowed
    assert not pol.check("export", turbine, approvals=["export_officer"], destination_country="CN").allowed
    assert pol.check("export", turbine, approvals=["export_officer"], destination_country="br").allowed
    assert not pol.check("train", classification.DataLabel(licence="CC-BY-NC-4.0")).allowed
    with pytest.raises(ValueError):
        classification.DataLabel(level="secret")


def test_compliance_assessment():
    r = compliance.assess({"audit_log": {"ok": True}, "manifest": True, "signature": {"ok": False}},
                          frameworks=["21 CFR Part 11"])
    by = {c["id"]: c["status"] for c in r["controls"]}
    assert by == {"11.10(c)": "partial", "11.10(e)": "covered", "11.70": "partial"}
    assert "21 CFR Part 11" in compliance.report_markdown(r)
    assert {c["id"] for c in compliance.controls_for("physics_anomaly")} >= {"DE.CM", "FR3"}
    assert all(set(c["evidence"]) <= set(compliance.CAPABILITIES) for c in compliance.CONTROLS)
    with pytest.raises(KeyError):
        compliance.assess({"magic": True})


def test_sbom_and_file_attestation(tmp_path):
    doc = attestation.sbom(packages=["numpy", "pytest"])
    names = {c["name"].lower() for c in doc["components"]}
    assert doc["bomFormat"] == "CycloneDX" and {"numpy", "pytest"} <= names
    assert all(c["purl"].startswith("pkg:pypi/") for c in doc["components"])
    data, out = tmp_path / "train.csv", tmp_path / "model.pt"
    data.write_text("x\n1\n")
    out.write_bytes(b"w")
    st = attestation.attest_files({"model.pt": str(out)}, {"epochs": 10}, inputs=[str(data)], repo=None)
    assert st["_type"] == attestation.STATEMENT_TYPE and st["predicateType"] == attestation.SLSA_PROVENANCE
    assert st["subject"][0]["digest"]["sha256"] == attestation.sha256_file(str(out))
    assert st["predicate"]["buildDefinition"]["resolvedDependencies"][0]["name"] == "train.csv"


def test_experiment_attestation_signed(tmp_path):
    pytest.importorskip("cryptography")
    import numpy as np

    import pinneapple as pp
    from pinneapple_physics.closed_form.burgers import burgers_sine_exact
    from pinneapple_security import integrate

    nu = 0.01 / np.pi
    exp = pp.Experiment(pp.PhysicalProblem.from_preset("burgers_1d", nu=nu), method="exact",
                        options={"fn": lambda X: burgers_sine_exact(X[:, 0], X[:, 1], nu)[:, None]},
                        reference="analytic", n_eval=100)
    res = exp.run()
    key = signing.generate_key()
    env = integrate.attest_and_sign_experiment(res, key, directory=str(tmp_path / "run"), repo=None)
    st = signing.verify_json(env, key.public())
    names = [s["name"] for s in st["subject"]]
    assert "experiment-record.json" in names and "result.json" in names
    assert st["predicate"]["buildDefinition"]["externalParameters"]["method"] == "exact"
    assert os.path.exists(tmp_path / "run" / "attestation.json")


def test_secure_model_store_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cryptography")
    from pinneapple_registry.model_store import ModelStore
    from pinneapple_security import integrate
    from pinneapple_security.audit import AuditLog

    store, key, log = ModelStore(str(tmp_path / "store")), signing.generate_key(), AuditLog(str(tmp_path / "audit.jsonl"))
    v = integrate.secure_save(store, "burgers", torch.nn.Linear(2, 1), key, metadata={"metrics": {"rel_l2": 0.01}},
                              actor="ana", audit=log, reason="first baseline")
    assert integrate.verify_version(store, "burgers", v, key.public())["ok"]
    assert not integrate.verify_version(store, "burgers", v, signing.generate_key().public())["ok"]
    store.promote("burgers", v, "staging")                # rewrites metadata.json: no longer matches the signature
    r = integrate.verify_version(store, "burgers", v, key.public())
    assert not r["ok"] and r["files"]["changed"] == ["metadata.json"]
    integrate.resign_version(store, "burgers", v, key, actor="ana", audit=log, reason="promoted to staging")
    assert integrate.verify_version(store, "burgers", v, key.public())["ok"]
    with open(store.checkpoint_path("burgers", v), "ab") as f:
        f.write(b"x")
    assert integrate.verify_version(store, "burgers", v, key.public())["files"]["changed"] == ["model.pt"]
    assert [r["action"] for r in log.records()] == ["model.save", "model.resign"] and log.verify()["ok"]


def test_model_card_signature_covers_trust_report():
    pytest.importorskip("cryptography")
    from pinneapple_hub.model_card import ModelCard
    from pinneapple_security import integrate

    card = ModelCard(name="m", architecture="mlp", validation_metrics={"rel_l2": 0.01}, reference_source="analytic")
    key = signing.generate_key()
    env = integrate.sign_model_card(card, key)
    integrate.verify_model_card(card, env, key.public())
    card.trust_override = {"reason": "edited after signing"}
    with pytest.raises(signing.VerificationError):
        integrate.verify_model_card(card, env, key.public())


def test_cli(tmp_path, capsys):
    pytest.importorskip("cryptography")
    d = tmp_path / "ds"
    d.mkdir()
    (d / "a.txt").write_text("1")
    assert cli(["manifest", str(d), "-o", str(tmp_path / "m.json")]) == 0
    assert cli(["verify-manifest", str(d), str(tmp_path / "m.json")]) == 0
    (d / "a.txt").write_text("2")
    assert cli(["verify-manifest", str(d), str(tmp_path / "m.json")]) == 1
    assert cli(["keygen", str(tmp_path / "k.json")]) == 0
    pub = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    (tmp_path / "pub.json").write_text(json.dumps(pub))
    assert cli(["sign", str(d / "a.txt"), str(tmp_path / "k.json")]) == 0
    assert cli(["verify", str(d / "a.txt"), str(tmp_path / "pub.json")]) == 0
    (d / "a.txt").write_text("3")
    assert cli(["verify", str(d / "a.txt"), str(tmp_path / "pub.json")]) == 1
    log = tmp_path / "log.txt"
    log.write_text("Case: /home/maria/acme/run\n")
    assert cli(["sanitize-log", str(log), "--hide", "acme"]) == 0
    out = capsys.readouterr().out
    assert "maria" not in out and "acme" not in out


def test_example_secure_pipeline_runs(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("cryptography")
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "security", "01_secure_physics_ai_pipeline.py")
    spec = importlib.util.spec_from_file_location("secure_pipeline_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.main(str(tmp_path))
    assert r["verified"] and r["audit_ok"] and r["epsilon"] < 10 and r["rmse"] < 1.0
    assert r["detect_delay"] is not None and r["detect_delay"] < 30
    assert r["summary"]["no evidence"] == 0

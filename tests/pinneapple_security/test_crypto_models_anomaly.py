"""Encryption at rest, checkpoint scanning and safe loading, adversarial sensitivity, physics-residual detection."""
import os
import pickle

import numpy as np
import pytest

from pinneapple_security import anomaly, encryption, model_security

pytest.importorskip("cryptography")


def test_encrypt_round_trip_and_tamper_detection(tmp_path):
    blob = encryption.encrypt_bytes(b"pressure field", "correct horse battery")
    assert encryption.decrypt_bytes(blob, "correct horse battery") == b"pressure field"
    with pytest.raises(encryption.DecryptionError):
        encryption.decrypt_bytes(blob, "wrong passphrase!")
    bad = bytearray(blob)
    bad[-1] ^= 1
    with pytest.raises(encryption.DecryptionError):
        encryption.decrypt_bytes(bytes(bad), "correct horse battery")
    bad = bytearray(blob)
    bad[8] ^= 1                    # salt is authenticated too
    with pytest.raises(encryption.DecryptionError):
        encryption.decrypt_bytes(bytes(bad), "correct horse battery")
    key = encryption.new_key()
    a = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    np.testing.assert_array_equal(encryption.decrypt_array(encryption.encrypt_array(a, key), key), a)
    f = tmp_path / "case.csv"
    f.write_text("p,u\n1,2\n")
    enc = encryption.encrypt_file(str(f), key, remove_plain=True)
    assert not f.exists() and (os.stat(enc).st_mode & 0o077) == 0
    assert open(encryption.decrypt_file(enc, key)).read() == "p,u\n1,2\n"


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_scan_flags_code_execution_and_passes_plain_weights(tmp_path):
    torch = pytest.importorskip("torch")
    good = tmp_path / "good.pt"
    torch.save(torch.nn.Linear(2, 1).state_dict(), good)
    r = model_security.scan_checkpoint(str(good))
    assert r.safe and r.weights_only_loadable, r.to_dict()
    assert set(model_security.safe_load(str(good))) == {"weight", "bias"}

    evil = tmp_path / "evil.pt"
    torch.save({"state_dict": {}, "hook": _Evil()}, evil)        # saving does not run it
    r = model_security.scan_checkpoint(str(evil))
    assert not r.safe and any(g.endswith("system") for g in r.dangerous)
    with pytest.raises(model_security.UnsafeModelError):
        model_security.safe_load(str(evil))
    raw = tmp_path / "evil.pkl"
    raw.write_bytes(pickle.dumps(_Evil(), protocol=2))
    assert not model_security.scan_checkpoint(str(raw)).safe
    assert not model_security.scan_pickle_bytes(pickle.dumps(_Evil(), protocol=4)).safe


def test_safe_load_checks_the_expected_digest(tmp_path):
    torch = pytest.importorskip("torch")
    from pinneapple_security.integrity import sha256_file
    p = tmp_path / "w.pt"
    torch.save({"w": torch.ones(2)}, p)
    model_security.safe_load(str(p), expected_sha256=sha256_file(str(p)))
    with pytest.raises(model_security.UnsafeModelError, match="sha256"):
        model_security.safe_load(str(p), expected_sha256="0" * 64)


def test_adversarial_sensitivity_beats_random_noise():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    # steep in x0, flat in x1: the worst-case perturbation must find x0
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[10.0, 0.01]]))
        model[0].bias.fill_(1.0)
    X = torch.rand(64, 2)
    r = model_security.adversarial_sensitivity(model, X, eps=0.05, steps=10)
    assert r["max_rel_change"] > r["random_rel_change"] and r["amplification"] > 1.5


def test_cusum_catches_slow_bias_injection_not_noise():
    rng = np.random.default_rng(0)
    trusted = rng.normal(0, 0.2, 2000)
    det = anomaly.ResidualCUSUM().fit(trusted)
    clean = det.run(rng.normal(0, 0.2, 300))
    assert clean["n_alarms"] <= 1
    attack = rng.normal(0, 0.2, 600)
    attack[300:] += np.linspace(0, 0.4, 300)            # slow ramp, under 1σ for the first ~150 samples
    r = det.run(attack)
    before = [a["t"] for a in r["alarms"] if a["t"] < 300]
    after = [a["t"] for a in r["alarms"] if a["t"] >= 300]
    assert len(before) <= 1                                # in-control ARL ~465 samples at k=0.5, h=5
    assert after and after[0] < 450


def test_mass_balance_residual_and_replay():
    rng = np.random.default_rng(1)
    q_in = 10 + rng.normal(0, 0.05, 500)
    q_out = q_in + rng.normal(0, 0.05, 500)
    r = anomaly.balance_residual([q_in], [q_out])
    det = anomaly.ResidualCUSUM().fit(r[:200])
    spoofed = q_out.copy()
    spoofed[350:] -= 0.3          # a spoofed outlet meter hides a 3 % loss
    assert det.run(anomaly.balance_residual([q_in], [spoofed])[200:])["first_alarm"] is not None
    live = rng.normal(0, 1, 400)
    replayed = np.concatenate([live, live[100:150]])
    assert anomaly.replay_score(replayed, window=50)["suspicious"]
    assert not anomaly.replay_score(rng.normal(0, 1, 450), window=50)["suspicious"]

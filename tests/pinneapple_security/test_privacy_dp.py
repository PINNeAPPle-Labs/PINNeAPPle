"""PII detection (CPF/CNPJ/card/IBAN validated), pseudonymisation, k-anonymity, log sanitising, differential privacy."""
import math

import numpy as np
import pytest

from pinneapple_security import dp, privacy


def test_brazilian_ids_and_checksums():
    assert privacy.valid_cpf("529.982.247-25") and not privacy.valid_cpf("529.982.247-24")
    assert not privacy.valid_cpf("111.111.111-11")
    assert privacy.valid_cnpj("11.222.333/0001-81") and not privacy.valid_cnpj("11.222.333/0001-80")
    assert privacy.luhn_ok("4111 1111 1111 1111") and not privacy.luhn_ok("4111 1111 1111 1112")
    assert privacy.iban_ok("GB82 WEST 1234 5698 7654 32") and not privacy.iban_ok("GB82 WEST 1234 5698 7654 33")


def test_find_and_redact_pii_only_valid_ids():
    t = ("Contato: ana.souza@empresa.com.br, tel +55 11 98765-4321, CPF 529.982.247-25, "
         "CNPJ 11.222.333/0001-81, cartão 4111 1111 1111 1111, servidor 10.0.0.12. "
         "Medida 529.982.247-24 não é CPF; run 12345678901 também não.")
    kinds = [f.kind for f in privacy.find_pii(t)]
    assert kinds == ["email", "phone", "cpf", "cnpj", "card", "ipv4"]
    r = privacy.redact(t)
    assert "[CPF]" in r and "[EMAIL]" in r and "529.982.247-24" in r and "ana.souza" not in r


def test_pseudonymize_is_keyed_and_deterministic():
    k1, k2 = b"a" * 32, b"b" * 32
    assert privacy.pseudonymize("well-17", k1) == privacy.pseudonymize("well-17", k1)
    assert privacy.pseudonymize("well-17", k1) != privacy.pseudonymize("well-17", k2)
    assert privacy.pseudonymize("well-17", k1) != privacy.pseudonymize("well-18", k1)
    with pytest.raises(ValueError):
        privacy.pseudonymize("x", b"short")


def test_k_anonymity_reports_small_groups():
    rows = [{"plant": "A", "region": "SP", "y": i} for i in range(6)] + [{"plant": "B", "region": "RJ", "y": 1}]
    r = privacy.k_anonymity(rows, ["plant", "region"], k=5)
    assert r["k"] == 1 and not r["ok"] and r["small_groups"] == [{"values": {"plant": "B", "region": "RJ"}, "size": 1}]
    assert r["rows_at_risk_pct"] == pytest.approx(100 / 7)


def test_sanitize_solver_log():
    log = ("Build  : v2306\nHost   : node17.cluster.internal\nCase   : /home/jsilva/clientX/pipe_bend\n"
           "nProcs : 8\nmailto jsilva@clientx.com from 192.168.1.20\nC:\\Users\\jsilva\\run\nTime = 0.1\n")
    clean, n = privacy.sanitize_log(log, extra=["clientX"])
    for leaked in ("jsilva", "node17", "clientX", "192.168.1.20", "clientx.com"):
        assert leaked.lower() not in clean.lower(), leaked
    assert "Time = 0.1" in clean and "nProcs : 8" in clean and n["user_path"] == 2


def test_laplace_and_gaussian_noise_scale():
    rng = np.random.default_rng(0)
    x = np.array([dp.laplace_mechanism(0.0, 1.0, 0.5, rng) for _ in range(20000)])
    assert np.mean(np.abs(x)) == pytest.approx(2.0, rel=0.05)          # Laplace scale b = Δ/ε, E|X| = b
    s = dp.gaussian_sigma(1.0, 0.5, 1e-5)
    assert s == pytest.approx(math.sqrt(2 * math.log(1.25e5)) / 0.5)
    with pytest.raises(ValueError):
        dp.gaussian_sigma(1.0, 2.0, 1e-5)


def test_dp_mean_and_histogram():
    x = np.random.default_rng(1).uniform(0, 10, 10000)
    m = dp.dp_mean(x, 0, 10, epsilon=1.0, rng=2)
    assert abs(m - x.mean()) < 0.05
    h = dp.dp_histogram(x, np.linspace(0, 10, 11), epsilon=1.0, rng=3)
    assert (h >= 0).all() and abs(h.sum() - 10000) < 50


def test_rdp_accountant_known_limits_and_monotonicity():
    a = dp.RDPAccountant().step(q=1.0, sigma=2.0, steps=1)
    assert a.rdp[0] == pytest.approx(2 / (2 * 4.0))                    # q=1: Gaussian RDP α/(2σ²) at α=2
    e1 = dp.RDPAccountant().step(0.01, 1.1, 1000).epsilon(1e-5)["epsilon"]
    e2 = dp.RDPAccountant().step(0.01, 1.1, 4000).epsilon(1e-5)["epsilon"]
    e3 = dp.RDPAccountant().step(0.01, 2.0, 1000).epsilon(1e-5)["epsilon"]
    assert 0 < e3 < e1 < e2
    # sub-sampling amplifies privacy: far less than the same steps without sampling
    assert e1 < dp.RDPAccountant().step(1.0, 1.1, 1000).epsilon(1e-5)["epsilon"] / 10


def test_dpsgd_clips_adds_noise_and_still_learns():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    X = torch.rand(512, 2)
    y = (X[:, :1] * 2 - X[:, 1:] + 0.5)
    model = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    eng = dp.DPSGD(model, opt, lambda out, t: ((out - t) ** 2).mean(dim=1), noise_multiplier=0.8,
                   max_grad_norm=1.0, sample_rate=64 / 512, seed=0)
    base = float(((model(X) - y) ** 2).mean())
    for _epoch in range(15):
        for i in range(0, 512, 64):
            info = eng.step(X[i:i + 64], y[i:i + 64])
    after = float(((model(X) - y) ** 2).mean())
    assert after < 0.3 * base
    eps = eng.epsilon(1e-5)
    assert 0 < eps["epsilon"] < 20 and 0 <= info["clipped_fraction"] <= 1

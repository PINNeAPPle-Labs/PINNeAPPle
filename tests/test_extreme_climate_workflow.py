"""The extreme-climate workflow (#1198): GEV / GPD fits against Coles (2001), profile intervals, non-stationary
selection, coverage of the intervals on synthetic samples, copula and attribution against a known truth."""
import numpy as np
import pytest
from scipy import stats

from pinneapple_analysis.extremes import (
    block_maxima,
    decluster_runs,
    extreme_climate_workflow,
    fit_copula,
    fit_gev,
    fit_gpd,
    joint_exceedance,
    load_case,
)
from pinneapple_analysis.extremes.copula import gumbel_copula_sample
from pinneapple_analysis.extremes.models import gev_sample
from pinneapple_analysis.extremes.workflow import coverage_study

FAST = dict(gof_bootstrap=49, attribution_bootstrap=100, compound_bootstrap=100, calibration_replicates=0)


def test_port_pirie_matches_coles_and_scipy():
    d = load_case("portpirie")
    f = fit_gev(d["values"])
    c, se = f.coefficients(), f.standard_errors()
    # Coles (2001) §3.4.1: mu 3.87 (0.028), sigma 0.198 (0.020), xi -0.050 (0.098), nll -4.34
    assert abs(c["mu0"] - 3.87) < 0.01 and abs(c["sigma"] - 0.198) < 0.001 and abs(c["xi"] + 0.050) < 0.001
    assert abs(se["mu0"] - 0.028) < 0.001 and abs(se["sigma"] - 0.020) < 0.001 and abs(se["xi"] - 0.098) < 0.001
    assert abs(f.nll + 4.34) < 0.01
    # 100-year level 4.69, delta-method interval [4.38, 5.00]
    z = f.return_level(100)
    s = f.return_level_se(100)
    assert abs(z - 4.69) < 0.01 and abs(z - 1.96 * s - 4.38) < 0.01 and abs(z + 1.96 * s - 5.00) < 0.01
    lo, hi = f.profile_return_level(100)
    assert lo < z < hi and hi - z > z - lo               # the profile interval is skewed up, as in the book
    cs, loc, sc = stats.genextreme.fit(d["values"])
    assert abs(loc - c["mu0"]) < 1e-3 and abs(sc - c["sigma"]) < 1e-3 and abs(-cs - c["xi"]) < 1e-3


def test_fremantle_trend_and_soi_as_in_coles():
    d = load_case("fremantle")
    z, cov = d["values"], d["covariates"]
    f0 = fit_gev(z)
    f1 = fit_gev(z, cov, mu=["year"], start=f0)
    f2 = fit_gev(z, cov, mu=["year", "soi"], start=f1)
    # Coles (2001) §6.3.1: nll -43.6 -> -49.9 (trend 0.00203 m/yr) -> -53.9 with the SOI (0.055)
    assert abs(f0.nll + 43.6) < 0.05 and abs(f1.nll + 49.9) < 0.05 and abs(f2.nll + 53.9) < 0.05
    assert abs(f1.coefficients()["mu_year"] - 0.00203) < 1e-5
    assert abs(f2.coefficients()["mu_soi"] - 0.055) < 1e-3


def test_rain_gpd_matches_coles():
    r = load_case("rain")["values"]
    f = fit_gpd(r, 30.0)
    # Coles (2001) §4.4.1: 152 exceedances, sigma 7.44 (0.958), xi 0.184 (0.101), 100-year level 106.3
    assert f.n_exceed == 152
    assert abs(f.sigma - 7.44) < 0.01 and abs(f.xi - 0.184) < 0.001
    se = np.sqrt(np.diag(f.cov))
    assert abs(se[0] - 0.958) < 0.002 and abs(se[1] - 0.101) < 0.001
    assert abs(f.return_level(100 * 365) - 106.3) < 0.1


def test_declustering_and_block_maxima():
    x = np.array([0, 5, 6, 0, 0, 7, 0, 9, 8, 0], float)
    d = decluster_runs(x, 4.0, run=1)
    assert d["n_exceed"] == 5 and d["n_clusters"] == 3 and list(d["peaks"]) == [6, 7, 9]
    d2 = decluster_runs(x, 4.0, run=2)
    assert d2["n_clusters"] == 2 and list(d2["peaks"]) == [6, 9]
    bm = block_maxima(np.arange(10.0), np.repeat([1, 2], 5))
    assert list(bm["maximum"]) == [4.0, 9.0]


@pytest.mark.parametrize("xi", [0.1, -0.2])
def test_intervals_cover_at_the_nominal_rate(xi):
    """200 synthetic samples of 50 maxima: the 95 % intervals of the parameters (Wald) and of the 100-year level
    (profile likelihood) hold the truth at a rate inside the binomial 99.7 % band [0.904, 0.996]."""
    r = coverage_study(10.0, 2.0, xi, 50, replicates=200, seed=3)
    lo, hi = r["band"]
    for k, v in r["coverage"].items():
        assert lo <= v <= hi, (k, v, r["band"])


def test_gumbel_copula_recovers_theta_and_joint_probability():
    rng = np.random.default_rng(0)
    s = gumbel_copula_sample(2000, 2.0, rng)
    assert abs(stats.kendalltau(s[:, 0], s[:, 1]).statistic - 0.5) < 0.03
    f = fit_copula(s[:, 0], s[:, 1])
    assert f.family == "gumbel" and abs(f.param - 2.0) < 0.12
    exact = 1 - 2 * 0.99 + np.exp(-((2 * (-np.log(0.99)) ** 2.0) ** 0.5))
    assert abs(float(joint_exceedance(f, 0.99, 0.99)) - exact) / exact < 0.1


def test_synthetic_case_finds_the_trend_the_copula_and_the_probability_ratio():
    r = extreme_climate_workflow(dataset="synthetic", **FAST)
    assert r["passed"], [c for c in r["checks"] if c["status"] == "fail"]
    assert "gmst" in r["fit"]["mu_covariates"]
    a = r["attribution"]
    lo, hi = a["probability_ratio_interval"]
    assert lo <= a["probability_ratio_true"] <= hi and lo > 1
    assert r["compound"]["copula"]["family"] == "gumbel"
    codes = {c["code"] for c in r["checks"]}
    assert {"synthetic.probability_ratio", "synthetic.copula", "gof.anderson_darling"} <= codes


@pytest.mark.parametrize("case", ["portpirie", "fremantle", "rain"])
def test_bundled_cases_pass_every_check(case):
    r = extreme_climate_workflow(dataset=case, **FAST)
    assert r["passed"], [c for c in r["checks"] if c["status"] == "fail"]
    assert any(c["code"].startswith("published.") for c in r["checks"])
    rl = r["return_levels"]
    assert all(a["level"] < b["level"] for a, b in zip(rl, rl[1:]))


def test_csv_input_daily_dates(tmp_path):
    rng = np.random.default_rng(1)
    days = np.arange("1980-01-01", "2020-01-01", dtype="datetime64[D]")
    x = rng.gumbel(20, 4, len(days))
    p = tmp_path / "daily.csv"
    p.write_text("date,tmax\n" + "".join(f"{d},{v:.3f}\n" for d, v in zip(days, x)))
    r = extreme_climate_workflow(csv_path=str(p), value_column="tmax", time_column="date", **FAST)
    assert r["extraction"]["blocks"] == 40
    assert r["fit"]["distribution"] == "GEV"
    with pytest.raises(ValueError):
        extreme_climate_workflow(csv_path=str(p), value_column="nope")


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        extreme_climate_workflow(dataset="nope")
    with pytest.raises(ValueError):
        extreme_climate_workflow(dataset="portpirie", return_periods=[1.0], **FAST)
    with pytest.raises(ValueError):
        fit_gev([1.0, 2.0])
    with pytest.raises(ValueError):
        fit_gpd(gev_sample(100, 0, 1, 0.1, np.random.default_rng(0)), 1e9)


def test_the_planner_proposes_the_workflow_in_portuguese_and_english():
    import pinneapple_flow.nodes_ext  # noqa: F401
    from pinneapple_flow.planner.extensions import match
    for text in ("quero os níveis de retorno de 100 anos com GEV e atribuição",
                 "extreme value analysis of annual maxima: return levels and return periods"):
        assert match(text)[0][1].id == "climate.extremes"

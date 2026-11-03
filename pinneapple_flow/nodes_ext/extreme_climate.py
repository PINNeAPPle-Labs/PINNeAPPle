"""The standard extreme-climate workflow (:mod:`pinneapple_analysis.extremes.workflow`, #1198): block maxima or
peaks over a threshold with declustering, GEV / GPD by maximum likelihood with profile-likelihood intervals,
non-stationary location and scale selected by likelihood ratio, return levels, compound events through a copula,
a probability-ratio attribution, QQ / return-level / Anderson-Darling diagnostics and a coverage study."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("climate.extremes", "Extreme climate: GEV/GPD return levels, non-stationarity, compound events, attribution",
      category="Time series", outputs=[Port("metrics", "metrics")],
      params=[Param("dataset", "enum", "fremantle", choices=["fremantle", "portpirie", "rain", "synthetic"],
                    help="bundled case (Coles 2001 data, or a synthetic warming case with known truth)"),
              Param("csv_path", "str", "", help="instead of a bundled case: a CSV with the series"),
              Param("value_column", "str", "", help="CSV column of the variable"),
              Param("time_column", "str", "", help="CSV column of the year (or ISO dates for daily data)"),
              Param("covariates", "list", [], help="covariate columns tried in the GEV location and scale "
                                                   "(empty: every covariate of the bundled case)"),
              Param("second_column", "str", "", help="second variable of a compound event (one value per block)"),
              Param("method", "enum", "auto", choices=["auto", "block", "pot"]),
              Param("threshold", "float", 0.0, minimum=0.0, help="POT threshold (0: the threshold-stability rule)"),
              Param("run", "int", 1, minimum=0, help="runs declustering: gap that separates clusters (0: none)"),
              Param("return_periods", "list", [2, 5, 10, 20, 50, 100]),
              Param("level", "float", 0.95, minimum=0.5, maximum=0.999),
              Param("lr_alpha", "float", 0.05, minimum=1e-4, maximum=0.5, help="LR-test level of a covariate"),
              Param("factual", "dict", {}, help="covariate values of the factual climate (attribution)"),
              Param("counterfactual", "dict", {}, help="covariate values of the counterfactual climate"),
              Param("event", "float", 0.0, help="event magnitude for the attribution (0: the largest observation)"),
              Param("gof_bootstrap", "int", 199, minimum=0, advanced=True),
              Param("attribution_bootstrap", "int", 300, minimum=0, advanced=True),
              Param("compound_bootstrap", "int", 200, minimum=0, advanced=True),
              Param("calibration_replicates", "int", 100, minimum=0, advanced=True,
                    help="synthetic samples of the coverage study (0: skip)"),
              Param("seed", "int", 0, advanced=True)],
      calls="pinneapple_analysis.extremes.workflow.extreme_climate_workflow")
def climate_extremes(ctx, inputs, params):
    from pinneapple_analysis.extremes.plots import diagnostic_figure
    from pinneapple_analysis.extremes.workflow import as_tables, extreme_climate_workflow
    r = extreme_climate_workflow(
        dataset=str(params["dataset"]), csv_path=str(params.get("csv_path") or ""),
        value_column=str(params.get("value_column") or ""), time_column=str(params.get("time_column") or ""),
        covariates=list(params.get("covariates") or []), second_column=str(params.get("second_column") or ""),
        method=str(params["method"]), threshold=float(params["threshold"]), run=int(params["run"]),
        return_periods=[float(t) for t in params["return_periods"]], level=float(params["level"]),
        lr_alpha=float(params["lr_alpha"]), factual=dict(params.get("factual") or {}) or None,
        counterfactual=dict(params.get("counterfactual") or {}) or None, event=float(params["event"]),
        gof_bootstrap=int(params["gof_bootstrap"]), attribution_bootstrap=int(params["attribution_bootstrap"]),
        compound_bootstrap=int(params["compound_bootstrap"]),
        calibration_replicates=int(params["calibration_replicates"]), seed=int(params["seed"]))
    f = r["fit"]
    ctx.metric("data", r["data"]["name"], r["data"]["description"])
    ctx.metric("distribution", f["distribution"], r["method"])
    for k, v in f["coefficients"].items():
        ctx.metric(f"fit.{k}", v, f"+- {f['standard_errors'].get(k, float('nan')):.3g} (delta method)")
    lo, hi = f["shape_profile_interval"]
    ctx.metric("fit.xi_profile_interval", [lo, hi], f"{r['level']:.0%} profile likelihood")
    for row in r["return_levels"]:
        ctx.metric(f"return_level.{row['T']:g}", row["level"],
                   f"profile [{row['profile'][0]:.4g}, {row['profile'][1]:.4g}], delta [{row['delta'][0]:.4g}, "
                   f"{row['delta'][1]:.4g}]")
    t = as_tables(r)
    ctx.table("return levels", ["T (years)", "level", "delta interval", "profile interval"], t["return_levels"])
    if "selection" in t:
        ctx.table("non-stationarity (likelihood ratio)", ["model", "nll", "parameters", "deviance", "p", "kept"],
                  t["selection"])
    if "compound" in t:
        cp = r["compound"]["copula"]
        ctx.metric("compound.copula", cp["family"], f"param {cp['param']:.3g}, Kendall tau {cp['kendall_tau']:.3g}, "
                                                   f"upper-tail dependence {cp['upper_tail_dependence']:.3g}")
        ctx.table("compound events (both exceed their T-year level)",
                  ["T", "P(both)", "joint return period", "interval", "if independent"], t["compound"])
    a = r.get("attribution")
    if a and "probability_ratio" in a:
        ctx.metric("attribution.probability_ratio", a["probability_ratio"],
                   f"[{a['probability_ratio_interval'][0]:.3g}, {a['probability_ratio_interval'][1]:.3g}] for "
                   f"z > {a['event']:.4g}, {a['factual']} vs {a['counterfactual']}")
        ctx.metric("attribution.intensity_change", a["intensity_change"],
                   f"{a['return_period']:g}-year level, [{a['intensity_change_interval'][0]:.3g}, "
                   f"{a['intensity_change_interval'][1]:.3g}]")
    g = r["diagnostics"]
    ctx.metric("gof.anderson_darling", g["anderson_darling"], f"bootstrap p = {g['ad_p_value']:.3f}")
    if "calibration" in r:
        c = r["calibration"]
        ctx.table("coverage study", ["quantity", "coverage", "nominal", "band"],
                  [[k, round(v, 4), c["nominal"], f"[{c['band'][0]:.3f}, {c['band'][1]:.3f}]"]
                   for k, v in c["coverage"].items()])
    ctx.table("checks", ["check", "status", "detail"], t["checks"])
    ctx.figure(diagnostic_figure(r), "extremes_diagnostics", "QQ plot and return-level curve")
    ctx.metric("passed", r["passed"])
    ctx.summary(r["summary"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Extreme climate: return levels, non-stationarity, compound events and attribution"
    fb = FlowBuilder(title, "Block maxima or peaks over a threshold, GEV / GPD by maximum likelihood with "
                            "profile-likelihood intervals, covariates in the location and scale selected by "
                            "likelihood ratio, return levels, a copula for compound events, a probability-ratio "
                            "attribution, QQ / return-level / Anderson-Darling diagnostics and a coverage study "
                            "(pinneapple_analysis.extremes.workflow).")
    fb.add("extremes", "climate.extremes", dict(params), label="Extreme-value analysis")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("extremes", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="climate.extremes", title="Extreme climate (standard workflow)", domain="climate",
        keywords={"extreme value": 3, "valor extremo": 3, "valores extremos": 3, "return level": 3,
                  "nível de retorno": 3, "níveis de retorno": 3, "período de retorno": 3, "return period": 3, "gev": 3, "gpd": 3,
                  "generalized extreme value": 3, "peaks over threshold": 3, "picos acima do limiar": 3,
                  "block maxima": 3, "máximos anuais": 2, "annual maxima": 2, "clima extremo": 3,
                  "extreme climate": 3, "eventos extremos": 2, "extreme events": 2, "compound event": 2,
                  "evento composto": 2, "attribution": 1, "atribuição": 1, "cheia": 1, "flood": 1,
                  "onda de calor": 1, "heatwave": 1},
        min_score=3, nodes=("climate.extremes",), build=lambda text, det: build_flow(),
        inputs=("a station or gridded series (annual maxima or daily values) or a CSV", "covariates (time, "
                "global temperature, an index)", "the return periods"),
        uncited="the method follows Coles (2001), An Introduction to Statistical Modeling of Extreme Values, cited "
                "in the library module; it is not in the flow reference file yet",
        description="pinneapple_analysis.extremes.workflow.extreme_climate_workflow"))


_register_capabilities()

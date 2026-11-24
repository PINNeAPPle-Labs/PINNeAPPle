"""The digital twin end to end (#84): a simulated sensor stream with a known truth, and a real public gauge.

**The simulated stream** is the heated channel -- two temperature sensors around a fouling wall conductance -- with
process noise, a known drift of that conductance and an injected spike. The truth is known by construction, so every
check the workflow makes is *scored*: NIS against its chi-square band, NEES over independent plants, the drift
detected within the stated delay, the inverse workflow recovering the parameter, the anomaly flagged, the forecast's
coverage of the readings it never saw. That is the demonstration the issue asks for, and it is synthetic: the script
says so, in the output and in the experiment record.

**The real gauge** is USGS site 01646500, Potomac River near Washington, DC: discharge every 5-15 minutes from the
USGS Water Services instantaneous-values API (public domain, no key, downloaded at run time and never stored here).
The twin is a linear reservoir (baseflow recession) commissioned on the record's first half -- ``k`` and both noise
levels identified from the data, see ``pinneapple_systems.digital_twin.end_to_end.river_model`` -- then run
unassisted over the whole thing. Readings the provisional feed does not deliver stay gaps; nothing is interpolated.
The drift verdict at the end is the trust module's answer to the one question a commissioned twin cannot stop
asking: are the residuals still the residuals it was validated on?

Run: ``python examples/digital_twin_end_to_end/twin_on_real_sensors.py``
     ``python examples/digital_twin_end_to_end/twin_on_real_sensors.py --source synthetic``  (no network: the
     labelled synthetic recession instead of the gauge, which the output says is synthetic)
     ``python examples/digital_twin_end_to_end/twin_on_real_sensors.py --only river``

Outputs in ``examples/digital_twin_end_to_end/_out/``: one folder per case with the workflow's ``dashboard.html``
and the provenance page ``twin_end_to_end.html``.
"""
from __future__ import annotations

import argparse
import os

import pinneapple_lab
from pinneapple_systems.digital_twin.end_to_end import end_to_end_twin

HERE = os.path.dirname(os.path.abspath(__file__))


def _print_case(name: str, r: dict) -> None:
    o = r["origin"]
    print(f"\n== {name}: {r['model']['name']} ==")
    print(f"   readings : {r['n_steps']} at dt = {o['dt']:.6g} {o['dt_unit']}"
          + (f", {len(r['gaps'])} incomplete" if r["gaps"] else ""))
    print(f"   from     : {o['kind']} -- {o['site_name']}" + (f" ({o['url']})" if o.get("url") else ""))
    print(f"   source   : {o['source']}")
    if o.get("note"):
        print(f"   note     : {o['note']}")
    print(f"   verdict  : {r['verdict']} -- {r['summary']}")
    for c in r["checks"]:
        print(f"     [{c['status']:>12}] {c['code']}: {c['detail']}")
    d = r.get("drift") or {}
    if d.get("status") == "ok":
        print(f"   drift    : {d['verdict']} over {d['n_windows']} windows ({d['flagged']} with an alarm, "
              f"{d['retrain']} would retrain); mean |z| = {d['mean_abs_innovation']:.3g}")
        print(f"              {d['reason']}")
    else:
        print(f"   drift    : not run -- {d.get('reason')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("usgs", "synthetic"), default="usgs",
                    help="the river case: the live USGS gauge, or the labelled synthetic recession")
    ap.add_argument("--only", choices=("both", "simulated", "river"), default="both")
    ap.add_argument("--members", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    a = ap.parse_args()
    cases: dict[str, dict] = {}
    with pinneapple_lab.experiment(
            "examples/digital_twin_end_to_end",
            title="Digital twin end to end: a simulated stream with a truth, and a real gauge",
            description="The heated-channel twin (two sensors, fouling conductance, known drift and spike) scored "
                        "against its truth, and the same workflow on a linear reservoir commissioned on a live USGS "
                        "gauge, with the drift verdict of pinneapple_analysis.trust on the residuals.",
            domain="digital twins",
            methods=["ensemble transform Kalman filter", "NIS / NEES consistency", "Page CUSUM",
                     "inverse workflow recalibration", "Kolmogorov-Smirnov drift monitoring"],
            modules=["pinneapple_systems.digital_twin.end_to_end", "pinneapple_analysis.trust.drift"],
            out_dir=HERE, register=False,
            links={"guide": "docs/guides/digital_twin_end_to_end.md",
                   "script": "examples/digital_twin_end_to_end/twin_on_real_sensors.py"}) as ex:
        if a.only in ("both", "simulated"):
            r = end_to_end_twin(source="simulated", n_steps=300, members=a.members, nees_runs=20, horizon=60,
                                recalibration_window=50, out_dir=os.path.join(a.out, "simulated"))
            cases["simulated"] = r
            _print_case("simulated sensor stream (SYNTHETIC: the truth is known)", r)
        if a.only in ("both", "river"):
            src = "usgs" if a.source == "usgs" else "synthetic_river"
            r = end_to_end_twin(source=src, members=a.members, out_dir=os.path.join(a.out, "river"))
            cases["river"] = r
            label = "real USGS gauge" if src == "usgs" else "synthetic recession (SYNTHETIC)"
            _print_case(f"river ({label})", r)

        for name, r in cases.items():
            o = r["origin"]
            ex.log_metric(f"{name}.readings", r["n_steps"], "readings", f"dt {o['dt']:.6g} {o['dt_unit']}")
            ex.log_metric(f"{name}.checks_passed", sum(c["status"] == "pass" for c in r["checks"]), "checks",
                          f"{len(r['checks'])} checks, {sum(c['status'] == 'inconclusive' for c in r['checks'])} "
                          f"inconclusive")
            for c in r["checks"]:
                if c["value"] is not None:
                    ex.log_metric(f"{name}.{c['code']}", float(c["value"]), "", c["detail"])
            d = r.get("drift") or {}
            if d.get("status") == "ok":
                ex.log_metric(f"{name}.drift.mean_abs_innovation", d["mean_abs_innovation"], "",
                              f"{d['n_windows']} windows, {d['flagged']} with an alarm; {d['verdict']}")
            ex.check(f"{name}: provenance stated", bool(o.get("kind") and o.get("source")),
                     detail=f"{o['kind']}: {o['site_name']}", kind="reproducibility")
            for f in (r.get("dashboard"), r.get("origin_page")):
                if f:
                    ex.add_file(f)

        sim = cases.get("simulated")
        if sim is not None:
            ex.check("simulated stream: every truth-scored check passes", sim["verdict"] == "pass",
                     detail=", ".join(sim["failed"]) or "every check passes", kind="verification")
        river = cases.get("river")
        if river is not None:
            d = river.get("drift") or {}
            ex.check("river: the drift block ran on the residuals", d.get("status") == "ok",
                     detail=str(d.get("reason", ""))[:200], kind="verification")
            ex.check("river: no reading was invented", not any(
                c["code"] == "twin.missing_readings" and c["status"] == "fail" for c in river["checks"]),
                detail=f"{len(river['gaps'])} incomplete readings carried as gaps", kind="physics")
        print(f"\noutputs in {a.out}")


if __name__ == "__main__":
    main()

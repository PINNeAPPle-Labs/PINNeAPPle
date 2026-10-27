"""End-to-end demo: commission the HX-201 twin, put it on line, file the evidence.

Run it:  python apps/twin_ops/demo.py

What it does, in order, and what each step is *for*:

1. imports the signed contract from ``spec_hx201.yaml`` and prints its digest — the digest is
   the handle every later artifact is filed under, so "which revision did we accept" has an
   answer;
2. generates 72 h of plant history (fouling, flow wander, instrument noise) — **synthetic**,
   and the report says so;
3. runs the preflight: it must pass, because everything after it is meaningless otherwise;
4. runs the backtest against the redundant outlet element, which the filter never sees;
5. puts the twin on line over the last part of the history and shows the drift ladder;
6. writes ``twin_evidence_hx201.json`` and prints the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))      # the library, uninstalled

from pinneapple_systems.digital_twin.product import (  # noqa: E402
    TwinSpec,
    commission,
    run_live,
)
from pinneapple_systems.operations import (  # noqa: E402
    PlantConfig,
    exchanger_model,
    fouling_at,
    generate_history,
)

SPEC_PATH = HERE / "spec_hx201.yaml"
EVIDENCE_PATH = HERE / "twin_evidence_hx201.json"


def build_model(ua_clean: float = 10_200.0):
    """The model as the twin sees it: ``model(X, theta)`` with fouling as the latent state."""
    def model(X: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
        return exchanger_model(X, theta, ua_clean=ua_clean)
    return model


def main() -> int:
    spec = TwinSpec.from_yaml(SPEC_PATH)
    print(f"spec {spec.digest()}  ·  {spec.asset_id}  ·  {spec.asset_name}")
    print(f"inputs: {spec.input_names}")
    print(f"gates : RMSE ≤ {100 * spec.gates.max_rmse_frac:.1f}% of span, "
          f"coverage ≥ {100 * spec.gates.min_coverage:.0f}%, "
          f"lead ≥ {spec.gates.min_alarm_lead_s:.0f} s\n")

    history = generate_history(hours=72.0, sampling_s=spec.sampling_s)
    print(f"history: {len(history)} frames over {72:.0f} h (synthetic — clearly labelled)\n")

    model = build_model()
    evidence = commission(
        spec, model, history,
        n_ens=64, seed=0, warmup_steps=12,
        notes="demo on synthetic history; not a plant commissioning record",
    )

    print(evidence.preflight.render() if evidence.preflight else "no preflight")
    print()
    if evidence.tracking:
        print(evidence.tracking.render())
    print()
    print(f"COMMISSIONED: {'YES' if evidence.commissioned else 'NO'}\n")

    # --- on line over the last third of the history -----------------------
    tail = history[-int(len(history) * 0.35):]
    theta0 = [evidence.initial_state[n]["value"] for n in spec.field_names
              if n in evidence.initial_state] or None
    run = run_live(spec, model, tail, n_ens=64, seed=0, evidence=evidence, theta0=theta0)
    latent = run.latent
    true_rf = fouling_at(72 * 3600.0, PlantConfig())

    print("On line")
    print(f"  steps {run.n_steps} · final mode {run.final_mode.value} · "
          f"rolling coverage {run.coverage:.3f}" if run.coverage is not None else
          f"  steps {run.n_steps} · final mode {run.final_mode.value}")
    print(f"  modes seen: {run.modes_seen}")
    for name, v in latent.items():
        print(f"  {name} = {v['value']:.6f} ± {v['std']:.6f} {v['unit']}"
              f"   (true {true_rf:.6f})")
    if run.decisions:
        print(f"  {len(run.decisions)} transition(s)")
        for d in run.decisions[:5]:
            print(f"    {d['from']} -> {d['to']} at coverage {d['coverage']}")

    print()
    if evidence.unverified:
        print("Not verified")
        for u in evidence.unverified:
            print(f"  - {u}")

    evidence.save(str(EVIDENCE_PATH))
    print(f"\nevidence written to {EVIDENCE_PATH.relative_to(HERE.parent.parent)}")
    return 0 if evidence.commissioned else 1


if __name__ == "__main__":
    raise SystemExit(main())

---
title: "Promote the classic CFD/FEA reproduction to an example with a Lab manifest, an inventory entry and a hub flow"
labels: [enhancement, examples]
---

**Who has it:** `experiments/toolchain_reproduction` — the script that runs a lid-driven cavity, a water hammer, a
cantilever and a plate with a hole, and checks each against its published value. It is the only place in the repository
where five solvers of four different families are held to a textbook number in one run.

**What PINNeAPPle already has:** the script itself, with `run.py` (five models, 20 checks), `make_report.py` (the PDF
summary) and `results/` (figures, `results.json`, the Lab manifest). It writes a valid `experiment.json` and uses
`pinneapple_lab.experiment`, so everything it produces is already in the Lab's schema.

**What is missing:** it lives outside `examples/`, so none of the gates see it — `scripts/check_experiment_manifests.py`
does not validate it, `pinneapple_flow.examples.inventory()` has no entry and no flow, the Hub does not list it, and
nothing runs it in CI. A reproduction that nobody re-runs stops being a reproduction.

**Proposal:**

1. Move it to `examples/toolchain_reproduction/` (or keep the folder and register it) so the manifest gate covers it.
2. Add the `pinneapple_flow` flow and its inventory note, the way the other example folders have them.
3. Run it in CI with `--smoke` (coarse meshes, ~15 s on a free machine, no GPU, no external solver).
4. Keep `--no-3d` and the full meshes for a manual run, and keep `make_report.py` as the way the PDF summary is
   produced — it reads `results.json`, so the summary cannot drift from the run.

**Acceptance (rule D3 — the repository's own gates):** `python scripts/check_experiment_manifests.py` reports no
problem; `pinneapple_flow.examples.flows()` lists the flow; `python run_all_examples.py --smoke` runs it green; the
Hub entry resolves.

**Origin:** reproducing the "blank machine + agent + open-source toolchain" experiment, 2026-11-14. The script and its
results exist; only the gates are missing.

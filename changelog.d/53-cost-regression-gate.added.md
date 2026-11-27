- The measured-cost regression gate (#53, roadmap D2), local because GitHub Actions is off for billing:
  `make cost-gate` (or `python scripts/measure_method_costs.py --gate`, `ONLY="S9 S19"` for a couple of methods)
  measures the catalog's probed methods now and compares with the versioned
  `pinneapple_catalog/method_costs_baseline.json`. **FLOPs gate strictly** — a rise of more than 5 % fails the run with
  `REGRESSION S9: flops 3.173e+07 -> 3.526e+07 (x1.11, limit 5 % (FLOPs are deterministic))` — while wall time and
  memory gate loosely at 50 %, since they move between runs on the same machine. A table measured on another machine
  is refused rather than compared: the same code on two CPUs is two measurements, not a regression. `--check` is the
  other half of the gate: every catalog method has to be measured or carry a written reason, so a new solver cannot
  slip in unaccounted for.

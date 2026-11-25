# Stirred exothermic reactor — a digital-twin benchmark case

An exothermic reactor with a three-blade impeller and five cooling tubes, reduced from the
third-party two-dimensional model issue #179 cites to a well-mixed model this repository runs in
seconds, and the digital twin that follows it through three perturbations: sensor noise, a failed
cooling tube (and a stopped impeller, which is the same loss of cooling by another route), and the
safety margin against a temperature alarm.

```
python examples/reactor_twin/reactor_twin_benchmark.py           # the full protocol, about a minute
python examples/reactor_twin/reactor_twin_benchmark.py --smoke   # four cases, one hour
```

It prints the reference's own checks, the post's two operating points against what this model does
with them, the results table (state error, latency, alarm detection) and the case-by-case table,
and writes `_out/report.md`, `_out/benchmark.json` and the figures under `_out/figures/`.

The model lives in `pinneapple_systems.digital_twin.stirred_reactor`; the guide is
[docs/guides/stirred_reactor_twin.md](../../docs/guides/stirred_reactor_twin.md) and the tests are
`tests/test_stirred_reactor_twin.py`.

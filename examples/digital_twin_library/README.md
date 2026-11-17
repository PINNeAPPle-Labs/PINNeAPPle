# Digital Twin Library — nine reference systems

FPSO and subsea production, a blast furnace, a chemical synthesis loop, a reusable booster, a
lunar base, a gas pipeline network, a compression station, a green hydrogen plant and a seawater
desalination train.

```
python examples/digital_twin_library/01_reference_systems.py           # all nine, full report
python examples/digital_twin_library/01_reference_systems.py --smoke   # two of them, quick
```

Each system prints its outputs, its conservation balances with their residuals, the invariants it
must satisfy, the limits of its own physics and the list of what was *not* verified. The runs are
also written to `_out/<system>.json` with their parameters and provenance.

The models live in `pinneapple_systems.digital_twin.library`; the guide is
[docs/guides/digital_twin_library.md](../../docs/guides/digital_twin_library.md) and the tests are
`tests/test_digital_twin_library.py`.

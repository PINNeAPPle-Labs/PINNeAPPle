# Refractory / liner wear twins (steel ladle, pig-iron ladle, BOF, RH, Oxy-Red)

Five 3D digital twins of metallurgical vessels built with `pinneapple_twin3d.wear` from a *spec*, not from
hand-written 3D code. Each shows the wear of the lining per cell over time, a remaining-life forecast and a
maintenance schedule recommendation.

```bash
python examples/use_cases/refractory_wear_twin/run_use_cases.py out/wear_twins --serve bof_converter
python examples/use_cases/refractory_wear_twin/custom_part.py out/tank    # a new part in ~25 lines
python -c "from pinneapple_twin3d import gallery; gallery.export_all('out/gallery')"   # rocket, airliner, drone, rover, ...
```

All wear data is **synthetic** (hot spots + growth law) and every output says so. Geometry is derived from the
proportions of the reference twins and is indicative. See the guide:
[Recreate a 3D wear twin](../../../docs/guides/recreate_a_wear_twin.md).

Tests: `pytest tests/test_twin3d_wear.py` (forecast, optimizer, geometry, data contract, the four use cases end to
end; set `DIGITAL_SOLUTIONS_DIR` to also check Python against the JavaScript core of the web suite).

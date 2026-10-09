"""Generate the four reference wear twins (steel ladle, pig-iron ladle, BOF, RH degasser).

    python examples/use_cases/refractory_wear_twin/run_use_cases.py [out_dir] [--serve bof_converter]

Each twin is a folder with the 3D viewer scene, ``wear_dataset.json`` (hub contract) and ``report.json``
(observed / predicted / recommended). All data here is SYNTHETIC and labelled as such.
"""
import os
import sys

from pinneapple_twin3d import serve
from pinneapple_twin3d.wear import OptimizerConfig, export_wear_twin, presets, synthetic_campaign

HEATS_PER_DAY = {"steel_ladle": 24, "pig_iron_ladle": 12, "bof_converter": 30, "rh_degasser": 20, "oxyred_reactor": 6}


def build(out_dir: str, key: str) -> str:
    spec, hotspots = presets.get(key)
    extra = {}
    if key == "pig_iron_ladle":  # one gunning at heat 60 in the slag line, to exercise repair detection
        extra = dict(repairs=(60,), repair_zones=("linha_escoria",))
    ds = synthetic_campaign(spec, hotspots, n_readings=41, x_max=120, **extra)
    cfg = OptimizerConfig(horizon=300, heats_per_day=HEATS_PER_DAY[key])
    return export_wear_twin(ds, os.path.join(out_dir, key), cfg=cfg)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "out/wear_twins"
    for k in sorted(presets.PRESETS):
        print("wrote", build(out, k))
    if "--serve" in sys.argv:
        serve(os.path.join(out, sys.argv[sys.argv.index("--serve") + 1]))

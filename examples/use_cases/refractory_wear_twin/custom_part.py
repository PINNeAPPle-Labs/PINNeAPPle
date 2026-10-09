"""Recreate a twin for a NEW part in ~25 lines: a tundish-like tank with a slag band and an impact pad.

    python examples/use_cases/refractory_wear_twin/custom_part.py out/tank
"""
import sys

from pinneapple_twin3d.wear import HotSpot, VesselSpec, ZoneSpec, export_wear_twin, forecast, optimize, synthetic_campaign


def tank_spec() -> VesselSpec:
    r, h = 1.2, 2.0  # inner radius and height [m]
    return VesselSpec("tank", "Cylindrical tank", (
        # a wall zone: profile (r, y) from bottom to top, 12 rows x 24 sectors, 200 mm new, 80 mm minimum
        ZoneSpec("wall", "Wall", ((r, 0.0), (r, h)), 12, 24, 200, 80, group="wall"),
        # a floor zone: profile along r at fixed y
        ZoneSpec("floor", "Floor", ((0.0, 0.0), (r, 0.0)), 5, 24, 250, 100, group="floor"),
    ), x_name="batch")


def build(out: str) -> str:
    spec = tank_spec()
    hot = (HotSpot("wall", 0.7, None, 0.1, amp=0.6),          # a band all around at 70 % of the height
           HotSpot("floor", 0.4, 90, 0.3, 35, amp=0.7))      # an impact pad at 90 deg, 40 % of the radius
    ds = synthetic_campaign(spec, hot, n_readings=25, x_max=80)
    print({k: round(v.min_remaining, 1) for k, v in forecast(ds).items()}, "->", optimize(forecast(ds)).recommended.name)
    return export_wear_twin(ds, out)


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "out/tank"))

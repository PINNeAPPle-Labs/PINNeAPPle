"""Stage 6 of the Ahmed-body study: Blender Cycles stills and video, the three.js viewer and an OpenUSD stage.

    python examples/end_to_end/ahmed_body/06_visuals.py [--quick]

Stills of the 25 deg validation run (Cp with streamlines from the front and the rear, mid-plane speed, wake
vorticity) on a dark stage; a video (an orbit of that CFD run, then the slant swept 0-40 deg through the Cp
surrogate); ``results/visuals/viewer`` (three.js) and ``ahmed_body.usda`` with the trust decision. Needs ``bpy``.
"""
import json
import os
import sys

from pinneapple_autolab.domains import ahmed_study as S

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "_runs", "cfd"), os.path.join(HERE, "results")
quick = "--quick" in sys.argv
out = S.visuals(S.load_records(RUNS, ok_only=True), RUNS, RES, samples=16 if quick else 64,
                size=(960, 540) if quick else (1600, 900), frames_orbit=12 if quick else 48,
                frames_sweep=9 if quick else 41)
print(json.dumps({k: v for k, v in out.items() if k != "usd"}, indent=1), out.get("usd"))
os._exit(0)

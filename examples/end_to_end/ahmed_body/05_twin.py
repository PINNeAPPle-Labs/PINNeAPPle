"""Stage 5 of the Ahmed-body study: the digital-twin page (static copy in ``results/twin``).

    python examples/end_to_end/ahmed_body/05_twin.py

Sliders for slant, clearance and speed; the selected Cp model on the body in 3-D (mean or 90 % half width); CD and
CL with their calibrated band; a warning beyond 0.25 (unit cube) from the nearest CFD run. The verifying-CFD button
needs the app: ``cd apps/ahmed_twin && uvicorn ahmed_twin.api:app --port 8171``.
"""
import os

from pinneapple_autolab.domains import ahmed_study as S

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "_runs", "cfd"), os.path.join(HERE, "results")
if not os.path.isdir(RUNS):                       # a fresh checkout: the committed copy of the records
    RUNS = os.path.join(RES, "cfd_runs")
print(S.twin(S.load_records(RUNS, ok_only=True), RES, os.path.join(RES, "twin")))
os._exit(0)

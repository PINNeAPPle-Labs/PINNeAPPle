- `pinneapple_studio`, the library side of the PINNeAPPle Studio (`pinneapple studio`, `python -m pinneapple_studio`):
  - every public class and function of the library as a tool, classified from the code as executable, composite or
    reference, with the reason;
  - a form generated from the signature (units from the docstring, nested dataclasses, numeric arrays);
  - status from the tests (verified / tested / discovered, the Engineering API's rules), the Portuguese prose of the
    map, tasks, areas and products, and related tools from the import graph;
  - presets and science experiments as tools;
  - search in Portuguese and English;
  - a gallery of every example, reproduction and app, with the requirements read from its scripts and the fix for
    each one that is missing;
  - a job runner (own process, no credentials in the environment, limits, evidence record) and a FastAPI router.

  `docs/pt/_mapa/inventario.py` now uses `pinneapple_studio.inventory`. The Flow Studio's routes are a reusable
  `APIRouter` (`create_router`), and `pinneapple_flow.studio.serve` takes extra routers.

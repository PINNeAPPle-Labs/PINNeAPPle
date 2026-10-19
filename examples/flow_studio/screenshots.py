"""Screenshots of Flow Studio for the docs (docs/flow_studio/): the canvas of a planned flow, a node's result panel
with the three.js viewer, the PINN node's architecture and validation, the Lab seal, the decision records and the
equation editor.

Start the studio first (``python -m pinneapple_flow studio --no-browser``), then::

    python examples/flow_studio/screenshots.py [--url http://127.0.0.1:8765] [--chromium /path/to/chrome]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[2] / "docs" / "flow_studio"


def wait_done(page, timeout_ms=2_400_000):
    """Every node in an end state (done, cached, degraded, not_run, failed, cancelled)."""
    page.wait_for_function("""() => [...document.querySelectorAll('[data-testid=node-status]')]
                              .every(e => ['done','cached','degraded','not_run','failed','cancelled']
                                          .includes(e.textContent))""", timeout=timeout_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--chromium", default="/opt/pw-browsers/chromium")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.chromium if Path(a.chromium).exists() else None)
        page = b.new_page(viewport={"width": 1600, "height": 960}, device_scale_factor=1)
        page.goto(a.url)
        page.wait_for_selector("[data-testid=palette-section-problem]")
        # 0. the palette: favourites, sections in workflow order, search, an unavailable node with its reason
        for t in ("train.pinn", "solver.classical", "validate.compare"):
            page.hover(f"[data-testid='palette-{t}']")
            page.click(f"[data-testid='palette-star-{t}']")
        page.wait_for_selector("[data-testid=palette-favourites] [data-testid='palette-validate.compare']")
        page.mouse.move(900, 500)
        page.eval_on_selector("[data-testid=palette] .overflow-y-auto", "e => e.scrollTop = 0")
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "flow_studio_palette.png"), clip={"x": 0, "y": 0, "width": 760, "height": 960})
        page.fill("[data-testid=palette-search]", "kalman filter")
        page.wait_for_selector("[data-testid=palette-results] [data-testid='palette-assimilate.filter']")
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "flow_studio_palette_search.png"),
                        clip={"x": 0, "y": 0, "width": 760, "height": 960})
        page.fill("[data-testid=palette-search]", "")
        off = page.locator("[data-testid=palette-section-solvers] [data-available=no]")
        if off.count():
            off.first.scroll_into_view_if_needed()
            off.first.hover()
            page.wait_for_selector("[data-testid=palette-tooltip]")
            page.screenshot(path=str(OUT / "flow_studio_palette_unavailable.png"),
                            clip={"x": 0, "y": 0, "width": 760, "height": 960})
        page.mouse.move(900, 500)
        # 1. the planner and the canvas it fills
        page.click("[data-testid=open-planner]")
        page.fill("[data-testid=planner-text]", "Steady heat conduction in a square plate with internal heat "
                                                "generation and the edges held at 0 degrees; compare finite elements,"
                                                " finite differences and a PINN. Then switch the plate from 2D to 3D.")
        page.click("[data-testid=planner-go]")
        page.wait_for_selector("[data-testid=proposal]")
        page.screenshot(path=str(OUT / "flow_studio_planner.png"))
        page.click("[data-testid=proposal-open] >> nth=0")
        page.wait_for_selector("[data-testid=planner-badge]")
        page.click(".react-flow__node:has-text('equations')")
        page.wait_for_selector("[data-testid=eq-preview]")
        page.screenshot(path=str(OUT / "flow_studio_canvas.png"))
        # 2. run it and open the results
        page.click("[data-testid=run]")
        wait_done(page)
        page.click("[data-testid='node-3-D viewer']")
        page.click("[data-testid=tab-results]")
        page.wait_for_selector("[data-testid=viewer3d] canvas", timeout=60_000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "flow_studio_result_3d.png"))
        page.click("[data-testid='node-Validation']")
        page.wait_for_selector("[data-testid=figure]")
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "flow_studio_result_validation.png"))
        page.click("[data-testid='node-PINN']")
        page.wait_for_selector("[data-testid=layer-graph]")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "flow_studio_result_pinn.png"))
        # the Lab seal of the validated item and what the next letter needs
        page.click("[data-testid='node-Lab seal']")
        page.click("[data-testid=tab-results]")
        page.wait_for_selector("[data-testid=seal-panel]")
        page.click("[data-testid=seal-next-steps]")
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "flow_studio_lab_seal.png"))
        # decision records: one node's, then the flow's
        page.click(".react-flow__node:has-text('equations')")
        page.click("[data-testid=tab-decisions]")
        page.wait_for_selector("[data-testid=decision]")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "flow_studio_decisions_node.png"))
        page.click("[data-testid=open-decisions]")
        page.wait_for_selector("[data-testid=decisions-report]")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "flow_studio_decisions.png"))
        page.keyboard.press("Escape")
        page.mouse.click(5, 5)
        # 3. the architecture editor of a surrogate
        page.click("[data-testid=open-lists]")
        page.click("[data-testid=example-planned_hybrid_surrogate_hpo]")
        page.wait_for_selector(".react-flow__node")
        page.click(".react-flow__node:has-text('HPO')")
        page.click("[data-testid=tab-params]")
        page.wait_for_selector("[data-testid=layer-graph]")
        page.click("[data-testid=add-block]")
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "flow_studio_architecture.png"))
        b.close()
    print(f"screenshots in {OUT}")


if __name__ == "__main__":
    main()

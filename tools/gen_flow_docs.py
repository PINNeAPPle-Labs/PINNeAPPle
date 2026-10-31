"""MkDocs gen-files script: the generated Flow Studio pages, written into the site at build time (never committed).

    flow_studio/COVERAGE.md            python -m pinneapple_flow coverage -o docs/flow_studio/COVERAGE.md
    flow_studio/EXAMPLES_AS_FLOWS.md   python -m pinneapple_flow example-flows doc
    studio/COVERAGE.md                 python -m pinneapple_studio coverage -o docs/studio/COVERAGE.md

Both need the package importable (``pip install -e ".[flow]"``). Without it a local ``mkdocs serve`` gets a page saying
how to generate them; with ``PINNEAPPLE_DOCS_STRICT=1`` (the docs and test workflows) a failure fails the build.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
STRICT = os.environ.get("PINNEAPPLE_DOCS_STRICT") == "1"


def _coverage() -> str:
    from pinneapple_flow.coverage import markdown
    return markdown()


def _examples() -> str:
    from pinneapple_flow.examples import markdown
    return markdown()


def _studio() -> str:
    from pinneapple_studio import catalog, coverage, gallery
    return coverage.markdown(catalog.load(), gallery.load())


PAGES = {
    "flow_studio/COVERAGE.md": (_coverage, "python -m pinneapple_flow coverage -o docs/flow_studio/COVERAGE.md"),
    "flow_studio/EXAMPLES_AS_FLOWS.md": (_examples, "python -m pinneapple_flow example-flows doc"),
    "studio/COVERAGE.md": (_studio, "python -m pinneapple_studio coverage -o docs/studio/COVERAGE.md"),
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for page, (render, command) in PAGES.items():
    try:
        text = render()
    except Exception:
        if STRICT:
            raise
        traceback.print_exc()
        if (ROOT / "docs" / page).is_file():      # generated earlier with the command: mkdocs serves that file
            continue
        text =(f"# {Path(page).stem}\n\nThis page is generated when the docs are built with PINNeAPPle installed "
                f"(`pip install -e \".[flow]\"`). Generate it locally with `{command}`.\n")
    with mkdocs_gen_files.open(page, "w") as f:
        f.write(text)

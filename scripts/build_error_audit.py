#!/usr/bin/env python3
"""Write ``docs/dev/errors_that_teach.md`` from ``pinneapple_core.errors.AUDITED`` (#46, roadmap A5).

The table is data in the library, not prose in a document, so the page is generated instead of maintained by hand:
``tests/test_errors_that_teach.py`` fails when the two disagree.

    python scripts/build_error_audit.py           # write the page
    python scripts/build_error_audit.py --check   # exit 1 if the page is stale (the gate)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pinneapple_core.errors import audit_markdown  # noqa: E402

OUT = ROOT / "docs" / "dev" / "errors_that_teach.md"

if __name__ == "__main__":
    text = audit_markdown()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is stale: run python scripts/build_error_audit.py")
            sys.exit(1)
        print(f"{OUT.relative_to(ROOT)} matches the audit table")
        sys.exit(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")

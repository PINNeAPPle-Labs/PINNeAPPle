"""``python -m pinneapple_systems.digital_twin.library`` — the registry's status check.

The command lives here rather than in :mod:`.scenarios` so that running it does not import the
module twice: ``python -m pkg.mod`` imports the package (which imports ``mod``) and then executes
``mod`` again as ``__main__``, and Python warns about it. A ``__main__.py`` is imported once.

    python -m pinneapple_systems.digital_twin.library --check
"""
from __future__ import annotations

import sys

from .scenarios import _main

if __name__ == "__main__":  # pragma: no cover - thin entry point
    sys.exit(_main())

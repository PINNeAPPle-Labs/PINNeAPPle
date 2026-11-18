"""``python -m pinneapple_intent``: compile, run and replay a study from the shell."""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

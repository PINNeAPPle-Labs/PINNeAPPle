"""The repository root goes on ``sys.path`` for the whole suite.

Several test modules import things that are part of this checkout but are not installed: the
``benchmarks`` package, ``pinneapple_app``, and the ``examples`` trees they read. pytest puts the
rootdir on ``sys.path`` only when a conftest lives there, and this file is that conftest, so the
same suite passes from a checkout and, where a test needs a file that only exists in a checkout,
that test can say so.

Nothing is added when the interpreter is already importing the package from somewhere else and the
checkout is not the current directory: the path is this file's folder, which is the checkout by
construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

"""``python -m pinneapple_unity`` -> :func:`pinneapple_unity.cli.main`."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

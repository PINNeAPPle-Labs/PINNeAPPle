"""Write the generated copies of the house theme, and check that they still match ``theme.json`` (#65).

``pinneapple_tools/visualization/theme.json`` is the single source. The browser cannot read it directly -- the
viewers are static files, and ``pp.viz``'s single-file page inlines its modules -- so the theme is copied into the
three places that need it in their own language:

    pinneapple_tools/visualization/studio/web/theme.js   imported by studio-core.js (studio viewer, the apps)
    pinneapple_twin3d/viewer/theme.js                    imported by viewer.js (twin3d)
    apps/_shared/static/shared/theme.css                 CSS custom properties, imported by app.css

    python scripts/build_theme.py            # rewrite them from theme.json
    python scripts/build_theme.py --check    # fail if one has drifted (this is the gate)

The generated files are checked in so a page needs no fetch to draw, and ``--check`` is what keeps "one theme file"
true: a colour changed in ``theme.json`` and not rebuilt is a consumer drawing yesterday's palette, which is exactly
the drift the issue was opened for.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinneapple_tools.visualization.theme import generated  # noqa: E402


def drift() -> list[str]:
    """The generated files that do not match what ``theme.json`` says today."""
    out = []
    for rel, content in generated().items():
        path = ROOT / rel
        if not path.is_file():
            out.append(f"{rel}: missing — run python scripts/build_theme.py")
        elif path.read_text(encoding="utf-8") != content:
            out.append(f"{rel}: does not match theme.json — run python scripts/build_theme.py")
    return out


def write() -> list[Path]:
    written = []
    for rel, content in generated().items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_theme", description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report drift and change nothing")
    a = ap.parse_args(argv)
    if a.check:
        problems = drift()
        for line in problems:
            print(line)
        print(f"{len(problems)} problem(s)")
        return 1 if problems else 0
    for path in write():
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

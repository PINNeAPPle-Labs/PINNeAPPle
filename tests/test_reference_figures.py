"""What `scripts/reference/build.py` is allowed to delete, and what it must not touch (#67, #625).

The reference folder holds two kinds of file: the text artefacts the build derives from the source (`index.html`,
`data/*.json`) and the figures, which belong to `run_examples.py` — the image and, next to it, the caption sidecar the
figure carries. The build prunes images no example lists any more; it cannot regenerate a caption, so it must leave
those alone. It did not, once: `build.py` deleted fifteen sidecars that had just landed (#2344) and the caption test
stayed red until the cause was found.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "reference"))

import build  # noqa: E402


def _folder(tmp_path: Path) -> Path:
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in ("used.png", "orphan.png", "used.json", "orphan.json"):
        (figures / name).write_bytes(b"x")
    return figures


def test_an_image_no_example_lists_is_removed(tmp_path):
    figures = _folder(tmp_path)
    build.prune_figures(str(tmp_path), {"used.png"})
    assert not (figures / "orphan.png").exists()


def test_an_image_an_example_lists_stays(tmp_path):
    figures = _folder(tmp_path)
    build.prune_figures(str(tmp_path), {"used.png"})
    assert (figures / "used.png").exists()


def test_a_caption_sidecar_is_never_removed(tmp_path):
    """The point of the guard: build.py cannot regenerate a caption, so it does not get to delete one."""
    figures = _folder(tmp_path)
    build.prune_figures(str(tmp_path), {"used.png"})
    assert (figures / "used.json").exists() and (figures / "orphan.json").exists()


def test_a_missing_figures_folder_is_not_an_error(tmp_path):
    build.prune_figures(str(tmp_path), set())              # nothing to prune, nothing to say

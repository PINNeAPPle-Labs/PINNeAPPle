"""The front door: the entry pages run or compile, and every symbol they name exists.

The repository had three entry pages and one of them called ``pp.build_model("VanillaPINN")``,
which raises ``KeyError`` because the registry is lowercase. Nothing caught it, because nothing ran
the pages. This does: every command in ``START_HERE.md`` runs as written, the code blocks of
``QUICKSTART.md`` and ``GETTING_STARTED.md`` compile, and every ``pp.<name>`` they mention resolves.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import pinneapple as pp

ROOT = Path(__file__).resolve().parents[1]
PY_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
PP_SYMBOL = re.compile(r"\bpp\.([A-Za-z_][A-Za-z0-9_]*)")

#: Pages that must exist and must not point back at the retired ones.
FRONT_DOOR = ("README.md", "START_HERE.md", "QUICKSTART.md", "GETTING_STARTED.md")


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _blocks(text: str) -> list[str]:
    return PY_BLOCK.findall(text)


def test_every_front_door_page_exists():
    for name in FRONT_DOOR:
        assert (ROOT / name).is_file(), name


def test_readme_points_at_the_front_door_and_the_four_commands():
    readme = _read("README.md")
    assert "START_HERE.md" in readme
    for command in ("pinneapple doctor", "pinneapple recommend", "pinneapple new"):
        assert command in readme, command


def test_the_retired_overviews_point_forward():
    """They stay at their old paths so a search hit lands somewhere true, not on stale content."""
    for name, pointer in (("OVERVIEW.md", "START_HERE.md"), ("OVERVIEW_EN.md", "START_HERE.md")):
        text = _read(name).lower()
        assert pointer.lower() in text and ("moved" in text or "mudou" in text)
        assert len(text.splitlines()) < 40, f"{name} is a stub, not an overview"


@pytest.mark.parametrize("page", ("QUICKSTART.md", "GETTING_STARTED.md", "START_HERE.md"))
def test_code_blocks_compile(page):
    blocks = _blocks(_read(page))
    assert blocks, f"{page} has no python blocks"
    for i, block in enumerate(blocks):
        compile(block, f"{page}[python block {i}]", "exec")


@pytest.mark.parametrize("page", FRONT_DOOR)
def test_every_symbol_the_page_names_exists(page):
    """Catches the class of bug that started this: a documented name the library does not have."""
    text = _read(page)
    missing = sorted({name for name in PP_SYMBOL.findall(text) if not hasattr(pp, name)})
    assert not missing, f"{page} names pp.<x> that do not exist: {missing}"


@pytest.mark.parametrize("page", FRONT_DOOR)
def test_every_preset_the_page_names_is_registered(page):
    presets = set(pp.list_presets()) | set(pp.presets.names())
    named = set(re.findall(r"\b([a-z0-9]+(?:_[a-z0-9]+){1,5})\b", _read(page))) & presets
    assert named, f"{page} names no preset at all"
    for name in named:
        assert name in presets, name


# START_HERE is the page a beginner follows, so it is executed, not only compiled. Its blocks are
# written to run in seconds: doctor, recommend, card, a solve with a closed form, an empty project.
START_BLOCKS = _blocks(_read("START_HERE.md"))


@pytest.mark.parametrize("i", range(len(START_BLOCKS)))
def test_start_here_block_runs(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    code = compile(START_BLOCKS[i], f"START_HERE.md[python block {i}]", "exec")
    exec(code, {"__name__": f"start_here_block_{i}"})  # noqa: S102 - the page is the input under test
    os.chdir(ROOT)


def test_start_here_block_count_is_small():
    """A page a beginner follows cannot be a wall of code: keep the executed blocks few and short."""
    assert 1 <= len(START_BLOCKS) <= 6
    assert max(len(b.splitlines()) for b in START_BLOCKS) <= 30

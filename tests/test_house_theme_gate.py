"""Every example that draws a figure is under the house theme, and stays there (#918).

The acceptance for the issue is "every figure added from now on uses ``use_theme``, enforced by the figure gate".
``scripts/apply_house_theme.py`` is both halves: the codemod that came before, and the gate that reads the files.
These tests hold the gate — that it is clean today, that it fails for a new figure outside the theme, and that the
migration it performs is an edit somebody can review: valid Python, sorted imports, no line that only restates the
theme, and nothing inserted with trailing whitespace.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_house_theme as theme  # noqa: E402


def test_every_example_figure_is_under_the_house_theme():
    problems = theme.problems()
    assert not problems, "examples drawing with matplotlib defaults:\n" + "\n".join(problems)


def test_the_gate_notices_a_new_figure_outside_the_theme(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "novo.py").write_text("import matplotlib.pyplot as plt\nplt.savefig('a.png')\n")
    assert theme.problems(examples, tmp_path / ".house_theme_baseline") == [
        "examples/novo.py: draws with matplotlib and neither use_theme nor use_cfd_style is called"]


def test_the_gate_notices_a_baseline_line_that_is_already_fixed(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "novo.py").write_text("from pinneapple_tools.visualization.style import use_theme\nuse_theme()\n")
    baseline = tmp_path / ".house_theme_baseline"
    baseline.write_text("# migrado\nexamples/novo.py\n")
    assert any("already themed but still listed" in p for p in theme.problems(examples, baseline))


def test_a_file_that_disappeared_does_not_stay_in_the_baseline(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    baseline = tmp_path / ".house_theme_baseline"
    baseline.write_text("examples/sumiu.py\n")
    assert theme.problems(examples, baseline) == [
        "examples/sumiu.py: listed in the baseline but the file is gone"]


def _migrated(tmp_path: Path, source: str) -> str:
    path = tmp_path / "exemplo.py"
    path.write_text(source)
    assert theme.migrate(path)
    return path.read_text()


def test_a_module_level_import_gets_the_theme_after_the_import_block(tmp_path):
    text = _migrated(tmp_path, "import matplotlib.pyplot as plt\nimport numpy as np\n\nplt.savefig('a.png')\n")
    assert text == ('import matplotlib.pyplot as plt\nimport numpy as np\n'
                    'from pinneapple_tools.visualization.style import use_theme\n\n'
                    'use_theme("report")   # one theme for every figure of the library (#918)\n'
                    "\nplt.savefig('a.png')\n")
    ast.parse(text)


def test_a_pyplot_import_inside_a_try_block_gets_the_theme_there(tmp_path):
    """The theme call has to sit where matplotlib was imported, or a missing matplotlib turns into a missing theme."""
    text = _migrated(tmp_path, "try:\n    import matplotlib\n    import matplotlib.pyplot as plt\n"
                                "    HAS = True\nexcept ImportError:\n    HAS = False\n")
    assert ("    import matplotlib.pyplot as plt\n\n"
            "    from pinneapple_tools.visualization.style import use_theme\n"
            '    use_theme("report")   # one theme for every figure of the library (#918)\n'
            "    HAS = True") in text
    ast.parse(text)


def test_an_import_that_spans_lines_is_not_cut_in_half(tmp_path):
    """The first version pasted the theme lines after the line the ``from`` was on, inside a parenthesised import."""
    text = _migrated(tmp_path, "import matplotlib.pyplot as plt\n"
                                "from pinneapple_design.geometry.csg import (\n"
                                "    CSGRectangle,\n    CSGCircle,\n)\n"
                                "plt.savefig('a.png')\n")
    assert "    CSGCircle,\n)\nfrom pinneapple_tools" in text
    compile(text, "exemplo.py", "exec")


def test_the_inserted_lines_carry_no_trailing_whitespace(tmp_path):
    text = _migrated(tmp_path, "def figure():\n    import matplotlib.pyplot as plt\n    plt.savefig('a.png')\n")
    assert not [line for line in text.splitlines() if line != line.rstrip()]


def test_an_rc_update_that_only_repeats_the_theme_is_dropped(tmp_path):
    text = _migrated(tmp_path, "def figure():\n    import matplotlib.pyplot as plt\n"
                                '    plt.rcParams.update({"font.size": 10, "axes.spines.top": False})\n'
                                "    plt.savefig('a.png')\n")
    assert "rcParams.update" not in text


def test_an_rc_update_that_chooses_something_is_kept(tmp_path):
    text = _migrated(tmp_path, "def figure():\n    import matplotlib.pyplot as plt\n"
                                '    plt.rcParams.update({"font.size": 14, "lines.linewidth": 3})\n'
                                "    plt.savefig('a.png')\n")
    assert 'plt.rcParams.update({"font.size": 14, "lines.linewidth": 3})' in text


def test_migrating_twice_changes_nothing(tmp_path):
    path = tmp_path / "exemplo.py"
    path.write_text("import matplotlib.pyplot as plt\nplt.savefig('a.png')\n")
    assert theme.migrate(path)
    once = path.read_text()
    assert theme.migrate(path) is False
    assert path.read_text() == once


def test_a_module_that_does_not_draw_is_left_alone(tmp_path):
    path = tmp_path / "sem_figura.py"
    path.write_text("import numpy as np\nprint(np.zeros(3))\n")
    assert theme.migrate(path) is False
    assert path.read_text() == "import numpy as np\nprint(np.zeros(3))\n"


def test_the_redundant_table_is_what_the_theme_actually_sets():
    """``_REDUNDANT`` is a copy of the theme's rc; if the theme changes, the copy has to change with it."""
    from pinneapple_tools.visualization.style import _theme_rc

    rc = _theme_rc("report", False)
    for key, value in theme._REDUNDANT.items():
        assert key in rc, f"{key} is not in the theme: the migration cannot drop it"
        assert rc[key] == value, f"the theme sets {key}={rc[key]!r}, the table says {value!r}"


#: a few real examples that already carry the theme, to check the codemod on real code and not only on fixtures
_THEMED = [p for p in sorted((ROOT / "examples").glob("**/*.py")) if theme.themed(p)][:8]


@pytest.mark.parametrize("path", _THEMED, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_the_migration_is_idempotent_on_a_real_example(path):
    """A themed example is left byte for byte alone — the codemod must never fight a hand edit."""
    before = path.read_text()
    assert theme.migrate(path) is False
    assert path.read_text() == before

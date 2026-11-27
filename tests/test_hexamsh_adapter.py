from pathlib import Path
from unittest.mock import patch

import pytest

from pinneapple_design.fitmesh.hexamsh import generate_hex_mesh


def test_missing_executable(tmp_path):
    cad = tmp_path / "part.step"
    cad.touch()
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Python 3.12"):
            generate_hex_mesh(cad, tmp_path / "out.msh")


def test_reject_wrong_cad(tmp_path):
    cad = tmp_path / "part.stl"
    cad.touch()
    with pytest.raises(ValueError, match="STEP"):
        generate_hex_mesh(cad, tmp_path / "out.msh")


def test_isolated_launch(tmp_path):
    cad = tmp_path / "part.step"
    cad.touch()
    dest = tmp_path / "out.msh"

    def run(args, **kwargs):
        work = Path(kwargs["cwd"])
        assert (work / cad.name).exists()
        assert work != tmp_path
        (work / "mesh.msh").touch()

    with patch("shutil.which", return_value="/bin/hexamsh"), patch(
        "subprocess.run", side_effect=run
    ), patch("pinneapple_design.fitmesh.hexamsh.import_hex_mesh", return_value="ok"):
        assert generate_hex_mesh(cad, dest) == "ok"

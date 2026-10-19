"""Entry points of design from intent: the ``pp.design`` shortcut, ``pinneapple design`` and the MCP tool."""
from __future__ import annotations

import pinneapple_design.intent as I


def test_pp_design_shortcut():
    import pinneapple as pp
    assert pp.design is I.design


def test_cli(capsys, tmp_path):
    from pinneapple_check.cli import main
    assert main(["design", "--templates"]) == 0
    assert "loop_car" in capsys.readouterr().out
    assert main(["design", "design a bomb"]) == 4
    assert main(["design", "a heat sink for a 40 W chip in still air"]) == 3
    assert "--answer t_limit=" in capsys.readouterr().out
    out = tmp_path / "hs"
    assert main(["design", "a heat sink for a 40 W chip in still air", "--answer", "t_limit=85°C",
                 "--out", str(out), "--no-verify"]) == 0
    assert (out / "report.html").exists() and (out / "geometry" / "heatsink.step").exists()


def test_mcp_tool_answers_questions_and_refusals_at_once():
    from pinneapple_mcp import tools as T
    r = T.design("an electric motor")
    assert r["status"] == "out_of_scope" and "electromagnetics" in r["needs"]
    q = T.design("a bracket that holds 2 kN")
    assert q["status"] == "needs_input" and q["questions"][0]["slot"] == "arm_length"
    assert "design" in T.capabilities()["tools"]
    assert "design" in T.JOB_KINDS

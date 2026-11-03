"""study_steps.table_step (#1171): an uploaded table as named columns with the units its header states, and the
``series`` / ``measurements`` the time-series and identification steps take. A unit is never guessed."""
from __future__ import annotations

import pytest

from pinneapple_flow.study_steps import split_header, table_step


@pytest.mark.parametrize("header,expected", [
    ("x_m", ("x", "m")), ("p_Pa", ("p", "Pa")), ("T_K", ("T", "K")), ("k_1_s", ("k", "1/s")),
    ("tensao_MPa", ("tensao", "MPa")), ("t_h", ("t", "h")), ("T [K]", ("T", "K")), ("p (bar)", ("p", "bar")),
    ("MWh", ("MWh", "MWh")), ("dia", ("dia", "d")),
    ("a_rms_g", ("a_rms_g", None)), ("deformacao", ("deformacao", None)),        # g or gram? none: never guessed
])
def test_headers_state_their_units(header, expected):
    assert split_header(header) == expected


def test_two_columns_give_the_series_and_the_measurements(tmp_path):
    p = tmp_path / "energia.csv"
    p.write_text("dia,MWh\n" + "".join(f"{d},{50 + d % 7}\n" for d in range(30)))
    r = table_step(str(p), x="dia", y="MWh")
    assert r["rows"] == 30 and r["units"] == {"dia": "d", "MWh": "MWh"} and r["series"][:2] == [50.0, 51.0]
    assert r["measurements"]["inputs"]["dia"][:3] == [0.0, 1.0, 2.0] and len(r["measurements"]["values"]) == 30
    assert [(c["code"], c["status"]) for c in r["checks"]] == [("dataset.complete", "pass"), ("dataset.units", "pass")]


def test_a_multi_column_table_with_gaps_and_text_and_no_unit(tmp_path):
    p = tmp_path / "planta.tsv"
    rows = "".join(f"{h}\t{'AB'[h % 2]}\t{20 + h}\t{'' if h == 5 else 2 + h / 10}\n" for h in range(12))
    p.write_text("t_h\tunit\tT_in_C\tflow\n" + rows)                 # one gap in twelve rows
    r = table_step(str(p), x=["t_h", "T_in_C"], y="flow")
    assert r["skipped"] == ["unit"] and set(r["columns"]) == {"t_h", "T_in_C", "flow"}
    assert r["units"]["T_in_C"] == "°C" and r["units"]["flow"] is None
    status = {c["code"]: c["status"] for c in r["checks"]}
    assert status == {"dataset.complete": "fail", "dataset.units": "inconclusive"}
    with pytest.raises(ValueError, match="no numeric column"):
        table_step(str(p), y="unit")


def test_the_measurements_take_the_shape_identify_step_reads(tmp_path):
    p = tmp_path / "cinetica.csv"
    p.write_text("T_K,k_1_s\n320,0.00057\n330,0.0011\n340,0.0021\n350,0.0039\n")
    r = table_step(str(p), x="T_K", y="k_1_s", sigma=1e-4)
    assert r["measurements"] == {"inputs": {"T_K": [320.0, 330.0, 340.0, 350.0]},
                                 "values": [0.00057, 0.0011, 0.0021, 0.0039], "sigma": 1e-4}
    assert r["units"] == {"T_K": "K", "k_1_s": "1/s"}

"""The stage map: the stages are the palette's, the wiring claims are checkable, the licences say how they were read.

Every test here pins something a hand-written table gets wrong: a stage id that drifted from the Flow palette, a
"wired" claim pointing at a module that does not exist, a licence with no provenance, a tool duplicated from the
recommendation catalogue instead of referenced. None of them re-asserts the content of a row — that is the map's
business — they assert that the map cannot lie about its own shape.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from pinneapple_catalog import stages

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_the_stages_are_the_flow_palette_sections():
    from pinneapple_flow import palette

    assert stages.STAGE_IDS == tuple(palette.SECTION_IDS), (
        "the map's stages and the Flow palette's sections have drifted; one of them was edited alone")


def test_every_stage_has_a_question_except_the_palette_drawer():
    for st in stages.stages():
        if st.kind == "internal":
            assert st.question == "", f"{st.id} is internal and should ask nothing"
        else:
            assert st.question.endswith("?"), f"{st.id} has no question an engineer would ask: {st.question!r}"


def test_a_stage_id_that_is_not_a_stage_raises_instead_of_returning_nothing():
    with pytest.raises(KeyError):
        stages.stage("sovlers")
    with pytest.raises(KeyError):
        stages.datasets("sovlers")
    with pytest.raises(KeyError):
        stages.coverage("sovlers")
    with pytest.raises(ValueError):
        stages.tools(integration="wired-ish")


def test_a_wired_tool_points_at_a_file_or_package_that_exists():
    """The one claim about this repository: a module is named, so it can be opened.

    The whole dotted path has to resolve — a check that stopped at ``pinneapple_simulation`` would accept a surface
    pointing at a module that was never written, which is the failure this test exists to prevent.
    """
    missing = []
    for tool in stages.tools(integration="wired"):
        assert tool.surface, f"{tool.id} claims to be wired and names no surface"
        rel = REPO.joinpath(*tool.surface.split("."))
        if not (rel.with_suffix(".py").exists() or rel.is_dir()):
            missing.append((tool.id, tool.surface))
    assert not missing, "tools claiming a code path that is not in the tree: " + repr(missing)


def test_a_named_tool_points_somewhere_too_and_a_candidate_points_nowhere():
    for tool in stages.tools():
        if tool.integration == "candidate":
            assert not tool.surface, f"{tool.id} is a candidate but names a surface"
        else:
            assert tool.surface, f"{tool.id} is {tool.integration} but names no surface"


def test_every_tool_says_where_its_licence_was_read():
    for tool in stages.tools():
        assert tool.license, f"{tool.id} has no licence"
        assert tool.license_source, f"{tool.id} does not say where the licence was read"
        assert tool.homepage.startswith("http"), f"{tool.id} has no homepage"


def test_an_unverified_licence_never_points_at_the_github_licence_api():
    """``license_verified`` is only True when the read happened; the two must not contradict each other."""
    for tool in stages.tools():
        if not tool.license_verified:
            assert "GitHub licence API" not in tool.license_source, (
                f"{tool.id} says its licence was read through the API but is not marked verified")


def test_a_licence_that_forbids_commercial_use_is_not_counted_as_open():
    """FluidX3D is source-available and free for research, and its licence says commercial use is not allowed, so
    the row cannot answer that it is open just because it is not called proprietary."""
    assert not stages.get("fluidx3d").is_open
    assert stages.get("genesis").is_open, "an Apache-2.0 entry has to stay open for this to mean anything"


def test_the_licence_of_an_entry_that_overlaps_the_recommendation_catalogue_is_really_there():
    """No duplicated tool pages: an entry that overlaps ``TOOL_CATALOG`` names the id, and the id must exist."""
    from pinneapple_analysis.verification.tool_recommendation import TOOL_CATALOG

    overlapping = [t for t in stages.tools() if t.also_in]
    assert overlapping, "no entry overlaps TOOL_CATALOG; the reference would be untested"
    for tool in overlapping:
        assert tool.also_in in TOOL_CATALOG, f"{tool.id} points at TOOL_CATALOG['{tool.also_in}'], which is absent"


def test_every_dataset_is_tagged_and_its_tags_are_stages():
    from pinneapple_data.datasets.catalog import CATALOG

    untagged = [e.id for e in CATALOG.values() if not e.stages]
    assert not untagged, f"catalogued datasets invisible from every stage: {untagged}"
    for entry in CATALOG.values():
        unknown = [s for s in entry.stages if s not in stages.STAGE_IDS]
        assert not unknown, f"{entry.id} names stages that do not exist: {unknown}"


def test_the_datasets_a_stage_lists_are_the_catalogues_own():
    """The data half of the map is a filter over the catalogue, not a second copy of it."""
    from pinneapple_data.datasets.catalog import entries

    for st in stages.stages():
        assert [d["id"] for d in stages.datasets(st.id)] == [e.id for e in entries(stage=st.id)]


def test_coverage_counts_what_the_filters_return():
    totals = stages.coverage()["totals"]
    assert totals["tools"] == len(stages.TOOLS)
    assert totals["wired"] == len(stages.tools(integration="wired"))
    assert totals["named"] == len(stages.tools(integration="named"))
    assert totals["candidate"] == len(stages.tools(integration="candidate"))
    assert totals["wired"] + totals["named"] + totals["candidate"] == totals["tools"]
    per_stage = stages.coverage()["stages"]
    assert set(per_stage) == set(stages.STAGE_IDS)
    for stage_id, row in per_stage.items():
        assert row["tools"] == len(stages.tools(stage_id=stage_id))
        assert row["wired"] == len(stages.tools(stage_id=stage_id, integration="wired"))


def test_the_palette_drawer_carries_no_tools():
    assert stages.tools(stage_id="library") == []
    assert stages.coverage("library")["tools"] == 0


def test_the_map_is_json_serialisable_and_one_stage_matches_its_row():
    payload = stages.to_json()
    json.dumps(payload)
    assert [s["id"] for s in payload["stages"]] == list(stages.STAGE_IDS)
    one = stages.to_json("solvers")
    assert one["id"] == "solvers" and one["tools"] and one["counts"]["tools"] == len(one["tools"])


def test_a_tool_serving_several_stages_appears_in_each_of_them():
    multi = [t for t in stages.tools() if len(t.stages) > 1]
    assert multi, "no tool serves more than one stage; the stage column would be meaningless"
    tool = max(multi, key=lambda t: len(t.stages))
    for stage_id in tool.stages:
        assert tool.id in [t.id for t in stages.tools(stage_id=stage_id)]


def test_no_tool_is_catalogued_twice_under_two_names():
    names = [t.name for t in stages.tools()]
    assert len(names) == len(set(names))
    homepages = [t.homepage.rstrip("/") for t in stages.tools()]
    duplicates = {h for h in homepages if homepages.count(h) > 1}
    assert not duplicates, f"two entries share a homepage: {duplicates}"

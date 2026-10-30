"""The first hour: doctor, recommend, card and new.

These are the four things a new user meets before any physics, so they are tested on what they
promise: the snippet ``card`` prints compiles, ``recommend`` answers the descriptions a person would
actually type (English and Portuguese), ``doctor`` reports rather than guesses, and a project from
``pp.new`` runs from any directory.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import pinneapple as pp
from pinneapple_core.onboarding import catalog, cli, newproject
from pinneapple_core.onboarding.doctor import profiles as doctor_profiles

# --------------------------------------------------------------------------- card


#: A sample that carries both states: what runs here without anything installed, what needs a solver
#: or a geometry, and a preset with no solver at all. Probing every preset would run every cheap
#: solver in the library (minutes); these five are enough to hold the contract.
SAMPLE = ("burgers_1d", "poisson_2d", "cpu_heatsink_thermal", "pipe_flow_3d", "transient_heat_3d")


def test_tagged_presets_with_their_own_geometry_are_runnable(checked_cards):
    """A preset whose tags map onto its own box or cylinder needs nothing from the user.

    ``pipe_flow_3d`` and ``transient_heat_3d`` bind their boundaries to named tags and are not in the
    set the classical solver covers; they used to be reported as unusable. ``pp.solve`` now sends the
    geometry the preset already declares (pde_environment/presets/tag_geometry.py).
    """
    for name in ("pipe_flow_3d", "transient_heat_3d"):
        card = checked_cards[name]
        assert card.available, f"{name}: {card.error}"
        assert card.geometry, name
        assert card.method == "pinn", name
        assert "no mesh needed" in card.snippet, name


def test_a_tagged_preset_trains_from_its_own_geometry():
    """The claim the card makes, trained for a handful of epochs: it really produces a field."""
    sol = pp.solve("transient_heat_3d", method="pinn", epochs=3)
    assert sol.info.get("geometry", "").startswith("box")
    values = sol.predict(np.array([[0.5] * len(sol.problem.coords)]))
    assert values.shape == (1, len(sol.problem.fields))


@pytest.fixture(scope="session")
def checked_cards():
    """The sample presets as cards, with runnability probed once for the whole test session."""
    return {name: pp.card(name) for name in SAMPLE}


def test_card_describes_a_preset_and_its_snippet_compiles():
    c = pp.card("burgers_1d")
    assert c.coordinates == ("x", "t") and c.fields == ("u",)
    assert c.method in ("analytic", "reference", "pinn")
    compile(c.snippet, "<card snippet>", "exec")
    assert "pp.solve('burgers_1d'" in c.snippet
    assert "burgers_1d" in str(c)


def test_every_preset_has_a_card_with_a_compilable_snippet():
    for card in catalog.catalogue():
        assert card.name and card.description, card.name
        assert card.coordinates, card.name
        compile(card.snippet, f"<card {card.name}>", "exec"), card.name


def test_a_preset_that_does_not_run_here_offers_a_program_that_does(checked_cards):
    """Every card either produces a field on this machine or hands over something the user can run."""
    unavailable = [c for c in checked_cards.values() if not c.available]
    assert unavailable, "the sample must contain a preset that does not run here"
    for c in unavailable:
        assert c.error, f"{c.name} is unavailable without saying why"
        assert c.fallback, f"{c.name} is unavailable and offers no alternative program"
        compile(c.fallback, f"<fallback {c.name}>", "exec")
    if unavailable:
        assert any("the same physics as" in c.fallback for c in unavailable)


def test_the_fallback_says_it_is_not_the_presets_own_geometry(checked_cards):
    for c in checked_cards.values():
        if c.fallback:
            assert "the same physics as" in c.fallback, c.name
            assert "on a box" in c.fallback, c.name


def test_card_unknown_name_lists_what_exists():
    with pytest.raises(KeyError) as e:
        pp.card("nope_not_a_preset")
    assert "nope_not_a_preset" in str(e.value)


def test_domains_cover_every_preset():
    grouped = catalog.domains()
    names = {n for group in grouped.values() for n in group}
    assert set(pp.list_presets()) <= names


def test_snippet_point_lies_inside_the_domain():
    """The point in the snippet must lie inside the preset's own bounds, or the program is not runnable."""
    for name in ("burgers_1d", "poisson_2d"):
        card = pp.card(name)
        point = card.snippet.split("np.array([[")[1].split("]]")[0]
        values = [float(v) for v in point.split(",")]
        assert len(values) == len(card.coordinates)
        for coord, value in zip(card.coordinates, values, strict=True):
            lo, hi = card.domain_bounds[coord]
            assert lo <= value <= hi, (name, coord, value, (lo, hi))


# --------------------------------------------------------------------- recommend

# The descriptions a user actually types, and the preset the catalogue should reach.
QUERIES = [
    ("heat sink cooled by a fan", "cpu_heatsink_thermal"),
    ("conducao de calor em uma aleta", "cpu_heatsink_thermal"),
    ("stress in a bracket under load", "linear_elasticity_3d_industry"),
    ("tensao em uma viga", "plane_stress_2d"),
    ("escoamento em tubulacao", "pipe_flow_3d"),
    ("pressure drop in a pipe", "pipe_flow_3d"),
    ("orbita de satelite", "kepler_two_body_orbit"),
    ("crack in a plate", "material_fracture_2d"),
    ("unsteady flow past a cylinder", "channel_flow_3d"),
]


@pytest.mark.parametrize("text,expected", QUERIES)
def test_recommend_reaches_the_expected_preset(text, expected):
    """The expected preset is the choice, or the closer match that was passed over for one that runs."""
    r = pp.recommend(text)
    assert expected in (r.preset, r.swapped_from), (
        f"{text!r} -> {r.preset} (swapped from {r.swapped_from}; wanted {expected}); matched {r.matched}")
    assert r.preset is not None


def test_recommend_only_offers_a_preset_that_runs_or_a_program_that_does():
    """The promise: what comes back can produce a field. Either the preset runs here, or the program
    is the same physics written so it runs anywhere, and the preset itself is never presented as
    ready."""
    for text, _ in QUERIES:
        r = pp.recommend(text)
        card = pp.card(r.preset)
        compile(r.snippet, f"<recommend {r.preset}>", "exec")
        if not card.available:
            assert r.snippet == card.fallback, text
            assert r.method == "pinn", text
            compile(r.snippet, f"<fallback {r.preset}>", "exec")


def test_recommend_explains_itself_and_prints_a_runnable_program():
    r = pp.recommend("heat sink cooled by a fan")
    assert r.matched and r.reason and r.confidence > 0
    compile(r.snippet, "<recommend snippet>", "exec")
    text = r.explain()
    for fragment in ("preset", "method", "sure", "run it now", "next"):
        assert fragment in text


def test_recommend_says_so_when_nothing_matches():
    r = pp.recommend("my foot hurts")
    assert r.preset is None and r.confidence == 0.0
    assert "no preset is close" in r.explain()
    compile(r.snippet, "<no match snippet>", "exec")


def test_recommend_prefers_training_when_the_user_asks_to_train():
    r = pp.recommend("train a PINN on steady heat conduction in 3d")
    assert r.method == "pinn"
    assert "epochs" in r.snippet


def test_recommend_needs_a_description():
    with pytest.raises(ValueError):
        pp.recommend("   ")


def test_recommend_model_is_always_in_the_registry():
    for text, _ in QUERIES:
        assert pp.recommend(text).model in pp.list_models()


# ------------------------------------------------------------------------ doctor


def test_doctor_reports_and_finds_nothing_missing_on_the_core_profile():
    report = pp.doctor("start", verbose=False)
    assert report.ok, [c.name for c in report.failures]
    assert any(c.name == "pinneapple" for c in report.checks)


def test_doctor_json_round_trips():
    report = pp.doctor("start", verbose=False)
    payload = report.as_dict()
    assert payload["profile"] == "start" and isinstance(payload["checks"], list)
    assert all({"name", "ok", "detail", "fix", "required"} <= set(c) for c in payload["checks"])


def test_doctor_rejects_an_unknown_profile():
    with pytest.raises(ValueError) as e:
        pp.doctor("astrology", verbose=False)
    assert "astrology" in str(e.value)


def test_doctor_profiles_are_the_documented_ones():
    assert set(doctor_profiles()) == {"start", "cfd", "research", "enterprise", "all"}


def test_doctor_never_claims_a_check_it_did_not_run():
    """A check that cannot run is ``None``, not ``ok``: an unverified installation is not a pass."""
    for check in pp.doctor("start", verbose=False).checks:
        assert check.ok in (True, False, None)
        if check.ok is False:
            assert check.fix, f"{check.name} failed without telling the user how to fix it"


# --------------------------------------------------------------------------- new


def test_new_plan_writes_one_entry_point_and_a_case_file_per_template():
    for template, entry in (("plain", "run.py"), ("research", "study.py"), ("cfd", "sweep.py")):
        spec = newproject.plan("demo", template=template, preset="poisson_2d")
        assert entry in spec["files"]
        assert "case.yaml" in spec["files"] and "README.md" in spec["files"]
        for name, content in spec["files"].items():
            if name.endswith(".py"):
                compile(content, f"<{template}/{name}>", "exec")


def test_new_refuses_a_name_that_is_not_a_slug(tmp_path):
    with pytest.raises(ValueError):
        pp.new("Not A Slug", dir=str(tmp_path), verbose=False)


def test_new_refuses_to_overwrite(tmp_path):
    pp.new("demo", dir=str(tmp_path), verbose=False)
    with pytest.raises(FileExistsError):
        pp.new("demo", dir=str(tmp_path), verbose=False)


def test_generated_project_runs_from_any_directory(tmp_path):
    """The promise of the template: one command, from anywhere, and a field at the end."""
    pp.new("demo", dir=str(tmp_path), template="plain", preset="poisson_2d", verbose=False)
    project = tmp_path / "demo"
    assert (project / "case.yaml").exists()

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = subprocess.run([sys.executable, str(project / "run.py")], cwd=elsewhere,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (project / "results" / "solution.npz").exists()
    assert (project / "results" / "report.md").exists()
    data = np.load(project / "results" / "solution.npz")
    assert data["values"].size and data["X"].shape[1] == 2


def test_research_project_records_metrics_with_their_unit_and_how(tmp_path):
    pp.new("study", dir=str(tmp_path), template="research", preset="burgers_1d", verbose=False)
    result = subprocess.run([sys.executable, str(tmp_path / "study" / "study.py")],
                            cwd=str(tmp_path), capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    import json

    payload = json.loads((tmp_path / "study" / "results" / "metrics.json").read_text())
    assert payload["preset"] == "burgers_1d"
    assert payload["metrics"]
    for metric in payload["metrics"]:
        assert metric["unit"] is not None and metric["how"], metric


# --------------------------------------------------------------------------- cli


def test_cli_lists_the_four_commands():
    assert cli.main([]) == 2
    assert cli.main(["--help"]) == 0


def test_cli_card_and_recommend_exit_zero(capsys):
    assert cli.main(["card", "burgers_1d"]) == 0
    assert "burgers_1d" in capsys.readouterr().out
    assert cli.main(["recommend", "heat sink cooled by a fan"]) == 0
    assert "cpu_heatsink_thermal" in capsys.readouterr().out


def test_cli_card_json_is_machine_readable(capsys):
    import json

    assert cli.main(["card", "burgers_1d", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "burgers_1d" and payload["fields"] == ["u"]


def test_cli_doctor_quiet_uses_the_exit_code():
    assert cli.main(["doctor", "start", "-q"]) == 0


def test_cli_recommend_without_text_is_a_usage_error(capsys):
    assert cli.main(["recommend"]) == 2


def test_cli_new_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["new", "demo", "--dry-run"]) == 0
    assert not (tmp_path / "demo").exists()
    assert "would create" in capsys.readouterr().out

# ------------------------------------------------------------------ the preset set

def test_most_presets_produce_a_field_without_extra_work():
    """A floor under the onboarding promise, checked without running any solver.

    Before the preset geometry was wired into ``pp.solve``, 15 of 72 presets could produce a field
    here. The number is not the point; that it must not silently fall back is. This counts what the
    library can reach today: a closed form, a solver this machine has, or a preset geometry it can
    train with.
    """
    from pinneapple_core.onboarding import catalog as catalog_mod
    from pinneapple_physics.solving import analytic_for

    names = pp.list_presets()
    reachable = [n for n in names
                 if analytic_for(n)
                 or catalog_mod.solver_available(catalog_mod._solver_of(n))
                 or catalog_mod.trainable(n)]
    assert len(reachable) >= 60, f"only {len(reachable)} of {len(names)} presets are reachable"


def test_a_preset_knows_which_preset_it_is():
    """The solver needs the registry key to find a preset's geometry fixtures."""
    spec = pp.get_preset("poisson_2d")
    assert spec.problem_id == "poisson_2d"
    assert spec.meta["preset"] == "poisson_2d"
    spec = pp.get_preset("burgers_1d", nu=0.02)          # with arguments, still identifiable
    assert spec.problem_id == "burgers_1d"
    problem = pp.PhysicalProblem.from_preset("transient_heat_3d")
    assert problem.to_pde_spec().meta["preset"] == "transient_heat_3d"

"""The plugin system (#100) and its diagnostics (#1474).

The plugins here are not installed: each test hands :class:`pinneapple_plugins.Entry` a loader, which is exactly how
an installed distribution resolves its entry point. So what runs is the real discovery-free half of the loader —
import, contract, dependencies, configuration, initialize — and the diagnostics built on it.

The two acceptance lines these tests exist for:

* #100 — a third-party package registers a solver/model and it shows up in ``pp.list_models()`` and in the benchmark
  without a change to the core;
* #1474 — an installed-but-broken plugin is listed with the reason while every other plugin still shows up, and
  ``doctor`` tells "not installed" from "installed but failing" from "disabled".
"""
from __future__ import annotations

import functools
import json
import sys
import textwrap

import pytest

from pinneapple_plugins import (
    CONTRACT_VERSION,
    AmbiguousCapability,
    Entry,
    InvalidDeclaration,
    Plugin,
    PluginContext,
    PluginManager,
    UnknownPlugin,
    check_declaration,
    contract_compatible,
    satisfies,
)
from pinneapple_plugins import cli as plugins_cli


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture
def config_file(tmp_path):
    return tmp_path / "plugins.json"


@pytest.fixture
def manager(config_file):
    """A manager with no installed plugins at all: each test adds the entries it needs."""
    return PluginManager(entries=[], config=config_file)


def _module(tmp_path, monkeypatch, name: str, source: str):
    """Write a module on disk, import it, and make it importable by name for the rest of the test."""
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    sys.modules.pop(name, None)
    return importlib.import_module(name)


class TinySolver(Plugin):
    name = "tiny"
    version = "0.1.0"
    contract_version = CONTRACT_VERSION
    capabilities = {"solver": ["cfd", "steady_state"], "geometry": ["stl", "step"]}
    permissions = ("read", "run")

    def initialize(self, context):
        context.log("tiny initialized")
        self.context = context

    def shutdown(self):
        self.closed = True


# ------------------------------------------------------------------------------------------------ the contract
def test_contract_compatibility_names_both_versions():
    assert contract_compatible("1.0")[0]
    assert contract_compatible("1.3", "1.4")[0], "a plugin older than the running contract is compatible"
    ok, why = contract_compatible("1.5", "1.2")
    assert not ok and "1.5" in why and "1.2" in why
    ok, why = contract_compatible("2.0")
    assert not ok and "different contract" in why


def test_specifier_subset_and_declaration_rules():
    assert satisfies("0.8.0", ">=0.8")
    assert not satisfies("0.7.9", ">=0.8")
    assert satisfies("0.8.4", ">=0.8,<0.9")
    assert not satisfies("0.9.0", ">=0.8,<0.9")
    assert satisfies("1.2.3", "~=1.2")
    with pytest.raises(InvalidDeclaration):
        satisfies("1.0", "=>1.0")
    good = check_declaration(TinySolver())
    assert good["name"] == "tiny" and good["capabilities"]["solver"] == ["cfd", "steady_state"]

    class Nameless(Plugin):
        name = ""

    with pytest.raises(InvalidDeclaration, match="non-empty `name`"):
        check_declaration(Nameless())

    class FlatCapabilities(Plugin):
        name = "flat"
        capabilities = {"solver": "cfd"}

    with pytest.raises(InvalidDeclaration, match="must be a list of strings"):
        check_declaration(FlatCapabilities())


# ------------------------------------------------------------------------------------------------ #100
def test_a_plugin_registers_a_model_visible_in_list_models_and_the_benchmark(tmp_path, monkeypatch):
    """#100: a third-party package registers, and the library lists it — no change to the core."""
    module = _module(tmp_path, monkeypatch, "pp_plugin_demo", """
        import torch
        import pinneapple as pp

        class DemoSolver(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class DemoMethod:
            id = "plugin_demo_method"

        class Demo(pp.plugins.Plugin):
            name = "demo"
            version = "1.2.0"
            capabilities = {"model": ["pointwise_coords"], "benchmark": ["demo"]}
            registers = {"model": ["plugin_demo_net"]}      # what initialize adds through pp.register

            def initialize(self, context):
                pp.register(DemoSolver, kind="model", name="plugin_demo_net", family="plugin")
                from pinneapple_analysis.benchmark import register_method
                register_method("plugin_demo_method")(DemoMethod)

        PLUGIN = Demo()
    """)
    import pinneapple as pp
    from pinneapple_analysis.benchmark.registry import _METHODS
    from pinneapple_neural.architectures.registry import ModelRegistry

    entry = Entry(name="demo", value="pp_plugin_demo:PLUGIN", distribution="pinneapple-demo", version="1.2.0",
                  loader=lambda: module.PLUGIN)
    manager = PluginManager(entries=[entry], config=tmp_path / "plugins.json", log=lambda line: None)
    try:
        manager.load_all()
        assert manager.records()["demo"].state == "ready"
        monkeypatch.setattr("pinneapple_plugins._ACTIVE", manager)
        assert "plugin_demo_net" in pp.list_models(), "a plugin's model shows up in pp.list_models()"
        assert "plugin_demo_method" in pp.benchmark.methods(), "and its method in the benchmark"
        assert pp.build_model("plugin_demo_net") is not None
        assert [m for m in manager.find(capability="model")] , "the capability is discoverable by what it does"
    finally:
        ModelRegistry._REGISTRY.pop("plugin_demo_net", None)
        _METHODS.pop("plugin_demo_method", None)
        sys.modules.pop("pp_plugin_demo", None)


def test_shutdown_is_called_for_every_initialized_plugin_even_when_a_later_one_fails(tmp_path):
    closed: list[str] = []

    class First(Plugin):
        name = "first"
        version = "1"

        def initialize(self, context):
            closed.append("first-init")

        def shutdown(self):
            closed.append("first-shutdown")

    class Second(Plugin):
        name = "second"
        version = "1"

        def initialize(self, context):
            raise RuntimeError("no backend")

        def shutdown(self):
            closed.append("second-shutdown")

    manager = PluginManager(entries=[Entry(name="first", loader=lambda: First()),
                                     Entry(name="second", loader=lambda: Second())],
                            config=tmp_path / "plugins.json")
    manager.load_all()
    assert manager.records()["first"].state == "ready"
    assert manager.records()["second"].state == "broken"
    assert manager.records()["second"].stage == "initialize"
    manager.shutdown_all()
    manager.shutdown_all()                                   # safe twice
    assert closed == ["first-init", "second-shutdown", "first-shutdown"]


# ------------------------------------------------------------------------------------------------ #1474
def test_a_broken_plugin_is_listed_with_the_reason_and_the_others_still_show(tmp_path):
    def explode():
        raise ImportError("No module named 'openfoamwrapper'")

    entries = [Entry(name="broken", value="nowhere:PLUGIN", distribution="pinneapple-broken", version="0.1",
                     loader=explode),
               Entry(name="tiny", value="tiny:PLUGIN", distribution="pinneapple-tiny", version="0.1.0",
                     loader=lambda: TinySolver())]
    manager = PluginManager(entries=entries, config=tmp_path / "plugins.json")
    records = manager.load_all()
    assert records["broken"].state == "broken"
    assert records["broken"].stage == "import"
    assert "openfoamwrapper" in records["broken"].reason
    assert records["tiny"].state == "ready", "one broken plugin does not take the others down"
    assert manager.capabilities()["solver"] == ["tiny"]


def test_doctor_tells_not_installed_from_failing_from_disabled(tmp_path, config_file):
    config_file.write_text(json.dumps({"disabled": ["tiny"]}), encoding="utf-8")

    class WrongContract(Plugin):
        name = "old"
        version = "0.9"
        contract_version = "9.0"

    entries = [Entry(name="tiny", value="tiny:PLUGIN", loader=lambda: TinySolver()),
               Entry(name="old", value="old:PLUGIN", loader=lambda: WrongContract())]
    manager = PluginManager(entries=entries, config=config_file)
    assert manager.inspect("tiny").state == "disabled"
    assert manager.inspect("old").state == "broken"
    assert manager.inspect("old").stage == "contract"
    assert "9.0" in manager.inspect("old").reason and CONTRACT_VERSION in manager.inspect("old").reason
    absent = manager.inspect("openfoam")
    assert absent.state == "not_installed" and "openfoam" in absent.reason

    states = {r.name: r.state for r in manager.doctor()}
    assert states == {"tiny": "disabled", "old": "broken"}
    assert manager.doctor("openfoam")[0].state == "not_installed"


def test_dependencies_are_reported_including_the_path_of_an_executable(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ccx = fake_bin / "ccx"
    ccx.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    ccx.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{tmp_path}")

    class WithExe(Plugin):
        name = "with-exe"
        version = "1"
        requires = {"executables": ["ccx", "blockMesh"], "python": ["numpy", "no_such_module_anywhere"]}

    manager = PluginManager(entries=[Entry(name="with-exe", loader=lambda: WithExe())],
                            config=tmp_path / "plugins.json")
    record = manager.load_all()["with-exe"]
    assert record.state == "broken" and record.stage == "dependencies"
    assert "blockMesh" in record.reason and "no_such_module_anywhere" in record.reason
    assert record.details["executables"]["ccx"] == str(ccx), "the path found is reported, not just 'ok'"
    assert record.details["executables"]["blockMesh"] == "not found on PATH"


def test_required_configuration_key_can_come_from_the_file_or_the_environment(tmp_path, config_file, monkeypatch):
    class NeedsKey(Plugin):
        name = "needs-key"
        version = "1"
        requires = {"config": ["case_root"]}

        def initialize(self, context):
            self.seen = dict(context.config)

    manager = PluginManager(entries=[Entry(name="needs-key", loader=lambda: NeedsKey())], config=config_file)
    broken = manager.load_all(force=True)["needs-key"]
    assert broken.state == "broken" and broken.stage == "config"
    assert "PINNEAPPLE_PLUGIN_NEEDS_KEY_CASE_ROOT" in broken.reason

    monkeypatch.setenv("PINNEAPPLE_PLUGIN_NEEDS_KEY_CASE_ROOT", "/data/cases")
    assert manager.load_all(force=True)["needs-key"].state == "ready"


def test_disable_writes_the_configuration_and_removes_the_capabilities(tmp_path, config_file):
    manager = PluginManager(entries=[Entry(name="tiny", value="tiny:PLUGIN", loader=lambda: TinySolver())],
                            config=config_file)
    manager.load_all()
    assert manager.capabilities()["solver"] == ["tiny"]
    path = manager.disable("tiny")
    assert json.loads(path.read_text())["disabled"] == ["tiny"]
    assert manager.capabilities() == {}, "a disabled plugin is out of lookup without touching site-packages"
    assert manager.find(capability="solver", domain="cfd") == []
    assert manager.records()["tiny"].state == "disabled"

    fresh = PluginManager(entries=[Entry(name="tiny", loader=lambda: TinySolver())], config=config_file)
    assert fresh.load_all()["tiny"].state == "disabled", "the next process reads the same file"

    fresh.enable("tiny")
    assert json.loads(config_file.read_text())["disabled"] == []
    assert fresh.load_all(force=True)["tiny"].state == "ready"

    with pytest.raises(UnknownPlugin):
        fresh.disable("not-a-plugin")


def test_find_matches_every_axis_and_refuses_to_pick_between_two(tmp_path):
    class Other(Plugin):
        name = "other"
        version = "1"
        capabilities = {"solver": ["cfd", "transient"]}

    manager = PluginManager(entries=[Entry(name="tiny", loader=lambda: TinySolver()),
                                     Entry(name="other", loader=lambda: Other())],
                            config=tmp_path / "plugins.json")
    manager.load_all()
    steady = manager.find(capability="solver", domain="cfd", steady_state=True)
    assert [m.plugin for m in steady] == ["tiny"], "the flag axis is a capability the plugin declared"
    assert [m.plugin for m in manager.find(capability="solver", domain="cfd")] == ["other", "tiny"]
    with pytest.raises(AmbiguousCapability) as err:
        manager.find_one(capability="solver", domain="cfd")
    assert "other" in str(err.value) and "tiny" in str(err.value)
    assert manager.find(capability="optimizer") == [], "nothing is inferred from a capability nobody declares"
    with pytest.raises(LookupError):
        manager.find_one(capability="optimizer")


# ------------------------------------------------------------------------------------------------ the CLI
def test_cli_lists_inspects_and_doctors(tmp_path, config_file, monkeypatch, capsys):
    monkeypatch.setenv("PINNEAPPLE_PLUGINS_CONFIG", str(config_file))     # never the developer's own file
    manager_entries = [Entry(name="tiny", value="tiny:PLUGIN", distribution="pinneapple-tiny", version="0.1.0",
                             loader=lambda: TinySolver()),
                       Entry(name="broken", value="nowhere:PLUGIN", distribution="pinneapple-broken", version="0.1",
                             loader=lambda: (_ for _ in ()).throw(ImportError("boom")))]
    monkeypatch.setattr("pinneapple_plugins.discovery.discover", lambda group=None: list(manager_entries))
    monkeypatch.setattr("pinneapple_plugins.cli.PluginManager",
                        functools.partial(PluginManager, entries=manager_entries))

    assert plugins_cli.main(["list"]) == 1, "a broken plugin is the exit code 1"
    out = capsys.readouterr().out
    assert "tiny" in out and "broken (import)" in out and "boom" in out

    assert plugins_cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "ok   dependencies" in out or "ok   import" in out

    assert plugins_cli.main(["inspect", "tiny"]) == 0
    out = capsys.readouterr().out
    assert "solver=['cfd', 'steady_state']" in out and "permissions     read, run" in out
    assert "registers       - (what initialize adds through pp.register)" in out, \
        "the diagnostics say what the plugin declared it registers, and that it declared nothing"

    assert plugins_cli.main(["inspect", "missing"]) == 1
    assert "not_installed" in capsys.readouterr().out

    assert plugins_cli.main(["disable", "tiny"]) == 0
    assert json.loads(config_file.read_text())["disabled"] == ["tiny"]
    assert plugins_cli.main(["enable", "tiny"]) == 0
    assert plugins_cli.main(["disable", "nope"]) == 2


def test_plugin_context_is_read_only_and_answers_its_paths(tmp_path):
    ctx = PluginContext(paths={"work": tmp_path}, config={"a": 1}, logger=lambda line: None)
    assert ctx.path("work") == tmp_path
    assert ctx.path("absent", str(tmp_path)) == tmp_path
    assert ctx.as_dict()["config_keys"] == ["a"]


# ------------------------------------------------------------------------------------------------ #1475
def _manager(tmp_path, *entries, running_version=None):
    return PluginManager(entries=list(entries), config=tmp_path / "plugins.json",
                         running_version=running_version)


def test_an_old_plugin_that_would_run_half_is_refused_by_its_version_specifier(tmp_path):
    """The dangerous plugin is the honest one written against last year's API: it installs and half of it works.

    Its `initialize` registers what still exists and silently skips what does not. The check that catches it is the
    version specifier it declared: the library it was written for is not the one installed, so the loading stops
    before `initialize` runs and nothing half-registered is ever in the registry.
    """
    calls: list[str] = []

    class OldApi(Plugin):
        name = "old-api"
        version = "0.4"
        contract_version = "1.0"
        requires = {"pinneapple": ">=0.6,<0.7"}
        capabilities = {"solver": ["cfd", "erosion"]}
        registers = {"solver": ["old_cfd", "old_erosion"]}

        def initialize(self, context):
            import pinneapple as pp

            pp.register(lambda problem, **options: 1.0, kind="solver", name="old_cfd")
            calls.append("old_cfd")
            # `register_erosion_model` was the 2025 API; the old code wrapped it in a `try/except ImportError: pass`

    manager = _manager(tmp_path, Entry(name="old-api", value="old:PLUGIN", loader=lambda: OldApi()),
                       running_version="0.8.0")
    record = manager.load_all()["old-api"]
    assert record.state == "broken" and record.stage == "dependencies"
    assert ">=0.6,<0.7" in record.reason and "0.8.0" in record.reason and "does not satisfy" in record.reason
    assert calls == [], "initialize never ran, so nothing half-loaded"
    assert "old_cfd" not in manager.capabilities().get("solver", [])
    assert not any("old_cfd" in names for names in manager.capabilities().values())


def test_a_plugin_that_declares_two_solvers_and_registers_one_is_refused(tmp_path):
    """The half plugin of this year's API: it loads, registers one of its two solvers, and would look successful."""
    from pinneapple_core.registry import registered as registered_names

    class HalfWritten(Plugin):
        name = "half"
        version = "2.0"
        capabilities = {"solver": ["cfd", "erosion"]}
        registers = {"solver": ["half_case", "half_erosion"]}

        def initialize(self, context):
            import pinneapple as pp

            pp.register(lambda problem, **options: 1.0, kind="solver", name="half_case")   # half_erosion: forgotten

    manager = _manager(tmp_path, Entry(name="half", value="half:PLUGIN", loader=lambda: HalfWritten()))
    record = manager.load_all()["half"]
    assert record.state == "broken" and record.stage == "registers"
    assert "declared in `registers` but not registered: solver=['half_erosion']" in record.reason
    assert "half of what it declares" in record.reason, "the message says why this is a refusal, not a warning"
    assert record.details["registered"] == {"solver": ["half_case"]}
    assert "half_case" not in registered_names("solver"), "what the half plugin did register was rolled back"


def test_a_plugin_that_registers_what_it_never_declared_is_refused(tmp_path):
    from pinneapple_core.registry import registered as registered_names

    class Sneaky(Plugin):
        name = "sneaky"
        version = "1"
        registers = {"model": ["declared_net"]}

        def initialize(self, context):
            import torch

            import pinneapple as pp

            class Declared(torch.nn.Module):
                def forward(self, x):
                    return x

            pp.register(Declared, kind="model", name="declared_net", family="plugin")
            pp.register(Declared, kind="model", name="sneaky_net", family="plugin")      # never declared

    manager = _manager(tmp_path, Entry(name="sneaky", value="sneaky:PLUGIN", loader=lambda: Sneaky()))
    record = manager.load_all()["sneaky"]
    assert record.state == "broken" and record.stage == "registers"
    assert "registered but not declared in `registers`: model=['sneaky_net']" in record.reason
    assert "declared_net" not in registered_names("model") and "sneaky_net" not in registered_names("model")


def test_a_contract_1_0_plugin_loads_and_what_it_registers_is_recorded_not_verified(tmp_path):
    """Compatibility is additive: a plugin written against 1.0 has no `registers`, so there is nothing to check it
    against. Saying that plainly (recorded, not verified) is the honest answer, not a silent pass."""
    from pinneapple_core.registry import unregister

    class WithoutDeclarations(Plugin):
        name = "old-1-0"
        version = "1"
        contract_version = "1.0"

        def initialize(self, context):
            import pinneapple as pp

            pp.register(lambda problem, **options: 1.0, kind="solver", name="old_1_0_solver")

    manager = _manager(tmp_path, Entry(name="old-1-0", value="old:PLUGIN", loader=lambda: WithoutDeclarations()))
    record = manager.load_all()["old-1-0"]
    try:
        assert record.state == "ready" and record.contract_version == "1.0"
        assert record.details["registered"] == {"solver": ["old_1_0_solver"]}
        assert "declared nothing: recorded, not verified" in record.check("registers")["detail"]
        assert "contract_note" in record.details
    finally:
        unregister("solver", "old_1_0_solver")


def test_the_registry_listener_never_breaks_a_registration():
    """A watcher with a bug is dropped, not allowed to make pp.register fail: the loader's check must not be able to
    break a plugin that is fine."""
    from pinneapple_core.registry import observe, register, registered, unregister

    def explode(action, kind, name):
        raise RuntimeError("a watcher with a bug")

    with observe(explode):
        register(lambda problem, **options: 2.0, kind="solver", name="listener_survivor")
    try:
        assert "listener_survivor" in registered("solver")
    finally:
        unregister("solver", "listener_survivor")


def test_provenance_names_each_ready_plugin_with_its_distribution(tmp_path, monkeypatch):
    entries = [Entry(name="tiny", value="tiny:PLUGIN", distribution="pinneapple-tiny", version="0.1.0",
                     loader=lambda: TinySolver()),
               Entry(name="broken", value="nowhere:PLUGIN", distribution="pinneapple-broken", version="0.1",
                     loader=lambda: (_ for _ in ()).throw(ImportError("boom")))]
    manager = _manager(tmp_path, *entries)
    manager.load_all()
    monkeypatch.setattr("pinneapple_plugins._ACTIVE", manager)
    import pinneapple_plugins as plugins

    assert plugins.provenance() == [{"name": "tiny", "version": "0.1.0", "contract_version": CONTRACT_VERSION,
                                     "source": "pinneapple-tiny 0.1.0", "entry_point": "tiny:PLUGIN"}], \
        "a broken plugin took part in nothing, so it is not in the provenance"


def test_the_policy_is_the_same_words_in_the_cli_and_the_page(capsys):
    from pinneapple_plugins.policy import CHECKED, NOT_PROVIDED, ONE_SENTENCE, RECORDED

    assert plugins_cli.main(["policy"]) == 0
    out = capsys.readouterr().out
    page = _repo_file("docs/core_concepts/plugins.md").read_text(encoding="utf-8")
    flat_page = " ".join(page.replace("> ", "").split())
    for sentence in [text for _, text in CHECKED] + list(NOT_PROVIDED) + list(RECORDED) + [ONE_SENTENCE]:
        one_line = " ".join(sentence.split())
        assert one_line in out, f"the CLI prints it: {one_line[:60]}"
        assert one_line in flat_page, f"the page quotes it word for word: {one_line[:60]}"


def test_nothing_describes_loading_a_plugin_as_sandboxed():
    """#1475: the acceptance line is also that no text claims isolation. Each mention has to be a denial."""
    import re

    pages = sorted(_repo_file("pinneapple_plugins").glob("*.py")) + [_repo_file("docs/core_concepts/plugins.md")]
    for path in pages:
        flat = " ".join(path.read_text(encoding="utf-8").split()).lower()
        for match in re.finditer("sandbox", flat):
            window = flat[max(0, match.start() - 150): match.end() + 150]
            assert any(denial in window for denial in ("not a sandbox", "no sandbox", "does not sandbox")), \
                f"{path}: 'sandbox' is used without denying it: …{flat[max(0, match.start() - 60):match.end() + 60]}…"


def _repo_file(relative: str):
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / relative


def test_a_registers_key_that_is_not_a_registry_kind_says_so(tmp_path):
    class Typo(Plugin):
        name = "typo"
        version = "1"
        registers = {"solvers": ["typo_case"]}          # the kind is "solver"

        def initialize(self, context):
            import pinneapple as pp

            pp.register(lambda problem, **options: 1.0, kind="solver", name="typo_case")

    from pinneapple_core.registry import registered as registered_names

    manager = _manager(tmp_path, Entry(name="typo", value="typo:PLUGIN", loader=lambda: Typo()))
    record = manager.load_all()["typo"]
    assert record.state == "broken" and record.stage == "registers"
    assert "solvers" in record.reason and "known kinds" in record.reason
    assert "typo_case" not in registered_names("solver")

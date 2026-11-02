"""Deprecation helpers and the frozen core API snapshot (#1128)."""
import json
import warnings

import pytest

from pinneapple_stability import api
from pinneapple_stability.deprecation import (
    PinneappleDeprecationWarning,
    deprecated,
    deprecated_alias,
    message,
    renamed_kwargs,
)


def test_deprecated_function_and_class_warn_and_still_work():
    @deprecated(since="0.9", removed_in="1.1", use="new_f")
    def old_f(x):
        return 2 * x

    with pytest.warns(PinneappleDeprecationWarning, match=r"old_f is deprecated since 0.9 .* removed in 1.1; use new_f"):
        assert old_f(3) == 6

    @deprecated(since="0.9", removed_in="1.1")
    class Old:
        def __init__(self, a):
            self.a = a

    with pytest.warns(PinneappleDeprecationWarning, match="Old is deprecated"):
        assert Old(4).a == 4


def test_alias_and_renamed_kwargs():
    def new_f(hidden_dim=1):
        return hidden_dim

    old = deprecated_alias(new_f, "pkg.old_f", since="0.9", removed_in="1.1")
    with pytest.warns(PinneappleDeprecationWarning, match=r"pkg.old_f is deprecated .* use .*new_f"):
        assert old(hidden_dim=5) == 5
    assert old.__name__ == "old_f"

    f = renamed_kwargs(since="0.9", removed_in="1.1", hidden="hidden_dim")(new_f)
    with pytest.warns(PinneappleDeprecationWarning, match="argument 'hidden'"):
        assert f(hidden=7) == 7
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f(hidden_dim=8) == 8                                     # the new name does not warn
    with pytest.raises(TypeError, match="both"):
        f(hidden=1, hidden_dim=2)


def test_the_warning_is_shown_by_default():
    assert issubclass(PinneappleDeprecationWarning, FutureWarning)    # Python shows FutureWarning by default


def test_message_literal_call():
    """Plain-string call (also the Studio's smoke input)."""
    assert message("pp.old", "0.9", "1.1", "pp.new") == "pp.old is deprecated since 0.9 and will be removed in 1.1; use pp.new instead"


def test_core_api_matches_the_committed_snapshot():
    """A change to a core name or signature must show up as a diff of api/core.json in the pull request
    (`python -m pinneapple_stability.api --write`), reviewed against docs/api_stability.md."""
    committed = json.loads((api.repo_root() / api.SNAPSHOT_PATH).read_text(encoding="utf-8"))
    d = api.diff(committed, api.snapshot())
    assert not any(d.values()), json.dumps(d, indent=1)[:4000]


def test_snapshot_covers_the_facade_and_diff_reports_changes():
    snap = api.snapshot()
    assert "MISSING" not in snap["pp"].values()
    assert snap["pp"]["solve"] == "from pinneapple_physics.solving import solve"
    assert "pinneapple_design.geometry.brep.study" in snap["modules"]
    old = {"pp": {"a": "def(x)", "b": "def()"}, "modules": {"m": {"C": {"class": [], "methods": {"run": "(n=1)"}}}}}
    new = {"pp": {"a": "def(x, y)", "c": "def()"}, "modules": {"m": {"C": {"class": [], "methods": {"run": "(n=2)"}}}}}
    d = api.diff(old, new)
    assert d["added"] == ["pp.c"] and d["removed"] == ["pp.b"]
    assert [c[0] for c in d["changed"]] == ["m.C.run", "pp.a"]

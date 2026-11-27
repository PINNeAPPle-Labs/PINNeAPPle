"""PC901 answers from the method catalog status, in both directions.

Kept out of ``test_physics_check.py`` on purpose: that module skips itself when ``meshio`` is missing, and this
rule needs nothing but the catalog — it is a lookup, not a fixture walk.
"""
from __future__ import annotations

from pathlib import Path


def test_pc901_follows_the_method_catalog_status():
    """The rule used to be exercised only through ``examples/physics_check/fail``: the kovasznay model declares
    E10, which the refreshed scan now finds validated, so that fixture stopped tripping PC901 (#1884).

    This keeps the rule covered whatever the scan says, by reading the same artifact the rule reads instead of
    hard-coding two ids that would go stale.
    """
    from pinneapple_catalog.methods import method_status
    from pinneapple_check.checks.models import _method
    from pinneapple_check.config import ModelTarget

    class Ctx:
        def __init__(self):
            self.out = []

        def emit(self, rule, target, ok, message):
            self.out.append((rule, ok, message))

        def note(self, _message):
            pass

    status = method_status()
    assert status, "the method catalog status file is missing"
    unvalidated = next(k for k, v in status.items() if v["status"] != "validated")
    validated = next(k for k, v in status.items() if v["status"] == "validated")

    for method_id, expected in ((unvalidated, False), (validated, True)):
        ctx = Ctx()
        _method(ctx, ModelTarget(name="m", path=Path("m.py:predict"), method=method_id))
        assert len(ctx.out) == 1, ctx.out
        rule, ok, message = ctx.out[0]
        assert rule == "PC901" and ok is expected, (method_id, ctx.out)
        assert method_id in message and status[method_id]["status"] in message


def test_pc901_reports_a_method_that_is_not_in_the_catalog():
    """A typo in the config is a violation, not a pass: the rule must not go quiet when the id is unknown."""
    from pinneapple_check.checks.models import _method
    from pinneapple_check.config import ModelTarget

    class Ctx:
        def __init__(self):
            self.out = []

        def emit(self, rule, target, ok, message):
            self.out.append((rule, ok, message))

        def note(self, _message):
            pass

    ctx = Ctx()
    _method(ctx, ModelTarget(name="m", path=Path("m.py:predict"), method="not-a-method-id"))
    assert len(ctx.out) == 1 and ctx.out[0][0] == "PC901" and ctx.out[0][1] is False

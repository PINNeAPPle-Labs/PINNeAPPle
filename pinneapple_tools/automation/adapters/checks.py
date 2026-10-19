"""The ``verify`` adapter: evaluate the workflow's acceptance criteria (``verify.criteria``) on the run's files,
and the ``web`` adapter: open a web UI with Playwright and capture it (``web.capture``)."""
from __future__ import annotations

import json
from typing import Any

from ..display import PlaywrightDriver
from ..record import Check, Evidence
from ..verify import evaluate_criterion
from .base import ActionFailed, ActionResult, Adapter, StepContext, register

__all__ = ["VerifyAdapter", "WebAdapter"]


class VerifyAdapter(Adapter):
    name = "verify"
    actions = ("criteria",)

    def do_criteria(self, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        res = ActionResult()
        evaluated = []
        for c in ctx.criteria:                                       # templates already filled by the runner
            p = dict(c.params)
            chk = evaluate_criterion(c.kind, p, ctx.run_dir, ctx.previous)
            chk.name = c.title or chk.name
            res.checks.append(chk)
            evaluated.append({"id": c.id, "kind": c.kind, "params": p, "passed": chk.passed, "value": chk.value,
                              "detail": chk.detail})
        if not ctx.criteria:
            res.checks.append(Check("criteria", False, "the workflow defines no acceptance criteria"))
        out = ctx.step_dir / "criteria.json"
        out.write_text(json.dumps(evaluated, indent=1, default=str))
        res.artifacts.append(out)
        res.outputs["criteria"] = {e["id"]: {"passed": e["passed"], "value": e["value"]} for e in evaluated}
        return res


class WebAdapter(Adapter):
    name = "web"
    mode = "gui"
    actions = ("capture",)

    def available(self, toolchain) -> tuple[bool, str]:
        return PlaywrightDriver.available()

    def do_capture(self, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        """Open ``url``, optionally click ``click`` selectors in order and wait for ``wait_text``; screenshot."""
        res = ActionResult()
        with PlaywrightDriver() as pw:
            pw.goto(params["url"])
            for sel in params.get("click", []):
                pw.click(sel)
            if params.get("wait_text"):
                pw.wait_text(params["wait_text"])
            shot = pw.screenshot(ctx.step_dir / "web.png")
            title = pw.page.title()
            ctx.commands.extend(f"playwright: {x}" for x in pw.log)
        if not shot.is_file():
            raise ActionFailed("no screenshot written")
        res.evidence.append(Evidence(ctx.rel(shot), "window", f"{title} ({params['url']})"))
        res.artifacts.append(shot)
        res.outputs["title"] = title
        if params.get("wait_text"):
            res.checks.append(Check(f"page shows {params['wait_text']!r}", True, title, source="Playwright"))
        return res


register(VerifyAdapter())
register(WebAdapter())

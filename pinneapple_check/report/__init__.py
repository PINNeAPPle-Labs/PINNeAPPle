"""Output formats: ``text`` (default), ``json``, ``sarif`` (GitHub code scanning), ``junit``, ``github``
(workflow-command annotations) and a Markdown ``summary`` (the job summary of a GitHub Action)."""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

from ..result import Report
from ..rules import RULES

__all__ = ["render", "FORMATS", "text", "to_json", "sarif", "junit", "github", "summary"]

FORMATS = ("text", "json", "sarif", "junit", "github", "summary")


def render(report: Report, fmt: str, **kw) -> str:
    return {"text": text, "json": to_json, "sarif": sarif, "junit": junit, "github": github,
            "summary": summary}[fmt](report, **kw)


# ---------------------------------------------------------------------------------------------- text
def _color(enabled: bool):
    codes = {"red": "31", "yellow": "33", "green": "32", "bold": "1", "dim": "2", "cyan": "36"}

    def c(s: str, *names: str) -> str:
        if not enabled:
            return s
        return "".join(f"\033[{codes[n]}m" for n in names) + s + "\033[0m"
    return c


def text(report: Report, color: bool | None = None, show_passed: bool = False, show_fixes: bool = True,
         **_) -> str:
    if color is None:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    c = _color(color)
    out = []
    shown = report.outcomes if show_passed else report.violations
    for o in sorted(shown, key=lambda o: (o.location.file, o.location.line, o.rule)):
        loc = c(f"{o.location.file}:{o.location.line}:{o.location.column}:", "bold")
        if o.passed:
            tag = c("ok", "green")
        else:
            tag = c(o.severity, "red" if o.severity == "error" else "yellow")
        out.append(f"{loc} {c(o.rule, 'cyan' if o.passed else 'red', 'bold')} [{tag}] {o.message}")
        if show_fixes and not o.passed:
            out.append(c(f"  = help: {o.fix}", "dim"))
    if out:
        out.append("")
    for n in report.notes:
        out.append(c(f"note: {n}", "dim"))
    if report.notes:
        out.append("")
    ne, nw, nb = len(report.errors), len(report.warnings), len(report.baselined)
    passed = sum(o.passed for o in report.outcomes)
    tgt = f"{len(report.targets)} target{'s' if len(report.targets) != 1 else ''}"
    if ne or nw:
        head = f"Found {ne} error{'s' if ne != 1 else ''} and {nw} warning{'s' if nw != 1 else ''} in {tgt}"
        out.append(c(head, "red" if ne else "yellow", "bold") + f" ({passed} checks passed"
                   + (f", {nb} baselined" if nb else "") + ").")
        rules = Counter(o.rule for o in report.violations)
        out.append("  " + ", ".join(f"{r} {RULES[r].name} x{n}" for r, n in sorted(rules.items())))
        out.append(f"  Explain a rule: pinneapple check --explain {sorted(rules)[0]}")
    elif report.targets:
        out.append(c(f"All checks passed: {passed} checks on {tgt}", "green", "bold")
                   + (f" ({nb} baselined)" if nb else "") + ".")
    else:
        out.append("Nothing to check: no pinneapple-check.toml targets and no results found.")
    return "\n".join(out) + "\n"


def to_json(report: Report, **_) -> str:
    return json.dumps(report.to_dict(), indent=1, default=str) + "\n"


# ---------------------------------------------------------------------------------------------- SARIF
_LEVEL = {"error": "error", "warning": "warning"}


def sarif(report: Report, **_) -> str:
    from .. import __version__
    rule_ids = sorted(RULES)
    rules = []
    for rid in rule_ids:
        r = RULES[rid]
        rules.append({
            "id": r.id, "name": "".join(w.capitalize() for w in r.name.split("-")),
            "shortDescription": {"text": r.title},
            "fullDescription": {"text": r.explanation},
            "help": {"text": f"{r.explanation}\n\nFix: {r.fix}",
                     "markdown": f"**{r.id} {r.name}**: {r.explanation}\n\n**Fix:** {r.fix}\n\n"
                                 f"[Documentation]({r.help_uri})"},
            "helpUri": r.help_uri,
            "defaultConfiguration": {"level": _LEVEL[r.severity]},
            "properties": {"tags": list(dict.fromkeys(["physics", r.stage])), "precision": "high",
                           "problem.severity": "error" if r.severity == "error" else "warning"},
        })
    results = []
    for o in report.outcomes:
        if o.passed:
            continue
        res = {
            "ruleId": o.rule, "ruleIndex": rule_ids.index(o.rule), "level": _LEVEL.get(o.severity, "warning"),
            "message": {"text": f"{o.message}\nFix: {o.fix}"},
            "locations": [{"physicalLocation": {
                "artifactLocation": {"uri": o.location.file, "uriBaseId": "%SRCROOT%"},
                "region": {"startLine": max(1, o.location.line), "startColumn": max(1, o.location.column)}}}],
            "partialFingerprints": {"pinneappleCheck/v1": o.fingerprint},
            "properties": {"target": o.target, "subject": o.subject, "value": _jsonable(o.value),
                           "threshold": _jsonable(o.threshold)},
        }
        if o.baselined:
            res["suppressions"] = [{"kind": "external", "justification": "in the pinneapple check baseline"}]
        results.append(res)
    doc = {"$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
           "runs": [{"tool": {"driver": {"name": "pinneapple-check", "semanticVersion": __version__,
                                         "informationUri": "https://github.com/PINNeAPPle-Labs/PINNeAPPle",
                                         "rules": rules}},
                     "automationDetails": {"id": "pinneapple-check/"},
                     "invocations": [{"executionSuccessful": True}],
                     "results": results}]}
    return json.dumps(doc, indent=1) + "\n"


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


# ---------------------------------------------------------------------------------------------- JUnit
def junit(report: Report, **_) -> str:
    root = ET.Element("testsuites", name="pinneapple check", tests=str(len(report.outcomes)),
                      failures=str(len(report.violations)))
    by: dict[str, list] = {}
    for o in report.outcomes:
        by.setdefault(o.target, []).append(o)
    for target, os_ in by.items():
        fails = [o for o in os_ if not o.passed and not o.baselined]
        suite = ET.SubElement(root, "testsuite", name=target, tests=str(len(os_)), failures=str(len(fails)),
                              errors="0", skipped=str(sum(o.baselined for o in os_)))
        for o in os_:
            case = ET.SubElement(suite, "testcase", classname=target, name=f"{o.rule} {RULES[o.rule].name}: {o.subject}",
                                 file=o.location.file, line=str(o.location.line))
            if o.baselined:
                ET.SubElement(case, "skipped", message=f"baselined: {o.message}")
            elif not o.passed:
                f = ET.SubElement(case, "failure", message=o.message, type=o.severity)
                f.text = f"{o.location.file}:{o.location.line}: {o.rule} {o.message}\nFix: {o.fix}"
            else:
                ET.SubElement(case, "system-out").text = o.message
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


# ---------------------------------------------------------------------------------------------- GitHub
def _esc_data(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _esc_prop(s: str) -> str:
    return _esc_data(s).replace(":", "%3A").replace(",", "%2C")


def github(report: Report, **_) -> str:
    lines = []
    for o in report.violations:
        kind = "error" if o.severity == "error" else "warning"
        title = _esc_prop(f"{o.rule} {RULES[o.rule].title}")
        lines.append(f"::{kind} file={_esc_prop(o.location.file)},line={o.location.line},col={o.location.column},"
                     f"title={title}::{_esc_data(o.message + chr(10) + 'Fix: ' + o.fix)}")
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------------------------- summary
_ICON = {"answered": "✅", "partial": "⚠️", "failed": "❌", "open": "▫️"}


def summary(report: Report, **_) -> str:
    ne, nw = len(report.errors), len(report.warnings)
    passed = sum(o.passed for o in report.outcomes)
    head = ("### ❌ Physics check failed" if ne else "### ⚠️ Physics check passed with warnings" if nw
            else "### ✅ Physics check passed")
    out = [head, "", f"{passed} checks passed, {ne} error(s), {nw} warning(s)"
           + (f", {len(report.baselined)} baselined" if report.baselined else "")
           + f" on {len(report.targets)} target(s).", ""]
    card = report.trust_card()
    out += ["| Target | Kind | Errors | Warnings | Data | Model | Physics | Benchmark |",
            "|---|---|---:|---:|:-:|:-:|:-:|:-:|"]
    for t in report.targets:
        n = t["name"]
        e = sum(1 for o in report.errors if o.target == n)
        w = sum(1 for o in report.warnings if o.target == n)
        st = card.get(n, {})
        icons = [_ICON[st[k]["status"]] if k in st else "" for k in ("data_geometry", "model", "physics", "benchmark")]
        out.append(f"| `{n}` | {t['kind']} | {e} | {w} | " + " | ".join(icons) + " |")
    if report.violations:
        out += ["", "| Rule | Location | Message |", "|---|---|---|"]
        for o in report.violations[:100]:
            msg = o.message.replace("|", "\\|")
            out.append(f"| [{o.rule}]({RULES[o.rule].help_uri}) {'❌' if o.severity == 'error' else '⚠️'} | "
                       f"`{o.location.file}:{o.location.line}` | {msg} |")
        if len(report.violations) > 100:
            out.append(f"| | | ... and {len(report.violations) - 100} more |")
    if report.notes:
        out += ["", "<details><summary>Not checked</summary>", ""] + [f"- {n}" for n in report.notes] + ["",
                                                                                                         "</details>"]
    out.append("")
    out.append("✅ answered · ⚠️ warnings only · ❌ failed · ▫️ nothing checked (trust-card stages)")
    return "\n".join(out) + "\n"

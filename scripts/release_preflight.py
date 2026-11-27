"""Release preflight: what still blocks the tag, per milestone, before anyone cuts it.

A release epic (#421 for 0.8.0, #422 for 0.9.0, #423 for 1.0.0) lists exit criteria, and the criteria are the whole
point: a tag has to mean that the items are closed **or explicitly moved** with the reason written in the item. That
is a decision for a person, but most of what blocks it is mechanical and can be checked: the version in every file
agreeing, the ``[Unreleased]`` section being what is about to ship, the fragments being folded, and the milestone
having no open item left.

``scripts/check_release_version.py`` already refuses a tag whose version files disagree. This is the wider gate: it
reports **every** blocker at once instead of the first one, for each of the three milestones, and it separates what it
can verify offline from what needs GitHub.

    python scripts/release_preflight.py 0.8.0              # local checks only
    python scripts/release_preflight.py 0.8.0 --github     # + the milestone's open items (needs `gh`)
    python scripts/release_preflight.py --all              # the three milestones of the plan

Exit code is 0 when nothing blocks, 1 when something does, so it can be a gate.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: the milestones the 2026 plan names, and the issue that tracks each one
MILESTONES = {"0.8.0": 421, "0.9.0": 422, "1.0.0": 423}


@dataclass
class Blocker:
    """One thing that stops the tag, and what to do about it."""
    kind: str
    what: str
    fix: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "what": self.what, "fix": self.fix}


@dataclass
class Report:
    version: str
    blockers: list[Blocker] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {"version": self.version, "ok": self.ok, "blockers": [b.as_dict() for b in self.blockers],
                "notes": self.notes}


# ------------------------------------------------------------------------------------------------------- the local reads
def _text(*parts: str) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def versions() -> dict[str, str]:
    """The version each file claims, under the names a reader recognises."""
    pyproject = _text("pyproject.toml")
    citation = _text("CITATION.cff")
    init = _text("pinneapple", "__init__.py")
    found = {}
    for name, pattern, text in (
            ("pyproject.toml", r'^version\s*=\s*"([^"]+)"', pyproject),
            ("CITATION.cff", r'^version:\s*"?([^"\s]+)"?', citation),
            ("pinneapple/__init__.py", r'^__version__\s*=\s*"([^"]+)"', init)):
        m = re.search(pattern, text, re.M)
        found[name] = m.group(1) if m else "?"
    return found


def unreleased() -> str:
    """The body of the ``## [Unreleased]`` section (empty when there is nothing pending)."""
    m = re.search(r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[)", _text("CHANGELOG.md"), re.M | re.S)
    return (m.group(1) if m else "").strip()


def released_heading(version: str) -> str | None:
    """The dated heading of that version, if the CHANGELOG already carries one."""
    m = re.search(rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})", _text("CHANGELOG.md"), re.M)
    return m.group(1) if m else None


def fragments() -> list[str]:
    folder = ROOT / "changelog.d"
    return sorted(p.name for p in folder.glob("*.md") if p.name != "README.md")


def _gh(*args: str, timeout: float = 30.0) -> str | None:
    """``gh``'s stdout, or ``None`` when it is missing, unauthenticated or the call fails."""
    try:
        out = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def milestone(version: str, timeout: float = 30.0) -> dict[str, object] | None:
    """The milestone's counts and its open item titles, through ``gh``. ``None`` when gh is not usable here.

    The milestone is found by its **title** (``v0.8.0``) and not by its number: the numbers are stable only by
    accident, and a preflight that reads the wrong milestone is worse than one that reads none.
    """
    raw = _gh("api", "repos/PINNeAPPle-Labs/PINNeAPPle/milestones?state=all&per_page=100", timeout=timeout)
    if raw is None:
        return None
    try:
        found = [m for m in json.loads(raw) if str(m.get("title", "")).lstrip("v") == version]
    except (TypeError, json.JSONDecodeError):
        return None
    if not found:
        return None
    data = found[0]
    items_raw = _gh("issue", "list", "--repo", "PINNeAPPle-Labs/PINNeAPPle", "--milestone", str(data["title"]),
                    "--state", "open", "--limit", "200", "--json", "number,title", timeout=timeout)
    items: list[str] = []
    if items_raw:
        try:
            items = [f"#{i['number']} {i['title']}" for i in json.loads(items_raw)]
        except (TypeError, json.JSONDecodeError):
            items = []
    return {"title": data["title"], "state": data["state"], "open": data["open_issues"],
            "closed": data["closed_issues"], "open_items": items}


# -------------------------------------------------------------------------------------------------------- the verdict
def preflight(version: str, with_github: bool = False) -> Report:
    """Every blocker for releasing ``version``, not just the first one."""
    report = Report(version=version)
    found = versions()
    if len(set(found.values()) | {version}) != 1:
        report.blockers.append(Blocker(
            "version", "the files disagree: " + ", ".join(f"{k}={v}" for k, v in found.items()) + f", pedido {version}",
            "alinhar pyproject.toml, CITATION.cff e pinneapple/__init__.py no mesmo número antes do tag"))
    pending = unreleased()
    if pending:
        report.blockers.append(Blocker(
            "changelog", f"o [Unreleased] tem {len(pending.splitlines())} linha(s) por lançar",
            "python scripts/changelog.py release " + version + " (dobra os fragmentos e a seção sob a data de hoje)"))
    frags = fragments()
    if frags:
        report.blockers.append(Blocker(
            "changelog", f"{len(frags)} fragmento(s) em changelog.d em aberto ({', '.join(frags[:3])}…)",
            "o `changelog.py release` dobra os fragmentos; nenhum pode sobrar depois do tag"))
    dated = released_heading(version)
    if dated is None:
        report.blockers.append(Blocker(
            "changelog", f"o CHANGELOG.md não tem cabeçalho datado para {version}",
            f"depois de dobrar, conferir '## [{version}] - AAAA-MM-DD'"))
    else:
        report.notes.append(f"o CHANGELOG.md já carrega '{version}' datado de {dated}")
    if with_github:
        data = milestone(version)
        if data is None:
            report.notes.append("o GitHub não respondeu (gh ausente ou sem autenticação): só os portões locais foram "
                                "avaliados")
        else:
            report.notes.append(f"milestone {data['title']}: {data['open']} aberto(s), {data['closed']} fechado(s)")
            if data["open"]:
                report.blockers.append(Blocker(
                    "milestone",
                    f"{data['open']} item(ns) aberto(s) no milestone, ex.: " + "; ".join(data["open_items"][:3]),
                    "fechar cada item com evidência, ou movê-lo para um milestone posterior com a razão escrita no "
                    "próprio item — a saída da issue de release exige uma das duas, não a contagem"))
            if data["state"] != "closed" and not data["open"]:
                report.notes.append("todos os itens fechados, mas o milestone ainda está aberto: feche-o no tag")
    return report


def render(reports: list[Report]) -> str:
    lines = ["# Preflight de release", ""]
    for r in reports:
        state = "pronto" if r.ok else f"{len(r.blockers)} bloqueio(s)"
        lines += [f"## {r.version} — {state}", ""]
        if r.blockers:
            lines += ["| bloqueio | o que é | o que fazer |", "|---|---|---|"]
            lines += [f"| `{b.kind}` | {b.what} | {b.fix} |" for b in r.blockers]
            lines.append("")
        for note in r.notes:
            lines.append(f"- {note}")
        if r.notes:
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", nargs="?", help="a versão a lançar, ex. 0.8.0")
    parser.add_argument("--all", action="store_true", help="as três versões do plano de 2026")
    parser.add_argument("--github", action="store_true", help="consultar o milestone (precisa de gh autenticado)")
    parser.add_argument("--json", action="store_true", help="saída em JSON")
    args = parser.parse_args(argv)
    targets = list(MILESTONES) if args.all else ([args.version] if args.version else [])
    if not targets:
        parser.error("diga a versão (ex. 0.8.0) ou --all")
    reports = [preflight(v, with_github=args.github) for v in targets]
    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2, ensure_ascii=False))
    else:
        print(render(reports))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())

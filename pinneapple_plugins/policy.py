"""What loading a plugin checks, and what it deliberately does not provide (#1475).

A plugin is an installed package: Python code that runs in this process with the privileges of whoever imports it.
The dangerous case is not the hostile plugin — it is the honest one written against last year's API, running half its
steps and looking successful. The policy below is what the loader does about that, and what it cannot do, in the words
the CLI prints (`pinneapple plugins policy`) and the documentation quotes (`docs/core_concepts/plugins.md`); a test
holds the page to those exact sentences.

Nothing here is enforced by the operating system. The list of what is *not* provided is part of the policy, not a
disclaimer: a deployment that needs process isolation needs a process, and no amount of checking at load time
substitutes for one.
"""
from __future__ import annotations

__all__ = ["CHECKED", "NOT_PROVIDED", "RECORDED", "ONE_SENTENCE", "policy_text", "policy_dict"]

#: What the loader checks, every time a plugin is loaded. ``(check, sentence)`` pairs, in the order they are applied.
CHECKED: tuple[tuple[str, str], ...] = (
    ("declared dependencies",
     "declared dependencies, resolved against the environment: the `pinneapple` version specifier, the other plugins "
     "it needs, the Python modules that must import, the executables that must be on PATH, and the configuration keys "
     "it requires"),
    ("contract version",
     "the contract version it was written against, refused by name with both versions before `initialize()` runs"),
    ("where it came from",
     "where the plugin came from — distribution name, version and entry-point value — recorded, and written into the "
     "provenance of every run it takes part in"),
    ("what it registers",
     "the registry entries it adds when `initialize()` runs, against the `registers` it declares: a plugin that "
     "declares `registers` and then registers something it did not declare, or declares something it did not "
     "register, is refused and its registrations are rolled back; a plugin that declares nothing has what it "
     "registers recorded and shown by `pinneapple plugins doctor`, and not verified"),
    ("its permissions",
     "the permissions it declares (`read`, `run`, `write`, `network`) — recorded and shown by `pinneapple plugins "
     "doctor`, never enforced"),
)

#: What loading a plugin does not provide. Stated, not implied: none of these is a setting that can be turned on.
NOT_PROVIDED: tuple[str, ...] = (
    "process isolation, resource limits or syscall filtering: a plugin runs in this process, with the privileges of "
    "whoever imports it",
    "a sandbox of any kind, including for a plugin started in a subprocess — a subprocess is not a sandbox either",
    "enforcement of the declared permissions: they are recorded, never applied, so a plugin that declares `read` can "
    "still write and a plugin that declares neither can still do both",
    "revocation: an installed plugin stays importable until it is disabled or uninstalled, and what a running plugin "
    "already registered in the library stays registered until the process ends",
    "protection from a plugin that passes every check: the checks are about declarations that match reality, not about "
    "what the code does after `initialize()` returns",
)

#: What is recorded about the plugin set of a run, so a result can be attributed to the code that produced it.
RECORDED: tuple[str, ...] = (
    "every plugin that took part in a run, with its name, its version, the contract version it was written against, "
    "the distribution and version it came from, and the entry-point value that was loaded",
    "what each plugin registered, and what it declared but did not register",
    "the plugins that failed to load, at which check, and why — a failure is a record, never an exception",
)

#: The policy in one sentence, for a report, a slide or a model card.
ONE_SENTENCE = ("Loading a plugin checks that its versions, dependencies and registrations match what it declares and "
                "records where it came from; it does not sandbox it — the plugin runs with the privileges of whoever "
                "imports it.")


def policy_dict() -> dict[str, list[str] | str]:
    """The policy as data (``--json``, and the page's table)."""
    return {"one_sentence": ONE_SENTENCE,
            "checked": [f"{name}: {sentence}" for name, sentence in CHECKED],
            "not_provided": list(NOT_PROVIDED),
            "recorded": list(RECORDED)}


def policy_text() -> str:
    """The policy as the CLI prints it and the documentation quotes it, sentence for sentence."""
    lines = [ONE_SENTENCE, "", "Checked at load time, every time:"]
    lines += [f"  - {sentence}" for _, sentence in CHECKED]
    lines += ["", "Not provided, by design:"]
    lines += [f"  - {sentence}" for sentence in NOT_PROVIDED]
    lines += ["", "Recorded, so a result can be attributed to the code that produced it:"]
    lines += [f"  - {sentence}" for sentence in RECORDED]
    return "\n".join(lines)

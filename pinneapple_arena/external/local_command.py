"""
Builds the exact command a user can run on their own machine to get a real
result for a model the hosted app can't run itself — because it's a heavy
GPU dependency, or a git-only / non-pip research repo, or fine-tuning simply
isn't wired up for it yet in-app.

This is what satisfies "when the app can't run it, tell me exactly how to
run it myself": every ModelResult for a not_installed / manual_install /
skipped_heavy / finetune_not_supported status carries one of these.
"""

from __future__ import annotations

from pinneapple_arena.external.schema import ModelSpec


def local_command_for(spec: ModelSpec, action: str = "infer", **params: object) -> str:
    """`action` is "infer" or "finetune": the command line of ``python -m pinneapple_arena.external``
    (PINNeAPPle installed), or the clone + build steps of a manual-install repository."""
    if spec.manual_only:
        clone_dir = spec.repo.rstrip("/").split("/")[-1]
        base = f"git clone {spec.repo} && cd {clone_dir}"
        return f"{base}  # then follow: {spec.manual_note}"

    flags = [f"--model {spec.slug!r}", f"--action {action}"]
    if spec.heavy:
        flags.append("--include-heavy")
    for key, value in params.items():
        flags.append(f"--{key.replace('_', '-')} {value}")
    return "python -m pinneapple_arena.external " + " ".join(flags)


def pip_install_command(spec: ModelSpec) -> str | None:
    if not spec.pip_package:
        return None
    return f"pip install {spec.pip_package}"

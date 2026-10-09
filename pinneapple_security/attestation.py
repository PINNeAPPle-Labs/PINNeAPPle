"""Provenance attestations (in-toto Statement v1 with a SLSA Provenance v1 predicate) and a CycloneDX SBOM.

An attestation answers "what produced this model / dataset / result, from what inputs, where": the *subjects* are the
artifacts with their digests, ``resolvedDependencies`` are the inputs (datasets, base checkpoints, the code commit),
``externalParameters`` is the run configuration and ``runDetails`` the builder (machine, library versions, times).
Signed with :func:`pinneapple_security.signing.sign_json`, it becomes evidence a third party can check without trusting
the person who trained the model.

The SBOM lists the installed Python distributions (name, version, purl, licence) in CycloneDX 1.5 JSON, the format
supply-chain scanners (Dependency-Track, Grype, Trivy) read.
"""
from __future__ import annotations

import os
import platform
import subprocess
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from .integrity import digest_record, sha256_file

__all__ = ["subject", "statement", "provenance", "attest_files", "attest_experiment", "sbom", "git_commit",
           "STATEMENT_TYPE", "SLSA_PROVENANCE"]

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
SLSA_PROVENANCE = "https://slsa.dev/provenance/v1"
BUILD_TYPE = "https://pinneapple.org/attestation/run/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def subject(name: str, path: str | None = None, sha256: str | None = None) -> dict[str, Any]:
    if sha256 is None:
        if path is None:
            raise ValueError("give the file path or its sha256")
        sha256 = sha256_file(path)
    return {"name": name, "digest": {"sha256": sha256}}


def git_commit(path: str = ".") -> dict[str, Any] | None:
    """Commit of the repository containing ``path`` and whether the work tree is dirty (None outside git)."""
    try:
        sha = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if sha.returncode:
            return None
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        url = subprocess.run(["git", "-C", path, "config", "--get", "remote.origin.url"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        return {"uri": url or None, "digest": {"gitCommit": sha.stdout.strip()}, "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return None


def _builder() -> dict[str, Any]:
    env = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    for mod in ("numpy", "torch", "pinneapple"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def provenance(external_parameters: dict[str, Any], dependencies: Sequence[dict[str, Any]] = (),
               started_on: str | None = None, finished_on: str | None = None,
               builder_id: str = "https://pinneapple.org/builder/local", internal_parameters: dict | None = None,
               build_type: str = BUILD_TYPE) -> dict[str, Any]:
    return {
        "buildDefinition": {"buildType": build_type, "externalParameters": external_parameters,
                            "internalParameters": internal_parameters or {}, "resolvedDependencies": list(dependencies)},
        "runDetails": {"builder": {"id": builder_id, "version": _builder()},
                       "metadata": {"invocationId": str(uuid.uuid4()), "startedOn": started_on or _now(),
                                    "finishedOn": finished_on or _now()}},
    }


def statement(subjects: Sequence[dict[str, Any]], predicate: dict[str, Any],
              predicate_type: str = SLSA_PROVENANCE) -> dict[str, Any]:
    if not subjects:
        raise ValueError("an attestation needs at least one subject")
    return {"_type": STATEMENT_TYPE, "subject": list(subjects), "predicateType": predicate_type, "predicate": predicate}


def attest_files(artifacts: dict[str, str] | Sequence[str], parameters: dict[str, Any],
                 inputs: dict[str, str] | Sequence[str] = (), repo: str | None = ".") -> dict[str, Any]:
    """Statement for output files (``name → path`` or paths) built from input files with ``parameters``."""
    def _pairs(x):
        return list(x.items()) if isinstance(x, dict) else [(os.path.basename(p), p) for p in x]

    deps = [{"name": n, "digest": {"sha256": sha256_file(p)}} for n, p in _pairs(inputs)]
    commit = git_commit(repo) if repo else None
    if commit:
        deps.append({"name": "source", **commit})
    return statement([subject(n, p) for n, p in _pairs(artifacts)], provenance(parameters, deps))


def attest_experiment(result: Any, artifacts: dict[str, str] | Sequence[str] = (),
                      inputs: dict[str, str] | Sequence[str] = (), repo: str | None = ".") -> dict[str, Any]:
    """Statement for a ``pp.Experiment`` result: the record itself (by digest) and any saved files are the subjects,
    the config and problem fingerprint are the parameters, the metrics go in ``internalParameters``."""
    rec = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    subjects = [subject("experiment-record.json", sha256=digest_record(rec))]
    pairs = list(artifacts.items()) if isinstance(artifacts, dict) else [(os.path.basename(p), p) for p in artifacts]
    subjects += [subject(n, p) for n, p in pairs]
    ipairs = list(inputs.items()) if isinstance(inputs, dict) else [(os.path.basename(p), p) for p in inputs]
    deps = [{"name": n, "digest": {"sha256": sha256_file(p)}} for n, p in ipairs]
    deps.append({"name": "physical-problem", "digest": {"sha256": str(rec.get("problem_fingerprint", ""))}})
    commit = git_commit(repo) if repo else None
    if commit:
        deps.append({"name": "source", **commit})
    pred = provenance(rec.get("config", {}), deps, started_on=rec.get("started_at"),
                      internal_parameters={"metrics": rec.get("metrics"), "wall_time_s": rec.get("wall_time_s")})
    return statement(subjects, pred)


def _licence(meta) -> str | None:
    lic = meta.get("License-Expression") or meta.get("License")
    if lic and len(lic) < 100 and "\n" not in lic:
        return lic
    for c in meta.get_all("Classifier") or []:
        if c.startswith("License ::"):
            return c.split("::")[-1].strip()
    return None


def sbom(name: str = "pinneapple-environment", version: str = "", packages: Iterable[str] | None = None) -> dict[str, Any]:
    """CycloneDX 1.5 SBOM of the installed distributions (all, or only ``packages``)."""
    from importlib import metadata

    wanted = {p.lower().replace("_", "-") for p in packages} if packages else None
    comps, seen = [], set()
    for d in metadata.distributions():
        n = (d.metadata.get("Name") or "").strip()
        key = n.lower().replace("_", "-")
        if not n or key in seen or (wanted is not None and key not in wanted):
            continue
        seen.add(key)
        c = {"type": "library", "name": n, "version": d.version, "purl": f"pkg:pypi/{key}@{d.version}",
             "bom-ref": f"pkg:pypi/{key}@{d.version}"}
        lic = _licence(d.metadata)
        if lic:
            c["licenses"] = [{"license": {"name": lic}}]
        comps.append(c)
    comps.sort(key=lambda c: c["name"].lower())
    return {"bomFormat": "CycloneDX", "specVersion": "1.5", "serialNumber": f"urn:uuid:{uuid.uuid4()}", "version": 1,
            "metadata": {"timestamp": _now(), "component": {"type": "application", "name": name, "version": version},
                         "tools": [{"vendor": "PINNeAPPle", "name": "pinneapple_security.sbom"}]},
            "components": comps}

"""Conservative candidate discovery from a repository tree."""
from pathlib import PurePosixPath


def discover_candidates(inventory, module_prefix="pinneapple_"):
    """Return candidate modules, never inferred implementation or validation status."""
    candidates = {}
    for path in inventory["paths"]:
        parts = PurePosixPath(path).parts
        if len(parts) == 2 and parts[0].startswith(module_prefix) and parts[1] == "__init__.py":
            candidates[parts[0]] = {"key": parts[0], "status": "unknown",
                                     "source": "repository_tree", "path": path,
                                     "repository": inventory["repository"], "ref": inventory["ref"]}
    return sorted(candidates.values(), key=lambda item: item["key"])

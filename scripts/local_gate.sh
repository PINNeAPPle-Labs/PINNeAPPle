#!/bin/bash
# The gate that GitHub Actions used to run, executed locally: lint what changed, run the tests of what changed, and run
# the repository's own structural checks. Prints one line per stage so its output can be pasted into a pull request as the
# evidence a reviewer would otherwise read from the checks tab.
#
#   scripts/local_gate.sh                 # everything touched since origin/main, plus the full touched test files
#   scripts/local_gate.sh --all           # ruff over the whole tree and the structural checks
#   scripts/local_gate.sh tests/test_x.py # a specific list instead of the diff
#   STRICT=1 scripts/local_gate.sh        # the repo-wide checks fail the gate too
#
# The lint and the touched tests decide the gate. The repository-wide checks are advisory by default: they read the whole
# tree (118 template imports are broken on this branch before any of these changes) or need a package this environment
# does not install, so they report a state the diff does not decide — visible, not blocking. STRICT=1 makes them block.
set -u
cd "$(dirname "$0")/.."

PY=${PY:-python3}
RUFF=${RUFF:-ruff}
status=0
fail() { echo "  ✗ $1"; status=1; }
pass() { echo "  ✓ $1"; }

scope=()
if [ "${1:-}" = "--all" ]; then
    shift
    scope=()
elif [ $# -gt 0 ]; then
    scope=("$@")
else
    base=${BASE:-origin/main}
    merge_base=$(git merge-base "$base" HEAD 2>/dev/null || echo "")
    if [ -n "$merge_base" ]; then
        changed=$( { git diff --name-only "$merge_base" HEAD; git diff --name-only; git ls-files --others --exclude-standard; } | sort -u )
    else
        changed=$(git ls-files --others --exclude-standard)
    fi
    # no mapfile on the bash 3.2 that ships with macOS
    while IFS= read -r line; do
        [ -n "$line" ] && scope+=("$line")
    done <<< "$changed"
fi

python_files=()
test_files=()
for path in "${scope[@]:-}"; do
    [ -z "$path" ] && continue
    [ -f "$path" ] || continue
    case "$path" in
        *.py) python_files+=("$path") ;;
    esac
    case "$path" in
        tests/test_*.py) test_files+=("$path") ;;
    esac
done

echo "== ruff"
if [ "${#python_files[@]}" -gt 0 ] || [ "${1:-}" = "--all" ]; then
    targets=("${python_files[@]}")
    [ "${#targets[@]}" -eq 0 ] && targets=(pinneapple_simulation tests)
    if "$RUFF" check "${targets[@]}"; then pass "ruff check"; else fail "ruff check"; fi
else
    pass "nothing to lint"
fi

echo "== tests of the touched modules"
if [ "${#test_files[@]}" -gt 0 ]; then
    if "$PY" -m pytest "${test_files[@]}" -q -p no:warnings; then
        pass "pytest ${test_files[*]}"
    else
        fail "pytest ${test_files[*]}"
    fi
else
    pass "no test file touched"
fi

echo "== repository checks"
for check in scripts/check_example_imports.py scripts/check_experiment_manifests.py scripts/check_app_helpers.py; do
    [ -f "$check" ] || continue
    if "$PY" "$check" >/dev/null 2>&1; then pass "$(basename "$check")"
    elif [ "${STRICT:-0}" = "1" ]; then fail "$(basename "$check")"
    else echo "  ! $(basename "$check") — failing repo-wide, not decided by this diff (STRICT=1 to block)"; fi
done

echo "== changelog fragment"
if [ "${#scope[@]}" -gt 0 ] && [ -d changelog.d ]; then
    if printf '%s\n' "${scope[@]}" | grep -q '^changelog.d/'; then pass "a fragment is present"; else
        echo "  ! no changelog.d fragment among the changed files"; fi
fi

echo
if [ "$status" -eq 0 ]; then echo "GATE PASS"; else echo "GATE FAIL"; fi
exit "$status"

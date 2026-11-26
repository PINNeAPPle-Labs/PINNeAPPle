#!/bin/bash
# The session that gets recorded: terminal on the left, figures on the right, silent on purpose.
#
# Each figure is announced with a line "[figure] <file>  <caption>". The renderer reads that line in the asciinema
# cast and switches the right-hand panel at the instant it appears -- the panel keeps no clock of its own, it follows
# the terminal. That is why the line is printed once the stage that measured the figure has finished, and not before.
set -u
export TERM=xterm-256color
export PYTHONUNBUFFERED=1
D=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$D/../../.." && pwd)
PY=${PYTHON:-python3}
cd "$ROOT" || exit 1
REL=${D#"$ROOT"/}

blue()  { printf '\033[1;38;2;88;166;255m%s\033[0m\n' "$1"; }
grey()  { printf '\033[38;2;139;148;158m%s\033[0m\n' "$1"; }
red()   { printf '\033[1;38;2;248;81;73m%s\033[0m\n' "$1"; }
rule()  { printf '\033[38;2;48;54;61m%s\033[0m\n' "$(printf '─%.0s' $(seq 1 112))"; }
from_readme() { sed -n "/^$1/,/^$2/p" "$D/README.md" | sed '$d' | head -18 \
    | sed 's/^### //; s/\*\*//g; s/`//g; s/^|/ /; s/|/ /g; s/^/  /'; }

# announce a figure: the right-hand panel switches on this line
panel() { printf '\033[38;2;88;166;255m[figure] %s\033[0m  \033[38;2;230;237;243m%s\033[0m\n' "$1" "$2"; }

# run a stage, showing that it is working, then show what it printed
stage() {
    local script=$1 pid
    "$PY" "$D/$script" > /tmp/stage.log 2>&1 &
    pid=$!
    printf '  \033[38;2;88;166;255m.\033[0m running %s ' "$script"
    while kill -0 "$pid" 2>/dev/null; do
        printf '.'
        sleep 0.5
    done
    wait "$pid"
    printf '\n\n'
    grep -vE "ExperimentalWarning|from pinneapple|^[[:space:]]*figures:" /tmp/stage.log | sed 's/^/  /'
}

clear
sleep 2
blue "Trust beyond accuracy"
grey "the PINNeAPPle verification pipeline, end to end, on a family of airfoils"
echo
printf '\033[1;97m  How do you know when to trust a Physics AI model?\033[0m\n'
echo
grey "every number in what follows is produced by the scripts in $REL"
grey "and every figure is the one the stage that measured it wrote"
sleep 5

panel doe_plan.png "stage 1 -- the designs the campaign actually ran"
sleep 4

clear
blue "The study"
rule
ls -1 "$D" | grep -v '^_\|__pycache__' | sed 's/^/  /'
echo
sed -n '3,13p' "$D/README.md" | sed 's/^/  /'
sleep 9

clear
blue "Stage 2 -- the reference is a measurement, not the truth"
rule
grey "four meshes, every spacing refined by the same factor 1.45; the grid-convergence study is inside this stage"
echo
stage 02_analysis.py
panel mesh_convergence.png "stage 2 -- four meshes, and what the finest triplet lets you claim"
sleep 7
panel pod_spectrum.png "stage 2 -- the basis, before any network is involved"
sleep 6
panel held_out_parity.png "stage 2 -- the sixteen designs that were never trained on"
sleep 7

clear
blue "What that says about the reference"
rule
from_readme "### 1. The reference" "### 2. The surrogate"
sleep 10

clear
blue "Stage 3 -- is it a flow, or only a fit?"
rule
grey "the verification pipeline's own gates, on a bundle built over the lattice"
echo
stage 03_physics_validation.py
panel physics_gates.png "stage 3 -- the surrogate beside the solver that made its training data"
sleep 7
panel fields.png "stage 3 -- where the model is wrong, point by point along the chord"
sleep 7
panel sensitivity.png "stage 3 -- robustness, measured rather than promised"
sleep 6

clear
blue "The gates, and what they are judged against"
rule
from_readme "### 4. Is it a flow" "### 5. Uncertainty"
sleep 11

clear
blue "Stage 4 -- where the evidence supports a prediction"
rule
grey "28 designs the model was not trained on, each with its error against its own simulation,"
grey "two physical checks and the ensemble spread"
echo
stage 04_validated_domain.py
panel domain_map.png "stage 4 -- the supported region is a shape, not a range"
sleep 9

clear
blue "Four questions a range check cannot tell apart"
rule
from_readme "### 6. Where the prediction is supported" "### 7. Opening a parameter"
sleep 12

clear
blue "Stage 6 -- evidence arriving after the model is frozen"
rule
grey "the same weights, twenty-one more independent references, and the answers that change"
echo
stage 06_revalidation.py
panel extend.png "stage 6 -- the same model, more evidence, and the geometries it unlocks"
sleep 9

clear
blue "Stage 7 -- the dossier"
rule
grey "a check reads passed only when a sealed record shows that it ran"
echo
stage 07_report.py
panel accuracy.png "stage 7 -- the field and the scalar disagree, and the report says so"
sleep 7
panel uncertainty.png "stage 7 -- what the ensemble spread is actually worth"
sleep 7

clear
blue "What the report says"
rule
sed -n '/^## Checks recorded/,/^## Reference evidence/p' "$D/results/dossier.md" | sed 's/^/  /'
echo
red "  Status: FAILED (technical)"
grey "  3 of 11 checks failed and 4 of the 16 held-out designs are above the declared 5 % lift tolerance."
sleep 10

clear
blue "What the report refuses to sign"
rule
sed -n '/^## Findings/,/^### VP-004/p' "$D/results/dossier.md" | grep -E "^### |^- \*\*What happened" | head -8 | sed 's/^/  /' | cut -c1-116
sleep 10

clear
blue "A model that tells you where its predictions are supported."
echo
grey "python $REL/run_all.py"
echo
grey "results/dossier.md -- every number above has a file behind it"
grey "results/figures/ -- and every figure was drawn by the stage that measured it"
sleep 6

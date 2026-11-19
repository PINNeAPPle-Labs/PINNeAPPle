#!/usr/bin/env bash
# The LPM campaign in one command: plan it, then start it detached so it outlives the terminal.
#
#   ./benchmarks/run_lpm_campaign.sh                 # plan, then start the whole campaign in its own session
#   ./benchmarks/run_lpm_campaign.sh --list          # the menu: what each design space predicts and how big it is
#   ./benchmarks/run_lpm_campaign.sh --status        # what already exists, and what each model scored
#   ./benchmarks/run_lpm_campaign.sh --plan          # the plan only, nothing starts
#   ./benchmarks/run_lpm_campaign.sh --stage dataset # only the datasets
#   ./benchmarks/run_lpm_campaign.sh --stage train   # only the fits (the datasets must be there)
#   ./benchmarks/run_lpm_campaign.sh --families pp_wing3d --geometries 800 --epochs 900   # one family, bigger
#   ./benchmarks/run_lpm_campaign.sh --families pp_fin pp_wing3d --scale smoke   # just those, tiny
#
# The sizes come from benchmarks/lpm_recipes.json; --geometries, --conditions, --epochs and --width override them for
# the families this run selects, without touching the file.
#
# The output goes OUTSIDE the repository by default: the datasets are the bulk of the campaign and a checkout is not
# the place for 30 GB. Override with OUT=... LOG=...
set -euo pipefail

PYTHON="${PYTHON:-python3.11}"
OUT="${OUT:-$HOME/lpm-campaign}"
LOG="${LOG:-$OUT/campaign.log}"
cd "$(dirname "$0")/.."

args=("$@")
# the read-only views answer straight away; nothing is planned and nothing is detached
for view in --list --status --plan; do
  if [[ " ${args[*]-} " == *" $view "* ]]; then
    exec "$PYTHON" -m benchmarks.lpm_campaign --out "$OUT" "${args[@]}"
  fi
done

"$PYTHON" -m benchmarks.lpm_campaign --out "$OUT" "${args[@]}" --plan
echo
echo "starting in its own session; output in $OUT"
"$PYTHON" -m benchmarks.lpm_campaign --out "$OUT" "${args[@]}" --detach "$LOG"
cat <<EOF

  follow:   tail -f $LOG
  summary:  $OUT/campaign.json     (held-out error, baseline and OOD probes per family)
  failures: $OUT/failures.json     (only if a family raised; the rest still ran)
  resume:   re-run this script — a dataset with a manifest and a model with a card are skipped

EOF

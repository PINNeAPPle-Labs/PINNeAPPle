#!/bin/bash
# Record the session headless, draw the figure panel from the cast, and put the two side by side.
#
#   bash examples/end_to_end/airfoil_trust/record_demo.sh [out_dir]
#
# Nothing opens a window: asciinema records without a terminal, agg turns the cast into video and ffmpeg does the
# hstack. The right-hand panel reads the instants the terminal itself wrote, so it stays aligned whatever the run
# cost. The files are large and are not committed; out_dir defaults to the temporary directory.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
OUT=${1:-${TMPDIR:-/tmp}/airfoil-trust-demo}
PY=${PYTHON:-python3}
CAST=$OUT/session.cast
TERM=$OUT/terminal-only.mp4
PANEL=$OUT/figures-only.mp4
VIDEO=$OUT/trust-beyond-accuracy-screen.mp4
mkdir -p "$OUT"

cd "$ROOT"
echo "== recording (headless, $(date +%H:%M:%S))"
asciinema rec --headless --window-size 106x48 --overwrite -c "bash $HERE/demo_session.sh" "$CAST"

echo "== terminal -> video"
agg --theme github-dark --font-size 15 --line-height 1.35 --idle-time-limit 1000 --last-frame-duration 1 \
    "$CAST" "$TERM"

echo "== figure panel"
"$PY" "$HERE/demo_panel.py" "$CAST" "$PANEL" --width 840 --height 1080

echo "== side by side"
ffmpeg -y -loglevel error -i "$TERM" -i "$PANEL" -filter_complex \
  "[0:v]scale=1036:1040:force_original_aspect_ratio=decrease,pad=1080:1080:22:20:color=0x0d1117[l];\
   [1:v]scale=840:1080[r];[l][r]hstack=inputs=2,format=yuv420p[v]" \
  -map "[v]" -an -c:v libx264 -preset medium -crf 20 "$VIDEO"

ls -lh "$VIDEO"
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$VIDEO"

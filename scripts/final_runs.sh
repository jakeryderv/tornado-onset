#!/usr/bin/env bash
# The baseline and the three experiments, back to back, then the summary table.
# Usage: bash scripts/final_runs.sh   (from the repo root; runs land under $EXP_DIR or experiments/)
set -u
cd "$(dirname "$0")/.."
export EXP_DIR="${EXP_DIR:-$PWD/experiments}"
mkdir -p "$EXP_DIR"
log() { echo "=== $(date -Is) $*"; }
for cfg in baseline frozen_encoder one_scan extrapolate; do
  log "run $cfg"
  if uv run onset-train "configs/$cfg.json" > "$EXP_DIR/$cfg.log" 2>&1; then log "done $cfg"; else log "RUN FAILED $cfg"; fi
done
log "all done"
uv run onset-summary "$EXP_DIR" 2>&1 | grep -v -i warning

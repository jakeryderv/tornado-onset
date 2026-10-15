#!/usr/bin/env bash
# Run configs back to back on a GPU pod (repo synced to $REPO, data and weights from the Hub),
# then print the summary table. The pod is NOT stopped here: experiments/ lives on the container
# disk, so pull it first, then terminate the pod.
# Usage: cd $REPO && UV_PROJECT_ENVIRONMENT=/root/venv setsid nohup \
#   bash scripts/pod_run.sh configs/a.json "configs/b.json --set seed=2" > /root/pod_run.log 2>&1 < /dev/null &
set -uo pipefail
REPO="${REPO:-/root/tornado-onset}"
export EXP_DIR="${EXP_DIR:-$REPO/experiments}"
cd "$REPO"
log() { echo "=== $(date -Is) $*"; }
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
log "syncing environment"; uv sync --quiet || { log "uv sync failed"; exit 1; }
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
mkdir -p "$EXP_DIR"
for spec in "$@"; do
  log "run $spec"
  # shellcheck disable=SC2086
  if uv run onset-train $spec > "$EXP_DIR/$(basename "${spec%% *}" .json).$(date +%H%M%S).log" 2>&1; then
    log "done $spec"
  else
    log "RUN FAILED $spec"
  fi
done
log "all done"
uv run onset-summary "$EXP_DIR" 2>&1 | grep -v -i warning

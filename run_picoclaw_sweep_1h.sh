#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --- User-tunable knobs ---
TOTAL_SECONDS="${TOTAL_SECONDS:-3500}"               # ~58 min default
PER_RUN_TIMEOUT_SECONDS="${PER_RUN_TIMEOUT_SECONDS:-900}" # 15 min per individual run safety
LOGDIR_BASE="${LOGDIR_BASE:-runs}"

# Suite attack pairs run in this order; script stops when TOTAL_SECONDS elapse.
SUITES=(banking workspace slack travel)

# Representative injection task per suite to keep runtime down.
INJECTION_TASK_BY_SUITE_banking="injection_task_0"
INJECTION_TASK_BY_SUITE_workspace="injection_task_0"
INJECTION_TASK_BY_SUITE_slack="injection_task_1"
INJECTION_TASK_BY_SUITE_travel="injection_task_0"

ATTACKS=(tool_knowledge important_instructions dos)

if ! command -v docker >/dev/null 2>&1; then
  echo "Missing 'docker' on PATH" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Missing 'uv' on PATH" >&2
  exit 1
fi

# --- Environment assumptions ---
# 1) You already activated the right virtualenv in your shell, e.g.:
#      source .venv/bin/activate
# 2) You already exported AZURE_API_KEY, e.g. using your helper one-liner.
if [[ -z "${AZURE_API_KEY:-}" ]]; then
  echo "AZURE_API_KEY is not set. Export it before running this script." >&2
  exit 1
fi

# --- Logging ---
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$LOGDIR_BASE/picoclaw_sweep_1h_$TS"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run.log"

echo "Writing logs to: $LOG_FILE"
echo "Total budget seconds: $TOTAL_SECONDS"

START_EPOCH="$(date +%s)"
END_EPOCH="$((START_EPOCH + TOTAL_SECONDS))"

run_one() {
  local suite="$1"
  local attack="$2"
  local injection_tasks="$3"

  # Stop if we ran out of budget.
  local now
  now="$(date +%s)"
  if (( now >= END_EPOCH )); then
    echo "Time budget exhausted before suite=$suite attack=$attack; stopping."
    return 1
  fi

  echo "============================================================" | tee -a "$LOG_FILE"
  echo "Running: suite=$suite attack=$attack injection_tasks=$injection_tasks" | tee -a "$LOG_FILE"
  echo "Start: $(date)" | tee -a "$LOG_FILE"
  echo "============================================================" | tee -a "$LOG_FILE"

  # Ensure no stale container lingers.
  docker rm -f picoclaw-bench >/dev/null 2>&1 || true

  # Run inside the repo env with uv.
  set +e
  timeout "$PER_RUN_TIMEOUT_SECONDS" uv run python run_picoclaw_benchmark.py \
    --suite "$suite" \
    --attack "$attack" \
    --injection-tasks "$injection_tasks" \
    --logdir "$LOGDIR_BASE" \
    2>&1 | tee -a "$LOG_FILE"
  local rc="${PIPESTATUS[0]:-0}"
  set -e

  # Cleanup container even if the process timed out.
  docker rm -f picoclaw-bench >/dev/null 2>&1 || true

  echo "End: $(date) (exit_code=$rc)" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"

  # If the benchmark process timed out, keep going (best-effort) until budget runs out.
  return 0
}

for suite in "${SUITES[@]}"; do
  for attack in "${ATTACKS[@]}"; do
    injection_tasks_var="INJECTION_TASK_BY_SUITE_${suite}"
    injection_tasks="${!injection_tasks_var}"
    # shellcheck disable=SC2154
    run_one "$suite" "$attack" "$injection_tasks" || exit 0
  done
done

echo "All scheduled runs completed (or time budget exhausted)."
echo "Logs: $LOG_FILE"


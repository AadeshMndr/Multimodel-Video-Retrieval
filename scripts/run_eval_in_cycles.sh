#!/usr/bin/env bash

set -euo pipefail

# Defaults can be overridden via env vars.
# Prefer minute-based knobs; keep second-based overrides for compatibility.
RUN_MINUTES="${RUN_MINUTES:-7}"
WAIT_MINUTES="${WAIT_MINUTES:-10}"
RUN_SECONDS="${RUN_SECONDS:-$((RUN_MINUTES * 60))}"
WAIT_SECONDS="${WAIT_SECONDS:-$((WAIT_MINUTES * 60))}"
EVAL_CMD="${EVAL_CMD:-python scripts/evaluate_retrieval.py --enable-random-search-cv --random-seed 42}"
MAX_GRACEFUL_STOP_SECONDS="${MAX_GRACEFUL_STOP_SECONDS:-30}"

if ! [[ "${RUN_SECONDS}" =~ ^[0-9]+$ ]] || (( RUN_SECONDS <= 0 )); then
  echo "[cycle-runner] RUN_SECONDS must be a positive integer (current: ${RUN_SECONDS})."
  exit 2
fi

if ! [[ "${WAIT_SECONDS}" =~ ^[0-9]+$ ]] || (( WAIT_SECONDS <= 0 )); then
  echo "[cycle-runner] WAIT_SECONDS must be a positive integer (current: ${WAIT_SECONDS})."
  exit 2
fi

child_pid=""

cleanup() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    echo "[cycle-runner] Interrupt received. Stopping active evaluation process ${child_pid}..."
    kill -INT "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  echo "[cycle-runner] Exiting."
}

trap cleanup INT TERM

cycle=0

while true; do
  cycle=$((cycle + 1))
  start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  run_minutes_display=$((RUN_SECONDS / 60))
  wait_minutes_display=$((WAIT_SECONDS / 60))
  echo "[cycle-runner] Cycle ${cycle} started at ${start_ts}"
  echo "[cycle-runner] Running for up to ${RUN_SECONDS}s (~${run_minutes_display}m): ${EVAL_CMD}"

  bash -lc "${EVAL_CMD}" &
  child_pid=$!

  elapsed=0
  while (( elapsed < RUN_SECONDS )); do
    if ! kill -0 "${child_pid}" 2>/dev/null; then
      break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  if kill -0 "${child_pid}" 2>/dev/null; then
    echo "[cycle-runner] Time slice reached (${RUN_SECONDS}s). Sending SIGINT for graceful checkpoint + stop..."
    kill -INT "${child_pid}" 2>/dev/null || true

    grace=0
    while (( grace < MAX_GRACEFUL_STOP_SECONDS )); do
      if ! kill -0 "${child_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
      grace=$((grace + 1))
    done

    if kill -0 "${child_pid}" 2>/dev/null; then
      echo "[cycle-runner] Process did not stop after ${MAX_GRACEFUL_STOP_SECONDS}s; sending SIGTERM..."
      kill -TERM "${child_pid}" 2>/dev/null || true
    fi
  fi

  # wait also collects the true exit code in all cases.
  set +e
  wait "${child_pid}"
  exit_code=$?
  set -e
  child_pid=""

  end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[cycle-runner] Cycle ${cycle} finished at ${end_ts} with exit code ${exit_code}"

  # Exit code 0 means evaluator finished naturally (nothing left / done).
  if [[ ${exit_code} -eq 0 ]]; then
    echo "[cycle-runner] Evaluation finished naturally. Stopping cycle runner."
    break
  fi

  echo "[cycle-runner] Waiting ${WAIT_SECONDS}s (~${wait_minutes_display}m) before restarting..."
  sleep "${WAIT_SECONDS}"
done

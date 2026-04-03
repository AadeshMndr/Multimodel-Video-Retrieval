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
MAX_CONSECUTIVE_OUTPUT_PARSER_ERRORS="${MAX_CONSECUTIVE_OUTPUT_PARSER_ERRORS:-3}"
MAX_CONSECUTIVE_PAYLOAD_TOO_LARGE_ERRORS="${MAX_CONSECUTIVE_PAYLOAD_TOO_LARGE_ERRORS:-3}"

REPORTS_DIR="${REPORTS_DIR:-evaluation/reports}"
RUNS_DIR="${RUNS_DIR:-${REPORTS_DIR}/runs}"

if ! [[ "${RUN_SECONDS}" =~ ^[0-9]+$ ]] || (( RUN_SECONDS <= 0 )); then
  echo "[cycle-runner] RUN_SECONDS must be a positive integer (current: ${RUN_SECONDS})."
  exit 2
fi

if ! [[ "${WAIT_SECONDS}" =~ ^[0-9]+$ ]] || (( WAIT_SECONDS <= 0 )); then
  echo "[cycle-runner] WAIT_SECONDS must be a positive integer (current: ${WAIT_SECONDS})."
  exit 2
fi

if ! [[ "${MAX_CONSECUTIVE_OUTPUT_PARSER_ERRORS}" =~ ^[0-9]+$ ]] || (( MAX_CONSECUTIVE_OUTPUT_PARSER_ERRORS < 0 )); then
  echo "[cycle-runner] MAX_CONSECUTIVE_OUTPUT_PARSER_ERRORS must be a non-negative integer (current: ${MAX_CONSECUTIVE_OUTPUT_PARSER_ERRORS})."
  exit 2
fi

if ! [[ "${MAX_CONSECUTIVE_PAYLOAD_TOO_LARGE_ERRORS}" =~ ^[0-9]+$ ]] || (( MAX_CONSECUTIVE_PAYLOAD_TOO_LARGE_ERRORS < 0 )); then
  echo "[cycle-runner] MAX_CONSECUTIVE_PAYLOAD_TOO_LARGE_ERRORS must be a non-negative integer (current: ${MAX_CONSECUTIVE_PAYLOAD_TOO_LARGE_ERRORS})."
  exit 2
fi

child_pid=""
output_parser_streak=0
payload_too_large_streak=0
latest_run_state_path=""
last_error_kind=""
last_error_message=""

log_stage() {
  echo "[cycle-runner] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

normalize_text() {
  tr '[:upper:]' '[:lower:]'
}

read_latest_run_state_error() {
  latest_run_state_path=""
  last_error_kind=""
  last_error_message=""

  if [[ ! -d "${RUNS_DIR}" ]]; then
    return 0
  fi

  local py_out=""
  local py_status=0
  set +e
  py_out="$(python3 - "${RUNS_DIR}" <<'PY'
import glob
import json
import os
import sys

runs_dir = sys.argv[1]
paths = glob.glob(os.path.join(runs_dir, "*", "run_state.json"))
if not paths:
    print("")
    print("")
    print("")
    sys.exit(0)

latest = max(paths, key=os.path.getmtime)
kind = ""
message = ""
try:
    with open(latest, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    last_error = payload.get("last_error") or {}
    if isinstance(last_error, dict):
        kind = str(last_error.get("kind") or "")
        message = str(last_error.get("message") or "")
except Exception:
    pass

print(latest)
print(kind)
print(message.replace("\n", " ").strip())
PY
)"
  py_status=$?
  set -e

  if (( py_status != 0 )); then
    return 0
  fi

  latest_run_state_path="$(printf '%s\n' "${py_out}" | sed -n '1p')"
  last_error_kind="$(printf '%s\n' "${py_out}" | sed -n '2p')"
  last_error_message="$(printf '%s\n' "${py_out}" | sed -n '3p')"
}

is_payload_too_large_error() {
  local kind="$1"
  local message_lower
  message_lower="$(printf '%s' "$2" | normalize_text)"

  if [[ "${kind}" != "APIStatusError" ]]; then
    return 1
  fi

  if [[ "${message_lower}" == *"request entity too large"* ]] || \
     [[ "${message_lower}" == *"payload too large"* ]] || \
     [[ "${message_lower}" == *"request_too_large"* ]] || \
     [[ "${message_lower}" == *"entity too large"* ]]; then
    return 0
  fi

  return 1
}

cleanup() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    log_stage "Interrupt received. Stopping active evaluation process ${child_pid}..."
    kill -INT "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  log_stage "Exiting."
}

trap cleanup INT TERM

cycle=0

while true; do
  cycle=$((cycle + 1))
  start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  run_minutes_display=$((RUN_SECONDS / 60))
  wait_minutes_display=$((WAIT_SECONDS / 60))
  log_stage "Cycle ${cycle} started at ${start_ts}."
  log_stage "Stage: launch evaluator for up to ${RUN_SECONDS}s (~${run_minutes_display}m)."
  log_stage "Command: ${EVAL_CMD}"

  bash -lc "${EVAL_CMD}" &
  child_pid=$!

  log_stage "Stage: monitoring evaluator process ${child_pid}."
  elapsed=0
  while (( elapsed < RUN_SECONDS )); do
    if ! kill -0 "${child_pid}" 2>/dev/null; then
      break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  if kill -0 "${child_pid}" 2>/dev/null; then
    log_stage "Time slice reached (${RUN_SECONDS}s). Sending SIGINT for graceful checkpoint + stop..."
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
      log_stage "Process did not stop after ${MAX_GRACEFUL_STOP_SECONDS}s; sending SIGTERM..."
      kill -TERM "${child_pid}" 2>/dev/null || true
    fi
  fi

  # wait also collects the true exit code in all cases.
  log_stage "Stage: collecting evaluator exit code."
  set +e
  wait "${child_pid}"
  exit_code=$?
  set -e
  child_pid=""

  end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  log_stage "Cycle ${cycle} finished at ${end_ts} with exit code ${exit_code}."

  # Exit code 0 means evaluator finished naturally (nothing left / done).
  if [[ ${exit_code} -eq 0 ]]; then
    log_stage "Evaluation finished naturally with no errors. Stopping cycle runner."
    break
  fi

  log_stage "Stage: inspecting latest run_state.json for error classification."
  read_latest_run_state_error

  if [[ -n "${latest_run_state_path}" ]]; then
    log_stage "Latest run_state: ${latest_run_state_path}"
  fi
  log_stage "Detected error kind='${last_error_kind}' message='${last_error_message}'"

  if [[ "${last_error_kind}" == "RateLimitError" ]]; then
    output_parser_streak=0
    payload_too_large_streak=0
    log_stage "Waiting for ${WAIT_SECONDS}s (~${wait_minutes_display}m) because of RateLimitError."
    sleep "${WAIT_SECONDS}"
    continue
  fi

  if [[ "${last_error_kind}" == "OutputParserException" ]]; then
    output_parser_streak=$((output_parser_streak + 1))
    payload_too_large_streak=0
    log_stage "OutputParserException streak=${output_parser_streak}."

    if (( output_parser_streak > MAX_CONSECUTIVE_OUTPUT_PARSER_ERRORS )); then
      log_stage "OutputParserException happened more than ${MAX_CONSECUTIVE_OUTPUT_PARSER_ERRORS} times in a row. Stopping cycle runner."
      break
    fi

    log_stage "Restarting immediately after OutputParserException."
    continue
  fi

  if is_payload_too_large_error "${last_error_kind}" "${last_error_message}"; then
    payload_too_large_streak=$((payload_too_large_streak + 1))
    output_parser_streak=0
    log_stage "APIStatusError(payload too large) streak=${payload_too_large_streak}."

    if (( payload_too_large_streak > MAX_CONSECUTIVE_PAYLOAD_TOO_LARGE_ERRORS )); then
      log_stage "APIStatusError payload-too-large happened more than ${MAX_CONSECUTIVE_PAYLOAD_TOO_LARGE_ERRORS} times in a row. Stopping cycle runner."
      break
    fi

    log_stage "Retrying immediately after payload-too-large APIStatusError."
    continue
  fi

  output_parser_streak=0
  payload_too_large_streak=0
  log_stage "Non-zero exit with unhandled/other error type. Waiting ${WAIT_SECONDS}s (~${wait_minutes_display}m) before restarting..."
  sleep "${WAIT_SECONDS}"
done

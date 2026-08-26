#!/usr/bin/env bash
#
# monitor.sh — show GPU/MIG utilisation and follow the latest training log.
#
# Prints a one-shot nvidia-smi snapshot (with per-MIG memory), the tmux session
# state, then tails the newest run's log. Ctrl-C stops the tail; it does NOT
# stop training (that runs inside the detached tmux session).
#
# Usage (on the server):
#     bash training/server/monitor.sh
#
# Environment overrides:
#     SESSION   tmux session name (default: sorbed-train)
#     REPO_DIR  repo checkout     (default: this script's repo root)
#     LOG_DIR   log directory     (default: <REPO_DIR>/logs)
#     LOG_FILE  specific log file (default: <LOG_DIR>/latest.log)
#     LINES     history lines to show before following (default: 40)
#     NO_FOLLOW set to 1 to print the snapshot and exit without tailing
#
set -euo pipefail

SESSION="${SESSION:-sorbed-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/latest.log}"
LINES="${LINES:-40}"

echo "=================================================================="
echo " GPU / MIG status"
echo "=================================================================="
if command -v nvidia-smi >/dev/null 2>&1; then
    # Top-level utilisation + memory, then the MIG instance breakdown.
    nvidia-smi
    echo
    echo "----- MIG instances (nvidia-smi -L) -----"
    nvidia-smi -L
else
    echo "nvidia-smi not found on this host."
fi

echo
echo "=================================================================="
echo " tmux session: ${SESSION}"
echo "=================================================================="
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux list-windows -t "${SESSION}"
else
    echo "No live tmux session '${SESSION}' (it may have finished or not started)."
fi

echo
echo "=================================================================="
echo " Log: ${LOG_FILE}"
echo "=================================================================="
if [[ ! -e "${LOG_FILE}" ]]; then
    echo "No log yet at ${LOG_FILE}."
    echo "Recent logs in ${LOG_DIR}:"
    ls -1t "${LOG_DIR}" 2>/dev/null | head -n 10 || echo "  (none)"
    exit 0
fi

if [[ "${NO_FOLLOW:-0}" == "1" ]]; then
    tail -n "${LINES}" "${LOG_FILE}"
else
    echo "(following — Ctrl-C stops the tail, not the training run)"
    echo
    tail -n "${LINES}" -f "${LOG_FILE}"
fi

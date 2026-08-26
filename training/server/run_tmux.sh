#!/usr/bin/env bash
#
# run_tmux.sh — launch a segmentation training run inside a detached tmux
# session on the H200, pinned to a single MIG slice, logging to a file.
#
# MIG is enabled on this H200, so the GPU is exposed as MIG *instances*, not as
# device index 0. You MUST target a MIG instance by its UUID. List them with:
#     nvidia-smi -L
# Example line:
#     GPU 0: NVIDIA H200 NVL (UUID: GPU-xxxx...)
#       MIG 3g.40gb Device 0: (UUID: MIG-1a2b3c4d-....)
# Pass that MIG-.... UUID as MIG_UUID.
#
# Usage (on the server):
#     MIG_UUID=MIG-1a2b3c4d-... bash training/server/run_tmux.sh
#
# Environment overrides:
#     MIG_UUID     (REQUIRED) MIG instance UUID for CUDA_VISIBLE_DEVICES
#     VENV_DIR     virtualenv          (default: /data/briefer/sorbed-venv)
#     REPO_DIR     repo checkout       (default: this script's repo root)
#     SESSION      tmux session name   (default: sorbed-train)
#     CONFIG       training YAML config
#                  (default: training/configs/seg_unetpp_effnet.yaml)
#     IMAGES_DIR   RGB image directory (default: <REPO_DIR>/data/fuseg/train/images)
#     MASKS_DIR    mask directory      (default: <REPO_DIR>/data/fuseg/train/labels)
#     OUT_DIR      artifact directory  (default: <REPO_DIR>/artifacts/seg_unetpp)
#     LOG_DIR      log directory       (default: <REPO_DIR>/logs)
#     EXTRA_ARGS   extra flags appended to the training command (e.g. "--epochs 40")
#
set -euo pipefail

VENV_DIR="${VENV_DIR:-/data/briefer/sorbed-venv}"
SESSION="${SESSION:-sorbed-train}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

CONFIG="${CONFIG:-training/configs/seg_unetpp_effnet.yaml}"
IMAGES_DIR="${IMAGES_DIR:-${REPO_DIR}/data/fuseg/train/images}"
MASKS_DIR="${MASKS_DIR:-${REPO_DIR}/data/fuseg/train/labels}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/artifacts/seg_unetpp}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"

# --- Preconditions -----------------------------------------------------------
if [[ -z "${MIG_UUID:-}" ]]; then
    echo "ERROR: MIG_UUID is not set. Find it with 'nvidia-smi -L' and re-run:" >&2
    echo "       MIG_UUID=MIG-xxxx... bash training/server/run_tmux.sh" >&2
    exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed on this server." >&2
    exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "ERROR: no virtualenv at ${VENV_DIR}. Run setup_env.sh first." >&2
    exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "ERROR: tmux session '${SESSION}' already exists." >&2
    echo "       Attach with: tmux attach -t ${SESSION}" >&2
    echo "       Or kill it:  tmux kill-session -t ${SESSION}" >&2
    exit 1
fi
if [[ ! -d "${IMAGES_DIR}" || ! -d "${MASKS_DIR}" ]]; then
    echo "ERROR: images/masks directory missing." >&2
    echo "       images: ${IMAGES_DIR}" >&2
    echo "       masks:  ${MASKS_DIR}" >&2
    echo "       Prepare data first (see training/RUNBOOK.md)." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/${SESSION}-${TIMESTAMP}.log"
# Stable symlink so monitor.sh can always find the newest run's log.
ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest.log"

# The training command. `stdbuf -oL -eL` keeps stdout line-buffered so `tail -f`
# sees progress immediately; `2>&1 | tee` mirrors to the log and the pane.
TRAIN_CMD=$(cat <<EOF
cd '${REPO_DIR}' \
  && source '${VENV_DIR}/bin/activate' \
  && export CUDA_VISIBLE_DEVICES='${MIG_UUID}' \
  && export PYTHONUNBUFFERED=1 \
  && export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  && echo "[run_tmux] session=${SESSION} mig=${MIG_UUID} log=${LOG_FILE}" \
  && stdbuf -oL -eL python -m training.train_seg \
       --config '${CONFIG}' \
       --images '${IMAGES_DIR}' \
       --masks '${MASKS_DIR}' \
       --out-dir '${OUT_DIR}' \
       --device cuda \
       ${EXTRA_ARGS:-} \
     2>&1 | tee '${LOG_FILE}' ; \
  echo "[run_tmux] training process exited with status \${PIPESTATUS[0]}"
EOF
)

echo "==> Launching training in detached tmux session '${SESSION}'"
echo "    config: ${CONFIG}"
echo "    images: ${IMAGES_DIR}"
echo "    masks:  ${MASKS_DIR}"
echo "    out:    ${OUT_DIR}"
echo "    mig:    ${MIG_UUID}"
echo "    log:    ${LOG_FILE}"

# Keep the pane alive after the process exits so you can read the final output.
tmux new-session -d -s "${SESSION}" -n train
tmux set-option -t "${SESSION}" remain-on-exit on
tmux send-keys -t "${SESSION}:train" "${TRAIN_CMD}" C-m

echo "==> Started. Useful commands:"
echo "    Attach:   tmux attach -t ${SESSION}   (detach again with Ctrl-b then d)"
echo "    Monitor:  bash training/server/monitor.sh"
echo "    Stop:     tmux kill-session -t ${SESSION}"

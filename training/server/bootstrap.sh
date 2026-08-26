#!/usr/bin/env bash
#
# bootstrap.sh — one-shot "curl and go" bring-up of a Sorbed segmentation
# training run on the H200 (MIG) server, straight from GitHub.
#
# It clones (or fast-forwards) the repo, builds the CUDA venv via setup_env.sh,
# fetches the open FUSeg foot-ulcer data via scripts/fetch_fuseg.py, and launches
# training in a detached tmux session via run_tmux.sh — then prints the monitor /
# attach commands. Every heavy step is delegated to a committed script, so this
# file is just the glue that makes the whole sequence a single pipe.
#
# Run it on the server (inside tmux is recommended for the clone/setup phase):
#
#     export MIG_UUID=MIG-xxxxxxxx-....      # from: nvidia-smi -L
#     curl -fsSL \
#       https://raw.githubusercontent.com/ArioMoniri/Sorbed/claude/bedsore-grading-system-wcd4ol/training/server/bootstrap.sh \
#       | bash
#
# Because the body is streamed to `bash` over a pipe (no local file, no argv),
# configuration is entirely by environment variable:
#
#     MIG_UUID     (REQUIRED unless SKIP_TRAIN=1) MIG instance UUID to pin the run
#     REPO_DIR     checkout location   (default: /data/briefer/sorbed)
#     BRANCH       branch to track     (default: claude/bedsore-grading-system-wcd4ol)
#     REPO_URL     git remote          (default: https://github.com/ArioMoniri/Sorbed)
#     CONFIG       training YAML        (default: training/configs/seg_unetpp_effnet.yaml)
#     CUDA_TAG     torch wheel index    (default: cu128 — driver 570 / CUDA 12.8)
#     VENV_DIR     virtualenv location  (default: /data/briefer/sorbed-venv)
#     DATA_DIR     FUSeg output dir     (default: <REPO_DIR>/data/fuseg)
#     TRAIN_LIMIT  train images to pull (default: 1000)
#     VAL_LIMIT    val images to pull   (default: 200)
#     SKIP_SETUP=1 skip setup_env.sh (reuse an existing venv)
#     SKIP_DATA=1  skip the FUSeg download (data already placed)
#     SKIP_TRAIN=1 stop after setup+data; do not launch training
#
# The script is idempotent: an existing checkout is fast-forwarded, an existing
# venv is reused, and already-present data is not re-downloaded. If a
# `sorbed-train` tmux session is already running, run_tmux.sh refuses to start a
# second one (re-run with SKIP_TRAIN=1, or kill the session first).
#
set -euo pipefail

# --- Configuration (all overridable via the environment) ---------------------
REPO_URL="${REPO_URL:-https://github.com/ArioMoniri/Sorbed}"
BRANCH="${BRANCH:-claude/bedsore-grading-system-wcd4ol}"
REPO_DIR="${REPO_DIR:-/data/briefer/sorbed}"
CONFIG="${CONFIG:-training/configs/seg_unetpp_effnet.yaml}"
CUDA_TAG="${CUDA_TAG:-cu128}"
VENV_DIR="${VENV_DIR:-/data/briefer/sorbed-venv}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/fuseg}"
TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
VAL_LIMIT="${VAL_LIMIT:-200}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- Preconditions -----------------------------------------------------------
command -v git >/dev/null 2>&1 || die "git is not installed on this server."
if [[ -z "${SKIP_TRAIN:-}" && -z "${MIG_UUID:-}" ]]; then
    die "MIG_UUID is not set. Find it with 'nvidia-smi -L', then re-run with
       MIG_UUID=MIG-xxxx... (or set SKIP_TRAIN=1 to only clone/setup/fetch)."
fi

log "Sorbed one-shot bootstrap"
echo "    repo url:  ${REPO_URL}"
echo "    branch:    ${BRANCH}"
echo "    checkout:  ${REPO_DIR}"
echo "    config:    ${CONFIG}"
echo "    cuda tag:  ${CUDA_TAG}"
echo "    venv:      ${VENV_DIR}"
echo "    data dir:  ${DATA_DIR}"

# --- 1. Clone or fast-forward the repo at the target branch ------------------
if [[ -d "${REPO_DIR}/.git" ]]; then
    log "Updating existing checkout at ${REPO_DIR}"
    git -C "${REPO_DIR}" remote set-url origin "${REPO_URL}"
    git -C "${REPO_DIR}" fetch --depth 1 origin "${BRANCH}"
    git -C "${REPO_DIR}" checkout -B "${BRANCH}" "origin/${BRANCH}"
    git -C "${REPO_DIR}" reset --hard "origin/${BRANCH}"
else
    log "Cloning ${REPO_URL} (${BRANCH}) into ${REPO_DIR}"
    mkdir -p "$(dirname "${REPO_DIR}")"
    git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi
echo "    at commit: $(git -C "${REPO_DIR}" rev-parse --short HEAD)"

# --- 2. Build the CUDA venv (delegates to the committed setup script) --------
if [[ -n "${SKIP_SETUP:-}" ]]; then
    log "SKIP_SETUP=1 — reusing venv at ${VENV_DIR} (no install)"
else
    log "Setting up the training venv (CUDA_TAG=${CUDA_TAG})"
    REPO_DIR="${REPO_DIR}" VENV_DIR="${VENV_DIR}" CUDA_TAG="${CUDA_TAG}" \
        bash "${REPO_DIR}/training/server/setup_env.sh"
fi

# --- 3. Fetch the open FUSeg foot-ulcer data ---------------------------------
if [[ -n "${SKIP_DATA:-}" ]]; then
    log "SKIP_DATA=1 — not downloading FUSeg (expecting data under ${DATA_DIR})"
elif [[ -d "${DATA_DIR}/train/images" && -d "${DATA_DIR}/validation/images" ]]; then
    log "FUSeg data already present at ${DATA_DIR} — skipping download"
else
    log "Fetching FUSeg (train=${TRAIN_LIMIT}, validation=${VAL_LIMIT}) into ${DATA_DIR}"
    "${VENV_DIR}/bin/python" "${REPO_DIR}/scripts/fetch_fuseg.py" \
        --out "${DATA_DIR}" --split train --limit "${TRAIN_LIMIT}"
    "${VENV_DIR}/bin/python" "${REPO_DIR}/scripts/fetch_fuseg.py" \
        --out "${DATA_DIR}" --split validation --limit "${VAL_LIMIT}"
fi

# --- 4. Launch training (delegates to run_tmux.sh) ---------------------------
if [[ -n "${SKIP_TRAIN:-}" ]]; then
    log "SKIP_TRAIN=1 — setup complete, not launching training"
    echo "    Launch later with:"
    echo "      MIG_UUID=MIG-xxxx... REPO_DIR=${REPO_DIR} \\"
    echo "        bash ${REPO_DIR}/training/server/run_tmux.sh"
    exit 0
fi

log "Launching training in detached tmux session 'sorbed-train'"
REPO_DIR="${REPO_DIR}" VENV_DIR="${VENV_DIR}" CONFIG="${CONFIG}" \
    IMAGES_DIR="${DATA_DIR}/train/images" MASKS_DIR="${DATA_DIR}/train/labels" \
    MIG_UUID="${MIG_UUID}" \
    bash "${REPO_DIR}/training/server/run_tmux.sh"

# --- 5. Next steps -----------------------------------------------------------
log "Bootstrap complete. Next steps:"
echo "    Monitor (Ctrl-C stops only the tail, not the run):"
echo "      bash ${REPO_DIR}/training/server/monitor.sh"
echo "    Attach the live pane (detach again with Ctrl-b then d):"
echo "      tmux attach -t sorbed-train"
echo "    Stop the run:"
echo "      tmux kill-session -t sorbed-train"
echo "    Exported model lands at:"
echo "      ${REPO_DIR}/artifacts/seg_unetpp/model.onnx"

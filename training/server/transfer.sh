#!/usr/bin/env bash
#
# transfer.sh — rsync this repo from a laptop to the H200 server over SSH.
#
# Pushes the working tree to /data/briefer/<repo-name> on the server, excluding
# version control, virtualenvs, caches, and large local data/artifacts so only
# source + configs go over the wire. Data is prepared separately on the server
# (see training/DATA_README.md and training/RUNBOOK.md).
#
# Run this FROM THE LAPTOP, from anywhere:
#     bash training/server/transfer.sh
#
# Environment overrides (all optional):
#     SSH_HOST    server host           (default: 10.6.110.10)
#     SSH_PORT    server ssh port       (default: 30405)
#     SSH_USER    server user           (default: root)
#     REMOTE_DIR  destination directory (default: /data/briefer/<repo-name>)
#     LOCAL_DIR   source repo root      (default: this script's repo root)
#     DRY_RUN     set to 1 to preview   (default: unset)
#
# No secrets are embedded: the host is an env var / default and SSH auth is
# whatever your ssh-agent / ~/.ssh/config already provides.
#
set -euo pipefail

SSH_HOST="${SSH_HOST:-10.6.110.10}"
SSH_PORT="${SSH_PORT:-30405}"
SSH_USER="${SSH_USER:-root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${LOCAL_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO_NAME="$(basename "${LOCAL_DIR}")"
REMOTE_DIR="${REMOTE_DIR:-/data/briefer/${REPO_NAME}}"

# Trailing slash on the source: copy the *contents* of LOCAL_DIR into REMOTE_DIR.
SRC="${LOCAL_DIR%/}/"
DEST="${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"

RSYNC_OPTS=(
    --archive          # preserve perms/times/symlinks
    --compress         # compress in transit
    --human-readable
    --progress
    --delete           # mirror: remove server files no longer present locally
    --delete-excluded  # also drop previously-synced files now excluded
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    RSYNC_OPTS+=(--dry-run --verbose)
    echo "==> DRY RUN (no files will be written)"
fi

# Exclusions: VCS, virtualenvs, tool caches, compiled artifacts, and heavy local
# data/outputs. The directive pack under var/ is git-ignored institutional
# content and is deliberately NOT synced — rebuild it on the server from its
# source PDF (scripts/build_directive_pack.py) if the pipeline needs it.
EXCLUDES=(
    --exclude ".git/"
    --exclude ".venv/"
    --exclude "sorbed-venv/"
    --exclude "**/__pycache__/"
    --exclude "*.pyc"
    --exclude ".mypy_cache/"
    --exclude ".ruff_cache/"
    --exclude ".pytest_cache/"
    --exclude ".hypothesis/"
    --exclude "data/"
    --exclude "datasets/"
    --exclude "manifests/"
    --exclude "artifacts/"
    --exclude "runs/"
    --exclude "logs/"
    --exclude "var/directive_packs/"
    --exclude "*.onnx"
    --exclude "*.pt"
    --exclude "*.pth"
)

echo "==> Transferring repo to the server"
echo "    from: ${SRC}"
echo "    to:   ${DEST}"
echo "    ssh:  -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST}"

# Ensure the destination exists before rsync writes into it.
ssh -p "${SSH_PORT}" "${SSH_USER}@${SSH_HOST}" "mkdir -p '${REMOTE_DIR}'"

rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" \
    -e "ssh -p ${SSH_PORT}" \
    "${SRC}" "${DEST}"

echo "==> Transfer complete."
echo "    Next, on the server: cd ${REMOTE_DIR} && bash training/server/setup_env.sh"

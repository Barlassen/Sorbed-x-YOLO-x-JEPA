#!/usr/bin/env bash
#
# setup_env.sh — create the training virtualenv on the H200 server and install
# the CUDA 12.x PyTorch stack plus the Sorbed training requirements.
#
# Target: 1x NVIDIA H200 NVL, MIG enabled (~40 GB usable slice), CUDA 12.8,
# driver 570. Run this ONCE per fresh checkout, inside tmux, as root, on the
# server. It is idempotent: re-running upgrades packages in place.
#
# Usage (on the server, e.g. inside tmux):
#     cd /data/briefer/sorbed
#     bash training/server/setup_env.sh
#
# Environment overrides (all optional):
#     VENV_DIR    virtualenv location      (default: /data/briefer/sorbed-venv)
#     REPO_DIR    repo checkout to install (default: script's repo root)
#     CUDA_TAG    torch wheel index tag    (default: cu124; use cu128 if desired)
#     PYTHON      base interpreter         (default: python3.11, then python3)
#
set -euo pipefail

VENV_DIR="${VENV_DIR:-/data/briefer/sorbed-venv}"
CUDA_TAG="${CUDA_TAG:-cu124}"

# Resolve the repo root from this script's location unless told otherwise.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# Pick a base interpreter: prefer 3.11 (repo's target), fall back to python3.
if [[ -z "${PYTHON:-}" ]]; then
    if command -v python3.11 >/dev/null 2>&1; then
        PYTHON="python3.11"
    else
        PYTHON="python3"
    fi
fi

echo "==> Sorbed training environment setup"
echo "    repo:      ${REPO_DIR}"
echo "    venv:      ${VENV_DIR}"
echo "    python:    ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "    torch tag: ${CUDA_TAG}"

# Sorbed requires Python >= 3.11 (uses datetime.UTC and enum.StrEnum). Fail here
# with actionable steps rather than deep inside the editable install.
PYVER="$(${PYTHON} -c 'import sys; print("%d %d" % sys.version_info[:2])')"
read -r PYMAJ PYMIN <<<"${PYVER}"
if (( PYMAJ < 3 || (PYMAJ == 3 && PYMIN < 11) )); then
    echo "ERROR: Sorbed needs Python >= 3.11, but ${PYTHON} is 3.${PYMIN}." >&2
    echo "       Install 3.11 (Ubuntu 22.04 ships 3.10), then delete this venv and re-run:" >&2
    echo "         apt-get update && apt-get install -y software-properties-common" >&2
    echo "         add-apt-repository -y ppa:deadsnakes/ppa && apt-get update" >&2
    echo "         apt-get install -y python3.11 python3.11-venv" >&2
    echo "         rm -rf ${VENV_DIR}" >&2
    echo "       Or point PYTHON at an existing >=3.11 interpreter:" >&2
    echo "         PYTHON=/usr/bin/python3.12 bash training/server/setup_env.sh" >&2
    exit 1
fi

if [[ ! -f "${REPO_DIR}/pyproject.toml" ]]; then
    echo "ERROR: ${REPO_DIR} does not look like the Sorbed repo (no pyproject.toml)." >&2
    echo "       Set REPO_DIR to the checkout you transferred with transfer.sh." >&2
    exit 1
fi

# 1. Create the virtualenv (skip if it already exists).
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "==> Creating virtualenv at ${VENV_DIR}"
    "${PYTHON}" -m venv "${VENV_DIR}"
else
    echo "==> Reusing existing virtualenv at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

# Preflight: prefer prebuilt wheels below (--prefer-binary) so nothing needs a C
# compiler. If a dep still has no wheel for this platform it falls back to a
# source build, which needs a compiler + Python headers — warn if those are
# missing so the failure isn't cryptic (as stringzilla's "Python.h: No such
# file" is). These are system build tools, not venv packages.
INCLUDEPY="$(python -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')"
if [[ ! -f "${INCLUDEPY}/Python.h" ]] || ! command -v cc >/dev/null 2>&1; then
    echo "WARNING: C build toolchain incomplete (need a compiler + ${INCLUDEPY}/Python.h)." >&2
    echo "         Most installs use prebuilt wheels, but if one builds from source" >&2
    echo "         and fails, install the headers/compiler and re-run:" >&2
    echo "           apt-get install -y python3.11-dev build-essential" >&2
fi

# 2. Install the CUDA-matched PyTorch build. Driver 570 / CUDA 12.8 runs both the
#    cu124 and cu128 wheels; cu124 is the safe default. Override with CUDA_TAG.
echo "==> Installing torch + torchvision from the ${CUDA_TAG} wheel index"
python -m pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    torch torchvision

# 3. Install the training requirements (segmentation stack, augmentation, ONNX).
#    --prefer-binary makes pip pick an older wheel over a newer sdist, so a
#    transitive dep like stringzilla installs prebuilt instead of compiling.
echo "==> Installing training requirements"
python -m pip install --prefer-binary -r "${REPO_DIR}/training/requirements-train.txt"

# 4. Install the sorbed package itself (editable) so `from sorbed ...` resolves.
#    The [ml] extra adds onnxruntime + scikit-learn for the CPU inference path.
echo "==> Installing sorbed (editable, with [ml] extra)"
python -m pip install --prefer-binary -e "${REPO_DIR}[ml]"

# 5. Sanity check: confirm torch sees the (MIG) GPU.
echo "==> Verifying CUDA visibility"
python - <<'PY'
import torch

print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device count: {torch.cuda.device_count()}")
    print(f"device 0: {torch.cuda.get_device_name(0)}")
else:
    print(
        "NOTE: torch.cuda.is_available() is False. If MIG is enabled you must set\n"
        "      CUDA_VISIBLE_DEVICES to a MIG UUID (see 'nvidia-smi -L'). run_tmux.sh\n"
        "      does this for the training run; this bare check does not."
    )
PY

echo "==> Done. Activate with: source ${VENV_DIR}/bin/activate"

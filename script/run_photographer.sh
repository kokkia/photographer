#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="${PHOTOGRAPHER_CONDA_ENV:-camera310}"

find_conda_init() {
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base)"
        if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
            printf '%s\n' "${conda_base}/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    local candidate
    for candidate in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"
    do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

if ! CONDA_INIT="$(find_conda_init)"; then
    echo "エラー: Condaが見つかりません。MinicondaまたはAnacondaをインストールしてください。" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_INIT}"

if ! conda activate "${CONDA_ENV_NAME}"; then
    echo "エラー: Conda環境 '${CONDA_ENV_NAME}' を有効化できませんでした。" >&2
    echo "READMEの環境構築手順を実行してください。" >&2
    exit 1
fi

MPL_CACHE_DIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/photographer-matplotlib-${UID}}"
mkdir -p -- "${MPL_CACHE_DIR}"
export MPLCONFIGDIR="${MPL_CACHE_DIR}"

cd -- "${PROJECT_ROOT}"
exec python -m src.gui "$@" --camera /dev/camera_front

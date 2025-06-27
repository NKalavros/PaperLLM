#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Pick mamba if installed, otherwise conda
if command -v mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi

# default channels
CHANNELS=(-c conda-forge -c pytorch)

# Usage: ./setup_transcription_env.sh [env_name]
ENV_NAME="${1:-transcription-env}"

# Ensure conda/mamba is available
command -v "$CONDA_CMD" >/dev/null 2>&1 || { echo "ERROR: $CONDA_CMD not found"; exit 1; }

# Create env with channels for arm64 support
echo "Creating environment '$ENV_NAME' with Python 3.10..."
"$CONDA_CMD" create -n "$ENV_NAME" python=3.10 "${CHANNELS[@]}" -y

# Install mamba into the new environment
echo "Installing mamba into the environment..."
"$CONDA_CMD" install -n "$ENV_NAME" "${CHANNELS[@]}" mamba -y

echo "Activating environment '$ENV_NAME'..."
# disable unbound‐variable checks in Conda scripts
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

# Install FFmpeg, PyTorch, torchvision, torchaudio using the same channels
echo "Installing FFmpeg, PyTorch, torchvision, torchaudio..."
mamba install ffmpeg pytorch torchvision torchaudio "${CHANNELS[@]}" -y

# Install Python packages via pip
echo "Installing transcription Python dependencies..."
pip install faster-whisper deepmultilingualpunctuation openai

echo "Setup complete. Activate with: conda activate $ENV_NAME"

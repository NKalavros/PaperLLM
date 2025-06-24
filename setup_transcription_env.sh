#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Pick mamba if installed, otherwise conda
if command -v mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi
echo "Using package manager: $CONDA_CMD"

# Usage: ./setup_transcription_env.sh [env_name]
ENV_NAME="${1:-transcription-env}"

# Ensure conda/mamba is available
command -v "$CONDA_CMD" >/dev/null 2>&1 || { echo "ERROR: $CONDA_CMD not found"; exit 1; }

echo "Creating environment '$ENV_NAME' with Python 3.10..."
"$CONDA_CMD" create -n "$ENV_NAME" python=3.10 -y

# Install mamba into the new environment
echo "Installing mamba into the environment..."
"$CONDA_CMD" install -n "$ENV_NAME" -c conda-forge mamba -y

echo "Activating environment '$ENV_NAME'..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing FFmpeg (via conda-forge)..."
"$CONDA_CMD" install -c conda-forge ffmpeg -y

echo "Installing PyTorch + torchvision + torchaudio..."
"$CONDA_CMD" install -c pytorch pytorch torchvision torchaudio -y

echo "Installing transcription dependencies (faster-whisper, punctuation, OpenAI API)..."
pip install faster-whisper deepmultilingualpunctuation openai

echo "Setup complete. Activate with: conda activate $ENV_NAME"

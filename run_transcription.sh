#!/bin/bash
set -euo pipefail
IFS=$'\n\t'
trap 'echo "Interrupted. Exiting."; exit 1' SIGINT SIGTERM

# Load .env if present
if [ -f ".env" ]; then
  echo "Loading configuration from .env"
  set -o allexport; source .env; set +o allexport
fi

# Allow overrides via env or defaults
PYTHON_EXE="${PYTHON_EXE:-python3}"
SCRIPT_NAME="${SCRIPT_NAME:-transcribe_monitor.py}"
TARGET_DIR="${TARGET_DIR:-complete_transcription}"

# Dependency checks
command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found in PATH"; exit 1; }
command -v "$PYTHON_EXE" >/dev/null 2>&1 || { echo "ERROR: $PYTHON_EXE not found"; exit 1; }

# Arg check
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <transcription_base_name>"
    exit 1
fi

TRANSCRIPTION_NAME="$1"
echo "Starting transcription process for: $TRANSCRIPTION_NAME"

mkdir -p "$TARGET_DIR"
FINAL_OUTPUT_PATH="$TARGET_DIR/$TRANSCRIPTION_NAME.txt"
echo "Output will be: $FINAL_OUTPUT_PATH"

[ -f "$SCRIPT_NAME" ] || { echo "Error: $SCRIPT_NAME not found"; exit 1; }

# If not already set, list and ask for ffmpeg format & device
system="$(uname)"
if [ -z "${FFMPEG_INPUT_FORMAT:-}" ]; then
    echo "Detected platform: $system"
    if [ "$system" = "Darwin" ]; then
        echo "Available audio devices (ffmpeg avfoundation):"
        ffmpeg -f avfoundation -list_devices true -i "" 2>&1 \
          | grep '\[AVFoundation indev' || true
    elif [ "$system" = "Linux" ]; then
        echo "Available audio devices (arecord):"
        arecord -l
    fi
    read -rp "Enter capture format (e.g. avfoundation, alsa): " FFMPEG_INPUT_FORMAT
    echo "Selected input format: $FFMPEG_INPUT_FORMAT"
fi

if [ -z "${FFMPEG_DEVICE:-}" ]; then
    if [ "$FFMPEG_INPUT_FORMAT" = "avfoundation" ]; then
        read -rp "Enter audio device index number (e.g. 1 for [1] MacBook Air Microphone): " idx
        # prefix with ':' if missing
        FFMPEG_DEVICE=":${idx#*:}"
    else
        read -rp "Enter ffmpeg audio device (e.g. 'hw:1,0'): " FFMPEG_DEVICE
    fi
    echo "Selected audio device: $FFMPEG_DEVICE"
fi

# Build optional Python args from env vars:
PYTHON_ARGS=()
if [ -n "${FFMPEG_INPUT_FORMAT:-}" ]; then
    PYTHON_ARGS+=(--input-format "$FFMPEG_INPUT_FORMAT")
fi
if [ -n "${FFMPEG_DEVICE:-}" ]; then
    PYTHON_ARGS+=(--audio-device "$FFMPEG_DEVICE")
fi

echo "Running transcription script..."
"$PYTHON_EXE" "$SCRIPT_NAME" "$FINAL_OUTPUT_PATH" "${PYTHON_ARGS[@]}"
EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo "Python script failed (status $EXIT_STATUS)."
    exit $EXIT_STATUS
fi

echo "Transcription saved to: $FINAL_OUTPUT_PATH"
exit 0
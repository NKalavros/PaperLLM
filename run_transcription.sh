#!/bin/bash

# --- Configuration ---
PYTHON_EXE="python3" # Or just "python" if that's your env
SCRIPT_NAME="transcribe_monitor.py"
TARGET_DIR="complete_transcription" # Directory for final outputs

# --- Argument Check ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <transcription_base_name>"
    echo "  Example: $0 my_meeting_notes"
    exit 1
fi

TRANSCRIPTION_NAME="$1"
echo "Starting transcription process for: $TRANSCRIPTION_NAME"

# --- Prepare Output Path ---
echo "Creating output directory (if needed): $TARGET_DIR"
mkdir -p "$TARGET_DIR" # Create directory and parent dirs if they don't exist

# Construct the full path for the final .txt file
FINAL_OUTPUT_PATH="$TARGET_DIR/$TRANSCRIPTION_NAME.txt"
echo "Final transcription will be saved to: $FINAL_OUTPUT_PATH"

# --- Check if Python script exists ---
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Python script '$SCRIPT_NAME' not found in the current directory."
    exit 1
fi

# --- Execute Python Script ---
echo "Running Python transcription script..."
echo "Press Ctrl+C in the terminal running the Python script to stop recording."

# Execute the python script, passing the final desired output path as an argument
"$PYTHON_EXE" "$SCRIPT_NAME" "$FINAL_OUTPUT_PATH"

# Capture the exit status of the python script
EXIT_STATUS=$?

# --- Check Exit Status ---
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Python script finished successfully."
    echo "Final transcription saved to: $FINAL_OUTPUT_PATH"
else
    echo "Error: Python script exited with status $EXIT_STATUS."
    # Note: The python script might have still created a partial file or logs.
    # The temporary directory might still exist if cleanup is disabled or failed.
    exit $EXIT_STATUS # Propagate the error status
fi

echo "Transcription process complete for: $TRANSCRIPTION_NAME"
exit 0
#!/usr/bin/env python3

import subprocess
import time
import os
import signal
import sys
import platform
import threading
import queue
import argparse
from pathlib import Path
from datetime import datetime

# --- Configuration ---
RECORD_INTERVAL_SECONDS = 30
OUTPUT_BASE_DIR = Path("./audio_transcription_output_concurrent")
AUDIO_SEGMENTS_DIR = OUTPUT_BASE_DIR / "audio_segments"
TRANSCRIPTION_SEGMENTS_DIR = OUTPUT_BASE_DIR / "transcription_segments"
FINAL_TRANSCRIPTION_FILE = OUTPUT_BASE_DIR / "transcription.txt"

# --- Choose Transcription Method ---
# 'faster-whisper' (recommended for local), 'api'
# 'local' (original openai-whisper) is removed in this version for clarity
TRANSCRIPTION_MODE = 'faster-whisper' # CHANGE AS NEEDED ('api')

# --- Faster Whisper Configuration ---
# Model size: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
#             'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3'
#             'distil-large-v2', 'distil-medium.en', 'distil-small.en' (faster distilled models)
LOCAL_WHISPER_MODEL = "large-v3" # Good balance for speed/accuracy, especially distilled

# Compute type: 'float16' (good for MPS/GPU), 'int8' (faster CPU, less accurate),
#               'float32' (CPU default accuracy), 'int8_float16', 'int16'
# Let's auto-select based on device
FASTER_WHISPER_COMPUTE_TYPE_MPS = "float16"
FASTER_WHISPER_COMPUTE_TYPE_CPU = "float32" # Use "float32" for better accuracy on CPU

# Beam size for transcription (higher = potentially more accurate but slower)
BEAM_SIZE = 5

# --- OpenAI API Configuration --- (Only if TRANSCRIPTION_MODE = 'api')
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
OPENAI_API_MODEL = "whisper-1"

# --- Platform Specific Defaults ---
# (get_platform_defaults function remains the same as previous version)
def get_platform_defaults():
    """Detects OS and returns appropriate ffmpeg format and default device."""
    system = platform.system()
    ffmpeg_format = None
    audio_device = None
    needs_manual_config = False
    print("-" * 30)
    if system == "Darwin": # macOS
        print("Detected macOS.")
        ffmpeg_format = "avfoundation"
        audio_device = ":0" # Default macOS input
        print(f"Using ffmpeg format: '{ffmpeg_format}', device: '{audio_device}'")
        print("(Verify using: ffmpeg -f avfoundation -list_devices true -i \"\")")
    elif system == "Linux":
        print("Detected Linux.")
        ffmpeg_format = "alsa"
        audio_device = "hw:0,0" # Common ALSA default
        print(f"Using ffmpeg format: '{ffmpeg_format}', device: '{audio_device}'")
        print("(Verify using: arecord -l)")
    else:
        print(f"Detected unsupported OS: {system}. Using Linux ALSA defaults.")
        ffmpeg_format = "alsa"
        audio_device = "hw:0,0"
        needs_manual_config = True
    print("-" * 30)
    if not ffmpeg_format or not audio_device:
         print("ERROR: Could not determine platform defaults. Please configure manually.")
         sys.exit(1)
    return ffmpeg_format, audio_device, needs_manual_config

FFMPEG_INPUT_FORMAT, AUDIO_DEVICE, PLATFORM_NEEDS_MANUAL_CONFIG = get_platform_defaults()

# --- FFmpeg Configuration ---
# Manually override FFMPEG_INPUT_FORMAT and AUDIO_DEVICE here if needed

if PLATFORM_NEEDS_MANUAL_CONFIG:
    print("WARNING: Please verify FFMPEG settings below.")
    # time.sleep(3)

FFMPEG_COMMAND_BASE = [
    'ffmpeg', '-loglevel', 'error', '-f', FFMPEG_INPUT_FORMAT,
    '-i', AUDIO_DEVICE, '-t', str(RECORD_INTERVAL_SECONDS),
    '-codec:a', 'libmp3lame', '-q:a', '4', # MP3 output
]

# --- Global Variables ---
keep_running = True
segment_counter = 0
# Queue to hold paths of audio files waiting for transcription
transcription_queue = queue.Queue()

# --- Helper Functions ---

def setup_directories():
    """Creates necessary output directories."""
    print("Setting up directories...")
    AUDIO_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTION_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f" - Audio segments: {AUDIO_SEGMENTS_DIR}")
    print(f" - Transcription segments: {TRANSCRIPTION_SEGMENTS_DIR}")
    print(f" - Final transcription: {FINAL_TRANSCRIPTION_FILE}")

def record_audio(output_filename):
    """Records audio using ffmpeg. Returns True on success, False on failure."""
    command = FFMPEG_COMMAND_BASE + [str(output_filename)]
    print(f"Recording segment to {output_filename.name}...")
    # print(f"  > Executing: {' '.join(command)}") # Can be verbose
    try:
        # Use Popen for non-blocking execution if needed, but run waits for completion
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"  > Recording successful: {output_filename.name}")
        return True
    except FileNotFoundError:
        print("\nERROR: 'ffmpeg' command not found. Is it installed and in PATH?")
        # Stop the whole process if ffmpeg is missing
        global keep_running
        keep_running = False
        return False
    except subprocess.CalledProcessError as e:
        print(f"  > ERROR during recording for {output_filename.name}:")
        print(f"    Stderr: {e.stderr.strip()}")
        print("    (Check audio device permissions and settings)")
        return False
    except Exception as e:
        print(f"  > An unexpected error occurred during recording: {e}")
        return False

# --- Transcription Functions ---

def transcribe_api(audio_path):
    """Transcribes audio using the OpenAI Whisper API."""
    # (This function remains the same as the previous version)
    try:
        from openai import OpenAI
    except ImportError:
        print("\nERROR: 'openai' library not found. (pip install -U openai)")
        return None # Indicate failure

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY environment variable not set.")
        return None

    print(f"Transcribing {audio_path.name} (API)...")
    try:
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=OPENAI_API_MODEL,
                file=audio_file
            )
        transcription = response.text.strip()
        print(f"  > API Transcription successful: {audio_path.name}")
        return transcription
    except Exception as e:
        print(f"  > ERROR during API transcription for {audio_path.name}: {e}")
        return None

def transcribe_faster_whisper(model, audio_path):
    """
    Transcribes audio using the loaded faster-whisper model.
    Handles potential errors during transcription of a single file.
    """
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        print(f"Skipping transcription: Audio file {audio_path.name} is missing or empty.")
        return None

    print(f"Transcribing {audio_path.name} (faster-whisper)...")
    try:
        # Use VAD filter for potentially faster processing of silence
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=BEAM_SIZE,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500) # Adjust VAD as needed
        )
        # Concatenate segments into a single string
        full_transcription = " ".join([segment.text for segment in segments]).strip()

        # print(f"  > Detected language '{info.language}' with probability {info.language_probability:.2f}")
        print(f"  > faster-whisper Transcription successful: {audio_path.name}")
        return full_transcription
    except Exception as e:
        print(f"  > ERROR during faster-whisper transcription for {audio_path.name}: {e}")
        # Log the error details if needed
        # Consider specific error handling, e.g., for CUDA/MPS errors
        return None # Indicate failure for this file


# --- Worker Thread Function ---

def transcription_worker(model):
    """
    Worker thread function. Continuously gets audio paths from the queue
    and transcribes them using the provided model or API function.
    """
    print("Transcription worker started.")
    while True:
        try:
            # Wait indefinitely until an item is available
            audio_path = transcription_queue.get()

            if audio_path is None: # Sentinel value received
                print("Transcription worker received stop signal.")
                transcription_queue.task_done()
                break # Exit the loop

            start_time = time.time()
            transcription = None
            if TRANSCRIPTION_MODE == 'faster-whisper':
                 if model: # Ensure model was loaded successfully
                    transcription = transcribe_faster_whisper(model, audio_path)
                 else:
                    print(f"Skipping {audio_path.name}: faster-whisper model not loaded.")
            elif TRANSCRIPTION_MODE == 'api':
                transcription = transcribe_api(audio_path)
            else:
                 print(f"ERROR: Invalid TRANSCRIPTION_MODE '{TRANSCRIPTION_MODE}' in worker.")
                 # Optionally signal main thread to stop?

            if transcription is not None:
                # Generate corresponding transcription filename
                transcription_filename = f"{audio_path.stem}.txt"
                transcription_filepath = TRANSCRIPTION_SEGMENTS_DIR / transcription_filename
                save_transcription_segment(transcription, transcription_filepath)
            else:
                 print(f"Transcription failed or skipped for {audio_path.name}.")
                 # Optionally save a placeholder error file

            processing_time = time.time() - start_time
            print(f"Worker processed {audio_path.name} in {processing_time:.2f}s")
            transcription_queue.task_done() # Signal that this task is complete

        except Exception as e:
             # Catch unexpected errors in the worker loop itself
             print(f"FATAL ERROR in transcription worker: {e}")
             # Decide if the worker should stop or try to continue
             # Signal main thread?
             # For now, let's try to continue after marking task done if possible
             if audio_path is not None: # Avoid task_done if error getting from queue
                 transcription_queue.task_done()
             # Maybe break here to stop the worker on fatal errors
             # break

    print("Transcription worker finished.")


def save_transcription_segment(text, output_filename):
    """Saves a single transcription segment to a file."""
    # print(f"Saving transcription segment to {output_filename.name}...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(text + "\n")
        # print(f"  > Save successful: {output_filename.name}")
    except IOError as e:
        print(f"  > ERROR saving transcription segment {output_filename.name}: {e}")

# --- Collation Function ---
def collate_transcriptions(final_output_path): # <-- Added argument
    """
    Collates all transcription segments into the final file (specified by final_output_path)
    as a single line of text.
    """
    print(f"\nCollating transcriptions into a single line at: {final_output_path}")
    try:
        if not TRANSCRIPTION_SEGMENTS_DIR.exists():
             print(f"Temporary transcription segments directory not found: {TRANSCRIPTION_SEGMENTS_DIR}")
             print("No segments to collate.")
             # Ensure the target directory exists even if there's nothing to collate
             final_output_path.parent.mkdir(parents=True, exist_ok=True)
             # Write an empty file to the target path
             with open(final_output_path, 'w', encoding='utf-8') as outfile:
                 outfile.write("")
             return

        segment_files = sorted(TRANSCRIPTION_SEGMENTS_DIR.glob("segment_*.txt"))

        if not segment_files:
            print("No transcription segments found to collate.")
            # Ensure the target directory exists
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            # Write an empty file
            with open(final_output_path, 'w', encoding='utf-8') as outfile:
                outfile.write("")
            return

        all_text_segments = []
        for segment_file in segment_files:
            try:
                with open(segment_file, 'r', encoding='utf-8') as infile:
                    content = infile.read().strip()
                    if content:
                         all_text_segments.append(content)
            except IOError as e: print(f"  > WARNING: Skipping {segment_file.name}: {e}")
            except Exception as e: print(f"  > WARNING: Error processing {segment_file.name}: {e}")

        final_text = " ".join(all_text_segments)

        # Ensure the target directory exists before writing
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the single concatenated line to the final output file path provided
        with open(final_output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(final_text)

        print(f"Successfully collated {len(all_text_segments)} segments into {final_output_path}")

    except IOError as e:
        print(f"ERROR: Could not write final transcription file {final_output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during collation: {e}")


# --- Signal Handler ---
def signal_handler(sig, frame):
    """Handles Ctrl+C gracefully."""
    global keep_running
    if keep_running:
        print("\nCtrl+C detected. Stopping recording loop...")
        keep_running = False
        # Don't put None on queue here, main loop will do it after exiting

# --- Main Execution ---

def main(args): # <-- Added args parameter
    global segment_counter
    global keep_running

    signal.signal(signal.SIGINT, signal_handler)
    setup_directories() # Sets up temporary directories

    # --- Load Model or Prep API (Keep as is) ---
    whisper_model = None
    worker_thread = None
    if TRANSCRIPTION_MODE == 'faster-whisper':
        try:
            from faster_whisper import WhisperModel
            device = "cpu" # Force CPU as per previous request
            compute_type = FASTER_WHISPER_COMPUTE_TYPE_CPU
            system = platform.system()
            print(f"Configuring faster-whisper for CPU execution.")
            if system == "Linux": # Optional CUDA check for Linux
                 try:
                      import torch
                      if torch.cuda.is_available():
                           print("NVIDIA CUDA available, attempting to use GPU.")
                           device = "cuda"; compute_type = FASTER_WHISPER_COMPUTE_TYPE_MPS
                      else: print("CUDA not available, using CPU.")
                 except ImportError: print("PyTorch not found, cannot check CUDA. Using CPU.")
            print(f"Loading faster-whisper model '{LOCAL_WHISPER_MODEL}'...")
            cpu_threads = 0 # Auto-detect
            print(f"Using device: '{device}', compute_type: '{compute_type}', cpu_threads: {cpu_threads if cpu_threads > 0 else 'auto'}")
            whisper_model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
            print("Model loaded successfully.")
        except ImportError: print("\nERROR: 'faster-whisper' not found."); sys.exit(1)
        except Exception as e: print(f"\nERROR loading faster-whisper model: {e}"); sys.exit(1)
        worker_thread = threading.Thread(target=transcription_worker, args=(whisper_model,), daemon=True); worker_thread.start()
    elif TRANSCRIPTION_MODE == 'api':
        worker_thread = threading.Thread(target=transcription_worker, args=(None,), daemon=True); worker_thread.start()
    else: print(f"ERROR: Unknown TRANSCRIPTION_MODE: {TRANSCRIPTION_MODE}"); sys.exit(1)

    # --- Recording Loop (Keep as is) ---
    print(f"\nStarting continuous recording (Interval: {RECORD_INTERVAL_SECONDS}s). Press Ctrl+C to stop.")
    print("Attempting minimal gap between recordings.")
    print("-" * 30)
    while keep_running:
        loop_start_time = time.time()
        if not keep_running: break
        segment_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"segment_{segment_counter:04d}_{timestamp}"
        audio_filepath = AUDIO_SEGMENTS_DIR / f"{base_filename}.mp3" # Use temp dir
        recording_start_time = time.time()
        success = record_audio(audio_filepath)
        recording_end_time = time.time()
        actual_recording_duration = recording_end_time - recording_start_time
        print(f"  > Segment {segment_counter} recorded in {actual_recording_duration:.2f}s (Target: {RECORD_INTERVAL_SECONDS}s)")
        if success:
            print(f"  > Queueing {audio_filepath.name} for transcription.")
            transcription_queue.put(audio_filepath)
        else: print(f"Recording failed for segment {segment_counter}, skipping.")
        loop_end_time = time.time()
        loop_overhead = loop_end_time - recording_end_time
        print(f"  > Loop overhead before next recording: {loop_overhead:.4f}s")

    # --- Shutdown Sequence (MODIFIED) ---
    print("\nRecording loop finished.")
    print("Signaling transcription worker to stop...")
    transcription_queue.put(None)
    if worker_thread:
         print("Waiting for transcription worker to finish...")
         worker_thread.join()
         print("Transcription worker finished.")

    # Call collation with the final path from arguments
    collate_transcriptions(args.final_output_path) # <-- Use parsed arg

    # Optional: Clean up temporary directory
    # print(f"Cleaning up temporary directory: {OUTPUT_BASE_DIR}")
    # try:
    #     import shutil
    #     shutil.rmtree(OUTPUT_BASE_DIR)
    #     print("Temporary directory removed.")
    # except Exception as e:
    #     print(f"Warning: Could not remove temporary directory {OUTPUT_BASE_DIR}: {e}")

    print("Script finished.")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Record audio segments and transcribe them.")
    parser.add_argument(
        "final_output_path",
        type=Path, # Use pathlib for path handling
        help="The final path (including filename) for the collated transcription text file."
    )
    parsed_args = parser.parse_args()

    # --- Run Main Function ---
    main(parsed_args) # Pass parsed arguments to main

#!/usr/bin/env python3
"""
Example: Stream raw system audio to stdout in 16kHz mono s16le format

This example records system audio and outputs raw PCM samples directly to stdout.
Format: 16kHz sample rate, mono (1 channel), signed 16-bit little-endian (s16le)

Usage:
    python raw_stdout.py > output.raw
    python raw_stdout.py | ffplay -f s16le -ar 16000 -ac 1 -
    python raw_stdout.py | aplay -f S16_LE -r 16000 -c 1
"""

from pysysaudio import SystemAudioRecorder
import sys


def main():
    # Create recorder with 16kHz mono s16le format
    # - sample_rate=16000: 16kHz
    # - channels=1: mono
    # - format="bytes": raw PCM bytes
    # - dtype="int16": signed 16-bit (s16le on little-endian systems)
    # Audio is automatically resampled from system's native 48kHz stereo to 16kHz mono
    recorder = SystemAudioRecorder(sample_rate=16000, channels=1, format="bytes", dtype="int16")

    # Log to stderr (stdout is for raw audio data)
    print("Starting system audio capture...", file=sys.stderr)
    print("Format: 16kHz, mono, s16le", file=sys.stderr)
    print("Press Ctrl+C to stop", file=sys.stderr)
    print("", file=sys.stderr)

    try:
        # Start recording without saving to file (stream only)
        recorder.start_recording(output_path=None)

        # Stream audio chunks to stdout
        for chunk in recorder.stream():
            # Write raw PCM data directly to stdout
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
        if recorder.is_recording():
            recorder.stop_recording()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

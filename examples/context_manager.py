#!/usr/bin/env python3
"""
Example: Using context manager for automatic cleanup
"""

from pysysaudio import SystemAudioRecorder
import time


def main():
    print("Recording with Context Manager")
    print("=" * 50)

    output_file = "context_manager_recording.wav"

    print(f"Recording to: {output_file}")
    print("Recording for 5 seconds...")

    # Using context manager ensures proper cleanup
    with SystemAudioRecorder(sample_rate=48000, channels=2) as recorder:
        recorder.start_recording(output_file)

        for i in range(5, 0, -1):
            print(f"{i}...", end=" ", flush=True)
            time.sleep(1)

        print("\nStopping...")
        output_path = recorder.stop_recording()

    # Recording is automatically stopped when exiting the context
    print(f"✅ Saved to: {output_path}")


if __name__ == "__main__":
    main()

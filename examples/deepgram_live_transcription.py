#!/usr/bin/env python3
"""
Example: Live transcription by streaming audio to Deepgram Nova-3

Requirements:
- Environment variable DEEPGRAM_API_KEY must be set
- pip install deepgram-sdk
- pip install numpy

Notes:
- We request 16kHz mono in int16 format (PCM16/linear16)
- pysysaudio automatically resamples (48kHz→16kHz) and converts to int16
- Audio frames are sent as PCM16 (little-endian) raw bytes
- Deepgram provides real-time transcription
- Interim results show partial transcriptions as speech is in progress

This is a minimal example intended for quick integration testing.
"""

import os
import sys
import threading
from typing import Any

from pysysaudio import SystemAudioRecorder

try:
    from deepgram import DeepgramClient
    from deepgram.core.events import EventType
except ImportError:
    print("This example requires deepgram-sdk: pip install deepgram-sdk")
    sys.exit(1)


DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise RuntimeError("Please set DEEPGRAM_API_KEY in your environment")


def main():
    print("pysysaudio - Deepgram Live Transcription Example")
    print("=" * 60)
    print("Model: flux-general-en")
    print("Press Ctrl+C to stop.\n")

    # Record at 16kHz mono in int16 format for Deepgram
    recorder = SystemAudioRecorder(
        sample_rate=16000,
        channels=1,
        format="bytes",  # Raw PCM bytes
        dtype="int16",  # 16-bit PCM (linear16)
    )

    try:
        # Configure Deepgram client
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        # Create a websocket connection to Deepgram
        with deepgram.listen.v2.connect(
            model="flux-general-en",
            #language="en-US",
            encoding="linear16",
            sample_rate="16000",
            #channels="1",
            # Interim results for real-time feedback
            #interim_results="true",
            # Utterance end detection
            #utterance_end_ms="1000",
            # VAD (Voice Activity Detection) events
            #vad_events="false",
            # Smart formatting
            # smart_format="true",
            # punctuate="true",
        ) as connection:

            # Event handlers
            def on_open(_):
                print("✓ Connected to Deepgram\n")

            # Track displayed interim text so we can print final output cleanly.
            last_partial = ""
            last_turn_index = None

            def on_message(message: Any):
                nonlocal last_partial, last_turn_index

                try:
                    message_type = getattr(message, "type", None)

                    # Flux model path: TurnInfo events.
                    if message_type == "TurnInfo":
                        event = getattr(message, "event", "")
                        transcript = (getattr(message, "transcript", "") or "").strip()
                        turn_index = getattr(message, "turn_index", None)

                        # New turn: clear any stale interim state.
                        if turn_index is not None and turn_index != last_turn_index:
                            last_partial = ""
                            last_turn_index = turn_index

                        if not transcript:
                            return

                        # EndOfTurn = final transcript for this turn.
                        if event == "EndOfTurn":
                            if last_partial:
                                # Clear the interim line before printing final.
                                print(" " * (len(last_partial) + 3), end="\r", flush=True)
                            print(f"📝 {transcript}")
                            last_partial = ""
                            return

                        # StartOfTurn/Update behave like interim text.
                        if transcript != last_partial:
                            last_partial = transcript
                            print(f"⏳ {transcript}", end="\r", flush=True)
                        return

                    # Legacy / Nova-style path: channel.alternatives transcript payload.
                    if hasattr(message, "channel") and hasattr(message.channel, "alternatives"):
                        sentence = (message.channel.alternatives[0].transcript or "").strip()
                        if not sentence:
                            return

                        if getattr(message, "is_final", False):
                            if last_partial:
                                print(" " * (len(last_partial) + 3), end="\r", flush=True)
                            print(f"📝 {sentence}")
                            last_partial = ""
                        else:
                            last_partial = sentence
                            print(f"⏳ {sentence}", end="\r", flush=True)
                except Exception:
                    # Keep the example resilient to SDK payload changes.
                    return

            def on_close(_):
                print("\nConnection closed")

            def on_error(error):
                print(f"\nError: {error}")

            # Register event handlers
            connection.on(EventType.OPEN, on_open)
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.CLOSE, on_close)
            connection.on(EventType.ERROR, on_error)

            # Start listening in a background thread (handles WebSocket receiving)
            def start_listening_thread():
                connection.start_listening()

            listen_thread = threading.Thread(target=start_listening_thread, daemon=True)
            listen_thread.start()

            print("Streaming audio...")

            # Start recording and stream audio
            recorder.start_recording()

            try:
                # Stream audio chunks to Deepgram
                for chunk in recorder.stream():
                    # Chunk is already PCM16 bytes, ready to send
                    connection.send_media(chunk)

            except KeyboardInterrupt:
                print("\n\nStopping...")
            finally:
                # Clean up
                if recorder.is_recording():
                    recorder.stop_recording()

            print("Finished")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        if recorder.is_recording():
            recorder.stop_recording()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")

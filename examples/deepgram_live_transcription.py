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

from pysysaudio import SystemAudioRecorder

try:
    from deepgram import DeepgramClient
    from deepgram.core.events import EventType
    from deepgram.extensions.types.sockets import ListenV1SocketClientResponse
except ImportError:
    print("This example requires deepgram-sdk: pip install deepgram-sdk")
    sys.exit(1)


DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise RuntimeError("Please set DEEPGRAM_API_KEY in your environment")


def main():
    print("pysysaudio - Deepgram Live Transcription Example")
    print("=" * 60)
    print("Model: Nova-3")
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
        with deepgram.listen.v1.connect(
            model="nova-3",
            language="en-US",
            encoding="linear16",
            sample_rate="16000",
            channels="1",
            # Interim results for real-time feedback
            interim_results="true",
            # Utterance end detection
            utterance_end_ms="1000",
            # VAD (Voice Activity Detection) events
            vad_events="false",
            # Smart formatting
            smart_format="true",
            punctuate="true",
        ) as connection:

            # Event handlers
            def on_open(_):
                print("✓ Connected to Deepgram\n")

            def on_message(message: ListenV1SocketClientResponse):
                try:
                    if hasattr(message, "channel") and hasattr(message.channel, "alternatives"):
                        sentence = message.channel.alternatives[0].transcript
                        if len(sentence) == 0:
                            return

                        if message.is_final:
                            # Final transcription
                            print(f"📝 {sentence}")
                        else:
                            # Interim result (partial transcription)
                            print(f"⏳ {sentence}", end="\r", flush=True)
                except Exception:
                    pass

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

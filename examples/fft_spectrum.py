#!/usr/bin/env python3
"""
Example: Real-time FFT Spectrum Analysis in Terminal

Displays a live frequency spectrum visualization of system audio using FFT.
The spectrum is shown as ASCII bar chart updated in real-time.

Requirements:
- pip install numpy

Usage:
    python examples/fft_spectrum.py
"""

import sys
import time

try:
    import numpy as np
except ImportError:
    print("❌ This example requires numpy: pip install numpy")
    sys.exit(1)

from pysysaudio import SystemAudioRecorder


def format_frequency(freq):
    """Format frequency for display (e.g., 1000 -> 1.0k)"""
    return f"{freq/1000:.1f}k" if freq >= 1000 else f"{int(freq)}"


def create_bar(value, width=50):
    """Create an ASCII bar for visualization"""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def main():
    print("pysysaudio - Real-time FFT Spectrum Analyzer")
    print("=" * 70)
    print("Displays live frequency spectrum of system audio")
    print("Press Ctrl+C to stop\n")

    # Configuration
    sample_rate = 48000
    channels = 1
    fft_size = 2048
    num_bands = 20
    min_freq = 20
    max_freq = min(20000, sample_rate // 2)

    # Create logarithmic frequency bands (better for audio perception)
    freq_bands = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands + 1)
    band_centers = [(freq_bands[i] + freq_bands[i + 1]) / 2 for i in range(num_bands)]

    # Create recorder
    recorder = SystemAudioRecorder(
        sample_rate=sample_rate, channels=channels, format="numpy", dtype="float32"
    )

    # State
    audio_buffer = np.array([], dtype=np.float32)
    peak_values = np.full(num_bands, -100.0)
    peak_decay = 0.95

    try:
        print("🎵 Starting audio capture... (play some music!)\n")
        recorder.start_recording()

        print("\033[2J\033[?25l", end="")  # Clear screen and hide cursor

        for chunk in recorder.stream():
            # Ensure chunk is 1D and add to buffer
            chunk = chunk.flatten() if chunk.ndim == 2 else chunk
            audio_buffer = np.concatenate([audio_buffer, chunk])

            # When we have enough samples, perform FFT
            if len(audio_buffer) >= fft_size:
                # Extract samples for FFT
                samples = audio_buffer[:fft_size]
                audio_buffer = audio_buffer[fft_size // 2 :]  # Overlap for smoother visualization

                # Apply window function to reduce spectral leakage
                window = np.hanning(fft_size)
                windowed_samples = samples * window

                # Perform FFT and calculate magnitude spectrum
                magnitude = np.abs(np.fft.rfft(windowed_samples))

                # Normalize: divide by window sum, multiply non-DC bins by 2
                magnitude = magnitude / np.sum(window)
                magnitude[1:-1] *= 2

                # Convert to dBFS (reference: RMS = 1.0)
                magnitude_db = 20 * np.log10(magnitude / np.sqrt(2) + 1e-20)

                # Frequency bins
                freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

                # Bin spectrum into frequency bands
                band_values = []
                for i in range(num_bands):
                    mask = (freqs >= freq_bands[i]) & (freqs < freq_bands[i + 1])
                    band_values.append(np.mean(magnitude_db[mask]) if np.any(mask) else -100)
                band_values = np.array(band_values)

                # Apply peak hold with smooth decay
                for i in range(num_bands):
                    if band_values[i] > peak_values[i]:
                        peak_values[i] = band_values[i]
                    else:
                        peak_values[i] = peak_values[i] * peak_decay + band_values[i] * (
                            1 - peak_decay
                        )

                # Normalize for display (-60 to -3 dBFS)
                normalized = np.clip((peak_values + 60) / 57, 0, 1)

                # Move cursor to top of screen
                print("\033[H", end="")

                # Display header
                print("\033[1m" + "=" * 70 + "\033[0m")
                print("\033[1m🎵 Real-time FFT Spectrum Analyzer\033[0m")
                print(
                    f"\033[90mFFT: {fft_size} | Bands: {num_bands} | Range: {min_freq}Hz-{format_frequency(max_freq)}Hz\033[0m"
                )
                print("\033[1m" + "=" * 70 + "\033[0m")
                print()

                # Display spectrum
                for i in range(num_bands):
                    freq_label = format_frequency(band_centers[i]).rjust(6)
                    db_value = peak_values[i]
                    bar = create_bar(normalized[i])
                    db_str = f"{db_value:>6.1f} dB" if db_value > -60 else "  -∞ dB"

                    # Color based on level
                    if db_value > -10:
                        color = "\033[91m"  # Red
                    elif db_value > -25:
                        color = "\033[93m"  # Yellow
                    elif db_value > -40:
                        color = "\033[92m"  # Green
                    else:
                        color = "\033[96m"  # Cyan

                    print(f"{freq_label} Hz │{color}{bar}\033[0m│ {db_str}")

                print("\n\033[90mPress Ctrl+C to stop\033[0m\n")

    except KeyboardInterrupt:
        print("\n\n✅ Stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("\033[?25h")  # Show cursor
        if recorder.is_recording():
            recorder.stop_recording()


if __name__ == "__main__":
    main()

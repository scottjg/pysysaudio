"""
High-level Python interface for system audio recording.
"""

import threading
import struct
import queue
from typing import Optional, Iterator, Literal, Union, Any
from pathlib import Path

# Type for audio data that can be yielded from the generator
AudioData = Union[bytes, Any]  # Any for numpy/mlx arrays


class SystemAudioRecorder:
    """
    Record system audio on macOS and Windows.

    Platform support:
        - macOS: Uses ScreenCaptureKit (requires macOS 13.0+)
        - Windows: Uses WASAPI loopback capture

    Example:
        recorder = SystemAudioRecorder()
        recorder.start_recording("output.wav")
        for chunk in recorder.stream():
            # process audio chunk
            pass
        recorder.stop_recording()
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 2,
        format: Literal["bytes", "numpy", "mlx"] = "bytes",
        dtype: Literal["float32", "int16", "int32"] = "float32",
        buffer_size: int = 100,
    ):
        """
        Initialize the system audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz (default: 48000)
            channels: Number of audio channels (default: 2 for stereo)
            format: Format for audio data yielded by stream():
                - "bytes": Raw PCM data as bytes (default, no dependencies)
                - "numpy": NumPy array, shape (frames, channels) (requires numpy)
                - "mlx": MLX array, shape (frames, channels) (requires mlx)
            dtype: Sample data type for output audio:
                - "float32": 32-bit float in range [-1.0, 1.0] (default)
                - "int16": 16-bit signed integer (PCM16, range [-32768, 32767])
                - "int32": 32-bit signed integer (PCM32)
                Note: dtype conversion for format="bytes" uses standard library (no numpy needed).
                      For format="numpy"/"mlx", numpy is used for better performance.
            buffer_size: Maximum number of audio chunks to buffer (default: 100)

        Note:
            Audio is automatically resampled/remixed to match the requested sample_rate
            and channels when the system provides different settings.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.dtype = dtype
        self._recording = False
        self._native_recorder = None
        self._lock = threading.Lock()
        self._audio_queue = queue.Queue(maxsize=buffer_size)
        self._stream_active = False
        self._actual_sample_rate = None  # Detected from system
        self._actual_channels = None  # Detected from first callback

        # Validate format and check dependencies
        # Note: numpy is always needed for automatic resampling, numpy format, or mlx format
        # For bytes format with int16/int32 and no resampling needed, we use pure Python (struct module)
        try:
            import numpy as np

            self._np = np
        except ImportError:
            if format == "numpy":
                raise ImportError(
                    "format='numpy' requires numpy. " "Install with: pip install numpy"
                )
            elif format == "mlx":
                raise ImportError(
                    "format='mlx' requires numpy for audio processing. "
                    "Install with: pip install numpy"
                )
            # For bytes format, we'll only raise error if resampling is actually needed
            # This will be checked when audio capture starts
            self._np = None

        if format == "mlx":
            try:
                import mlx.core as mx

                self._mx = mx
            except ImportError as e:
                raise ImportError(
                    "format='mlx' requires mlx. " f"Install with: pip install mlx. Error: {e}"
                )

    def start_recording(self, output_path: Optional[str] = None):
        """
        Start recording system audio.

        Args:
            output_path: Optional path where the audio file will be saved.
                        If None, audio is only available via stream().

        Note:
            If resampling is needed and output_path is provided, the file will be
            written at the system's native sample rate (typically 48kHz), but the
            stream() will yield audio at the requested sample rate.
            For resampled file output, use stream() and write the file manually.
        """
        with self._lock:
            if self._recording:
                raise RuntimeError("Recording is already in progress")

            # Reset actual rate/channels for new recording session
            self._actual_sample_rate = None
            self._actual_channels = None

            # Import the native module
            try:
                from . import _pysysaudio_native

                self._native_recorder = _pysysaudio_native.AudioRecorder(
                    output_path or "", self.sample_rate, self.channels  # Empty string if None
                )
            except ImportError as e:
                raise RuntimeError(
                    "Native audio recording module not available. "
                    "Make sure pysysaudio is properly installed with native extensions."
                ) from e

            # Always use callback internally to populate the queue
            wrapped_callback = self._internal_callback
            self._native_recorder.start_with_callback(wrapped_callback)
            self._recording = True
            self._stream_active = True

    def _internal_callback(self, data: bytes, sample_count: int, channels: int, sample_rate: float):
        """Internal callback that receives audio data from native layer and queues it."""
        if not self._stream_active:
            return

        # Track actual format from first callback
        if self._actual_channels is None:
            self._actual_channels = channels
            self._actual_sample_rate = int(
                sample_rate
            )  # Get actual sample rate from ScreenCaptureKit

        # Convert data to requested format (includes resampling if needed)
        converted_data = self._convert_audio_data(data, sample_count, channels)

        # Calculate output sample count after resampling
        output_sample_count = sample_count
        output_channels = channels
        if self._actual_channels != self.channels or self._actual_sample_rate != self.sample_rate:
            # Calculate output dimensions
            frames = sample_count // channels
            output_frames = int(frames * self.sample_rate / self._actual_sample_rate)
            output_sample_count = output_frames * self.channels
            output_channels = self.channels

        try:
            # Non-blocking put - if queue is full, drop the chunk
            self._audio_queue.put_nowait((converted_data, output_sample_count, output_channels))
        except queue.Full:
            # Queue is full, drop this chunk
            # This prevents blocking the audio recording thread
            pass

    def _resample_and_remix(self, data: bytes, sample_count: int, channels: int) -> bytes:
        """
        Resample and remix audio to match requested sample rate and channels.

        Args:
            data: Raw PCM bytes (float32)
            sample_count: Total number of samples (frames * channels)
            channels: Number of input channels

        Returns:
            Resampled/remixed audio as bytes (float32)
        """
        # Convert bytes to numpy array for processing
        audio = self._np.frombuffer(data, dtype=self._np.float32).copy()
        frames = sample_count // channels

        # Reshape to (frames, channels) for processing
        if channels > 1:
            audio = audio.reshape(frames, channels)
        else:
            audio = audio.reshape(frames, 1)

        # Channel mixing (e.g., stereo to mono or mono to stereo)
        if channels != self.channels:
            if self.channels == 1 and channels > 1:
                # Downmix to mono by averaging channels
                audio = audio.mean(axis=1, keepdims=True)
            elif self.channels == 2 and channels == 1:
                # Upmix mono to stereo by duplicating
                audio = self._np.tile(audio, (1, 2))
            else:
                # For other channel combinations, just truncate or pad
                if channels > self.channels:
                    audio = audio[:, : self.channels]
                else:
                    # Pad with zeros
                    padding = self._np.zeros(
                        (frames, self.channels - channels), dtype=self._np.float32
                    )
                    audio = self._np.concatenate([audio, padding], axis=1)

        # Sample rate conversion using linear interpolation for both up and down sampling
        if self._actual_sample_rate != self.sample_rate:
            ratio = self.sample_rate / self._actual_sample_rate
            new_frames = int(frames * ratio)

            if new_frames != frames:
                # Use linear interpolation to resample
                old_indices = self._np.linspace(0, frames - 1, frames)
                new_indices = self._np.linspace(0, frames - 1, new_frames)

                # Interpolate each channel
                resampled = self._np.zeros((new_frames, self.channels), dtype=self._np.float32)
                for ch in range(self.channels):
                    resampled[:, ch] = self._np.interp(new_indices, old_indices, audio[:, ch])
                audio = resampled

        # Flatten and convert back to bytes
        return audio.flatten().tobytes()

    def _convert_dtype(self, data: bytes) -> bytes:
        """
        Convert float32 audio data to int16 or int32.

        Args:
            data: Input audio data as float32 bytes

        Returns:
            Converted audio data as bytes in the target dtype
        """
        # Use numpy if available (faster), otherwise use struct (no dependencies)
        if hasattr(self, "_np"):
            # Fast numpy path
            audio = self._np.frombuffer(data, dtype=self._np.float32)
            audio = self._np.clip(audio, -1.0, 1.0)

            if self.dtype == "int16":
                audio_int = (audio * 32767.0).astype(self._np.int16)
            elif self.dtype == "int32":
                audio_int = (audio * 2147483647.0).astype(self._np.int32)
            else:
                return data

            return audio_int.tobytes()
        else:
            # Pure Python path using struct module (no dependencies)
            sample_count = len(data) // 4  # 4 bytes per float32

            # Unpack all float32 values
            floats = struct.unpack(f"<{sample_count}f", data)

            # Convert to target type
            if self.dtype == "int16":
                # Clip and scale to int16 range
                ints = [max(-32768, min(32767, int(f * 32767.0))) for f in floats]
                return struct.pack(f"<{sample_count}h", *ints)
            elif self.dtype == "int32":
                # Clip and scale to int32 range
                ints = [max(-2147483648, min(2147483647, int(f * 2147483647.0))) for f in floats]
                return struct.pack(f"<{sample_count}i", *ints)
            else:
                return data

    def _convert_audio_data(self, data: bytes, sample_count: int, channels: int) -> AudioData:
        """Convert raw bytes to the requested format, with automatic resampling."""
        # Apply resampling/remixing if needed
        if self._actual_channels != self.channels or self._actual_sample_rate != self.sample_rate:
            # Check if numpy is available for resampling
            if self._np is None:
                raise ImportError(
                    "Automatic resampling requires numpy. " "Install with: pip install numpy"
                )
            data = self._resample_and_remix(data, sample_count, channels)
            # After resampling, get actual dimensions from the resampled data
            channels = self.channels
            sample_count = len(data) // 4  # 4 bytes per float32, total samples after resampling

        # Convert dtype if needed (from float32 to int16/int32)
        if self.dtype != "float32":
            data = self._convert_dtype(data)

        # Convert to requested output format
        if self.format == "bytes":
            return data
        elif self.format == "numpy":
            # Determine dtype based on self.dtype
            np_dtype = {
                "float32": self._np.float32,
                "int16": self._np.int16,
                "int32": self._np.int32,
            }[self.dtype]

            array = self._np.frombuffer(data, dtype=np_dtype)
            # Reshape to (frames, channels) for multi-channel audio
            if channels > 1:
                frames = sample_count // channels
                array = array.reshape(frames, channels)
            return array
        elif self.format == "mlx":
            # First to numpy, then to MLX
            np_dtype = {
                "float32": self._np.float32,
                "int16": self._np.int16,
                "int32": self._np.int32,
            }[self.dtype]

            np_array = self._np.frombuffer(data, dtype=np_dtype)

            # Calculate actual sample count from byte length
            bytes_per_sample = {"float32": 4, "int16": 2, "int32": 4}[self.dtype]
            actual_sample_count = len(data) // bytes_per_sample

            # Reshape to (frames, channels) for multi-channel audio
            if channels > 1:
                frames = actual_sample_count // channels
                np_array = np_array.reshape(frames, channels)

            return self._mx.array(np_array)
        else:
            return data

    def stream(self, timeout: Optional[float] = 0.1) -> Iterator[AudioData]:
        """
        Generator that yields audio chunks as they are recorded.

        Args:
            timeout: Timeout in seconds for waiting for new chunks (default: 0.1).
                    None means wait indefinitely. Use a small timeout to allow
                    checking if recording has stopped.

        Yields:
            Audio data in the format specified during initialization (bytes, numpy, or mlx array)

        Example:
            recorder = SystemAudioRecorder(format="numpy")
            recorder.start_recording("output.wav")
            for chunk in recorder.stream():
                # Process audio chunk (numpy array)
                print(f"Received {len(chunk)} samples")
                if some_condition:
                    break
            recorder.stop_recording()
        """
        while self._recording or not self._audio_queue.empty():
            try:
                chunk_data, sample_count, channels = self._audio_queue.get(timeout=timeout)
                yield chunk_data
            except queue.Empty:
                # Timeout reached, check if still recording
                if not self._recording:
                    break
                continue

    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and finalize the audio file.

        Returns:
            Path to the recorded audio file, or empty string if no file was created
        """
        with self._lock:
            if not self._recording:
                raise RuntimeError("No recording in progress")

            self._stream_active = False  # Stop accepting new chunks
            output_path = self._native_recorder.stop()
            self._recording = False
            self._native_recorder = None

            return output_path

    def is_recording(self) -> bool:
        """Check if recording is currently in progress."""
        return self._recording

    @staticmethod
    def check_permission() -> bool:
        """
        Check if Screen Recording permission is granted (macOS only).

        On Windows, always returns True as WASAPI loopback doesn't require special permissions.

        Returns:
            True if permission is granted (or on Windows), False otherwise
        """
        try:
            from . import _pysysaudio_native

            return _pysysaudio_native.check_screen_recording_permission()
        except ImportError:
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops recording if in progress."""
        if self._recording:
            self.stop_recording()
        return False

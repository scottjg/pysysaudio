"""
High-level Python interface for system audio recording.
"""

import threading
import warnings
from typing import Any, Iterator, Literal, Optional, Union

# Type for audio data that can be yielded from the generator
AudioData = Union[bytes, Any]  # Any for numpy/mlx arrays

_FORMATS = ("bytes", "numpy", "mlx")
_DTYPES = ("float32", "int16", "int32")


class SystemAudioRecorder:
    """
    Record system audio on macOS and Windows.

    Platform support:
        - macOS: Uses CoreAudio process taps (requires macOS 14.2+)
        - Windows: Uses WASAPI loopback capture

    Threading model:
        The platform capture thread never runs Python. It pushes float32 samples
        into a native queue. stream() pulls from that queue, and the native layer
        converts to the requested sample rate, channel count and dtype on the
        calling thread using the platform's resampler (AudioConverter on macOS,
        the Media Foundation resampler on Windows). Heavy work inside a stream()
        loop can therefore never stall capture. If the loop falls more than
        buffer_seconds behind, capture drops chunks and dropped_frames counts them.

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
        buffer_seconds: float = 2.0,
        buffer_size: Optional[int] = None,
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
                Arrays are always 2-D, so mono audio is shaped (frames, 1).
            dtype: Sample data type for output audio:
                - "float32": 32-bit float in range [-1.0, 1.0] (default)
                - "int16": 16-bit signed integer (PCM16, range [-32768, 32767])
                - "int32": 32-bit signed integer (PCM32)
            buffer_seconds: How much captured audio to hold between the capture thread and
                stream() before chunks are dropped (default: 2.0). See dropped_frames.
            buffer_size: Deprecated and ignored. Use buffer_seconds.

        Note:
            Audio is resampled and remixed natively to match the requested sample_rate
            and channels when the system provides different settings. This applies to
            both stream() and the WAV file.
        """
        if buffer_size is not None:
            warnings.warn(
                "buffer_size is deprecated and has no effect; use buffer_seconds instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if format not in _FORMATS:
            raise ValueError(f"format must be one of {_FORMATS}, got {format!r}")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {dtype!r}")
        if int(sample_rate) <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
        if int(channels) <= 0:
            raise ValueError(f"channels must be positive, got {channels!r}")

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.format = format
        self.dtype = dtype
        self.buffer_seconds = buffer_seconds
        self._recording = False
        self._native_recorder = None
        self._lock = threading.Lock()
        self._actual_sample_rate = None  # What the system delivers, known at start
        self._actual_channels = None
        self._dropped_frames = 0  # Final count from the last recording

        # numpy is only needed to shape arrays for the numpy and mlx formats.
        self._np = None
        self._mx = None
        if format in ("numpy", "mlx"):
            try:
                import numpy as np
            except ImportError:
                raise ImportError(
                    f"format={format!r} requires numpy. Install with: pip install numpy"
                ) from None
            self._np = np
        if format == "mlx":
            try:
                import mlx.core as mx
            except ImportError as e:
                raise ImportError(
                    f"format='mlx' requires mlx. Install with: pip install mlx. Error: {e}"
                ) from None
            self._mx = mx

    def start_recording(self, output_path: Optional[str] = None):
        """
        Start recording system audio.

        Args:
            output_path: Optional path where a WAV file will be saved. If None, audio is
                only available via stream(). The file is written at the requested
                sample_rate and channels as 16-bit PCM, whatever dtype is set to.
        """
        with self._lock:
            if self._recording:
                raise RuntimeError("Recording is already in progress")

            try:
                from . import _pysysaudio_native
            except ImportError as e:
                raise RuntimeError(
                    "Native audio recording module not available. "
                    "Make sure pysysaudio is properly installed with native extensions."
                ) from e

            # Reset per-recording state
            self._actual_sample_rate = None
            self._actual_channels = None
            self._dropped_frames = 0

            native = _pysysaudio_native.AudioRecorder(
                output_path or "",
                self.sample_rate,
                self.channels,
                self.dtype,
                self.buffer_seconds,
            )
            native.start()

            # The capture format is known as soon as capture starts.
            self._actual_sample_rate = int(native.capture_sample_rate())
            self._actual_channels = int(native.capture_channels())

            self._native_recorder = native
            self._recording = True

    def _format_output(self, data: bytes) -> AudioData:
        """Shape converted bytes into the requested output format."""
        if self.format == "bytes":
            return data

        # numpy and mlx are always shaped (frames, channels), mono included, so callers
        # can index chunk[:, ch] without checking the channel count.
        np = self._np
        np_dtype = {"float32": np.float32, "int16": np.int16, "int32": np.int32}[self.dtype]
        array = np.frombuffer(data, dtype=np_dtype).reshape(-1, self.channels)

        if self.format == "mlx":
            return self._mx.array(array)
        return array

    def stream(self, timeout: Optional[float] = 0.1) -> Iterator[AudioData]:
        """
        Generator that yields audio chunks as they are recorded.

        Chunks are read from the native capture queue and converted on the calling
        thread, so processing in the loop body never stalls capture. If the loop
        falls more than buffer_seconds behind, chunks are dropped; see dropped_frames.

        Args:
            timeout: Timeout in seconds for waiting for new chunks (default: 0.1).
                    None means wait indefinitely. The generator ends once recording
                    has stopped and the remaining audio, including the resampler's
                    final few milliseconds, has been delivered.

        Yields:
            Audio data in the format specified during initialization (bytes, numpy, or mlx array)

        Example:
            recorder = SystemAudioRecorder(format="numpy")
            recorder.start_recording("output.wav")
            for chunk in recorder.stream():
                # Process audio chunk (numpy array)
                print(f"Received {len(chunk)} frames")
                if some_condition:
                    break
            recorder.stop_recording()
        """
        native = self._native_recorder
        if native is None:
            return

        wait = -1.0 if timeout is None else float(timeout)
        # Cap a single read at one second of capture so a consumer that fell behind
        # catches up in bounded pieces rather than one huge chunk.
        max_frames = max(self._actual_sample_rate or 48000, 1)

        while True:
            data = native.read(max_frames, wait)
            if data is None:
                # Stopped, and everything has been delivered.
                break
            if data:
                yield self._format_output(data)
            # Empty bytes: the wait timed out, or the resampler is still filling.

    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and finalize the audio file.

        A stream() loop running in another thread wakes up, drains what is left
        in the queue, and ends.

        Returns:
            Path to the recorded audio file, or empty string if no file was created
        """
        with self._lock:
            if not self._recording:
                raise RuntimeError("No recording in progress")

            native = self._native_recorder
            output_path = native.stop()
            self._dropped_frames = int(native.dropped_frames())
            self._recording = False
            self._native_recorder = None

            return output_path

    def is_recording(self) -> bool:
        """Check if recording is currently in progress."""
        return self._recording

    @property
    def dropped_frames(self) -> int:
        """
        Frames of captured audio discarded because stream() fell behind.

        Counted at the capture sample rate. Resets on each start_recording();
        after stop_recording() it holds the final count for the last recording.
        """
        native = self._native_recorder
        if native is not None:
            return int(native.dropped_frames())
        return self._dropped_frames

    @staticmethod
    def check_permission() -> bool:
        """
        Check if native system-capture support is available (macOS only).

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

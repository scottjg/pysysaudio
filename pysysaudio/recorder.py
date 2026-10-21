"""
High-level Python interface for system audio recording.
"""

import threading
import struct
import warnings
from array import array
from typing import Optional, Iterator, Literal, Union, Any
from pathlib import Path

# Type for audio data that can be yielded from the generator
AudioData = Union[bytes, Any]  # Any for numpy/mlx arrays


class SystemAudioRecorder:
    """
    Record system audio on macOS and Windows.

    Platform support:
        - macOS: Uses CoreAudio process taps (requires macOS 14.2+)
        - Windows: Uses WASAPI loopback capture

    Threading model:
        The platform capture thread never runs Python. It pushes float32 samples
        into a native queue; stream() pulls from that queue and does all format
        conversion on the calling thread. Heavy work inside a stream() loop can
        therefore never stall capture. If the loop falls more than buffer_seconds
        behind, capture drops chunks and dropped_frames counts them.

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
            dtype: Sample data type for output audio:
                - "float32": 32-bit float in range [-1.0, 1.0] (default)
                - "int16": 16-bit signed integer (PCM16, range [-32768, 32767])
                - "int32": 32-bit signed integer (PCM32)
                Note: dtype conversion for format="bytes" uses standard library (no numpy needed).
                      For format="numpy"/"mlx", numpy is used for better performance.
            buffer_seconds: How much captured audio to hold between the capture thread and
                stream() before chunks are dropped (default: 2.0). See dropped_frames.
            buffer_size: Deprecated and ignored. Use buffer_seconds.

        Note:
            Audio is automatically resampled/remixed to match the requested sample_rate
            and channels when the system provides different settings.
        """
        if buffer_size is not None:
            warnings.warn(
                "buffer_size is deprecated and has no effect; use buffer_seconds instead",
                DeprecationWarning,
                stacklevel=2,
            )
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.dtype = dtype
        self.buffer_seconds = buffer_seconds
        self._recording = False
        self._native_recorder = None
        self._lock = threading.Lock()
        self._actual_sample_rate = None  # Detected from the capture backend at start
        self._actual_channels = None
        self._dropped_frames = 0  # Final count from the last recording
        self._reset_resampler_state()

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
            self._reset_resampler_state()

            native = _pysysaudio_native.AudioRecorder(output_path or "", self.buffer_seconds)
            native.start()

            # The capture format is known as soon as capture starts.
            self._actual_sample_rate = int(native.sample_rate())
            self._actual_channels = int(native.channels())

            needs_resample = (
                self._actual_channels != self.channels
                or self._actual_sample_rate != self.sample_rate
            )
            if needs_resample and self._np is None and self.format != "bytes":
                native.stop()
                raise ImportError(
                    "Automatic resampling requires numpy. Install with: pip install numpy"
                )

            self._native_recorder = native
            self._recording = True

    def _reset_resampler_state(self) -> None:
        """Forget the carried-over frame and read position from a previous recording."""
        # Last remixed input frame of the previous chunk. Interpolation across a chunk
        # boundary reads from it, so the output stream is continuous.
        self._resample_prev: Optional[Any] = None
        # Read position of the next output sample, in units of 1/sample_rate input frames,
        # measured from the carried-over frame. Kept as an integer so it never drifts.
        self._resample_phase_num = 0

    def _resample_plan(self, available_frames: int) -> tuple:
        """
        Decide where in the current input buffer the output samples are read from.

        The buffer holds the carried-over frame (if any) followed by the current chunk.
        Output sample n is read at input position (start_num + n * src) / dst frames,
        where src and dst are the input and output sample rates.

        Returns (start_num, count) and advances the carried read position so the next
        chunk picks up exactly where this one left off.
        """
        src = self._actual_sample_rate
        dst = self.sample_rate
        start_num = self._resample_phase_num
        last = available_frames - 1

        if last < 0 or start_num > last * dst:
            count = 0
        else:
            count = (last * dst - start_num) // src + 1

        # Re-express the next read position relative to the final frame of this buffer,
        # which becomes index 0 of the next buffer.
        self._resample_phase_num = start_num + count * src - max(last, 0) * dst
        return start_num, count

    def _resample_and_remix(self, data: bytes, sample_count: int, channels: int) -> bytes:
        """
        Resample and remix audio to match requested sample rate and channels.

        Resampling is stateful: the fractional read position and the last input frame are
        carried from one chunk to the next, so consecutive chunks join seamlessly and the
        output rate is exact over time.

        Args:
            data: Raw PCM bytes (float32)
            sample_count: Total number of samples (frames * channels)
            channels: Number of input channels

        Returns:
            Resampled/remixed audio as bytes (float32)
        """
        if self._np is None:
            return self._resample_and_remix_pure_python(data, sample_count, channels)

        np = self._np
        frames = sample_count // channels
        audio = np.frombuffer(data, dtype=np.float32).reshape(frames, max(channels, 1))

        # Channel mixing (e.g., stereo to mono or mono to stereo)
        if channels != self.channels:
            if self.channels == 1 and channels > 1:
                # Downmix to mono by averaging channels
                audio = audio.mean(axis=1, keepdims=True, dtype=np.float32)
            elif self.channels == 2 and channels == 1:
                # Upmix mono to stereo by duplicating
                audio = np.tile(audio, (1, 2))
            elif channels > self.channels:
                audio = audio[:, : self.channels]
            else:
                padding = np.zeros((frames, self.channels - channels), dtype=np.float32)
                audio = np.concatenate([audio, padding], axis=1)

        if self._actual_sample_rate != self.sample_rate:
            audio = self._resample_numpy(audio)

        return np.ascontiguousarray(audio, dtype=np.float32).tobytes()

    def _resample_numpy(self, audio: Any) -> Any:
        """Linear-interpolation resampling of a (frames, channels) float32 array."""
        np = self._np
        if self._resample_prev is not None:
            buf = np.concatenate([self._resample_prev, audio], axis=0)
        else:
            buf = audio

        start_num, count = self._resample_plan(buf.shape[0])
        if buf.shape[0] > 0:
            self._resample_prev = buf[-1:].copy()

        if count == 0:
            return np.zeros((0, buf.shape[1]), dtype=np.float32)

        positions = (start_num + np.arange(count, dtype=np.int64) * self._actual_sample_rate) / (
            self.sample_rate
        )
        idx = np.minimum(positions.astype(np.int64), buf.shape[0] - 1)
        frac = (positions - idx).astype(np.float32)[:, None]
        nxt = np.minimum(idx + 1, buf.shape[0] - 1)
        return buf[idx] * (1.0 - frac) + buf[nxt] * frac

    def _resample_and_remix_pure_python(
        self, data: bytes, sample_count: int, channels: int
    ) -> bytes:
        """
        Pure Python fallback for resampling/remixing when numpy is unavailable.
        This is slower than the numpy implementation but avoids the dependency.
        """
        samples = array("f")
        samples.frombytes(data)

        frames = sample_count // channels if channels else 0
        working_channels = channels

        if working_channels != self.channels and frames > 0:
            samples = self._remix_channels_pure_python(samples, frames, working_channels)
            working_channels = self.channels
            frames = len(samples) // working_channels if working_channels else 0

        if self._actual_sample_rate != self.sample_rate and working_channels > 0:
            samples = self._resample_pure_python(samples, frames, working_channels)

        return samples.tobytes()

    def _remix_channels_pure_python(
        self, samples: array, frames: int, input_channels: int
    ) -> array:
        """Pure Python channel remixing."""
        target_channels = self.channels
        remixed = array("f")

        if input_channels == 0 or frames == 0:
            return remixed

        if target_channels == 1:
            for frame_idx in range(frames):
                start = frame_idx * input_channels
                total = 0.0
                for ch in range(input_channels):
                    total += samples[start + ch]
                remixed.append(total / input_channels)
            return remixed

        if input_channels == 1:
            for frame_idx in range(frames):
                value = samples[frame_idx]
                for _ in range(target_channels):
                    remixed.append(value)
            return remixed

        min_channels = min(input_channels, target_channels)
        for frame_idx in range(frames):
            start = frame_idx * input_channels
            for ch in range(min_channels):
                remixed.append(samples[start + ch])
            if target_channels > min_channels:
                remixed.extend([0.0] * (target_channels - min_channels))

        return remixed

    def _resample_pure_python(self, samples: array, frames: int, channels: int) -> array:
        """Pure Python linear-interpolation resampling of interleaved float samples."""
        # Prepend the carried-over frame so interpolation spans the chunk boundary.
        if self._resample_prev is not None:
            buf = array("f", self._resample_prev)
            buf.extend(samples)
            buf_frames = frames + 1
        else:
            buf = samples
            buf_frames = frames

        start_num, count = self._resample_plan(buf_frames)
        if buf_frames > 0:
            self._resample_prev = buf[(buf_frames - 1) * channels :]

        out = array("f")
        if count == 0:
            return out

        src = self._actual_sample_rate
        dst = self.sample_rate
        last = buf_frames - 1
        for n in range(count):
            num = start_num + n * src
            idx = min(num // dst, last)
            frac = (num - idx * dst) / dst
            nxt = min(idx + 1, last)
            base = idx * channels
            base_next = nxt * channels
            for ch in range(channels):
                a = buf[base + ch]
                b = buf[base_next + ch]
                out.append(a + (b - a) * frac)
        return out

    def _convert_dtype(self, data: bytes) -> bytes:
        """
        Convert float32 audio data to int16 or int32.

        Args:
            data: Input audio data as float32 bytes

        Returns:
            Converted audio data as bytes in the target dtype
        """
        # Use numpy if available (faster), otherwise use struct (no dependencies)
        if self._np is not None:
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
        needs_resample = (
            self._actual_channels != self.channels
            or self._actual_sample_rate != self.sample_rate
        )

        if needs_resample and self._np is None and self.format != "bytes":
            raise ImportError(
                "Automatic resampling requires numpy. " "Install with: pip install numpy"
            )

        # Apply resampling/remixing if needed (numpy or pure python path)
        if needs_resample:
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

        Chunks are read from the native capture queue and converted on the calling
        thread, so processing in the loop body never stalls capture. If the loop
        falls more than buffer_seconds behind, chunks are dropped; see dropped_frames.

        Args:
            timeout: Timeout in seconds for waiting for new chunks (default: 0.1).
                    None means wait indefinitely. The generator ends once recording
                    has stopped and the remaining chunks have been drained.

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
        native = self._native_recorder
        if native is None:
            return

        wait = -1.0 if timeout is None else float(timeout)
        channels = self._actual_channels or 1
        # Cap a single read at one second so a consumer that fell behind catches
        # up in bounded pieces rather than one huge chunk.
        max_frames = max(self._actual_sample_rate or 48000, 1)

        while True:
            data = native.read(max_frames, wait)
            if data:
                yield self._convert_audio_data(data, len(data) // 4, channels)
            elif not native.is_recording():
                # Stopped, and everything that was queued has been delivered.
                break
            # Otherwise the wait timed out while capture is still running; keep waiting.

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

"""
Tests for pysysaudio recorder functionality
"""

import math
import pytest
import tempfile
import wave
import os
import struct
import platform
import sys
import threading
import time
from pathlib import Path


def _macos_supports_coreaudio_taps() -> bool:
    """CoreAudio process taps are available on macOS 14.2+."""
    if sys.platform != "darwin":
        return False
    version = platform.mac_ver()[0]
    if not version:
        return False
    parts = [int(p) for p in version.split(".") if p]
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts[:2]) >= (14, 2)


def test_import():
    """Test that the module can be imported"""
    from pysysaudio import SystemAudioRecorder

    assert SystemAudioRecorder is not None


def test_recorder_initialization():
    """Test recorder initialization with different parameters"""
    from pysysaudio import SystemAudioRecorder

    # Default parameters
    recorder1 = SystemAudioRecorder()
    assert recorder1.sample_rate == 48000
    assert recorder1.channels == 2
    assert recorder1.format == "bytes"
    assert recorder1.dtype == "float32"

    # Custom parameters
    recorder2 = SystemAudioRecorder(sample_rate=44100, channels=1, format="bytes", dtype="int16")
    assert recorder2.sample_rate == 44100
    assert recorder2.channels == 1
    assert recorder2.format == "bytes"
    assert recorder2.dtype == "int16"


def test_recorder_not_recording_initially():
    """Test that recorder is not recording when first created"""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder()
    assert recorder.is_recording() is False


def test_check_permission():
    """Test checking screen recording permission"""
    from pysysaudio import SystemAudioRecorder

    # Should return a boolean
    permission = SystemAudioRecorder.check_permission()
    assert isinstance(permission, bool)


def test_context_manager():
    """Test using recorder as context manager"""
    from pysysaudio import SystemAudioRecorder

    with SystemAudioRecorder() as recorder:
        assert recorder is not None
        assert recorder.is_recording() is False


@pytest.mark.skipif(
    not _macos_supports_coreaudio_taps(),
    reason="CoreAudio process taps require macOS 14.2+",
)
def test_recording_lifecycle():
    """Test basic recording start/stop lifecycle"""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Should not be recording initially
        assert not recorder.is_recording()

        # Start recording
        recorder.start_recording(tmp_path)
        assert recorder.is_recording()

        # Stop recording
        output_path = recorder.stop_recording()
        assert not recorder.is_recording()
        assert output_path == tmp_path

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.mark.skipif(
    not _macos_supports_coreaudio_taps(),
    reason="CoreAudio process taps require macOS 14.2+",
)
def test_cannot_start_twice():
    """Test that starting recording twice raises an error"""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        recorder.start_recording(tmp_path)

        # Trying to start again should raise an error
        with pytest.raises(RuntimeError, match="already in progress"):
            recorder.start_recording(tmp_path)

        recorder.stop_recording()

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.mark.skipif(
    not _macos_supports_coreaudio_taps(),
    reason="CoreAudio process taps require macOS 14.2+",
)
def test_cannot_stop_without_start():
    """Test that stopping without starting raises an error"""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder()

    with pytest.raises(RuntimeError, match="No recording in progress"):
        recorder.stop_recording()


def test_stream_without_recording():
    """Test that stream() works but yields nothing when not recording"""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder()
    
    # Stream should not yield anything when not recording
    chunks = list(recorder.stream(timeout=0.1))
    assert len(chunks) == 0


def _native():
    from pysysaudio import _pysysaudio_native

    return _pysysaudio_native


def _pack(values):
    return struct.pack(f"<{len(values)}f", *values)


def _unpack(data):
    return struct.unpack(f"<{len(data) // 4}f", data)


def _sine(rate, seconds, freq, channels, amplitude=0.5):
    """Interleaved float32 bytes of a sine wave, plus the frame count."""
    frames = int(rate * seconds)
    values = []
    for i in range(frames):
        v = amplitude * math.sin(2 * math.pi * freq * i / rate)
        values.extend([v] * channels)
    return _pack(values), frames


def _convert(converter, data, frame_bytes, chunk_frames):
    """Feed data through a converter in fixed-size chunks, then flush."""
    out = bytearray()
    step = chunk_frames * frame_bytes
    for start in range(0, len(data), step):
        out += converter.process(data[start : start + step])
    out += converter.flush()
    return bytes(out)


def _rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


def test_converter_passthrough_is_exact():
    """Same rate and channel count with float32 out must not touch the samples."""
    data = _pack([0.0, 0.25, -0.5, 1.0, -1.0, 0.125])
    converter = _native().FormatConverter(48000, 2, 48000, 2, "float32")
    assert converter.process(data) == data
    assert converter.flush() == b""
    assert converter.flush() == b""  # flushing twice is harmless


@pytest.mark.parametrize(
    "dtype,fmt,expected",
    [
        ("int16", "h", (-32767, -32767, -16383, 0, 16383, 32767, 32767)),
        ("int32", "i", (-2147483647, -2147483647, -1073741823, 0, 1073741823, 2147483647, 2147483647)),
    ],
)
def test_converter_encodes_integers(dtype, fmt, expected):
    """Integers are clipped to [-1, 1], scaled by the type's max and truncated toward zero."""
    data = _pack([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    converter = _native().FormatConverter(48000, 1, 48000, 1, dtype)
    out = converter.process(data)
    assert struct.unpack(f"<{len(expected)}{fmt}", out) == expected


@pytest.mark.parametrize(
    "in_channels,out_channels,frame,expected",
    [
        (2, 1, [0.2, 0.6], [0.4]),  # downmix averages
        (1, 2, [0.3], [0.3, 0.3]),  # mono duplicates
        (1, 3, [0.3], [0.3, 0.3, 0.3]),
        (3, 2, [0.1, 0.2, 0.3], [0.1, 0.2]),  # extra channels dropped
        (2, 3, [0.1, 0.2], [0.1, 0.2, 0.0]),  # missing channels silent
    ],
)
def test_converter_remixes_channels(in_channels, out_channels, frame, expected):
    converter = _native().FormatConverter(48000, in_channels, 48000, out_channels, "float32")
    out = _unpack(converter.process(_pack(frame * 4)))
    assert out == pytest.approx(expected * 4)


def test_converter_rejects_bad_input():
    native = _native()
    with pytest.raises(ValueError):
        native.FormatConverter(48000, 2, 16000, 1, "float64")
    with pytest.raises(ValueError):
        native.FormatConverter(48000, 0, 16000, 1, "float32")
    converter = native.FormatConverter(48000, 2, 16000, 1, "float32")
    with pytest.raises(ValueError):
        converter.process(b"\x00" * 12)  # one and a half stereo frames


@pytest.mark.parametrize(
    "src_rate,dst_rate,in_channels,out_channels",
    [
        (48000, 16000, 2, 1),  # integer ratio with downmix (speech-to-text path)
        (48000, 44100, 1, 1),  # non-integer ratio down
        (44100, 48000, 1, 2),  # non-integer ratio up with upmix
    ],
)
def test_native_resampler_is_continuous_and_exact(
    src_rate, dst_rate, in_channels, out_channels
):
    """Chunked conversion matches whole-signal conversion, keeps the rate and the tone."""
    native = _native()
    freq = 440.0
    data, frames = _sine(src_rate, 0.25, freq, in_channels)
    frame_bytes = 4 * in_channels

    chunked = _convert(
        native.FormatConverter(src_rate, in_channels, dst_rate, out_channels, "float32"),
        data, frame_bytes, chunk_frames=512,
    )
    whole = _convert(
        native.FormatConverter(src_rate, in_channels, dst_rate, out_channels, "float32"),
        data, frame_bytes, chunk_frames=frames,
    )

    # Chunk boundaries leave no trace in the output.
    assert _unpack(chunked) == pytest.approx(_unpack(whole), abs=1e-6)

    # After flush, the output length matches the input duration at the new rate.
    out_frames = len(chunked) // (4 * out_channels)
    assert abs(out_frames - frames * dst_rate / src_rate) <= 4

    # Every output channel carries the same tone at the new rate. Allow for a fixed
    # filter delay, but no drift: one lag must fit the whole middle of the signal.
    first = _unpack(chunked)[::out_channels]
    margin = dst_rate // 20
    ideal = [0.5 * math.sin(2 * math.pi * freq * n / dst_rate) for n in range(len(first))]
    best = min(
        max(abs(first[n + lag] - ideal[n]) for n in range(margin, len(first) - margin - 64))
        for lag in range(0, 65)
    )
    assert best < 0.01


def test_native_resampler_rejects_aliases():
    """A tone above the new Nyquist frequency must be filtered out, not folded back."""
    native = _native()
    margin = 800

    data, _ = _sine(48000, 0.25, 10000.0, 1)  # 10 kHz is above 16 kHz's 8 kHz Nyquist
    aliased = _unpack(_convert(native.FormatConverter(48000, 1, 16000, 1), data, 4, 512))
    assert _rms(aliased[margin:-margin]) < 0.01

    data, _ = _sine(48000, 0.25, 1000.0, 1)  # in-band tone passes at full level
    passed = _unpack(_convert(native.FormatConverter(48000, 1, 16000, 1), data, 4, 512))
    assert _rms(passed[margin:-margin]) == pytest.approx(0.5 / math.sqrt(2), rel=0.02)


def test_buffer_size_is_deprecated():
    """The old chunk-count parameter is accepted but warns."""
    from pysysaudio import SystemAudioRecorder

    with pytest.warns(DeprecationWarning):
        recorder = SystemAudioRecorder(buffer_size=10)
    assert recorder.buffer_seconds == 2.0


def test_dropped_frames_before_recording():
    from pysysaudio import SystemAudioRecorder

    assert SystemAudioRecorder().dropped_frames == 0


@pytest.mark.skipif(
    not _macos_supports_coreaudio_taps(),
    reason="CoreAudio process taps require macOS 14.2+",
)
def test_stream_end_to_end():
    """Capture briefly and check stream() delivers converted audio without drops."""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder(sample_rate=16000, channels=1, format="bytes", dtype="int16")
    recorder.start_recording()
    try:
        # Capture format is known as soon as recording starts.
        assert recorder._actual_sample_rate > 0
        assert recorder._actual_channels > 0

        deadline = time.monotonic() + 0.5
        total_frames = 0
        for chunk in recorder.stream():
            assert isinstance(chunk, bytes)
            assert len(chunk) % 2 == 0  # whole int16 mono frames
            total_frames += len(chunk) // 2
            if time.monotonic() >= deadline:
                break
    finally:
        recorder.stop_recording()

    # At least half of the 0.5 s of 16 kHz audio should have arrived.
    assert total_frames >= 4000
    assert recorder.dropped_frames == 0


@pytest.mark.skipif(
    not _macos_supports_coreaudio_taps(),
    reason="CoreAudio process taps require macOS 14.2+",
)
def test_stream_ends_when_stopped_from_another_thread():
    """A stream() loop blocked with no timeout must wake and finish on stop."""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder()
    recorder.start_recording()
    threading.Timer(0.3, recorder.stop_recording).start()

    started = time.monotonic()
    chunks = list(recorder.stream(timeout=None))
    elapsed = time.monotonic() - started

    assert not recorder.is_recording()
    assert elapsed < 3.0
    assert all(isinstance(c, bytes) for c in chunks)


@pytest.mark.parametrize("dtype,size", [("float32", 4), ("int16", 2), ("int32", 4)])
@pytest.mark.parametrize("channels", [1, 2, 3])
def test_numpy_output_is_always_2d(dtype, size, channels):
    """numpy chunks are (frames, channels) for every channel count, mono included."""
    np = pytest.importorskip("numpy")
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder(channels=channels, format="numpy", dtype=dtype)
    chunk = recorder._format_output(bytes(480 * channels * size))

    assert chunk.shape == (480, channels)
    assert chunk.dtype == np.dtype(dtype)


def test_numpy_mono_values_land_in_column_zero():
    np = pytest.importorskip("numpy")
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder(channels=1, format="numpy")
    chunk = recorder._format_output(np.arange(5, dtype=np.float32).tobytes())

    assert chunk.shape == (5, 1)
    assert chunk[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("channels", [1, 2])
def test_mlx_output_is_always_2d(channels):
    """mlx chunks share the numpy shape contract."""
    pytest.importorskip("numpy")
    pytest.importorskip("mlx.core")
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder(channels=channels, format="mlx")
    chunk = recorder._format_output(bytes(480 * channels * 4))

    assert tuple(chunk.shape) == (480, channels)


def test_invalid_settings_are_rejected():
    from pysysaudio import SystemAudioRecorder

    with pytest.raises(ValueError):
        SystemAudioRecorder(dtype="float64")
    with pytest.raises(ValueError):
        SystemAudioRecorder(format="list")
    with pytest.raises(ValueError):
        SystemAudioRecorder(channels=0)


@pytest.mark.skipif(
    not _macos_supports_coreaudio_taps(),
    reason="CoreAudio process taps require macOS 14.2+",
)
def test_wav_file_uses_requested_format():
    """The file is written at the requested rate and channel count, as 16-bit PCM."""
    from pysysaudio import SystemAudioRecorder

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        recorder = SystemAudioRecorder(sample_rate=16000, channels=1, dtype="float32")
        recorder.start_recording(tmp_path)
        time.sleep(0.5)
        recorder.stop_recording()

        with wave.open(tmp_path) as w:
            assert w.getframerate() == 16000
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert 0.3 * 16000 <= w.getnframes() <= 0.8 * 16000
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for pysysaudio recorder functionality
"""

import pytest
import tempfile
import os
import struct
import platform
import sys
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


def test_bytes_resample_without_numpy():
    """Ensure bytes format can resample even when numpy is unavailable."""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder(sample_rate=24000, channels=1, format="bytes", dtype="float32")
    recorder._np = None  # Simulate numpy not being installed
    recorder._actual_sample_rate = 48000
    recorder._actual_channels = 2

    frames = 4
    input_channels = 2
    sample_count = frames * input_channels
    samples = [float(i) for i in range(sample_count)]
    data = struct.pack(f"<{sample_count}f", *samples)

    result = recorder._convert_audio_data(data, sample_count, input_channels)
    expected_frames = int(frames * recorder.sample_rate / recorder._actual_sample_rate)
    assert len(result) == expected_frames * recorder.channels * 4  # float32 bytes

    output_samples = struct.unpack(f"<{expected_frames * recorder.channels}f", result)
    # Mono frames are 0.5, 2.5, 4.5, 6.5; a 2:1 decimation reads frames 0 and 2.
    assert output_samples == pytest.approx((0.5, 4.5))


def test_bytes_int16_conversion_without_numpy():
    """Ensure dtype conversion works without numpy dependency."""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder(format="bytes", dtype="int16")
    recorder._np = None  # Simulate numpy not being installed

    data = struct.pack("<2f", -1.0, 1.0)
    converted = recorder._convert_dtype(data)

    ints = struct.unpack("<2h", converted)
    assert ints == (-32767, 32767)


def _make_recorder(src_rate, dst_rate, in_channels, out_channels, use_numpy):
    """Build a recorder with capture format pre-set, bypassing the native layer."""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder(
        sample_rate=dst_rate, channels=out_channels, format="bytes", dtype="float32"
    )
    if not use_numpy:
        recorder._np = None
    recorder._actual_sample_rate = src_rate
    recorder._actual_channels = in_channels
    return recorder


def _sine_frames(rate, seconds, freq, channels):
    """Interleaved float32 bytes of a sine wave, plus the frame count."""
    import math

    frames = int(rate * seconds)
    values = []
    for i in range(frames):
        v = math.sin(2 * math.pi * freq * i / rate)
        values.extend([v] * channels)
    return struct.pack(f"<{len(values)}f", *values), frames


def _resample_in_chunks(recorder, data, frames, channels, chunk_frames):
    """Push data through the resampler in fixed-size chunks and return the output."""
    out = bytearray()
    bytes_per_frame = channels * 4
    for start in range(0, frames, chunk_frames):
        chunk = data[start * bytes_per_frame : (start + chunk_frames) * bytes_per_frame]
        chunk_frame_count = len(chunk) // bytes_per_frame
        out += recorder._resample_and_remix(chunk, chunk_frame_count * channels, channels)
    return bytes(out)


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize(
    "src_rate,dst_rate,in_channels,out_channels",
    [
        (48000, 16000, 2, 1),  # integer ratio with downmix (speech-to-text path)
        (48000, 44100, 1, 1),  # non-integer ratio down
        (44100, 48000, 1, 2),  # non-integer ratio up with upmix
    ],
)
def test_resampler_is_continuous_across_chunks(
    src_rate, dst_rate, in_channels, out_channels, use_numpy
):
    """Chunked resampling must equal whole-signal resampling and keep an exact rate."""
    if use_numpy:
        pytest.importorskip("numpy")

    seconds = 0.2 if use_numpy else 0.05
    data, frames = _sine_frames(src_rate, seconds, 440.0, in_channels)

    chunked = _resample_in_chunks(
        _make_recorder(src_rate, dst_rate, in_channels, out_channels, use_numpy),
        data,
        frames,
        in_channels,
        chunk_frames=512,
    )
    whole = _resample_in_chunks(
        _make_recorder(src_rate, dst_rate, in_channels, out_channels, use_numpy),
        data,
        frames,
        in_channels,
        chunk_frames=frames,
    )

    # Identical bytes: chunk boundaries leave no trace in the output.
    assert chunked == whole

    # The output length tracks the input length at the target rate to within one frame.
    out_frames = len(chunked) // (out_channels * 4)
    expected_frames = frames * dst_rate / src_rate
    assert abs(out_frames - expected_frames) <= 1

    # The output is the same sine wave at the new rate: no drift, no pitch shift.
    import math

    samples = struct.unpack(f"<{out_frames * out_channels}f", chunked)
    worst = 0.0
    for n in range(out_frames):
        ideal = math.sin(2 * math.pi * 440.0 * n / dst_rate)
        for ch in range(out_channels):
            worst = max(worst, abs(samples[n * out_channels + ch] - ideal))
    assert worst < 0.01


def test_pure_python_resampler_matches_numpy():
    """The dependency-free fallback must produce the same samples as the numpy path."""
    pytest.importorskip("numpy")

    data, frames = _sine_frames(48000, 0.05, 440.0, 2)
    with_numpy = _resample_in_chunks(
        _make_recorder(48000, 16000, 2, 1, True), data, frames, 2, chunk_frames=500
    )
    pure = _resample_in_chunks(
        _make_recorder(48000, 16000, 2, 1, False), data, frames, 2, chunk_frames=500
    )

    assert len(with_numpy) == len(pure)
    a = struct.unpack(f"<{len(with_numpy) // 4}f", with_numpy)
    b = struct.unpack(f"<{len(pure) // 4}f", pure)
    assert a == pytest.approx(b, abs=1e-6)


def test_resampler_state_resets_between_recordings():
    """A new recording must not inherit the read position from the previous one."""
    recorder = _make_recorder(48000, 44100, 1, 1, False)
    data, frames = _sine_frames(48000, 0.01, 440.0, 1)
    recorder._resample_and_remix(data, frames, 1)
    assert recorder._resample_prev is not None

    recorder._reset_resampler_state()
    assert recorder._resample_prev is None
    assert recorder._resample_phase_num == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

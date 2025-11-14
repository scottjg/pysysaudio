"""
Tests for pysysaudio recorder functionality
"""

import pytest
import tempfile
import os
import struct
from pathlib import Path


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
    not os.path.exists("/System/Library/Frameworks/ScreenCaptureKit.framework"),
    reason="ScreenCaptureKit not available",
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
    not os.path.exists("/System/Library/Frameworks/ScreenCaptureKit.framework"),
    reason="ScreenCaptureKit not available",
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
    not os.path.exists("/System/Library/Frameworks/ScreenCaptureKit.framework"),
    reason="ScreenCaptureKit not available",
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
    assert output_samples == pytest.approx((0.5, 6.5))


def test_bytes_int16_conversion_without_numpy():
    """Ensure dtype conversion works without numpy dependency."""
    from pysysaudio import SystemAudioRecorder

    recorder = SystemAudioRecorder(format="bytes", dtype="int16")
    recorder._np = None  # Simulate numpy not being installed

    data = struct.pack("<2f", -1.0, 1.0)
    converted = recorder._convert_dtype(data)

    ints = struct.unpack("<2h", converted)
    assert ints == (-32767, 32767)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

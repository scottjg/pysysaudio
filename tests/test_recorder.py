"""
Tests for pysysaudio recorder functionality
"""

import pytest
import tempfile
import os
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

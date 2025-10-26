"""
pysysaudio - System audio recording for macOS and Windows

This module provides a Python interface to record system audio:
- macOS: Uses Apple's ScreenCaptureKit framework (macOS 13.0+)
- Windows: Uses WASAPI (Windows Audio Session API) with loopback capture
"""

from .recorder import SystemAudioRecorder

__version__ = "0.1.0"
__all__ = ["SystemAudioRecorder"]

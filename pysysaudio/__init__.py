"""
pysysaudio - System audio recording for macOS and Windows

This module provides a Python interface to record system audio:
- macOS: Uses Apple's CoreAudio process taps (macOS 14.2+)
- Windows: Uses WASAPI (Windows Audio Session API) with loopback capture
"""

from importlib.metadata import PackageNotFoundError, version as _package_version

from .recorder import SystemAudioRecorder

try:
    # Single source of truth is the version in pyproject.toml.
    __version__ = _package_version("pysysaudio")
except PackageNotFoundError:  # Running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["SystemAudioRecorder", "__version__"]

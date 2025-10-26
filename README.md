# pysysaudio

A Python library for recording system audio on **macOS** and **Windows**.

## Features

- 🎵 Record system audio (all sounds playing on your system)
- 🪟 **Cross-platform**: macOS (ScreenCaptureKit) and Windows (WASAPI)
- 🎚️ Configurable sample rate and channel count
- 📝 Save recordings to WAV format
- 🔧 Configurable output formats: bytes, NumPy, or MLX arrays
- 🎛️ Automatic resampling and channel remixing

## Requirements

### macOS
- macOS 13.0 (Ventura) or later
- Python 3.9+
- Screen Recording permission (macOS will prompt on first use)

### Windows
- Windows 10 or later
- Python 3.9+
- Visual Studio C++ build tools (for installation from source)

## Installation

### From PyPI (Recommended)

```bash
pip install pysysaudio
```

Pre-built wheels are available for:
- **macOS**: 13.0+ (Ventura and later) - Intel, Apple Silicon, and Universal2
- **Windows**: 10+ (64-bit)
- **Python**: 3.9, 3.10, 3.11, 3.12, 3.13

### From Source

```bash
git clone https://github.com/yourusername/pysysaudio.git
cd pysysaudio
pip install .
```

### For Development

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Recording

```python
from pysysaudio import SystemAudioRecorder
import time

# Create a recorder instance
recorder = SystemAudioRecorder(sample_rate=48000, channels=2)

# Start recording
recorder.start_recording("output.wav")

# Record for 10 seconds
time.sleep(10)

# Stop and save
output_path = recorder.stop_recording()
print(f"Recording saved to: {output_path}")
```

### Using Context Manager

```python
from pysysaudio import SystemAudioRecorder
import time

with SystemAudioRecorder() as recorder:
    recorder.start_recording("output.wav")
    time.sleep(10)
    # Automatically stops when context exits
```

### Real-time Audio Streaming

Stream audio chunks in real-time for processing, analysis, or streaming to APIs.

Choose your output format:

```python
from pysysaudio import SystemAudioRecorder

# Option 1: bytes format (default, no dependencies)
recorder = SystemAudioRecorder(format="bytes", dtype="int16")
recorder.start_recording()

for audio_data in recorder.stream():
    # audio_data is bytes (PCM int16)
    print(f"Got {len(audio_data)} bytes")
    if some_condition:
        break

recorder.stop_recording()

# Option 2: NumPy arrays (requires numpy)
recorder = SystemAudioRecorder(format="numpy", dtype="float32")
recorder.start_recording("output.wav")  # Save to file AND stream

for audio_chunk in recorder.stream():
    # audio_chunk is a numpy.ndarray with shape (frames, channels)
    print(f"NumPy: shape={audio_chunk.shape}, mean={audio_chunk.mean()}")
    if some_condition:
        break

recorder.stop_recording()

# Option 3: MLX arrays (requires mlx, Apple Silicon optimized)
import mlx.core as mx

recorder = SystemAudioRecorder(format="mlx", dtype="float32")
recorder.start_recording()  # Streaming only (no file)

for audio_chunk in recorder.stream():
    # audio_chunk is an mlx.core.array, runs on GPU/Neural Engine
    mean = mx.mean(audio_chunk)
    print(f"MLX: shape={audio_chunk.shape}, mean={float(mean)}")
    if some_condition:
        break

recorder.stop_recording()
```

### Check Permission Status

```python
from pysysaudio import SystemAudioRecorder

# Check if Screen Recording permission is granted
if SystemAudioRecorder.check_permission():
    print("Permission granted!")
else:
    print("Please grant Screen Recording permission in System Settings")
```

## API Reference

### `SystemAudioRecorder`

Main class for recording system audio.

#### Constructor

```python
SystemAudioRecorder(
    sample_rate: int = 48000, 
    channels: int = 2,
    format: Literal["bytes", "numpy", "mlx"] = "bytes",
    dtype: Literal["float32", "int16", "int32"] = "float32",
    buffer_size: int = 100
)
```

- `sample_rate`: Audio sample rate in Hz (default: 48000)
- `channels`: Number of audio channels - 1 for mono, 2 for stereo (default: 2)
- `format`: Format for audio data yielded by `stream()` (default: "bytes")
  - `"bytes"`: Raw PCM data as bytes (no dependencies)
  - `"numpy"`: NumPy ndarray with shape (frames, channels) (requires `numpy`)
  - `"mlx"`: MLX array with shape (frames, channels) (requires `mlx`, Apple Silicon)
- `dtype`: Sample data type for output audio (default: "float32")
  - `"float32"`: 32-bit float in range [-1.0, 1.0]
  - `"int16"`: 16-bit signed integer (PCM16, range [-32768, 32767])
  - `"int32"`: 32-bit signed integer (PCM32)
- `buffer_size`: Maximum number of audio chunks to buffer (default: 100)

#### Methods

##### `start_recording(output_path: Optional[str] = None)`

Start recording system audio.

- `output_path`: Optional path where the WAV file will be saved. If `None`, audio is only available via `stream()`.

You can save to a file, stream audio via `stream()`, or both simultaneously.

##### `stream(timeout: Optional[float] = 0.1) -> Iterator[AudioData]`

Generator that yields audio chunks as they are recorded.

- `timeout`: Timeout in seconds for waiting for new chunks (default: 0.1). `None` means wait indefinitely.

**Returns**: Iterator yielding audio data in the format specified during initialization (bytes, numpy array, or mlx array).

**Example**:
```python
recorder = SystemAudioRecorder(format="numpy")
recorder.start_recording()
for chunk in recorder.stream():
    # Process audio chunk
    if done:
        break
recorder.stop_recording()
```

##### `stop_recording() -> Optional[str]`

Stop recording and finalize the audio file.

Returns the path to the saved audio file, or empty string if no file was created (streaming-only mode).

##### `is_recording() -> bool`

Check if recording is currently in progress.

##### `check_permission() -> bool` (static)

Check if Screen Recording permission has been granted.

Returns `True` if permission is granted, `False` otherwise.

## Permissions

On first use, macOS will prompt you to grant Screen Recording permission to your application/terminal. This is required for ScreenCaptureKit to capture system audio.

To grant permission:
1. macOS will show a permission dialog on first use
2. Or manually: System Settings → Privacy & Security → Screen Recording
3. Enable permission for your Python interpreter or Terminal

## How It Works

`pysysaudio` uses platform-native APIs for optimal performance:

### macOS
Uses ScreenCaptureKit framework to capture system audio. ScreenCaptureKit is a modern API introduced in macOS 12.3, with audio capture support added in macOS 13.0.

### Windows  
Uses WASAPI (Windows Audio Session API) in loopback mode to capture audio from the default output device.

## Development

### Building

```bash
pip install -e ".[dev]"
```

## License

MIT License - see LICENSE file for details

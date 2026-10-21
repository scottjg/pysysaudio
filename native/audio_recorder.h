#ifndef AUDIO_RECORDER_H
#define AUDIO_RECORDER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pysysaudio {

// Captures system audio into an internal queue and, optionally, a WAV file.
//
// The capture thread never runs Python: it converts the platform's buffers to
// interleaved float32 and pushes them into a lock-free queue. The consumer
// pulls from that queue with read(), which blocks without the GIL and converts
// to the requested rate, channel count and sample type on the reader's thread
// using the platform's native resampler.
class AudioRecorder {
public:
    // sample_rate, channels, dtype: the format read() returns and the WAV file
    // is written in (the file is always 16-bit PCM). dtype is "float32",
    // "int16" or "int32".
    // buffer_seconds: how much captured audio to hold between the capture
    // thread and read() before chunks are dropped.
    AudioRecorder(const std::string& output_path, int sample_rate, int channels,
                  const std::string& dtype, double buffer_seconds);
    ~AudioRecorder();

    AudioRecorder(const AudioRecorder&) = delete;
    AudioRecorder& operator=(const AudioRecorder&) = delete;

    void start();
    std::string stop();
    bool is_recording() const;

    // Format the system delivers. Valid after start().
    double capture_sample_rate() const;
    int capture_channels() const;

    // Replace `out` with converted audio for up to max_frames captured frames.
    // Blocks until audio is available, the recorder is stopped, or
    // timeout_seconds passes (negative waits indefinitely); `out` is empty on
    // timeout. Returns false once the recorder is stopped and everything,
    // including the resampler's tail, has been returned. Single reader only.
    bool read(std::vector<uint8_t>& out, size_t max_frames, double timeout_seconds);

    // Captured frames discarded because read() did not keep up.
    size_t dropped_frames() const;

    static bool check_screen_recording_permission();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace pysysaudio

#endif // AUDIO_RECORDER_H

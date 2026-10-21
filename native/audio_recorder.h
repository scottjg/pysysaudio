#ifndef AUDIO_RECORDER_H
#define AUDIO_RECORDER_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace pysysaudio {

// Captures system audio into an internal queue and, optionally, a WAV file.
//
// The capture thread never runs Python: it converts the platform's buffers to
// interleaved float32 and pushes them into a lock-free queue. The consumer
// pulls from that queue with read(), which blocks without the GIL.
class AudioRecorder {
public:
    // buffer_seconds: how much captured audio to hold between the capture
    // thread and read() before chunks are dropped.
    AudioRecorder(const std::string& output_path, double buffer_seconds);
    ~AudioRecorder();

    AudioRecorder(const AudioRecorder&) = delete;
    AudioRecorder& operator=(const AudioRecorder&) = delete;

    void start();
    std::string stop();
    bool is_recording() const;

    // Capture format. Valid after start().
    double sample_rate() const;
    int channels() const;

    // Fill `out` with up to max_frames * channels() interleaved float32 samples.
    // Blocks until samples are available, the recorder is stopped, or
    // timeout_seconds passes (negative waits indefinitely). After stop(),
    // drains what remains and then returns an empty vector.
    void read(std::vector<float>& out, size_t max_frames, double timeout_seconds);

    // Frames discarded because read() did not keep up with capture.
    size_t dropped_frames() const;

    static bool check_screen_recording_permission();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace pysysaudio

#endif // AUDIO_RECORDER_H

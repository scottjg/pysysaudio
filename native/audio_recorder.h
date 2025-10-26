#ifndef AUDIO_RECORDER_H
#define AUDIO_RECORDER_H

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace pysysaudio {

// Callback function type: receives raw PCM data (float32 samples, normalized -1.0 to 1.0), sample count, channels, and actual sample rate
using AudioCallback = std::function<void(const float* samples, size_t sample_count, int channels, double sample_rate)>;

class AudioRecorder {
public:
    AudioRecorder(const std::string& output_path, int sample_rate, int channels);
    ~AudioRecorder();
    
    // Delete copy constructor and assignment operator
    AudioRecorder(const AudioRecorder&) = delete;
    AudioRecorder& operator=(const AudioRecorder&) = delete;
    
    void start();
    void start_with_callback(AudioCallback callback);
    std::string stop();
    bool is_recording() const;
    
    static bool check_screen_recording_permission();
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace pysysaudio

#endif // AUDIO_RECORDER_H


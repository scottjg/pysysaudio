#ifndef PYSYSAUDIO_FORMAT_CONVERTER_H
#define PYSYSAUDIO_FORMAT_CONVERTER_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pysysaudio {

enum class SampleType { Float32, Int16, Int32 };

inline SampleType parse_sample_type(const std::string& name) {
    if (name == "float32") return SampleType::Float32;
    if (name == "int16") return SampleType::Int16;
    if (name == "int32") return SampleType::Int32;
    throw std::invalid_argument("dtype must be 'float32', 'int16' or 'int32', got '" + name + "'");
}

inline size_t sample_size(SampleType type) {
    return type == SampleType::Int16 ? sizeof(int16_t) : 4;
}

// Map `frames` interleaved frames from in_channels to out_channels.
//   - same count: copy
//   - to mono: average all input channels
//   - from mono: duplicate the channel into every output channel
//   - otherwise: keep the first channels and pad any extra ones with silence
// `out` must hold frames * out_channels floats. Does not allocate, so it is
// safe on a real-time thread.
inline void remix_channels(const float* in, size_t frames, int in_channels,
                           float* out, int out_channels) noexcept {
    if (in_channels == out_channels) {
        std::memcpy(out, in, frames * static_cast<size_t>(in_channels) * sizeof(float));
        return;
    }
    for (size_t f = 0; f < frames; ++f) {
        const float* src = in + f * in_channels;
        float* dst = out + f * out_channels;
        if (out_channels == 1) {
            float sum = 0.0f;
            for (int c = 0; c < in_channels; ++c) sum += src[c];
            dst[0] = sum / static_cast<float>(in_channels);
        } else if (in_channels == 1) {
            for (int c = 0; c < out_channels; ++c) dst[c] = src[0];
        } else {
            const int shared = std::min(in_channels, out_channels);
            for (int c = 0; c < shared; ++c) dst[c] = src[c];
            for (int c = shared; c < out_channels; ++c) dst[c] = 0.0f;
        }
    }
}

// Append float samples to `out` as the given sample type (little-endian on
// every platform we build for). Integers are clipped to [-1, 1], scaled by the
// type's maximum and truncated toward zero.
inline void encode_samples(const float* samples, size_t count, SampleType type,
                           std::vector<uint8_t>& out) {
    const size_t start = out.size();
    out.resize(start + count * sample_size(type));
    uint8_t* dst = out.data() + start;

    switch (type) {
        case SampleType::Float32:
            std::memcpy(dst, samples, count * sizeof(float));
            break;
        case SampleType::Int16:
            for (size_t i = 0; i < count; ++i) {
                const float v = std::max(-1.0f, std::min(1.0f, samples[i]));
                const int16_t s = static_cast<int16_t>(v * 32767.0f);
                std::memcpy(dst + i * sizeof(int16_t), &s, sizeof(int16_t));
            }
            break;
        case SampleType::Int32:
            for (size_t i = 0; i < count; ++i) {
                const double v = std::max(-1.0, std::min(1.0, static_cast<double>(samples[i])));
                const int32_t s = static_cast<int32_t>(v * 2147483647.0);
                std::memcpy(dst + i * sizeof(int32_t), &s, sizeof(int32_t));
            }
            break;
    }
}

// Streaming sample-rate converter for interleaved float32, backed by the
// platform's own resampler: AudioConverter on macOS (resampler_mac.mm) and
// the Media Foundation resampler on Windows (resampler_win.cpp).
//
// State carries across process() calls, so a stream split into arbitrary
// chunks converts the same as if it arrived in one piece. Not thread-safe;
// use from one thread at a time. Not for real-time threads: it allocates.
class Resampler {
public:
    Resampler(double input_rate, double output_rate, int channels);
    ~Resampler();

    Resampler(const Resampler&) = delete;
    Resampler& operator=(const Resampler&) = delete;

    // Convert `frames` input frames and append whatever output is ready.
    void process(const float* input, size_t frames, std::vector<float>& out);

    // Mark end of stream and append the filter's remaining tail. Call once;
    // the resampler cannot take more input afterwards.
    void flush(std::vector<float>& out);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Remix, then resample, then encode: capture-format float32 in, requested
// format out. The resampler is skipped when the rates already match.
class FormatConverter {
public:
    FormatConverter(double input_rate, int input_channels,
                    double output_rate, int output_channels, SampleType type)
        : input_channels_(input_channels),
          output_channels_(output_channels),
          type_(type) {
        if (input_channels < 1 || output_channels < 1) {
            throw std::invalid_argument("channel counts must be at least 1");
        }
        if (input_rate <= 0.0 || output_rate <= 0.0) {
            throw std::invalid_argument("sample rates must be positive");
        }
        if (input_rate != output_rate) {
            resampler_ = std::make_unique<Resampler>(input_rate, output_rate, output_channels);
        }
    }

    int input_channels() const { return input_channels_; }
    int output_channels() const { return output_channels_; }
    SampleType sample_type() const { return type_; }

    // Replace `out` with the converted bytes for `frames` input frames. May be
    // empty while the resampler fills its filter.
    void process(const float* input, size_t frames, std::vector<uint8_t>& out) {
        out.clear();
        if (flushed_) {
            throw std::runtime_error("FormatConverter used after flush()");
        }
        if (frames == 0) {
            return;
        }

        remixed_.resize(frames * static_cast<size_t>(output_channels_));
        remix_channels(input, frames, input_channels_, remixed_.data(), output_channels_);

        if (!resampler_) {
            encode_samples(remixed_.data(), remixed_.size(), type_, out);
            return;
        }
        resampled_.clear();
        resampler_->process(remixed_.data(), frames, resampled_);
        encode_samples(resampled_.data(), resampled_.size(), type_, out);
    }

    // Replace `out` with the resampler's tail. Safe to call more than once;
    // later calls return nothing.
    void flush(std::vector<uint8_t>& out) {
        out.clear();
        if (flushed_) {
            return;
        }
        flushed_ = true;
        if (!resampler_) {
            return;
        }
        resampled_.clear();
        resampler_->flush(resampled_);
        encode_samples(resampled_.data(), resampled_.size(), type_, out);
    }

private:
    int input_channels_;
    int output_channels_;
    SampleType type_;
    bool flushed_ = false;
    std::unique_ptr<Resampler> resampler_;
    std::vector<float> remixed_;
    std::vector<float> resampled_;
};

}  // namespace pysysaudio

#endif  // PYSYSAUDIO_FORMAT_CONVERTER_H

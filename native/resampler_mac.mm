// Resampler backed by CoreAudio's AudioConverter.

#import <AudioToolbox/AudioToolbox.h>

#include "format_converter.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

// Returned from the input callback when the caller's chunk is used up but the
// stream has not ended. AudioConverterFillComplexBuffer stops, passes this
// status back with whatever it produced, and keeps its filter state so the
// next call resumes seamlessly.
constexpr OSStatus kNoMoreInputYet = 'psNI';

AudioStreamBasicDescription float_format(double rate, int channels) {
    AudioStreamBasicDescription f = {};
    f.mSampleRate = rate;
    f.mFormatID = kAudioFormatLinearPCM;
    f.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked;
    f.mBitsPerChannel = 32;
    f.mChannelsPerFrame = static_cast<UInt32>(channels);
    f.mFramesPerPacket = 1;
    f.mBytesPerFrame = f.mChannelsPerFrame * sizeof(float);
    f.mBytesPerPacket = f.mBytesPerFrame;
    return f;
}

[[noreturn]] void throw_osstatus(const char* what, OSStatus status) {
    throw std::runtime_error(std::string(what) + " (OSStatus=" + std::to_string(status) + ")");
}

}  // namespace

namespace pysysaudio {

class Resampler::Impl {
public:
    Impl(double input_rate, double output_rate, int channels)
        : channels_(channels), ratio_(output_rate / input_rate) {
        const AudioStreamBasicDescription in = float_format(input_rate, channels);
        const AudioStreamBasicDescription out = float_format(output_rate, channels);
        OSStatus status = AudioConverterNew(&in, &out, &converter_);
        if (status != noErr) {
            throw_osstatus("Failed to create AudioConverter", status);
        }

        // Best available band-limited filter. This runs on the reader's thread,
        // never the audio thread, so the extra cost is affordable.
        UInt32 quality = kAudioConverterQuality_Max;
        AudioConverterSetProperty(converter_, kAudioConverterSampleRateConverterQuality,
                                  sizeof(quality), &quality);
        UInt32 complexity = kAudioConverterSampleRateConverterComplexity_Mastering;
        AudioConverterSetProperty(converter_, kAudioConverterSampleRateConverterComplexity,
                                  sizeof(complexity), &complexity);
    }

    ~Impl() {
        if (converter_ != nullptr) {
            AudioConverterDispose(converter_);
        }
    }

    void process(const float* input, size_t frames, std::vector<float>& out) {
        pending_ = input;
        pending_frames_ = frames;
        end_of_stream_ = false;
        drain(out, frames);
    }

    void flush(std::vector<float>& out) {
        pending_ = nullptr;
        pending_frames_ = 0;
        end_of_stream_ = true;
        drain(out, 0);
    }

private:
    // Pull output until the converter has consumed all pending input (it asks
    // for more and we answer kNoMoreInputYet) or, at end of stream, until it
    // has nothing left to give.
    void drain(std::vector<float>& out, size_t input_frames) {
        const UInt32 capacity = static_cast<UInt32>(
            std::max<double>(1024.0, input_frames * ratio_ + 256.0));
        scratch_.resize(static_cast<size_t>(capacity) * channels_);

        while (true) {
            UInt32 frames = capacity;
            AudioBufferList list;
            list.mNumberBuffers = 1;
            list.mBuffers[0].mNumberChannels = static_cast<UInt32>(channels_);
            list.mBuffers[0].mDataByteSize =
                static_cast<UInt32>(scratch_.size() * sizeof(float));
            list.mBuffers[0].mData = scratch_.data();

            const OSStatus status =
                AudioConverterFillComplexBuffer(converter_, &Impl::input_proc, this,
                                                &frames, &list, nullptr);
            out.insert(out.end(), scratch_.data(),
                       scratch_.data() + static_cast<size_t>(frames) * channels_);

            if (status == kNoMoreInputYet) {
                // The converter asked for more input after using all of ours, so
                // it no longer refers to the caller's buffer.
                return;
            }
            if (status != noErr) {
                throw_osstatus("AudioConverter failed", status);
            }
            if (frames == 0) {
                // End of stream and the tail is fully drained.
                return;
            }
        }
    }

    static OSStatus input_proc(AudioConverterRef,
                               UInt32* io_packets,
                               AudioBufferList* io_data,
                               AudioStreamPacketDescription**,
                               void* user_data) {
        auto* self = static_cast<Impl*>(user_data);
        AudioBuffer& buffer = io_data->mBuffers[0];

        if (self->pending_frames_ == 0) {
            *io_packets = 0;
            buffer.mData = nullptr;
            buffer.mDataByteSize = 0;
            // noErr with zero packets tells the converter the stream has ended.
            return self->end_of_stream_ ? noErr : kNoMoreInputYet;
        }

        const UInt32 frames =
            static_cast<UInt32>(std::min<size_t>(*io_packets, self->pending_frames_));
        buffer.mNumberChannels = static_cast<UInt32>(self->channels_);
        buffer.mData = const_cast<float*>(self->pending_);
        buffer.mDataByteSize = frames * static_cast<UInt32>(self->channels_) * sizeof(float);
        *io_packets = frames;

        self->pending_ += static_cast<size_t>(frames) * self->channels_;
        self->pending_frames_ -= frames;
        return noErr;
    }

    AudioConverterRef converter_ = nullptr;
    int channels_;
    double ratio_;
    const float* pending_ = nullptr;
    size_t pending_frames_ = 0;
    bool end_of_stream_ = false;
    std::vector<float> scratch_;
};

Resampler::Resampler(double input_rate, double output_rate, int channels)
    : impl_(std::make_unique<Impl>(input_rate, output_rate, channels)) {}

Resampler::~Resampler() = default;

void Resampler::process(const float* input, size_t frames, std::vector<float>& out) {
    impl_->process(input, frames, out);
}

void Resampler::flush(std::vector<float>& out) { impl_->flush(out); }

}  // namespace pysysaudio

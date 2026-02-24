#import <Foundation/Foundation.h>
#import <CoreAudio/CoreAudio.h>
#import <CoreAudio/AudioHardware.h>
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#import <AudioToolbox/AudioToolbox.h>

#include "audio_recorder.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr AudioObjectPropertyAddress PropertyAddress(
    AudioObjectPropertySelector selector,
    AudioObjectPropertyScope scope = kAudioObjectPropertyScopeGlobal,
    AudioObjectPropertyElement element = kAudioObjectPropertyElementMain) noexcept {
    return {selector, scope, element};
}

inline void throw_osstatus(const std::string& prefix, OSStatus status) {
    throw std::runtime_error(prefix + " (OSStatus=" + std::to_string(status) + ")");
}

}  // namespace

namespace pysysaudio {

class AudioRecorder::Impl {
public:
    Impl(const std::string& output_path, int sample_rate, int channels)
        : output_path_(output_path),
          requested_sample_rate_(sample_rate),
          requested_channels_(channels),
          is_recording_(false),
          tap_id_(kAudioObjectUnknown),
          aggregate_device_id_(kAudioObjectUnknown),
          io_proc_id_(nullptr),
          audio_file_(nullptr),
          callback_(nullptr) {
        std::memset(&tap_format_, 0, sizeof(tap_format_));
        // Reserve a large buffer (~256KB) to avoid real-time reallocation.
        // 65536 frames * 1 channel * 4 bytes = 256KB.
        // Typical buffer is < 4096 frames.
        io_buffer_.reserve(65536);
    }

    ~Impl() {
        if (is_recording_.load(std::memory_order_relaxed)) {
            try {
                stop();
            } catch (...) {
                // Avoid throwing from the destructor.
            }
        }
        cleanup_audio_file();
        cleanup_device_and_tap();
    }

    void start() { start_internal(nullptr); }

    void start_with_callback(AudioCallback callback) { start_internal(callback); }

    std::string stop() {
        if (!is_recording_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Not currently recording");
        }

        // AudioDeviceStop is synchronous — IOProc will not fire after it returns.
        if (aggregate_device_id_ != kAudioObjectUnknown && io_proc_id_ != nullptr) {
            AudioDeviceStop(aggregate_device_id_, io_proc_id_);
            AudioDeviceDestroyIOProcID(aggregate_device_id_, io_proc_id_);
            io_proc_id_ = nullptr;
        }

        cleanup_audio_file();
        cleanup_device_and_tap();
        is_recording_.store(false, std::memory_order_relaxed);
        return output_path_;
    }

    bool is_recording() const { return is_recording_.load(std::memory_order_relaxed); }

private:
    static OSStatus io_proc(AudioObjectID,
                            const AudioTimeStamp*,
                            const AudioBufferList* inInputData,
                            const AudioTimeStamp*,
                            AudioBufferList*,
                            const AudioTimeStamp*,
                            void* inClientData) noexcept {
        auto* impl = static_cast<Impl*>(inClientData);
        return impl ? impl->handle_io(inInputData) : kAudioHardwareNoError;
    }

    void start_internal(AudioCallback callback) {
        if (is_recording_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Already recording");
        }
        if (@available(macOS 14.2, *)) {
            // Supported.
        } else {
            throw std::runtime_error("macOS 14.2 or later is required for CoreAudio taps");
        }

        callback_ = callback;

        @autoreleasepool {
            create_tap_and_aggregate_device();
            configure_output_file_if_needed();
        }

        OSStatus status = AudioDeviceCreateIOProcID(
            aggregate_device_id_, &Impl::io_proc, this, &io_proc_id_);
        if (status != noErr) {
            cleanup_audio_file();
            cleanup_device_and_tap();
            throw_osstatus("Failed to create IOProc", status);
        }

        // Set before AudioDeviceStart so the IOProc sees it immediately.
        is_recording_.store(true, std::memory_order_release);

        status = AudioDeviceStart(aggregate_device_id_, io_proc_id_);
        if (status != noErr) {
            is_recording_.store(false, std::memory_order_relaxed);
            AudioDeviceDestroyIOProcID(aggregate_device_id_, io_proc_id_);
            io_proc_id_ = nullptr;
            cleanup_audio_file();
            cleanup_device_and_tap();
            throw_osstatus("Failed to start CoreAudio tap device", status);
        }
    }

    void create_tap_and_aggregate_device() {
        NSArray<NSNumber*>* processes = @[];
        CATapDescription* tap_description =
            [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:processes];
        tap_description.name = @"pysysaudio tap";
        tap_description.privateTap = YES;
        tap_description.exclusive = YES;
        tap_description.muteBehavior = CATapUnmuted;

        OSStatus status = AudioHardwareCreateProcessTap(tap_description, &tap_id_);
        if (status != noErr || tap_id_ == kAudioObjectUnknown) {
            throw_osstatus("Failed to create process tap", status);
        }

        read_tap_format();

        NSString* tap_uid = tap_description.UUID.UUIDString;
        if (tap_uid == nil || tap_uid.length == 0) {
            cleanup_device_and_tap();
            throw std::runtime_error("Failed to get process tap UID");
        }

        NSDictionary* subtap = @{
            @kAudioSubTapUIDKey : tap_uid,
            @kAudioSubTapDriftCompensationKey : @YES,
        };

        NSString* aggregate_uid =
            [NSString stringWithFormat:@"com.pysysaudio.aggregate.%@", NSUUID.UUID.UUIDString];
        NSDictionary* aggregate_description = @{
            @kAudioAggregateDeviceNameKey : @"pysysaudio aggregate capture",
            @kAudioAggregateDeviceUIDKey : aggregate_uid,
            @kAudioAggregateDeviceTapListKey : @[ subtap ],
            @kAudioAggregateDeviceTapAutoStartKey : @NO,
            @kAudioAggregateDeviceIsPrivateKey : @YES,
        };

        status = AudioHardwareCreateAggregateDevice(
            (__bridge CFDictionaryRef)aggregate_description, &aggregate_device_id_);
        if (status != noErr || aggregate_device_id_ == kAudioObjectUnknown) {
            cleanup_device_and_tap();
            throw_osstatus("Failed to create aggregate device", status);
        }
    }

    void read_tap_format() {
        auto address = PropertyAddress(kAudioTapPropertyFormat);
        UInt32 size = sizeof(tap_format_);
        std::memset(&tap_format_, 0, sizeof(tap_format_));

        OSStatus status =
            AudioObjectGetPropertyData(tap_id_, &address, 0, nullptr, &size, &tap_format_);
        if (status != noErr || tap_format_.mChannelsPerFrame == 0 || tap_format_.mSampleRate <= 0.0) {
            tap_format_.mFormatID = kAudioFormatLinearPCM;
            tap_format_.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian |
                                       kAudioFormatFlagIsPacked;
            tap_format_.mSampleRate = requested_sample_rate_ > 0 ? requested_sample_rate_ : 48000;
            tap_format_.mChannelsPerFrame =
                requested_channels_ > 0 ? static_cast<UInt32>(requested_channels_) : 2;
            tap_format_.mBitsPerChannel = 32;
            tap_format_.mFramesPerPacket = 1;
            tap_format_.mBytesPerFrame = tap_format_.mChannelsPerFrame * sizeof(float);
            tap_format_.mBytesPerPacket = tap_format_.mBytesPerFrame;
        }
    }

    void configure_output_file_if_needed() {
        if (output_path_.empty()) {
            return;
        }

        NSString* output_path = [NSString stringWithUTF8String:output_path_.c_str()];
        NSURL* output_url = [NSURL fileURLWithPath:output_path];
        [[NSFileManager defaultManager] removeItemAtURL:output_url error:nil];

        AudioStreamBasicDescription file_format = {};
        file_format.mSampleRate = tap_format_.mSampleRate;
        file_format.mFormatID = kAudioFormatLinearPCM;
        file_format.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
        file_format.mBitsPerChannel = 16;
        file_format.mChannelsPerFrame = tap_format_.mChannelsPerFrame;
        file_format.mFramesPerPacket = 1;
        file_format.mBytesPerFrame = file_format.mChannelsPerFrame * sizeof(int16_t);
        file_format.mBytesPerPacket = file_format.mBytesPerFrame;

        OSStatus status = ExtAudioFileCreateWithURL((__bridge CFURLRef)output_url,
                                                    kAudioFileWAVEType,
                                                    &file_format,
                                                    nullptr,
                                                    kAudioFileFlags_EraseFile,
                                                    &audio_file_);
        if (status != noErr) {
            throw_osstatus("Failed to create output WAV file", status);
        }

        AudioStreamBasicDescription client_format = {};
        client_format.mSampleRate = tap_format_.mSampleRate;
        client_format.mFormatID = kAudioFormatLinearPCM;
        client_format.mFormatFlags =
            kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked;
        client_format.mBitsPerChannel = 32;
        client_format.mChannelsPerFrame = tap_format_.mChannelsPerFrame;
        client_format.mFramesPerPacket = 1;
        client_format.mBytesPerFrame = client_format.mChannelsPerFrame * sizeof(float);
        client_format.mBytesPerPacket = client_format.mBytesPerFrame;

        status = ExtAudioFileSetProperty(audio_file_,
                                         kExtAudioFileProperty_ClientDataFormat,
                                         sizeof(client_format),
                                         &client_format);
        if (status != noErr) {
            cleanup_audio_file();
            throw_osstatus("Failed to configure output WAV client format", status);
        }
    }

    OSStatus handle_io(const AudioBufferList* input_data) noexcept {
        if (!is_recording_.load(std::memory_order_acquire) ||
            input_data == nullptr || input_data->mNumberBuffers == 0) {
            return noErr;
        }

        UInt32 channels = 0;
        UInt32 frames = 0;
        if (!convert_input_to_interleaved_float(input_data, io_buffer_, channels, frames)) {
            return noErr;
        }

        if (callback_) {
            callback_(
                io_buffer_.data(), io_buffer_.size(), static_cast<int>(channels), tap_format_.mSampleRate);
        }

        if (audio_file_ != nullptr && frames > 0) {
            AudioBufferList list;
            list.mNumberBuffers = 1;
            list.mBuffers[0].mNumberChannels = channels;
            list.mBuffers[0].mDataByteSize =
                static_cast<UInt32>(io_buffer_.size() * sizeof(float));
            list.mBuffers[0].mData = io_buffer_.data();
            ExtAudioFileWriteAsync(audio_file_, frames, &list);
        }

        return noErr;
    }

    bool convert_input_to_interleaved_float(const AudioBufferList* input_data,
                                            std::vector<float>& out_samples,
                                            UInt32& out_channels,
                                            UInt32& out_frames) const noexcept {
        if (tap_format_.mFormatID != kAudioFormatLinearPCM) {
            return false;
        }

        const bool is_float = (tap_format_.mFormatFlags & kAudioFormatFlagIsFloat) != 0;
        const bool is_signed_int = (tap_format_.mFormatFlags & kAudioFormatFlagIsSignedInteger) != 0;
        const bool is_non_interleaved =
            (tap_format_.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
        const UInt32 bits = tap_format_.mBitsPerChannel;
        const UInt32 bytes_per_sample = bits / 8;

        if (bytes_per_sample == 0) {
            return false;
        }

        out_channels = tap_format_.mChannelsPerFrame;
        if (out_channels == 0) {
            return false;
        }

        if (!is_non_interleaved) {
            const AudioBuffer& buffer = input_data->mBuffers[0];
            UInt32 channels_in_buffer = buffer.mNumberChannels > 0 ? buffer.mNumberChannels : out_channels;
            if (channels_in_buffer == 0) {
                return false;
            }

            out_channels = channels_in_buffer;
            out_frames = buffer.mDataByteSize / (bytes_per_sample * out_channels);
            out_samples.resize(static_cast<size_t>(out_frames) * out_channels);

            const uint8_t* data = static_cast<const uint8_t*>(buffer.mData);
            for (UInt32 frame = 0; frame < out_frames; ++frame) {
                for (UInt32 ch = 0; ch < out_channels; ++ch) {
                    const size_t idx = static_cast<size_t>(frame) * out_channels + ch;
                    const uint8_t* sample_ptr = data + idx * bytes_per_sample;
                    out_samples[idx] = sample_to_float(sample_ptr, is_float, is_signed_int, bits);
                }
            }
            return true;
        }

        if (input_data->mNumberBuffers == 0) {
            return false;
        }

        out_channels = std::min(out_channels, input_data->mNumberBuffers);
        if (out_channels == 0) {
            return false;
        }

        out_frames = input_data->mBuffers[0].mDataByteSize / bytes_per_sample;
        out_samples.resize(static_cast<size_t>(out_frames) * out_channels);

        for (UInt32 ch = 0; ch < out_channels; ++ch) {
            const AudioBuffer& buffer = input_data->mBuffers[ch];
            const uint8_t* data = static_cast<const uint8_t*>(buffer.mData);
            UInt32 channel_frames = buffer.mDataByteSize / bytes_per_sample;
            UInt32 usable_frames = std::min(out_frames, channel_frames);

            for (UInt32 frame = 0; frame < usable_frames; ++frame) {
                const size_t idx = static_cast<size_t>(frame) * out_channels + ch;
                out_samples[idx] =
                    sample_to_float(data + (frame * bytes_per_sample), is_float, is_signed_int, bits);
            }
        }

        return true;
    }

    static float sample_to_float(const uint8_t* sample_ptr,
                                 bool is_float,
                                 bool is_signed_int,
                                 UInt32 bits) noexcept {
        if (is_float && bits == 32) {
            float value = 0.0f;
            std::memcpy(&value, sample_ptr, sizeof(float));
            return value;
        }
        if (is_float && bits == 64) {
            double value = 0.0;
            std::memcpy(&value, sample_ptr, sizeof(double));
            return static_cast<float>(value);
        }
        if (is_signed_int && bits == 16) {
            int16_t value = 0;
            std::memcpy(&value, sample_ptr, sizeof(int16_t));
            return static_cast<float>(value) / 32768.0f;
        }
        if (is_signed_int && bits == 32) {
            int32_t value = 0;
            std::memcpy(&value, sample_ptr, sizeof(int32_t));
            return static_cast<float>(static_cast<double>(value) / 2147483648.0);
        }
        return 0.0f;
    }

    void cleanup_audio_file() noexcept {
        if (audio_file_ != nullptr) {
            ExtAudioFileDispose(audio_file_);
            audio_file_ = nullptr;
        }
    }

    void cleanup_device_and_tap() noexcept {
        if (aggregate_device_id_ != kAudioObjectUnknown) {
            AudioHardwareDestroyAggregateDevice(aggregate_device_id_);
            aggregate_device_id_ = kAudioObjectUnknown;
        }
        if (tap_id_ != kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tap_id_);
            tap_id_ = kAudioObjectUnknown;
        }
    }

    std::string output_path_;
    int requested_sample_rate_;
    int requested_channels_;
    std::atomic<bool> is_recording_;
    AudioObjectID tap_id_;
    AudioObjectID aggregate_device_id_;
    AudioDeviceIOProcID io_proc_id_;
    ExtAudioFileRef audio_file_;
    AudioStreamBasicDescription tap_format_;
    AudioCallback callback_;
    std::vector<float> io_buffer_;  // Pre-allocated; only touched from IOProc.
};

AudioRecorder::AudioRecorder(const std::string& output_path, int sample_rate, int channels)
    : pImpl(std::make_unique<Impl>(output_path, sample_rate, channels)) {}

AudioRecorder::~AudioRecorder() = default;

void AudioRecorder::start() { pImpl->start(); }

void AudioRecorder::start_with_callback(AudioCallback callback) {
    pImpl->start_with_callback(callback);
}

std::string AudioRecorder::stop() { return pImpl->stop(); }

bool AudioRecorder::is_recording() const { return pImpl->is_recording(); }

bool AudioRecorder::check_screen_recording_permission() {
    // Kept for API compatibility with existing Python wrappers.
    // For the CoreAudio tap backend, probe permission by creating a short-lived tap.
    if (@available(macOS 14.2, *)) {
        // Supported.
    } else {
        return false;
    }

    CATapDescription* tap_description =
        [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:@[]];
    tap_description.privateTap = YES;
    tap_description.exclusive = NO; // Don't interrupt other taps for a probe.
    tap_description.muteBehavior = CATapUnmuted;
    tap_description.name = @"pysysaudio permission probe";

    AudioObjectID tap_id = kAudioObjectUnknown;
    OSStatus status = AudioHardwareCreateProcessTap(tap_description, &tap_id);
    if (status == noErr) {
        if (tap_id != kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tap_id);
        }
        return true;
    }

    if (status == kAudioDevicePermissionsError) {
        return false;
    }

    // Treat unexpected failures as unavailable permission/capture support.
    return false;
}

}  // namespace pysysaudio

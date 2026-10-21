// Resampler backed by the Media Foundation audio resampler DSP
// (CLSID_CResamplerMediaObject), used through its IMFTransform interface.

#ifdef _WIN32

#include "format_converter.h"

#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>
#include <mmreg.h>
#include <wmcodecdsp.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void throw_hresult(const char* what, HRESULT hr) {
    char code[16];
    std::snprintf(code, sizeof(code), "0x%08lX", static_cast<unsigned long>(hr));
    throw std::runtime_error(std::string(what) + " (HRESULT=" + code + ")");
}

void check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        throw_hresult(what, hr);
    }
}

template <typename T>
void release(T*& p) {
    if (p != nullptr) {
        p->Release();
        p = nullptr;
    }
}

// IWMResamplerProps, declared here because MinGW's wmcodecdsp.h omits it. Same
// IID and vtable as the Windows SDK declaration.
struct ResamplerProps : public IUnknown {
    virtual HRESULT STDMETHODCALLTYPE SetHalfFilterLength(LONG half_filter_length) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetUserChannelMtx(float* channel_matrix) = 0;
};
const IID kIidResamplerProps = {
    0xE7E9984F, 0xF09F, 0x4DA4, {0x90, 0x3F, 0x6E, 0x2E, 0x0E, 0xFE, 0x56, 0xB5}};

// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, spelled out so no extra import library is needed.
const GUID kSubtypeIeeeFloat = {
    0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};

}  // namespace

namespace pysysaudio {

class Resampler::Impl {
public:
    Impl(double input_rate, double output_rate, int channels)
        : channels_(channels), input_rate_(input_rate), output_rate_(output_rate) {
        // Media Foundation needs COM on this thread. RPC_E_CHANGED_MODE means the
        // thread already has an apartment, which works just as well.
        const HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (co == S_OK || co == S_FALSE) {
            com_initialized_ = true;
            com_thread_ = GetCurrentThreadId();
        } else if (co != RPC_E_CHANGED_MODE) {
            throw_hresult("Failed to initialize COM", co);
        }

        try {
            check(MFStartup(MF_VERSION, MFSTARTUP_LITE), "Failed to start Media Foundation");
            mf_started_ = true;

            IUnknown* unknown = nullptr;
            check(CoCreateInstance(CLSID_CResamplerMediaObject, nullptr, CLSCTX_INPROC_SERVER,
                                   IID_IUnknown, reinterpret_cast<void**>(&unknown)),
                  "Failed to create the Media Foundation resampler");
            HRESULT hr = unknown->QueryInterface(IID_PPV_ARGS(&transform_));

            // Longest filter the resampler offers (range 1..60), for the best
            // anti-aliasing. This runs on the reader's thread, not the audio thread.
            ResamplerProps* props = nullptr;
            if (SUCCEEDED(unknown->QueryInterface(kIidResamplerProps,
                                                  reinterpret_cast<void**>(&props)))) {
                props->SetHalfFilterLength(60);
                props->Release();
            }
            unknown->Release();
            check(hr, "Resampler does not expose IMFTransform");

            IMFMediaType* type = make_media_type(input_rate_);
            hr = transform_->SetInputType(0, type, 0);
            type->Release();
            check(hr, "Resampler rejected the input format");

            type = make_media_type(output_rate_);
            hr = transform_->SetOutputType(0, type, 0);
            type->Release();
            check(hr, "Resampler rejected the output format");

            check(transform_->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0),
                  "Resampler failed to begin streaming");
            check(transform_->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0),
                  "Resampler failed to start the stream");
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~Impl() { shutdown(); }

    void process(const float* input, size_t frames, std::vector<float>& out) {
        if (frames == 0) {
            return;
        }

        IMFSample* sample = make_input_sample(input, frames);
        HRESULT hr = transform_->ProcessInput(0, sample, 0);
        if (hr == MF_E_NOTACCEPTING) {
            // Output from an earlier chunk is still waiting; collect it, then retry.
            pull_output(out);
            hr = transform_->ProcessInput(0, sample, 0);
        }
        sample->Release();
        check(hr, "Resampler rejected input");

        pull_output(out);
    }

    void flush(std::vector<float>& out) {
        check(transform_->ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0),
              "Resampler failed to end the stream");
        check(transform_->ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0),
              "Resampler failed to drain");
        pull_output(out);
    }

private:
    IMFMediaType* make_media_type(double rate) const {
        WAVEFORMATEXTENSIBLE wfx = {};
        WAVEFORMATEX& base = wfx.Format;
        base.nChannels = static_cast<WORD>(channels_);
        base.nSamplesPerSec = static_cast<DWORD>(rate);
        base.wBitsPerSample = 32;
        base.nBlockAlign = static_cast<WORD>(channels_ * sizeof(float));
        base.nAvgBytesPerSec = base.nSamplesPerSec * base.nBlockAlign;

        UINT32 size = sizeof(WAVEFORMATEX);
        if (channels_ <= 2) {
            base.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
            base.cbSize = 0;
        } else {
            // More than two channels needs the extensible form with a channel mask.
            base.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
            base.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
            wfx.Samples.wValidBitsPerSample = 32;
            wfx.dwChannelMask = channels_ >= 32 ? 0xFFFFFFFFu : ((1u << channels_) - 1u);
            wfx.SubFormat = kSubtypeIeeeFloat;
            size = sizeof(WAVEFORMATEXTENSIBLE);
        }

        IMFMediaType* type = nullptr;
        check(MFCreateMediaType(&type), "Failed to create a media type");
        const HRESULT hr = MFInitMediaTypeFromWaveFormatEx(type, &base, size);
        if (FAILED(hr)) {
            type->Release();
            throw_hresult("Failed to describe the audio format", hr);
        }
        return type;
    }

    IMFSample* make_input_sample(const float* input, size_t frames) {
        const DWORD bytes = static_cast<DWORD>(frames * channels_ * sizeof(float));
        IMFSample* sample = nullptr;
        IMFMediaBuffer* buffer = nullptr;
        try {
            check(MFCreateSample(&sample), "Failed to create an input sample");
            check(MFCreateMemoryBuffer(bytes, &buffer), "Failed to allocate an input buffer");

            BYTE* data = nullptr;
            check(buffer->Lock(&data, nullptr, nullptr), "Failed to lock the input buffer");
            std::memcpy(data, input, bytes);
            buffer->Unlock();
            check(buffer->SetCurrentLength(bytes), "Failed to size the input buffer");
            check(sample->AddBuffer(buffer), "Failed to attach the input buffer");
            release(buffer);

            // Timestamps in 100 ns units, derived from the running frame count so
            // they never accumulate rounding error.
            const LONGLONG start = frames_to_hns(input_frames_);
            input_frames_ += frames;
            sample->SetSampleTime(start);
            sample->SetSampleDuration(frames_to_hns(input_frames_) - start);
            return sample;
        } catch (...) {
            release(buffer);
            release(sample);
            throw;
        }
    }

    LONGLONG frames_to_hns(unsigned long long frames) const {
        return static_cast<LONGLONG>(static_cast<double>(frames) * 10000000.0 / input_rate_);
    }

    // Collect output until the transform needs more input.
    void pull_output(std::vector<float>& out) {
        MFT_OUTPUT_STREAM_INFO info = {};
        check(transform_->GetOutputStreamInfo(0, &info), "Failed to query resampler output");
        const DWORD capacity = std::max<DWORD>(info.cbSize, static_cast<DWORD>(
                                                                8192 * channels_ * sizeof(float)));
        ensure_output_buffer(capacity);

        while (true) {
            check(output_buffer_->SetCurrentLength(0), "Failed to reset the output buffer");

            MFT_OUTPUT_DATA_BUFFER data = {};
            data.dwStreamID = 0;
            data.pSample = output_sample_;
            DWORD status = 0;
            const HRESULT hr = transform_->ProcessOutput(0, 1, &data, &status);
            if (data.pEvents != nullptr) {
                data.pEvents->Release();
            }
            if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) {
                return;
            }
            check(hr, "Resampler failed to produce output");

            BYTE* bytes = nullptr;
            DWORD length = 0;
            check(output_buffer_->Lock(&bytes, nullptr, &length), "Failed to lock the output buffer");
            const float* samples = reinterpret_cast<const float*>(bytes);
            out.insert(out.end(), samples, samples + length / sizeof(float));
            output_buffer_->Unlock();
        }
    }

    void ensure_output_buffer(DWORD capacity) {
        if (output_buffer_ != nullptr && output_capacity_ >= capacity) {
            return;
        }
        release(output_buffer_);
        release(output_sample_);
        check(MFCreateSample(&output_sample_), "Failed to create an output sample");
        check(MFCreateMemoryBuffer(capacity, &output_buffer_), "Failed to allocate an output buffer");
        check(output_sample_->AddBuffer(output_buffer_), "Failed to attach the output buffer");
        output_capacity_ = capacity;
    }

    void shutdown() {
        release(output_buffer_);
        release(output_sample_);
        release(transform_);
        if (mf_started_) {
            MFShutdown();
            mf_started_ = false;
        }
        // Only balance CoInitializeEx on the thread that made the call.
        if (com_initialized_ && GetCurrentThreadId() == com_thread_) {
            CoUninitialize();
        }
        com_initialized_ = false;
    }

    int channels_;
    double input_rate_;
    double output_rate_;
    bool com_initialized_ = false;
    DWORD com_thread_ = 0;
    bool mf_started_ = false;
    IMFTransform* transform_ = nullptr;
    IMFSample* output_sample_ = nullptr;
    IMFMediaBuffer* output_buffer_ = nullptr;
    DWORD output_capacity_ = 0;
    unsigned long long input_frames_ = 0;
};

Resampler::Resampler(double input_rate, double output_rate, int channels)
    : impl_(std::make_unique<Impl>(input_rate, output_rate, channels)) {}

Resampler::~Resampler() = default;

void Resampler::process(const float* input, size_t frames, std::vector<float>& out) {
    impl_->process(input, frames, out);
}

void Resampler::flush(std::vector<float>& out) { impl_->flush(out); }

}  // namespace pysysaudio

#endif  // _WIN32

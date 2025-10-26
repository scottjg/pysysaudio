#ifdef _WIN32

#include "audio_recorder.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <cstring>
#include <chrono>

// Windows Audio API includes
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <mmreg.h>  // For WAVE_FORMAT_* constants

// For WAV file writing
#pragma pack(push, 1)
struct WAVHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size = 0;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 2;
    uint32_t sample_rate = 48000;
    uint32_t byte_rate = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size = 0;
};
#pragma pack(pop)

namespace pysysaudio {

class AudioRecorder::Impl {
public:
    Impl(const std::string& output_path, int sample_rate, int channels)
        : output_path_(output_path)
        , sample_rate_(sample_rate)
        , channels_(channels)
        , is_recording_(false)
        , capture_thread_(nullptr)
        , callback_(nullptr)
        , pEnumerator_(nullptr)
        , pDevice_(nullptr)
        , pAudioClient_(nullptr)
        , pCaptureClient_(nullptr)
        , pRenderClient_(nullptr)
        , pRenderService_(nullptr)
        , bytes_recorded_(0) {
        
        // Initialize COM
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
            throw std::runtime_error("Failed to initialize COM");
        }
        com_initialized_ = (hr == S_OK || hr == S_FALSE);
    }
    
    ~Impl() {
        if (is_recording_) {
            try {
                stop();
            } catch (...) {
                // Ignore exceptions in destructor
            }
        }
        
        cleanup_resources();
        
        if (com_initialized_) {
            CoUninitialize();
        }
    }
    
    void start() {
        start_internal(nullptr);
    }
    
    void start_with_callback(AudioCallback callback) {
        start_internal(callback);
    }
    
    void start_internal(AudioCallback callback) {
        if (is_recording_) {
            throw std::runtime_error("Already recording");
        }
        
        callback_ = callback;
        
        HRESULT hr;
        
        // Create device enumerator
        hr = CoCreateInstance(
            __uuidof(MMDeviceEnumerator),
            nullptr,
            CLSCTX_ALL,
            __uuidof(IMMDeviceEnumerator),
            (void**)&pEnumerator_
        );
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create device enumerator");
        }
        
        // Get default audio render device (speakers/output)
        hr = pEnumerator_->GetDefaultAudioEndpoint(
            eRender,  // Render (output) device for loopback
            eConsole,
            &pDevice_
        );
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to get default audio device");
        }
        
        // Activate audio client
        hr = pDevice_->Activate(
            __uuidof(IAudioClient),
            CLSCTX_ALL,
            nullptr,
            (void**)&pAudioClient_
        );
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to activate audio client");
        }
        
        // Get the mix format
        WAVEFORMATEX* pwfx = nullptr;
        hr = pAudioClient_->GetMixFormat(&pwfx);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to get mix format");
        }
        
        // Store the actual format we're capturing
        actual_sample_rate_ = pwfx->nSamplesPerSec;
        actual_channels_ = pwfx->nChannels;
        
        // Optional: Uncomment for debugging audio format issues
        // std::cout << "WASAPI: " << actual_sample_rate_ << " Hz, " 
        //           << actual_channels_ << " ch, " << pwfx->wBitsPerSample << " bit" << std::endl;
        
        // Initialize audio client in loopback mode
        hr = pAudioClient_->Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            10000000,  // 1 second buffer
            0,
            pwfx,
            nullptr
        );
        
        if (FAILED(hr)) {
            CoTaskMemFree(pwfx);
            throw std::runtime_error("Failed to initialize audio client (loopback mode)");
        }
        
        // Get buffer size
        UINT32 bufferFrameCount;
        hr = pAudioClient_->GetBufferSize(&bufferFrameCount);
        if (FAILED(hr)) {
            CoTaskMemFree(pwfx);
            throw std::runtime_error("Failed to get buffer size");
        }
        
        // Get capture client
        hr = pAudioClient_->GetService(
            __uuidof(IAudioCaptureClient),
            (void**)&pCaptureClient_
        );
        if (FAILED(hr)) {
            CoTaskMemFree(pwfx);
            throw std::runtime_error("Failed to get capture client");
        }
        
        // Prepare WAV file if output path is provided
        if (!output_path_.empty()) {
            wav_file_.open(output_path_, std::ios::binary);
            if (!wav_file_.is_open()) {
                CoTaskMemFree(pwfx);
                throw std::runtime_error("Failed to open output file: " + output_path_);
            }
            
            // Write WAV header (will update at the end)
            WAVHeader header;
            header.num_channels = actual_channels_;
            header.sample_rate = actual_sample_rate_;
            header.bits_per_sample = 16;
            header.block_align = header.num_channels * header.bits_per_sample / 8;
            header.byte_rate = header.sample_rate * header.block_align;
            wav_file_.write(reinterpret_cast<char*>(&header), sizeof(header));
        }
        
        // Start audio client
        hr = pAudioClient_->Start();
        if (FAILED(hr)) {
            CoTaskMemFree(pwfx);
            throw std::runtime_error("Failed to start audio client");
        }
        
        is_recording_ = true;
        
        // Start capture thread BEFORE starting silent playback
        capture_thread_ = new std::thread(&Impl::capture_thread_func, this, *pwfx);
        
        // Give the capture thread time to initialize
        Sleep(100);
        
        // CRITICAL: Start a silent render stream to prime the loopback capture
        // WASAPI loopback only captures when audio is actively playing
        // Starting a silent stream ensures we always capture data
        start_silent_playback(pwfx);
        
        CoTaskMemFree(pwfx);
    }
    
    std::string stop() {
        if (!is_recording_) {
            throw std::runtime_error("Not currently recording");
        }
        
        // Signal the thread to stop
        is_recording_ = false;
        
        // Give thread a moment to exit gracefully
        Sleep(50);
        
        // Wait for capture thread to finish
        if (capture_thread_) {
            if (capture_thread_->joinable()) {
                // Give it up to 2 seconds to exit
                auto start = std::chrono::steady_clock::now();
                bool thread_exited = false;
                
                // Poll for thread completion with timeout
                while (true) {
                    Sleep(10);
                    
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
                    
                    if (elapsed > 2000) {
                        // Thread didn't exit in time - this shouldn't happen in normal operation
                        break;
                    }
                    
                    // Try to join - this will block until thread exits
                    capture_thread_->join();
                    thread_exited = true;
                    break;
                }
            }
            delete capture_thread_;
            capture_thread_ = nullptr;
        }
        
        // Stop audio clients
        if (pAudioClient_) {
            pAudioClient_->Stop();
        }
        if (pRenderClient_) {
            pRenderClient_->Stop();
        }
        
        // Flush and finalize WAV file
        if (wav_file_.is_open()) {
            // Flush any pending writes
            wav_file_.flush();
            
            // Update WAV header with final sizes
            wav_file_.seekp(0, std::ios::beg);
            WAVHeader header;
            header.num_channels = actual_channels_;
            header.sample_rate = actual_sample_rate_;
            header.bits_per_sample = 16;
            header.block_align = header.num_channels * header.bits_per_sample / 8;
            header.byte_rate = header.sample_rate * header.block_align;
            header.data_size = bytes_recorded_;
            header.file_size = bytes_recorded_ + sizeof(WAVHeader) - 8;
            wav_file_.write(reinterpret_cast<char*>(&header), sizeof(header));
            wav_file_.flush();
            wav_file_.close();
        }
        
        cleanup_resources();
        
        return output_path_;
    }
    
    bool is_recording() const {
        return is_recording_;
    }
    
private:
    void start_silent_playback(WAVEFORMATEX* pwfx) {
        // Start a silent render stream to ensure loopback captures data
        // This is a known WASAPI requirement for loopback capture
        
        HRESULT hr = pDevice_->Activate(
            __uuidof(IAudioClient),
            CLSCTX_ALL,
            nullptr,
            (void**)&pRenderClient_
        );
        
        if (FAILED(hr)) {
            // Silent playback failed - non-fatal, loopback may still work if audio is playing
            return;
        }
        
        // Initialize render client with same format
        hr = pRenderClient_->Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            0,  // No special flags
            10000000,  // 1 second buffer
            0,
            pwfx,
            nullptr
        );
        
        if (FAILED(hr)) {
            pRenderClient_->Release();
            pRenderClient_ = nullptr;
            return;
        }
        
        // Get the render service
        hr = pRenderClient_->GetService(
            __uuidof(IAudioRenderClient),
            (void**)&pRenderService_
        );
        
        if (FAILED(hr)) {
            pRenderClient_->Release();
            pRenderClient_ = nullptr;
            return;
        }
        
        // Get buffer and fill with silence
        UINT32 bufferFrameCount;
        pRenderClient_->GetBufferSize(&bufferFrameCount);
        
        BYTE* pData;
        hr = pRenderService_->GetBuffer(bufferFrameCount, &pData);
        if (SUCCEEDED(hr)) {
            // Fill with silence (zeros)
            memset(pData, 0, bufferFrameCount * pwfx->nBlockAlign);
            pRenderService_->ReleaseBuffer(bufferFrameCount, 0);
        }
        
        // Start the silent playback
        pRenderClient_->Start();
    }
    
    void cleanup_resources() {
        // Release render client (already stopped in stop())
        if (pRenderService_) {
            pRenderService_->Release();
            pRenderService_ = nullptr;
        }
        if (pRenderClient_) {
            pRenderClient_->Release();
            pRenderClient_ = nullptr;
        }
        
        if (pCaptureClient_) {
            pCaptureClient_->Release();
            pCaptureClient_ = nullptr;
        }
        if (pAudioClient_) {
            pAudioClient_->Release();
            pAudioClient_ = nullptr;
        }
        if (pDevice_) {
            pDevice_->Release();
            pDevice_ = nullptr;
        }
        if (pEnumerator_) {
            pEnumerator_->Release();
            pEnumerator_ = nullptr;
        }
    }
    
    void capture_thread_func(WAVEFORMATEX format) {
        // Set thread priority for real-time audio
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        
        // Determine if the format is float
        // WASAPI loopback typically delivers float32
        bool is_float = (format.wFormatTag == WAVE_FORMAT_IEEE_FLOAT);
        
        // For WAVE_FORMAT_EXTENSIBLE, we need to check the SubFormat GUID
        // But since format is passed by value, we can't safely cast to WAVEFORMATEXTENSIBLE
        // Solution: Assume 32-bit is float (which is standard for WASAPI loopback)
        if (format.wFormatTag == WAVE_FORMAT_EXTENSIBLE || 
            (format.wBitsPerSample == 32 && format.wFormatTag != WAVE_FORMAT_PCM)) {
            is_float = true;  // WASAPI loopback is almost always float32
        }
        
        int empty_iterations = 0;
        int total_chunks = 0;
        
        while (is_recording_.load()) {
            Sleep(5);  // Sleep for 5ms between checks (more responsive)
            
            UINT32 packetLength = 0;
            HRESULT hr = pCaptureClient_->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) {
                std::cerr << "Failed to get packet size" << std::endl;
                break;
            }
            
            if (packetLength == 0) {
                empty_iterations++;
            } else {
                empty_iterations = 0;
            }
            
            while (packetLength != 0) {
                BYTE* pData;
                UINT32 numFramesAvailable;
                DWORD flags;
                
                hr = pCaptureClient_->GetBuffer(
                    &pData,
                    &numFramesAvailable,
                    &flags,
                    nullptr,
                    nullptr
                );
                
                if (FAILED(hr)) {
                    std::cerr << "Failed to get buffer" << std::endl;
                    break;
                }
                
                if (numFramesAvailable > 0 && is_recording_.load()) {
                    size_t dataSize = numFramesAvailable * format.nBlockAlign;
                    
                    // Handle silence flag
                    if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                        pData = nullptr;  // Will be handled as silence
                    }
                    
                    if (pData || (flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
                        // Process the audio data (this calls the Python callback)
                        process_audio_data(pData, numFramesAvailable, format, is_float, 
                                         flags & AUDCLNT_BUFFERFLAGS_SILENT);
                        total_chunks++;
                    }
                }
                
                pCaptureClient_->ReleaseBuffer(numFramesAvailable);
                
                hr = pCaptureClient_->GetNextPacketSize(&packetLength);
                if (FAILED(hr)) {
                    break;
                }
            }
        }
    }
    
    void process_audio_data(BYTE* pData, UINT32 numFrames, const WAVEFORMATEX& format, 
                           bool is_float, bool is_silent) {
        
        // Check if we're still recording - avoid processing after stop() is called
        if (!is_recording_) {
            return;
        }
        
        size_t total_samples = numFrames * actual_channels_;
        
        // Convert to float32 for callback
        std::vector<float> float_samples(total_samples);
        
        if (is_silent) {
            // Fill with silence
            std::fill(float_samples.begin(), float_samples.end(), 0.0f);
        } else {
            if (is_float) {
                // Already float, just copy
                const float* float_data = reinterpret_cast<const float*>(pData);
                std::copy(float_data, float_data + total_samples, float_samples.begin());
            } else if (format.wBitsPerSample == 16) {
                // Convert int16 to float
                const int16_t* int16_data = reinterpret_cast<const int16_t*>(pData);
                for (size_t i = 0; i < total_samples; i++) {
                    float_samples[i] = static_cast<float>(int16_data[i]) / 32768.0f;
                }
            } else if (format.wBitsPerSample == 32) {
                // Assume int32, convert to float
                const int32_t* int32_data = reinterpret_cast<const int32_t*>(pData);
                for (size_t i = 0; i < total_samples; i++) {
                    float_samples[i] = static_cast<float>(int32_data[i]) / 2147483648.0f;
                }
            }
        }
        
        // Call Python callback if provided (check again in case stop was called during conversion)
        if (callback_ && is_recording_) {
            callback_(float_samples.data(), total_samples, actual_channels_, actual_sample_rate_);
        }
        
        // Write to WAV file if open
        if (wav_file_.is_open()) {
            std::lock_guard<std::mutex> lock(file_mutex_);
            
            // Convert float to int16 for WAV file
            std::vector<int16_t> int16_samples(total_samples);
            for (size_t i = 0; i < total_samples; i++) {
                float sample = float_samples[i];
                // Clamp to -1.0 to 1.0
                if (sample > 1.0f) sample = 1.0f;
                if (sample < -1.0f) sample = -1.0f;
                int16_samples[i] = static_cast<int16_t>(sample * 32767.0f);
            }
            
            wav_file_.write(reinterpret_cast<const char*>(int16_samples.data()), 
                           int16_samples.size() * sizeof(int16_t));
            bytes_recorded_ += int16_samples.size() * sizeof(int16_t);
        }
    }
    
    std::string output_path_;
    int sample_rate_;
    int channels_;
    std::atomic<bool> is_recording_;
    std::thread* capture_thread_;
    AudioCallback callback_;
    
    // Windows Audio API objects
    IMMDeviceEnumerator* pEnumerator_;
    IMMDevice* pDevice_;
    IAudioClient* pAudioClient_;
    IAudioCaptureClient* pCaptureClient_;
    
    // Silent render stream to prime loopback (WASAPI loopback requires active stream)
    IAudioClient* pRenderClient_;
    IAudioRenderClient* pRenderService_;
    
    // File writing
    std::ofstream wav_file_;
    std::mutex file_mutex_;
    uint32_t bytes_recorded_;
    
    // Actual format info
    int actual_sample_rate_;
    int actual_channels_;
    
    bool com_initialized_;
};

// AudioRecorder implementation
AudioRecorder::AudioRecorder(const std::string& output_path, int sample_rate, int channels)
    : pImpl(std::make_unique<Impl>(output_path, sample_rate, channels)) {
}

AudioRecorder::~AudioRecorder() = default;

void AudioRecorder::start() {
    pImpl->start();
}

void AudioRecorder::start_with_callback(AudioCallback callback) {
    pImpl->start_with_callback(callback);
}

std::string AudioRecorder::stop() {
    return pImpl->stop();
}

bool AudioRecorder::is_recording() const {
    return pImpl->is_recording();
}

bool AudioRecorder::check_screen_recording_permission() {
    // On Windows, WASAPI loopback doesn't require special permissions
    // Always return true
    return true;
}

} // namespace pysysaudio

#endif // _WIN32


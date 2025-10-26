#import <Foundation/Foundation.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreAudio/CoreAudio.h>

#include "audio_recorder.h"
#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>

// Forward declare the C++ callback type
namespace pysysaudio {
    using AudioCallback = std::function<void(const float* samples, size_t sample_count, int channels, double sample_rate)>;
}

// Objective-C delegate to handle audio samples (must be outside namespace)
@interface AudioSampleDelegate : NSObject <SCStreamDelegate, SCStreamOutput>

@property (nonatomic, strong) AVAssetWriter *assetWriter;
@property (nonatomic, strong) AVAssetWriterInput *audioInput;
@property (nonatomic, assign) BOOL isRecording;
@property (nonatomic, strong) dispatch_queue_t writingQueue;
@property (nonatomic, assign) int channels;
@property (nonatomic, copy) void(^audioCallback)(const float* samples, size_t sample_count, int channels, double sample_rate);

- (instancetype)initWithOutputPath:(NSString *)outputPath 
                        sampleRate:(int)sampleRate 
                          channels:(int)channels
                  audioCallback:(void(^)(const float* samples, size_t sample_count, int channels, double sample_rate))callback;
- (void)startRecording;
- (void)stopRecording;

@end

@implementation AudioSampleDelegate

- (instancetype)initWithOutputPath:(NSString *)outputPath 
                        sampleRate:(int)sampleRate 
                          channels:(int)channels
                  audioCallback:(void(^)(const float* samples, size_t sample_count, int channels, double sample_rate))callback {
    self = [super init];
    if (self) {
        _isRecording = NO;
        _channels = channels;
        _audioCallback = callback;
        _writingQueue = dispatch_queue_create("com.pysysaudio.writing", DISPATCH_QUEUE_SERIAL);
        
        // Only set up AVAssetWriter if output path is provided
        if (outputPath && outputPath.length > 0) {
            // Set up AVAssetWriter
            NSURL *outputURL = [NSURL fileURLWithPath:outputPath];
            NSError *error = nil;
            
            // Remove existing file if it exists
            [[NSFileManager defaultManager] removeItemAtURL:outputURL error:nil];
            
            _assetWriter = [[AVAssetWriter alloc] initWithURL:outputURL 
                                                     fileType:AVFileTypeWAVE 
                                                        error:&error];
            if (error) {
                NSLog(@"Error creating asset writer: %@", error);
                return nil;
            }
        
        // Configure audio settings
        AudioChannelLayout channelLayout;
        memset(&channelLayout, 0, sizeof(AudioChannelLayout));
        channelLayout.mChannelLayoutTag = (channels == 2) ? 
            kAudioChannelLayoutTag_Stereo : kAudioChannelLayoutTag_Mono;
        
        NSData *channelLayoutData = [NSData dataWithBytes:&channelLayout 
                                                   length:sizeof(channelLayout)];
        
        NSDictionary *audioSettings = @{
            AVFormatIDKey: @(kAudioFormatLinearPCM),
            AVSampleRateKey: @(sampleRate),
            AVNumberOfChannelsKey: @(channels),
            AVLinearPCMBitDepthKey: @(16),
            AVLinearPCMIsNonInterleaved: @NO,
            AVLinearPCMIsFloatKey: @NO,  // int16 for WAV file compatibility
            AVLinearPCMIsBigEndianKey: @NO,
            AVChannelLayoutKey: channelLayoutData
        };
            
            _audioInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeAudio
                                                             outputSettings:audioSettings];
            _audioInput.expectsMediaDataInRealTime = YES;
            
            if ([_assetWriter canAddInput:_audioInput]) {
                [_assetWriter addInput:_audioInput];
            } else {
                NSLog(@"Cannot add audio input to asset writer");
                return nil;
            }
        }
    }
    return self;
}

- (void)startRecording {
    self.isRecording = YES;
    if (self.assetWriter && self.assetWriter.status == AVAssetWriterStatusUnknown) {
        [self.assetWriter startWriting];
        [self.assetWriter startSessionAtSourceTime:kCMTimeZero];
    }
}

- (void)stopRecording {
    self.isRecording = NO;
    
    if (self.assetWriter) {
        dispatch_sync(self.writingQueue, ^{
            [self.audioInput markAsFinished];
            [self.assetWriter finishWritingWithCompletionHandler:^{
                NSLog(@"Finished writing audio file");
            }];
        });
    }
}

// SCStreamOutput protocol method
- (void)stream:(SCStream *)stream 
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer 
               ofType:(SCStreamOutputType)type API_AVAILABLE(macos(13.0)) {
    
    if (!self.isRecording) {
        return;
    }
    
    if (@available(macOS 13.0, *)) {
        if (type == SCStreamOutputTypeAudio) {
            // Retain the sample buffer before async dispatch
            CFRetain(sampleBuffer);
            
            // If we have a callback, extract the audio data and call it
            if (self.audioCallback) {
                CMBlockBufferRef blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer);
                if (blockBuffer) {
                    size_t length = 0;
                    char *dataPointer = NULL;
                    
                    OSStatus status = CMBlockBufferGetDataPointer(blockBuffer, 0, NULL, &length, &dataPointer);
                    
                    if (status == noErr && dataPointer) {
                        // Get the audio format description
                        CMFormatDescriptionRef formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer);
                        const AudioStreamBasicDescription *asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc);
                        
                        // Debug: Log format info on first callback
                        static bool loggedFormat = false;
                        if (!loggedFormat && asbd) {
                            NSLog(@"Audio format: sampleRate=%.0f, channels=%u, formatID=%u, formatFlags=%u, bitsPerChannel=%u, bytesPerFrame=%u",
                                  asbd->mSampleRate, asbd->mChannelsPerFrame, asbd->mFormatID, 
                                  asbd->mFormatFlags, asbd->mBitsPerChannel, asbd->mBytesPerFrame);
                            loggedFormat = true;
                        }
                        
                        // ScreenCaptureKit delivers Float32 by default - pass it through as-is
                        if (asbd && asbd->mFormatID == kAudioFormatLinearPCM) {
                            double actualSampleRate = asbd->mSampleRate;
                            
                            if (asbd->mFormatFlags & kAudioFormatFlagIsFloat) {
                                // Audio is in Float32 format (normalized -1.0 to 1.0)
                                const float *floatSamples = (const float *)dataPointer;
                                size_t totalFloats = length / sizeof(float);
                                
                                // Check if audio is non-interleaved (planar)
                                bool isNonInterleaved = (asbd->mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
                                
                                if (isNonInterleaved) {
                                    // Non-interleaved: [L L L L...] [R R R R...]
                                    // Need to convert to interleaved float: [L R L R L R...]
                                    size_t framesPerChannel = totalFloats / self.channels;
                                    std::vector<float> interleavedSamples(totalFloats);
                                    
                                    for (size_t frame = 0; frame < framesPerChannel; frame++) {
                                        for (int ch = 0; ch < self.channels; ch++) {
                                            interleavedSamples[frame * self.channels + ch] = 
                                                floatSamples[ch * framesPerChannel + frame];
                                        }
                                    }
                                    
                                    // Call callback with interleaved float samples and actual sample rate
                                    // totalFloats = total number of float values (frames * channels)
                                    self.audioCallback(interleavedSamples.data(), totalFloats, self.channels, actualSampleRate);
                                } else {
                                    // Already interleaved: [L R L R L R...]
                                    // Pass directly without any conversion, include actual sample rate
                                    // totalFloats = total number of float values (frames * channels)
                                    self.audioCallback(floatSamples, totalFloats, self.channels, actualSampleRate);
                                }
                            } else {
                                // If somehow we get int16 PCM format, convert to float
                                const int16_t *int16Samples = (const int16_t *)dataPointer;
                                size_t sampleCount = length / sizeof(int16_t);
                                std::vector<float> floatSamples(sampleCount);
                                
                                for (size_t i = 0; i < sampleCount; i++) {
                                    floatSamples[i] = (float)int16Samples[i] / 32768.0f;
                                }
                                
                                self.audioCallback(floatSamples.data(), sampleCount, self.channels, actualSampleRate);
                            }
                        }
                    }
                }
            }
            
            // Only write to file if we have an asset writer
            if (self.audioInput) {
                dispatch_async(self.writingQueue, ^{
                    if (self.audioInput.isReadyForMoreMediaData && self.isRecording) {
                        [self.audioInput appendSampleBuffer:sampleBuffer];
                    }
                    CFRelease(sampleBuffer);
                });
            } else {
                // No file writing, just release the sample buffer
                CFRelease(sampleBuffer);
            }
        }
    }
}

// SCStreamDelegate protocol methods
- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
    if (error) {
        NSLog(@"Stream stopped with error: %@", error);
    }
}

@end

// Now the C++ implementation in the namespace
namespace pysysaudio {

// C++ Implementation class
class AudioRecorder::Impl {
public:
    Impl(const std::string& output_path, int sample_rate, int channels)
        : output_path_(output_path)
        , sample_rate_(sample_rate)
        , channels_(channels)
        , is_recording_(false)
        , stream_(nil)
        , delegate_(nil)
        , callback_(nullptr) {
    }
    
    ~Impl() {
        if (is_recording_) {
            stop();
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
        
        if (@available(macOS 13.0, *)) {
            // Request permission and start capture
            dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
            __block NSError *permissionError = nil;
            __block AudioRecorder::Impl* implPtr = this;
            
            [SCShareableContent getShareableContentExcludingDesktopWindows:YES
                                                        onScreenWindowsOnly:YES
                                                          completionHandler:^(SCShareableContent *content, NSError *error) {
                permissionError = error;
                if (error) {
                    NSLog(@"Error getting shareable content: %@", error);
                    dispatch_semaphore_signal(semaphore);
                    return;
                }
                
                // Create stream configuration
                SCStreamConfiguration *config = [[SCStreamConfiguration alloc] init];
                config.capturesAudio = YES;
                config.sampleRate = implPtr->sample_rate_;
                config.channelCount = implPtr->channels_;
                
                // Create content filter - use display for audio capture
                if (content.displays.count > 0) {
                    SCDisplay *display = content.displays.firstObject;
                    SCContentFilter *filter = [[SCContentFilter alloc] initWithDisplay:display
                                                                     excludingWindows:@[]];
                    
                    // Create the stream
                    implPtr->stream_ = [[SCStream alloc] initWithFilter:filter 
                                                        configuration:config 
                                                             delegate:nil];
                    
                    // Create and set up delegate with optional callback
                    NSString *outputPath = nil;
                    if (!implPtr->output_path_.empty()) {
                        outputPath = [NSString stringWithUTF8String:implPtr->output_path_.c_str()];
                    }
                    
                    // Create Objective-C block that wraps C++ callback
                    void(^objcCallback)(const float*, size_t, int, double) = nil;
                    if (implPtr->callback_) {
                        // Capture the C++ callback in the Objective-C block
                        AudioCallback cppCallback = implPtr->callback_;
                        objcCallback = ^(const float* samples, size_t sample_count, int channels, double sample_rate) {
                            cppCallback(samples, sample_count, channels, sample_rate);
                        };
                    }
                    
                    implPtr->delegate_ = [[AudioSampleDelegate alloc] initWithOutputPath:outputPath
                                                                           sampleRate:implPtr->sample_rate_
                                                                             channels:implPtr->channels_
                                                                        audioCallback:objcCallback];
                    
                    if (!implPtr->delegate_) {
                        NSLog(@"Failed to create audio delegate");
                        dispatch_semaphore_signal(semaphore);
                        return;
                    }
                    
                    NSError *addOutputError = nil;
                    [implPtr->stream_ addStreamOutput:implPtr->delegate_
                                          type:SCStreamOutputTypeAudio
                                sampleHandlerQueue:dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0)
                                             error:&addOutputError];
                    
                    if (addOutputError) {
                        NSLog(@"Error adding stream output: %@", addOutputError);
                        dispatch_semaphore_signal(semaphore);
                        return;
                    }
                    
                    // Start the stream
                    [implPtr->stream_ startCaptureWithCompletionHandler:^(NSError *startError) {
                        if (startError) {
                            NSLog(@"Error starting capture: %@", startError);
                        } else {
                            implPtr->is_recording_ = true;
                            [implPtr->delegate_ startRecording];
                        }
                        dispatch_semaphore_signal(semaphore);
                    }];
                } else {
                    NSLog(@"No displays available");
                    dispatch_semaphore_signal(semaphore);
                }
            }];
            
            // Wait for completion
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
            
            if (permissionError) {
                throw std::runtime_error("Failed to start recording: " + 
                    std::string([permissionError.localizedDescription UTF8String]));
            }
            
            if (!is_recording_) {
                throw std::runtime_error("Failed to start recording");
            }
        } else {
            throw std::runtime_error("macOS 13.0 or later is required for audio capture");
        }
    }
    
    std::string stop() {
        if (!is_recording_) {
            throw std::runtime_error("Not currently recording");
        }
        
        [delegate_ stopRecording];
        
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        
        [stream_ stopCaptureWithCompletionHandler:^(NSError *error) {
            if (error) {
                NSLog(@"Error stopping capture: %@", error);
            }
            dispatch_semaphore_signal(semaphore);
        }];
        
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        
        is_recording_ = false;
        stream_ = nil;
        delegate_ = nil;
        
        return output_path_;
    }
    
    bool is_recording() const {
        return is_recording_;
    }
    
private:
    std::string output_path_;
    int sample_rate_;
    int channels_;
    bool is_recording_;
    SCStream * __strong stream_;
    AudioSampleDelegate * __strong delegate_;
    AudioCallback callback_;
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
    if (@available(macOS 13.0, *)) {
        __block bool hasPermission = false;
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        
        [SCShareableContent getShareableContentExcludingDesktopWindows:YES
                                                    onScreenWindowsOnly:YES
                                                      completionHandler:^(SCShareableContent *content, NSError *error) {
            if (error) {
                // Check if it's a permission error
                if (error.code == -3801) { // Permission denied
                    hasPermission = false;
                } else {
                    // Other errors might mean permission is granted but something else failed
                    hasPermission = true;
                }
            } else {
                hasPermission = true;
            }
            dispatch_semaphore_signal(semaphore);
        }];
        
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        return hasPermission;
    }
    return false;
}

} // namespace pysysaudio

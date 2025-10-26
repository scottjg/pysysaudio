#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "audio_recorder.h"

namespace py = pybind11;
using namespace pysysaudio;

PYBIND11_MODULE(_pysysaudio_native, m) {
#ifdef _WIN32
    m.doc() = "Native audio recording module using WASAPI (Windows Audio Session API)";
#else
    m.doc() = "Native audio recording module using ScreenCaptureKit (macOS)";
#endif

    // AudioRecorder class
    py::class_<AudioRecorder>(m, "AudioRecorder")
        .def(py::init<const std::string&, int, int>(),
             py::arg("output_path"),
             py::arg("sample_rate") = 48000,
             py::arg("channels") = 2,
             "Create an audio recorder instance")
        .def("start", &AudioRecorder::start,
             "Start recording system audio to file")
        .def("start_with_callback", 
             [](AudioRecorder& self, py::function callback) {
                 // Wrap Python callback in C++ lambda
                 self.start_with_callback([callback](const float* samples, size_t sample_count, int channels, double sample_rate) {
                     // Convert float* to Python bytes
                     py::gil_scoped_acquire acquire;
                     py::bytes data(reinterpret_cast<const char*>(samples), sample_count * sizeof(float));
                     callback(data, sample_count, channels, sample_rate);
                 });
             },
             py::arg("callback"),
             "Start recording with a callback function that receives (bytes, sample_count, channels, sample_rate)")
        .def("stop", 
             [](AudioRecorder& self) {
                 // Release GIL while stopping to avoid deadlock with callback thread
                 py::gil_scoped_release release;
                 return self.stop();
             },
             "Stop recording and return the output file path")
        .def("is_recording", &AudioRecorder::is_recording,
             "Check if currently recording")
        .def_static("check_screen_recording_permission", 
                    &AudioRecorder::check_screen_recording_permission,
                    "Check if Screen Recording permission is granted");

    // Standalone function for checking permission
    m.def("check_screen_recording_permission", 
          &AudioRecorder::check_screen_recording_permission,
          "Check if Screen Recording permission is granted");
}


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "audio_recorder.h"
#include <vector>

namespace py = pybind11;
using namespace pysysaudio;

PYBIND11_MODULE(_pysysaudio_native, m) {
#ifdef _WIN32
    m.doc() = "Native audio recording module using WASAPI (Windows Audio Session API)";
#else
    m.doc() = "Native audio recording module using CoreAudio taps (macOS)";
#endif

    // The capture thread never touches Python. Samples flow through a native
    // queue, and read() below blocks with the GIL released until some arrive.
    py::class_<AudioRecorder>(m, "AudioRecorder")
        .def(py::init<const std::string&, double>(),
             py::arg("output_path"),
             py::arg("buffer_seconds") = 2.0,
             "Create an audio recorder. output_path may be empty for streaming only; "
             "buffer_seconds is how much audio read() may fall behind before chunks are dropped")
        .def("start",
             [](AudioRecorder& self) {
                 py::gil_scoped_release release;
                 self.start();
             },
             "Start capturing system audio")
        .def("stop",
             [](AudioRecorder& self) {
                 // Release the GIL so a thread blocked in read() can wake and finish.
                 py::gil_scoped_release release;
                 return self.stop();
             },
             "Stop capturing and return the output file path")
        .def("is_recording", &AudioRecorder::is_recording,
             "Check if currently recording")
        .def("sample_rate", &AudioRecorder::sample_rate,
             "Sample rate of the captured audio (valid after start)")
        .def("channels", &AudioRecorder::channels,
             "Channel count of the captured audio (valid after start)")
        .def("dropped_frames", &AudioRecorder::dropped_frames,
             "Frames discarded because read() did not keep up with capture")
        .def("read",
             [](AudioRecorder& self, size_t max_frames, double timeout_seconds) {
                 std::vector<float> samples;
                 {
                     py::gil_scoped_release release;
                     self.read(samples, max_frames, timeout_seconds);
                 }
                 return py::bytes(reinterpret_cast<const char*>(samples.data()),
                                  samples.size() * sizeof(float));
             },
             py::arg("max_frames"),
             py::arg("timeout_seconds") = -1.0,
             "Read up to max_frames frames of interleaved float32 samples as bytes. Blocks "
             "until data is available, the recorder stops, or the timeout passes (negative "
             "waits indefinitely). Returns empty bytes on timeout or once stopped and drained")
        .def_static("check_screen_recording_permission",
                    &AudioRecorder::check_screen_recording_permission,
                    "Check if the current runtime supports system capture");

    // Standalone function for checking permission
    m.def("check_screen_recording_permission",
          &AudioRecorder::check_screen_recording_permission,
          "Check if the current runtime supports system capture");
}

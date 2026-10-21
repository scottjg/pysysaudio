#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "audio_recorder.h"
#include "format_converter.h"
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pysysaudio;

namespace {

py::bytes to_bytes(const std::vector<uint8_t>& data) {
    return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

}  // namespace

PYBIND11_MODULE(_pysysaudio_native, m) {
#ifdef _WIN32
    m.doc() = "Native audio recording module using WASAPI (Windows Audio Session API)";
#else
    m.doc() = "Native audio recording module using CoreAudio taps (macOS)";
#endif

    // The capture thread never touches Python. Samples flow through a native
    // queue; read() blocks with the GIL released and converts to the requested
    // format with the platform's resampler on the reader's thread.
    py::class_<AudioRecorder>(m, "AudioRecorder")
        .def(py::init<const std::string&, int, int, const std::string&, double>(),
             py::arg("output_path"),
             py::arg("sample_rate") = 48000,
             py::arg("channels") = 2,
             py::arg("dtype") = "float32",
             py::arg("buffer_seconds") = 2.0,
             "Create an audio recorder. read() returns audio at sample_rate, channels and "
             "dtype; the WAV file at output_path (empty for none) is written at the same "
             "rate and channel count as 16-bit PCM. buffer_seconds is how much audio read() "
             "may fall behind before chunks are dropped")
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
        .def("capture_sample_rate", &AudioRecorder::capture_sample_rate,
             "Sample rate the system delivers (valid after start)")
        .def("capture_channels", &AudioRecorder::capture_channels,
             "Channel count the system delivers (valid after start)")
        .def("dropped_frames", &AudioRecorder::dropped_frames,
             "Captured frames discarded because read() did not keep up")
        .def("read",
             [](AudioRecorder& self, size_t max_frames, double timeout_seconds) -> py::object {
                 std::vector<uint8_t> data;
                 bool more;
                 {
                     py::gil_scoped_release release;
                     more = self.read(data, max_frames, timeout_seconds);
                 }
                 if (!more) {
                     return py::none();
                 }
                 return to_bytes(data);
             },
             py::arg("max_frames"),
             py::arg("timeout_seconds") = -1.0,
             "Read converted audio for up to max_frames captured frames. Blocks until audio "
             "is available, the recorder stops, or the timeout passes (negative waits "
             "indefinitely). Returns bytes, empty on timeout, or None once stopped and fully "
             "drained")
        .def_static("check_screen_recording_permission",
                    &AudioRecorder::check_screen_recording_permission,
                    "Check if the current runtime supports system capture");

    // Standalone access to the same conversion read() uses, for tests and for
    // converting audio that did not come from the recorder.
    py::class_<FormatConverter>(m, "FormatConverter")
        .def(py::init([](double input_rate, int input_channels, double output_rate,
                         int output_channels, const std::string& dtype) {
                 return std::make_unique<FormatConverter>(input_rate, input_channels, output_rate,
                                                          output_channels, parse_sample_type(dtype));
             }),
             py::arg("input_rate"),
             py::arg("input_channels"),
             py::arg("output_rate"),
             py::arg("output_channels"),
             py::arg("dtype") = "float32",
             "Convert interleaved float32 audio: remix channels, resample with the platform's "
             "native resampler, then encode as dtype ('float32', 'int16' or 'int32')")
        .def("process",
             [](FormatConverter& self, py::bytes data) {
                 std::string input = data;
                 const size_t frame_bytes = sizeof(float) * self.input_channels();
                 if (input.size() % frame_bytes != 0) {
                     throw py::value_error("input must be whole frames of float32 samples");
                 }
                 std::vector<uint8_t> out;
                 {
                     py::gil_scoped_release release;
                     self.process(reinterpret_cast<const float*>(input.data()),
                                  input.size() / frame_bytes, out);
                 }
                 return to_bytes(out);
             },
             py::arg("data"),
             "Convert a chunk of interleaved float32 bytes; state carries across calls")
        .def("flush",
             [](FormatConverter& self) {
                 std::vector<uint8_t> out;
                 {
                     py::gil_scoped_release release;
                     self.flush(out);
                 }
                 return to_bytes(out);
             },
             "End the stream and return the resampler's remaining output");

    // Standalone function for checking permission
    m.def("check_screen_recording_permission",
          &AudioRecorder::check_screen_recording_permission,
          "Check if the current runtime supports system capture");
}

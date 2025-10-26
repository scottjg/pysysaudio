import os
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class get_pybind_include:
    """Helper class to determine the pybind11 include path"""

    def __str__(self):
        import pybind11

        return pybind11.get_include()


# Detect platform and configure build accordingly
system = platform.system()

if system == "Darwin":
    # macOS build configuration
    macos_version = platform.mac_ver()[0]
    major = int(macos_version.split(".")[0])
    if major < 13:
        raise RuntimeError("pysysaudio requires macOS 13.0 (Ventura) or later")

    ext_modules = [
        Extension(
            "pysysaudio._pysysaudio_native",
            sources=[
                "native/audio_recorder.mm",
                "native/bindings.cpp",
            ],
            include_dirs=[
                get_pybind_include(),
                "native",
            ],
            language="c++",
            extra_compile_args=[
                "-std=c++17",
                "-stdlib=libc++",
                "-mmacosx-version-min=13.0",
                "-fobjc-arc",  # Enable ARC for Objective-C
            ],
            extra_link_args=[
                "-framework",
                "ScreenCaptureKit",
                "-framework",
                "AVFoundation",
                "-framework",
                "CoreMedia",
                "-framework",
                "CoreAudio",
                "-framework",
                "AudioToolbox",
                "-framework",
                "Foundation",
                "-mmacosx-version-min=13.0",
            ],
        ),
    ]

elif system == "Windows":
    # Windows build configuration
    ext_modules = [
        Extension(
            "pysysaudio._pysysaudio_native",
            sources=[
                "native/audio_recorder_win.cpp",
                "native/bindings.cpp",
            ],
            include_dirs=[
                get_pybind_include(),
                "native",
            ],
            language="c++",
            extra_compile_args=[
                "/std:c++17",
                "/EHsc",  # Exception handling
            ],
            libraries=[
                "ole32",
                "oleaut32",
                "uuid",
            ],
        ),
    ]

else:
    raise RuntimeError(
        f"pysysaudio does not support {system}. Only macOS and Windows are supported."
    )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "unix": [],
    }
    l_opts = {
        "unix": [],
    }

    def build_extensions(self):
        # Register .mm files as C++ source files (macOS only)
        if platform.system() == "Darwin":
            self.compiler.src_extensions.append(".mm")

        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        for ext in self.extensions:
            ext.extra_compile_args = opts + ext.extra_compile_args
            ext.extra_link_args = link_opts + ext.extra_link_args

        build_ext.build_extensions(self)


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)

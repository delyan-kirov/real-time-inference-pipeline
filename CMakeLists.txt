# ---------------------------------------------------------------------------
# \file CMakeLists.txt
# \brief RFD (real-time face detection) - top-level build definition.
#
# \note
#   * `cmake --preset dev` reads this and GENERATES a build system (Makefiles)
#     into build/dev/. That step is called "configure".
#   * `cmake --build --preset dev` then runs make against those files.
#   * Re-running the build re-runs configure automatically if this file changed.
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.22)

project(rfd
  VERSION 0.1.0
  DESCRIPTION "Real-time face-recognition inference pipeline on GStreamer"
  LANGUAGES CXX)

# --- Global language settings ----------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # -std=c++20, not -std=gnu++20

# Emit build/dev/compile_commands.json. clangd reads this to power IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Put binaries somewhere predictable: build/<preset>/ rather than nested dirs.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Default to a real build type when the user configures without a preset.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# --- Layout ------------------------------------------------------------------
set(RFD_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# --- Dependencies ------------------------------------------------------------
find_package(PkgConfig REQUIRED)

pkg_check_modules(gstreamer      REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.20)
pkg_check_modules(gstreamer_base REQUIRED IMPORTED_TARGET gstreamer-base-1.0)
pkg_check_modules(gstreamer_app  REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_check_modules(gstreamer_video REQUIRED IMPORTED_TARGET gstreamer-video-1.0)

message(STATUS "GStreamer version: ${gstreamer_VERSION}")

# OpenCV supplies both inference models and the DNN runtime that executes them:
#   objdetect - FaceDetectorYN (YuNet) and FaceRecognizerSF (SFace) wrappers
#   dnn       - ONNX importer and inference engine behind both
#   imgcodecs - reading gallery images off disk
find_package(OpenCV 4.5.4 CONFIG REQUIRED COMPONENTS core imgproc imgcodecs dnn objdetect)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Metadata output and the on-disk gallery are both JSON. Header-only, so this
# costs nothing at link time.
find_package(nlohmann_json 3.9 CONFIG REQUIRED)

message(STATUS "nlohmann_json version: ${nlohmann_json_VERSION}")

# --- Model weights -------------------------------------------------
set(RFD_MODEL_DIR "${CMAKE_SOURCE_DIR}/external/models"
    CACHE PATH "Directory holding the vendored YuNet and SFace ONNX weights")

set(RFD_YUNET_MODEL "${RFD_MODEL_DIR}/face_detection_yunet_2023mar.onnx")
set(RFD_SFACE_MODEL "${RFD_MODEL_DIR}/face_recognition_sface_2021dec.onnx")

foreach(_model IN ITEMS RFD_YUNET_MODEL RFD_SFACE_MODEL)
  if(NOT EXISTS "${${_model}}")
    message(FATAL_ERROR
      "${_model} not found: ${${_model}}\n"
      "Expected vendored weights under RFD_MODEL_DIR (${RFD_MODEL_DIR}). "
      "See external/README.md to re-fetch them.")
  endif()
endforeach()

# An embedding is only comparable against others from the same weights, so the
# SFace digest is baked in and written into every gallery file. Reusing the
# SHA256SUMS that CI already verifies avoids both a second copy of the hash and
# a crypto dependency just to recompute it at runtime.
file(STRINGS "${RFD_MODEL_DIR}/SHA256SUMS" _rfd_sums)
foreach(_line IN LISTS _rfd_sums)
  if(_line MATCHES "^([0-9a-f]+) +face_recognition_sface")
    set(RFD_SFACE_SHA256 "${CMAKE_MATCH_1}")
  endif()
endforeach()
unset(_rfd_sums)

if(NOT RFD_SFACE_SHA256)
  message(FATAL_ERROR
    "could not read the SFace digest from ${RFD_MODEL_DIR}/SHA256SUMS")
endif()

add_library(rfd_models INTERFACE)
target_compile_definitions(rfd_models INTERFACE
  RFD_YUNET_MODEL="${RFD_YUNET_MODEL}"
  RFD_SFACE_MODEL="${RFD_SFACE_MODEL}"
  RFD_SFACE_SHA256="${RFD_SFACE_SHA256}")

# --- Warnings ---------------------------------------------------------------
set(RFD_WARNINGS
  -Wall -Wextra -Wpedantic
  -Wshadow -Wconversion -Wsign-conversion
  -Wnon-virtual-dtor -Woverloaded-virtual)

add_library(rfd_warnings INTERFACE)
target_compile_options(rfd_warnings INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:${RFD_WARNINGS} -Wold-style-cast>)

# Same set minus -Wold-style-cast, for translation units that use GObject's macro
add_library(rfd_warnings_gobject INTERFACE)
target_compile_options(rfd_warnings_gobject INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:${RFD_WARNINGS}>)

# --- Targets ----------------------------------------------------------------
add_subdirectory(source)

# --- Tests and convenience targets -------------------------------------------
# `ctest --preset dev`.
enable_testing()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Used by tests/.
set(RFD_DATA "${CMAKE_SOURCE_DIR}/external/data/gallery")
set(RFD_GALLERY "${CMAKE_BINARY_DIR}/gallery.json")

add_subdirectory(tests)

# --- Editor integration -----------------------------------------------------
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  file(CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    SYMBOLIC)
endif()

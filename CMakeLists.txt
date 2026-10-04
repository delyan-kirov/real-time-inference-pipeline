# ---------------------------------------------------------------------------
# Real-time inference pipeline - top-level build definition.
#
# \note:
#   * `cmake --preset dev` reads this and GENERATES a build system (Makefiles)
#     into build/dev/. That step is called "configure".
#   * `cmake --build --preset dev` then runs make against those files.
#   * Re-running the build re-runs configure automatically if this file changed.
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.22)

project(rtip
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

# --- Dependencies ------------------------------------------------------------
find_package(PkgConfig REQUIRED)

pkg_check_modules(gstreamer      REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.20)
pkg_check_modules(gstreamer_base REQUIRED IMPORTED_TARGET gstreamer-base-1.0)
pkg_check_modules(gstreamer_app  REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_check_modules(gstreamer_video REQUIRED IMPORTED_TARGET gstreamer-video-1.0)

message(STATUS "GStreamer version: ${gstreamer_VERSION}")

# OpenCV supplies both inference models and the DNN runtime that executes them:
#   objdetect - FaceDetectorYN (YuNet) and FaceRecognizerSF (SFace) wrappers
#   dnn       - the ONNX importer and inference engine behind both
#   imgcodecs - reading gallery images off disk
# Named as COMPONENTS rather than taken from opencv4.pc on purpose: the .pc file
# lists every module OpenCV was built with (~50 libraries), while find_package
# yields one imported target per module so we link only what we use.
# 4.5.4 is the floor: FaceDetectorYN and FaceRecognizerSF first appear there.
# It is also what Ubuntu 22.04 ships, which is the oldest image CI builds on.
find_package(OpenCV 4.5.4 CONFIG REQUIRED COMPONENTS core imgproc imgcodecs dnn objdetect)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --- Model weights -------------------------------------------------
set(RTIP_MODEL_DIR "${CMAKE_SOURCE_DIR}/external/models"
    CACHE PATH "Directory holding the vendored YuNet and SFace ONNX weights")

set(RTIP_YUNET_MODEL "${RTIP_MODEL_DIR}/face_detection_yunet_2023mar.onnx")
set(RTIP_SFACE_MODEL "${RTIP_MODEL_DIR}/face_recognition_sface_2021dec.onnx")

foreach(_model IN ITEMS RTIP_YUNET_MODEL RTIP_SFACE_MODEL)
  if(NOT EXISTS "${${_model}}")
    message(FATAL_ERROR
      "${_model} not found: ${${_model}}\n"
      "Expected vendored weights under RTIP_MODEL_DIR (${RTIP_MODEL_DIR}). "
      "See external/README.md to re-fetch them.")
  endif()
endforeach()

add_library(rtip_models INTERFACE)
target_compile_definitions(rtip_models INTERFACE
  RTIP_YUNET_MODEL="${RTIP_YUNET_MODEL}"
  RTIP_SFACE_MODEL="${RTIP_SFACE_MODEL}")

# --- Warnings ---------------------------------------------------------------
add_library(rtip_warnings INTERFACE)
target_compile_options(rtip_warnings INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:
    -Wall -Wextra -Wpedantic
    -Wshadow -Wconversion -Wsign-conversion
    -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual
  >)

# --- Targets ----------------------------------------------------------------
add_subdirectory(src)

# --- Editor integration -----------------------------------------------------
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  file(CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    SYMBOLIC)
endif()

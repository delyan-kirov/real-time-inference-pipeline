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

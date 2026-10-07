# ---------------------------------------------------------------------------
# \file source/CMakeLists.txt
# \brief The library, the GStreamer plugin and the executables.
#
# Headers live in ../include and are published through rfd_face's PUBLIC include
# directory, so every target here and in tests/ sees them the same way.
# ---------------------------------------------------------------------------

# --- Shared face library -----------------------------------------------------
# Detection, embedding and gallery matching, with no dependency on how frames
# arrive. The enrollment tool and the pipeline both link this, which is what
# guarantees a gallery is built with exactly the alignment the matcher expects.
add_library(rfd_face
  detector.cpp
  embedder.cpp
  gallery.cpp)

target_include_directories(rfd_face PUBLIC "${RFD_INCLUDE_DIR}")

# This static library is linked into libgstrfd.so below, so its objects have to
# be position independent or the shared link fails outright with "recompile with
# -fPIC". Nix only gets away without it because its gcc wrapper compiles PIE by
# default; a stock Ubuntu toolchain does not.
set_target_properties(rfd_face PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(rfd_face
  PUBLIC
    rfd_models
    opencv_core
    opencv_objdetect
  PRIVATE
    rfd_warnings
    opencv_imgproc
    opencv_dnn
    nlohmann_json::nlohmann_json)

# --- GStreamer element -------------------------------------------------------
# Built as a loadable plugin so `gst-inspect-1.0 rfdface` and
# `gst-launch-1.0 ... ! rfdface ! ...` work, which is what makes this a genuine
# GStreamer component rather than a class our own binary registers.
add_library(rfd_gst SHARED
  face_meta.cpp
  face_filter.cpp
  plugin.cpp)

# GStreamer loads plugins by filename pattern: libgst<name>.so.
set_target_properties(rfd_gst PROPERTIES OUTPUT_NAME gstrfd)

target_link_libraries(rfd_gst
  PUBLIC
    rfd_face
    PkgConfig::gstreamer
    PkgConfig::gstreamer_video
  PRIVATE
    rfd_warnings_gobject)

target_compile_definitions(rfd_gst PRIVATE RFD_VERSION="${PROJECT_VERSION}")

# --- Enrollment tool ---------------------------------------------------------
add_executable(rfd-enroll enroll.cpp)

target_link_libraries(rfd-enroll
  PRIVATE
    rfd_face
    rfd_warnings
    opencv_imgcodecs)

# --- The application ---------------------------------------------------------
# Builds the pipeline, paces it, and serialises the metadata rfdface attached.
# Uses rfd_warnings_gobject because it drives GStreamer through the same C macros.
add_executable(rfd rfd.cpp)

target_link_libraries(rfd
  PRIVATE
    rfd_gst
    rfd_warnings_gobject
    PkgConfig::gstreamer_app
    nlohmann_json::nlohmann_json)

# --- Smoke test --------------------------------------------------------------
# Proves the toolchain, GStreamer headers, linkage, the plugin registry and the
# vendored weights all work end to end.
add_executable(rfd-smoke smoke.cpp)

# It deliberately does not link rfd_face - the point is to test the toolchain,
# not our library - so the vocabulary header's directory is named explicitly.
target_include_directories(rfd-smoke PRIVATE "${RFD_INCLUDE_DIR}")

target_link_libraries(rfd-smoke
  PRIVATE
    rfd_warnings
    rfd_models
    PkgConfig::gstreamer
    PkgConfig::gstreamer_base
    PkgConfig::gstreamer_video
    opencv_core
    opencv_dnn
    opencv_objdetect)

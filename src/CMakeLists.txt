# The smoke-test executable. This will be replaced by the real pipeline; for now
# it exists to prove the toolchain, GStreamer headers, linkage and the plugin
# registry all work end to end.
add_executable(rtip-hello hello.cpp)

target_link_libraries(rtip-hello
  PRIVATE
    rtip_warnings
    PkgConfig::gstreamer
    PkgConfig::gstreamer_base
    PkgConfig::gstreamer_video)

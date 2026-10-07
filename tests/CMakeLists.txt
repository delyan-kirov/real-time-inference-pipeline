# ---------------------------------------------------------------------------
# \file tests/CMakeLists.txt
# \brief The verification suite, as CTest tests.
#
#   ctest --preset dev              everything
#   ctest --preset dev -R realtime  one thing
#   ctest --preset dev -L fast      skip the slow streaming tests
#
# Ordering is expressed with fixtures rather than test order, because almost
# everything needs a gallery and building one is itself a test worth failing.
# ---------------------------------------------------------------------------

# --- Environment check -------------------------------------------------------
add_test(NAME smoke COMMAND rfd-smoke)
set_tests_properties(smoke PROPERTIES
  LABELS "fast"
  PASS_REGULAR_EXPRESSION "ok - smoke test passed")

# --- The vendored weights still match what upstream published ----------------
add_test(NAME model-digests
  COMMAND ${CMAKE_COMMAND} -E chdir "${CMAKE_SOURCE_DIR}/external/models"
          sha256sum -c SHA256SUMS)
set_tests_properties(model-digests PROPERTIES LABELS "fast")

# --- Gallery, which most other tests need ------------------------------------
add_test(NAME enroll
  COMMAND rfd-enroll "${RFD_DATA}" -o "${RFD_GALLERY}")
set_tests_properties(enroll PROPERTIES
  LABELS "fast"
  FIXTURES_SETUP gallery
  # A collapsed separation margin means the gallery cannot be thresholded at all,
  # so the summary line is the assertion.
  PASS_REGULAR_EXPRESSION "separation margin 0\\.[1-9]")

# --- The committed data is usable, not merely present ------------------------
add_test(NAME validate-data
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/validate_data.py"
          --gallery "${RFD_DATA}" --json "${RFD_GALLERY}")
set_tests_properties(validate-data PROPERTIES
  LABELS "fast"
  FIXTURES_REQUIRED gallery
  PASS_REGULAR_EXPRESSION "PASSED")

# --- Pipeline checks ---------------------------------------------------------
# The element is loaded from the build tree, so GStreamer has to be told where
# to look; this is what makes gst-inspect and gst-launch work here.
set(RFD_CHECK_ENV "GST_PLUGIN_PATH=${CMAKE_BINARY_DIR}/source")

# Instrumented code is several times slower, so the capacity checks skip their
# absolute floors.
if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=")
  list(APPEND RFD_CHECK_ENV "RFD_SANITIZED=1")
endif()

# Stock gst-inspect-1.0/gst-launch-1.0 cannot dlopen an ASan-instrumented plugin
# ("ASan runtime does not come first"), so check_pipeline.py preloads the runtime
# into them. Resolved here, where the compiler is known.
if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=[^ ]*address")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_rfd_asan_soname "libasan.so")
  else()
    set(_rfd_asan_soname "libclang_rt.asan-${CMAKE_SYSTEM_PROCESSOR}.so")
  endif()

  # -print-file-name echoes the name back unchanged when it cannot resolve it.
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=${_rfd_asan_soname}"
    OUTPUT_VARIABLE _rfd_asan_runtime
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

  if(NOT IS_ABSOLUTE "${_rfd_asan_runtime}")
    message(FATAL_ERROR
      "cannot locate ${_rfd_asan_soname} for ${CMAKE_CXX_COMPILER}")
  endif()

  message(STATUS "ASan runtime for the stock gst tools: ${_rfd_asan_runtime}")
  list(APPEND RFD_CHECK_ENV "RFD_ASAN_PRELOAD=${_rfd_asan_runtime}")
endif()

# One CTest test per check, so a failure names the property that broke rather
# than just "the pipeline test failed".
set(RFD_PIPELINE_CHECKS plugin gst-launch caps-change throughput open-set degradation realtime)

foreach(check IN LISTS RFD_PIPELINE_CHECKS)
  add_test(NAME "pipeline-${check}"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/check_pipeline.py"
            --build-dir "${CMAKE_BINARY_DIR}" --gallery "${RFD_GALLERY}"
            --only "${check}")
  set_tests_properties("pipeline-${check}" PROPERTIES
    FIXTURES_REQUIRED gallery
    PASS_REGULAR_EXPRESSION "PASSED"
    ENVIRONMENT "${RFD_CHECK_ENV}")
endforeach()

# realtime streams five clips at source rate, so it cannot be fast by definition.
set_tests_properties(pipeline-realtime PROPERTIES TIMEOUT 240 LABELS "slow")
set_tests_properties(pipeline-degradation PROPERTIES LABELS "slow")
set_tests_properties(pipeline-plugin pipeline-gst-launch pipeline-caps-change
                     pipeline-throughput pipeline-open-set PROPERTIES LABELS "fast")

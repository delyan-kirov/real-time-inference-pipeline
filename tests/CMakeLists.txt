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
    # The element is loaded from the build tree, so GStreamer has to be told
    # where to look; this is what makes gst-inspect and gst-launch work here.
    ENVIRONMENT "GST_PLUGIN_PATH=${CMAKE_BINARY_DIR}/source")
endforeach()

# realtime streams five clips at source rate, so it cannot be fast by definition.
set_tests_properties(pipeline-realtime PROPERTIES TIMEOUT 240 LABELS "slow")
set_tests_properties(pipeline-degradation PROPERTIES LABELS "slow")
set_tests_properties(pipeline-plugin pipeline-gst-launch pipeline-caps-change
                     pipeline-throughput pipeline-open-set PROPERTIES LABELS "fast")

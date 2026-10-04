/**
 * @file smoke.cpp
 * @brief Workspace smoke test.
 *
 * Proves four things:
 *
 *   1. We compiled and linked against GStreamer at all.
 *   2. The runtime GStreamer matches the headers we built against.
 *   3. The plugin registry is populated - under Nix, plugins live in separate
 *      store paths and are only found via GST_PLUGIN_SYSTEM_PATH_1_0.
 *   4. The vendored YuNet and SFace weights load into OpenCV's DNN backend.
 *
 * It builds `videotestsrc num-buffers=N ! videoconvert ! fakesink`, runs it to
 * EOS, and reports. That exercises plugin discovery, element linking, state
 * changes and bus handling - the same machinery the real pipeline uses.
 */

#include <gst/gst.h>

#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/objdetect/face.hpp>
#include <string>

#include "util.hpp"

/// RFD's vocabulary (Str, Vec, UInt, ...) is used unqualified in this file;
/// see util.hpp.
using namespace RFD;

namespace {

/// Frames the smoke pipeline pushes before EOS.
constexpr int SMOKE_TEST_FRAMES = 30;

/// Prints the compiled-against and running-against GStreamer versions.
void report_versions() {
    guint major = 0, minor = 0, micro = 0, nano = 0;
    gst_version(&major, &minor, &micro, &nano);

    std::cout << "  compiled against : " << GST_VERSION_MAJOR << '.' << GST_VERSION_MINOR << '.'
              << GST_VERSION_MICRO << '\n'
              << "  running against  : " << major << '.' << minor << '.' << micro << '\n';

    if (major != GST_VERSION_MAJOR || minor != GST_VERSION_MINOR) {
        std::cout << "  WARNING: header/runtime version mismatch\n";
    }
}

/// Prints how many plugins the registry found, warning when it is empty.
void report_plugin_registry() {
    GList* plugins = gst_registry_get_plugin_list(gst_registry_get());
    const guint count = g_list_length(plugins);
    gst_plugin_list_free(plugins);

    std::cout << "  plugins found    : " << count << '\n';
    if (count == 0) {
        std::cout << "  WARNING: empty registry - is GST_PLUGIN_SYSTEM_PATH_1_0 set?\n";
    }
}

/**
 * @brief Imports both ONNX graphs. Paths come from CMake (see rfd_models).
 * @return True when both weight files loaded.
 */
bool check_models() {
    std::cout << "  opencv           : " << CV_VERSION << '\n';

    try {
        // Input size is a placeholder; the real pipeline sets it per negotiated
        // caps. Only the graph import is under test here.
        const auto detector = cv::FaceDetectorYN::create(RFD_YUNET_MODEL, "", cv::Size(320, 320));
        if (detector == nullptr) {
            std::cerr << "  YuNet            : create returned null\n";
            return false;
        }
        std::cout << "  yunet            : loaded\n";

        const auto recognizer = cv::FaceRecognizerSF::create(RFD_SFACE_MODEL, "");
        if (recognizer == nullptr) {
            std::cerr << "  SFace            : create returned null\n";
            return false;
        }
        std::cout << "  sface            : loaded\n";
    } catch (const cv::Exception& e) {
        std::cerr << "  failed to load model weights: " << e.what() << '\n'
                  << "  check external/models/ against its SHA256SUMS\n";
        return false;
    }

    return true;
}

/**
 * @brief Runs a trivial pipeline to EOS.
 * @return True on clean completion.
 */
bool run_smoke_pipeline() {
    const Str description =
        "videotestsrc pattern=ball num-buffers=" + std::to_string(SMOKE_TEST_FRAMES) +
        " ! video/x-raw,width=640,height=480,framerate=30/1"
        " ! videoconvert"
        " ! fakesink sync=false";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(description.c_str(), &error);
    if (pipeline == nullptr) {
        std::cerr << "  failed to build pipeline: "
                  << (error != nullptr ? error->message : "unknown") << '\n';
        g_clear_error(&error);
        return false;
    }

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "  failed to start pipeline\n";
        gst_object_unref(pipeline);
        return false;
    }

    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* message = gst_bus_timed_pop_filtered(
        bus, 10 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));

    bool ok = false;
    if (message == nullptr) {
        std::cerr << "  pipeline timed out\n";
    } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError* err = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(message, &err, &debug);
        std::cerr << "  pipeline error: " << err->message << '\n';
        if (debug != nullptr) {
            std::cerr << "  debug: " << debug << '\n';
        }
        g_clear_error(&err);
        g_free(debug);
    } else {
        ok = true;
    }

    if (message != nullptr) {
        gst_message_unref(message);
    }

    // Always walk back down to NULL before dropping the last reference,
    // otherwise elements can leak their allocated resources.
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(pipeline);
    return ok;
}

}  // namespace

/**
 * @brief Reports the environment, then runs the smoke pipeline.
 * @param argc Argument count; forwarded to gst_init.
 * @param argv Argument vector; forwarded to gst_init.
 * @return EXIT_SUCCESS when the weights load and the pipeline reaches EOS.
 */
int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    std::cout << "rfd - real-time face detection pipeline\n"
              << "environment check\n";
    report_versions();
    report_plugin_registry();
    const bool models_ok = check_models();

    std::cout << "smoke pipeline (" << SMOKE_TEST_FRAMES << " frames)\n";
    const bool pipeline_ok = run_smoke_pipeline();
    const bool ok = models_ok && pipeline_ok;
    std::cout << (ok ? "  ok - smoke test passed\n" : "  FAILED\n");

    gst_deinit();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

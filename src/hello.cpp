// ---------------------------------------------------------------------------
// Workspace smoke test:
//
//   1. We compiled and linked against GStreamer at all.
//   2. The runtime GStreamer matches the headers we built against.
//   3. The plugin registry is populated - under Nix, plugins live in separate
//      store paths and are only found via GST_PLUGIN_SYSTEM_PATH_1_0. A pipeline
//      that fails to parse here means the dev shell is misconfigured, not that
//      the code is wrong.
//
// It builds `videotestsrc num-buffers=N ! videoconvert ! fakesink`, runs it to
// EOS, and reports. That exercises plugin discovery, element linking, state
// changes and bus handling - the same machinery the real pipeline will use.
// ---------------------------------------------------------------------------

#include <gst/gst.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

constexpr int kSmokeTestFrames = 30;

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

void report_plugin_registry() {
    GList* plugins = gst_registry_get_plugin_list(gst_registry_get());
    const guint count = g_list_length(plugins);
    gst_plugin_list_free(plugins);

    std::cout << "  plugins found    : " << count << '\n';
    if (count == 0) {
        std::cout << "  WARNING: empty registry - is GST_PLUGIN_SYSTEM_PATH_1_0 set?\n";
    }
}

// Run a trivial pipeline to EOS. Returns true on clean completion.
bool run_smoke_pipeline() {
    const std::string description =
        "videotestsrc pattern=ball num-buffers=" + std::to_string(kSmokeTestFrames) +
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

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    std::cout << "rtip - real-time inference pipeline\n"
              << "environment check\n";
    report_versions();
    report_plugin_registry();

    std::cout << "smoke pipeline (" << kSmokeTestFrames << " frames)\n";
    const bool ok = run_smoke_pipeline();
    std::cout << (ok ? "  ok - hello, world\n" : "  FAILED\n");

    gst_deinit();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

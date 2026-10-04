/**
 * @file plugin.cpp
 * @brief Plugin entry point, so rfdface is a real GStreamer plugin and not
 *        merely a class our own binary happens to register.
 *
 * That distinction is the point: with libgstrfd.so on GST_PLUGIN_PATH,
 * `gst-inspect-1.0 rfdface` documents the element and
 * `gst-launch-1.0 ... ! rfdface ! fakesink` runs inference with no code of ours
 * driving it. An element that only works inside its own application has not
 * really joined the pipeline.
 *
 * The same translation unit also exposes rfd_register_static(), because the
 * application links the element directly and should not depend on finding a
 * shared object at runtime.
 */

#include <gst/gst.h>

#include "face_filter.hpp"

/// Plugin version string, normally supplied by CMake.
#ifndef RFD_VERSION
#define RFD_VERSION "0.1.0"
#endif

// GST_PLUGIN_DEFINE expects PACKAGE, which autotools-built plugins get from
// config.h. We build with CMake, so define it here.
#ifndef PACKAGE
#define PACKAGE "rfd"
#endif

/**
 * @brief GstPluginInitFunc: registers every element this plugin provides.
 *
 * GST_RANK_NONE: autoplugging must never insert face recognition into a pipeline
 * on its own. It is only ever used when asked for by name.
 *
 * @param plugin The plugin being loaded.
 * @return TRUE if registration succeeded.
 */
static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "rfdface", GST_RANK_NONE, GST_TYPE_RFD_FACE);
}

extern "C" {

/**
 * @brief Registers the element without a plugin file, for the in-process case.
 * @return TRUE if registration succeeded.
 */
gboolean rfd_register_static(void);

gboolean rfd_register_static(void) {
    return gst_element_register(nullptr, "rfdface", GST_RANK_NONE, GST_TYPE_RFD_FACE);
}
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, rfd, "Real-time face recognition elements",
                  plugin_init, RFD_VERSION, "LGPL", "rfd", "https://github.com/delyan-kirov")

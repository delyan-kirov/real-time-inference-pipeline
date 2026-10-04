/**
 * @file face_filter.hpp
 * @brief rfdface - a GStreamer element that recognises faces in the frames
 *        passing through it and attaches the results to those same buffers.
 *
 * This is a GstVideoFilter:
 *
 *   - it composes - `gst-launch-1.0 ... ! rfdface ! fakesink` works, and any
 *     downstream element can read the metadata without knowing we exist;
 *   - backpressure is GStreamer's problem, not ours. An upstream
 *     `queue leaky=downstream` drops frames when we cannot keep up, which is how
 *     the pipeline degrades gracefully instead of stalling or growing unbounded;
 *   - caps renegotiation arrives as `set_info`, so a mid-stream resolution
 *     change is an event we handle rather than an assumption we baked in at
 *     startup.
 *
 * Transform is *in place*: we annotate buffers, we never alter pixels. That
 * keeps us out of the buffer-copy path entirely.
 *
 * @see face_meta.hpp for the metadata this element attaches.
 */

#pragma once

#include <gst/video/gstvideofilter.h>

G_BEGIN_DECLS

/// The GType of the rfdface element, for gst_element_register().
#define GST_TYPE_RFD_FACE (gst_rfd_face_get_type())

/**
 * @brief Declares GstRfdFace, its class struct, cast macros and get_type().
 *
 * Final: the element is not designed to be subclassed
 */
G_DECLARE_FINAL_TYPE(GstRfdFace, gst_rfd_face, GST, RFD_FACE, GstVideoFilter)

G_END_DECLS

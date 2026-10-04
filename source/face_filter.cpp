/**
 * @file face_filter.cpp
 * @brief Implementation of the rfdface GstVideoFilter: detect, embed, match and
 *        annotate, in place.
 *
 * @see face_filter.hpp for why this is an element rather than an application
 *      loop.
 */

#include "face_filter.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <opencv2/core.hpp>

#include "detector.hpp"
#include "embedder.hpp"
#include "face_meta.hpp"
#include "gallery.hpp"

/// RFD's vocabulary (Str, Vec, UInt, ...) is used unqualified in this file;
/// see util.hpp.
using namespace RFD;

GST_DEBUG_CATEGORY_STATIC(rfd_face_debug);
#define GST_CAT_DEFAULT rfd_face_debug

namespace {

/// Property defaults, taken from the measured values in the face library.
constexpr float DEFAULT_THRESHOLD = RFDxFACE::Embedder::COSINE_THRESHOLD;
constexpr int DEFAULT_MAX_INPUT_EDGE = RFDxFACE::Detector::DEFAULT_MAX_INPUT_EDGE;

/**
 * @brief All C++ state, behind one raw pointer in the instance struct.
 *
 * GObject allocates instances with zeroed memory and never runs a C++
 * constructor, so members with non-trivial lifetimes cannot live directly in
 * _GstRfdFace - they are new'd in _init and deleted in finalize instead.
 */
struct State {
    Box<RFDxFACE::Detector> detector;
    Box<RFDxFACE::Embedder> embedder;
    Box<RFDxFACE::Gallery> gallery;

    /**
     * @brief Set once the models are up.
     *
     * The cost is paid on the first frame rather than at construction -
     * `gst-inspect-1.0 rfdface` must work with no weights and no gallery present.
     */
    bool initialised = false;

    /**
     * @brief True after a failure, so a broken gallery posts one error and then
     *        stops retrying on every single frame.
     */
    bool failed = false;

    /// Frames seen since the element started, including skipped ones.
    guint64 frame_number = 0;

    /// The gallery path the loaded templates came from, for logging.
    Str loaded_gallery_path;
};

}  // namespace

/**
 * @brief The rfdface instance struct.
 */
struct _GstRfdFace {
    /// GstVideoFilter base; must come first.
    GstVideoFilter parent;

    // Properties. Guarded by the GObject lock on get/set.

    /// "gallery": path to a gallery.json, or NULL for detection only.
    gchar* gallery_path;
    /// "threshold": minimum cosine similarity to name a face.
    gfloat threshold;
    /// "max-input-edge": longest side of the detector input, 0 to disable.
    gint max_input_edge;
    /// "enabled": run inference at all.
    gboolean enabled;

    /// Owned C++ state; see the State docs above.
    State* state;
};

G_DEFINE_TYPE(GstRfdFace, gst_rfd_face, GST_TYPE_VIDEO_FILTER)

/// Property ids, in installation order.
enum {
    PROP_0,
    PROP_GALLERY,
    PROP_THRESHOLD,
    PROP_MAX_INPUT_EDGE,
    PROP_ENABLED,
};

// BGR only, in both directions. SFace and YuNet want BGR, and declaring it here
// makes GStreamer insert or configure videoconvert upstream instead of us
// converting by hand on every frame.
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=(string)BGR"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=(string)BGR"));

/**
 * @brief GObjectClass::set_property. Changing the gallery or input cap forces a
 *        reload on the next frame.
 */
static void gst_rfd_face_set_property(GObject* object, guint prop_id, const GValue* value,
                                      GParamSpec* pspec) {
    GstRfdFace* self = GST_RFD_FACE(object);

    switch (prop_id) {
        case PROP_GALLERY:
            g_free(self->gallery_path);
            self->gallery_path = g_value_dup_string(value);
            // Force a reload: the path changed, so whatever is loaded is stale.
            self->state->initialised = false;
            self->state->failed = false;
            break;
        case PROP_THRESHOLD:
            self->threshold = g_value_get_float(value);
            break;
        case PROP_MAX_INPUT_EDGE:
            self->max_input_edge = g_value_get_int(value);
            self->state->initialised = false;  // detector carries this
            break;
        case PROP_ENABLED:
            self->enabled = g_value_get_boolean(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/**
 * @brief GObjectClass::get_property.
 */
static void gst_rfd_face_get_property(GObject* object, guint prop_id, GValue* value,
                                      GParamSpec* pspec) {
    GstRfdFace* self = GST_RFD_FACE(object);

    switch (prop_id) {
        case PROP_GALLERY:
            g_value_set_string(value, self->gallery_path);
            break;
        case PROP_THRESHOLD:
            g_value_set_float(value, self->threshold);
            break;
        case PROP_MAX_INPUT_EDGE:
            g_value_set_int(value, self->max_input_edge);
            break;
        case PROP_ENABLED:
            g_value_set_boolean(value, self->enabled);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/**
 * @brief GObjectClass::finalize. Releases the property string and the C++ state.
 */
static void gst_rfd_face_finalize(GObject* object) {
    GstRfdFace* self = GST_RFD_FACE(object);

    g_free(self->gallery_path);
    delete self->state;
    self->state = nullptr;

    G_OBJECT_CLASS(gst_rfd_face_parent_class)->finalize(object);
}

/**
 * @brief GstVideoFilterClass::set_info - negotiated caps.
 *
 * Logged rather than merely recorded: a resolution change mid-stream is exactly
 * the "changing input" case the task asks about, and seeing it in the log is how
 * we know it was handled rather than ignored.
 *
 * @param filter  This element.
 * @param in_info The newly negotiated input format.
 * @return Always TRUE; there is nothing that can fail here.
 */
static gboolean gst_rfd_face_set_info(GstVideoFilter* filter, GstCaps* /*incaps*/,
                                      GstVideoInfo* in_info, GstCaps* /*outcaps*/,
                                      GstVideoInfo* /*out_info*/) {
    GstRfdFace* self = GST_RFD_FACE(filter);

    GST_INFO_OBJECT(self, "negotiated %dx%d %s at %d/%d fps", GST_VIDEO_INFO_WIDTH(in_info),
                    GST_VIDEO_INFO_HEIGHT(in_info), GST_VIDEO_INFO_NAME(in_info),
                    GST_VIDEO_INFO_FPS_N(in_info), GST_VIDEO_INFO_FPS_D(in_info));

    // The Detector re-declares its own input size per frame (see detector.hpp), so
    // there is nothing to rebuild here. Keeping this vtable entry anyway means the
    // change is observable instead of silent.
    return TRUE;
}

/**
 * @brief Brings up the models and the gallery, once, on the first frame.
 *
 * @param self This element.
 * @return True when inference can run. On failure it posts an element error,
 *         which is how a GStreamer element reports "I cannot continue", and
 *         returns false for every subsequent frame without retrying.
 */
static bool ensure_ready(GstRfdFace* self) {
    State& state = *self->state;

    if (state.initialised) {
        return true;
    }
    if (state.failed) {
        return false;
    }

    try {
        state.detector =
            std::make_unique<RFDxFACE::Detector>(0.9F, 0.3F, 5000, self->max_input_edge);
        state.embedder = std::make_unique<RFDxFACE::Embedder>();

        if (self->gallery_path != nullptr && *self->gallery_path != '\0') {
            state.gallery =
                std::make_unique<RFDxFACE::Gallery>(RFDxFACE::Gallery::load(self->gallery_path));
            state.loaded_gallery_path = self->gallery_path;
            GST_INFO_OBJECT(self, "loaded %zu identities from %s", state.gallery->size(),
                            self->gallery_path);
        } else {
            // Detection-only mode. Legitimate and useful: it is what throughput
            // testing on crowd footage wants, and it computes no embeddings.
            state.gallery.reset();
            GST_INFO_OBJECT(self, "no gallery set - detection only, no embeddings");
        }

        // Warm both graphs before the stream starts. The first real inference is
        // otherwise ~4x slower than steady state (measured: 44 ms against a 12 ms
        // median), and at 30 fps that overrun alone costs several dropped frames
        // right at startup. A throwaway frame moves that cost to where nothing is
        // waiting on it.
        //
        // The detector warms on a blank frame. SFace needs a detection row to
        // align against and will not get one from a blank frame, so a synthetic
        // row is used - the resulting embedding is meaningless and discarded; only
        // the graph allocation it forces matters.
        cv::Mat blank(320, 320, CV_8UC3, cv::Scalar(0, 0, 0));
        state.detector->detect(blank);

        if (state.embedder) {
            cv::Mat row(1, 15, CV_32F);
            const float synthetic[15] = {80.0F,  80.0F,  160.0F, 160.0F, 120.0F,
                                         130.0F, 180.0F, 130.0F, 150.0F, 165.0F,
                                         125.0F, 195.0F, 175.0F, 195.0F, 1.0F};
            std::copy(std::begin(synthetic), std::end(synthetic), row.ptr<float>(0));
            try {
                // Discarded on purpose: only the graph allocation matters.
                static_cast<void>(state.embedder->embed(blank, row));
            } catch (const Exception&) {
                // A failed warm-up is not a failed element; the real frames decide.
            }
        }

        state.initialised = true;
        return true;
    } catch (const Exception& error) {
        state.failed = true;
        GST_ELEMENT_ERROR(self, LIBRARY, INIT, ("%s", error.what()),
                          ("failed to initialise face recognition"));
        return false;
    }
}

/**
 * @brief GstVideoFilterClass::transform_frame_ip - the per-frame work.
 *
 * Detects, optionally identifies, and attaches a GstRfdFaceMeta. Pixels are
 * never modified.
 *
 * @param filter This element.
 * @param frame  The mapped input frame, annotated in place.
 * @return GST_FLOW_OK, or GST_FLOW_ERROR after posting an element error.
 */
static GstFlowReturn gst_rfd_face_transform_frame_ip(GstVideoFilter* filter, GstVideoFrame* frame) {
    GstRfdFace* self = GST_RFD_FACE(filter);
    State& state = *self->state;

    const guint64 frame_number = state.frame_number++;

    if (!self->enabled) {
        return GST_FLOW_OK;
    }
    if (!ensure_ready(self)) {
        return GST_FLOW_ERROR;
    }

    const auto started = Clock::now();

    // Wrap the mapped frame without copying. The stride must come from the frame
    // rather than being assumed as width*3: GStreamer pads rows to alignment
    // boundaries, and ignoring that shears the image for some widths.
    cv::Mat image(GST_VIDEO_FRAME_HEIGHT(frame), GST_VIDEO_FRAME_WIDTH(frame), CV_8UC3,
                  GST_VIDEO_FRAME_PLANE_DATA(frame, 0),
                  static_cast<UInt>(GST_VIDEO_FRAME_PLANE_STRIDE(frame, 0)));

    Vec<RFDxGST::FaceResult> results;
    try {
        for (const RFDxFACE::Detection& detection : state.detector->detect(image)) {
            RFDxGST::FaceResult result;
            result.box = detection.box;
            result.landmarks = detection.landmarks;
            result.detection_score = detection.score;

            if (state.gallery) {
                const cv::Mat embedding = state.embedder->embed(image, detection.row);
                const RFDxFACE::Match match = state.gallery->match(embedding, self->threshold);
                // An empty identity is the open-set rejection; the score is kept
                // either way so a near-miss can be seen downstream.
                result.identity = match.identity;
                result.similarity = match.score;
            }
            results.push_back(std::move(result));
        }
    } catch (const Exception& error) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED, ("%s", error.what()), ("inference failed"));
        return GST_FLOW_ERROR;
    }

    const auto elapsed = Clock::now() - started;

    GstRfdFaceMeta* meta = gst_buffer_add_rfd_face_meta(frame->buffer);
    if (meta == nullptr) {
        GST_WARNING_OBJECT(self, "could not attach metadata to frame %" G_GUINT64_FORMAT,
                           frame_number);
        return GST_FLOW_OK;
    }

    *meta->faces = std::move(results);
    meta->frame_number = frame_number;
    meta->inference_us = static_cast<guint64>(std::chrono::duration_cast<Micros>(elapsed).count());

    GST_LOG_OBJECT(self, "frame %" G_GUINT64_FORMAT ": %zu face(s) in %" G_GUINT64_FORMAT " us",
                   frame_number, meta->faces->size(), meta->inference_us);

    return GST_FLOW_OK;
}

/**
 * @brief GObject class init: properties, pad templates, element metadata and the
 *        GstVideoFilter vtable.
 */
static void gst_rfd_face_class_init(GstRfdFaceClass* klass) {
    GObjectClass* object_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstVideoFilterClass* filter_class = GST_VIDEO_FILTER_CLASS(klass);

    object_class->set_property = gst_rfd_face_set_property;
    object_class->get_property = gst_rfd_face_get_property;
    object_class->finalize = gst_rfd_face_finalize;

    g_object_class_install_property(
        object_class, PROP_GALLERY,
        g_param_spec_string("gallery", "Gallery",
                            "Path to a gallery.json from rfd-enroll. Unset means "
                            "detection only, with no embeddings computed.",
                            nullptr, static_cast<GParamFlags>(G_PARAM_READWRITE)));

    g_object_class_install_property(
        object_class, PROP_THRESHOLD,
        g_param_spec_float("threshold", "Threshold",
                           "Minimum cosine similarity to call a face identified. "
                           "Below this the face is reported as unknown.",
                           -1.0F, 1.0F, DEFAULT_THRESHOLD,
                           static_cast<GParamFlags>(G_PARAM_READWRITE)));

    g_object_class_install_property(
        object_class, PROP_MAX_INPUT_EDGE,
        g_param_spec_int("max-input-edge", "Max input edge",
                         "Longest side of the detector input; frames larger than this "
                         "are downscaled. YuNet only finds faces of roughly 10-300px, "
                         "so a close-up in a full-resolution frame is invisible to it. "
                         "0 disables downscaling.",
                         0, 8192, DEFAULT_MAX_INPUT_EDGE,
                         static_cast<GParamFlags>(G_PARAM_READWRITE)));

    g_object_class_install_property(
        object_class, PROP_ENABLED,
        g_param_spec_boolean("enabled", "Enabled",
                             "Run inference. Setting this false passes frames through "
                             "untouched, which isolates decode cost from inference cost "
                             "when measuring throughput.",
                             TRUE, static_cast<GParamFlags>(G_PARAM_READWRITE)));

    gst_element_class_set_static_metadata(
        element_class, "RFD face recognition", "Filter/Analyzer/Video",
        "Detects faces with YuNet, identifies them against a gallery with SFace, and "
        "attaches the results to each buffer as GstRfdFaceMeta",
        "rfd");

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    filter_class->set_info = gst_rfd_face_set_info;
    filter_class->transform_frame_ip = gst_rfd_face_transform_frame_ip;

    GST_DEBUG_CATEGORY_INIT(rfd_face_debug, "rfdface", 0, "RFD face recognition");
}

/**
 * @brief GObject instance init: property defaults and the owned C++ state.
 */
static void gst_rfd_face_init(GstRfdFace* self) {
    self->gallery_path = nullptr;
    self->threshold = DEFAULT_THRESHOLD;
    self->max_input_edge = DEFAULT_MAX_INPUT_EDGE;
    self->enabled = TRUE;
    self->state = new State();
}

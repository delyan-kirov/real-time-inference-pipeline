/**
 * @file face_meta.hpp
 * @brief GstRfdFaceMeta - inference results
 *
 */

#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>

#include "detector.hpp"
#include "util.hpp"

/**
 * @namespace RFDxGST
 * @brief The GStreamer element and the per-buffer metadata it attaches.
 */
namespace RFDxGST {

/// RFD's vocabulary (Str, Vec, UInt, ...) is used unqualified throughout this
/// module; see util.hpp.
using namespace RFD;

/**
 * @brief One recognised (or rejected) face.
 */
struct FaceResult {
    /// Bounding box in frame coordinates.
    cv::Rect2f box;

    /// The five YuNet landmarks, in frame coordinates.
    Arr<cv::Point2f, RFDxFACE::LANDMARK_COUNT> landmarks{};

    /// Detector confidence, in 0..1.
    float detection_score = 0.0F;

    /**
     * @brief The matched identity, empty for an open-set rejection.
     *
     * Empty when no gallery template cleared the threshold. @ref similarity is
     * still the best score seen, so a near-miss can be logged rather than
     * silently discarded.
     */
    Str identity;

    /// Cosine similarity to the best gallery template, reported either way.
    float similarity = 0.0F;

    /// @return True when a gallery template cleared the threshold.
    [[nodiscard]] bool identified() const {
        return !identity.empty();
    }
};

}  // namespace RFDxGST

/**
 * @brief The C-facing metadata struct attached to each analysed buffer.
 *
 * GstMeta is a C API, so the struct is plain and the C++ vector lives behind a
 * pointer the meta owns and frees.
 */
struct GstRfdFaceMeta {
    /// GStreamer's own metadata header; must come first.
    GstMeta meta;

    /// Owned by this meta; allocated in init, destroyed in free.
    RFD::Vec<RFDxGST::FaceResult>* faces;

    /**
     * @brief Frame index since the stream started.
     *
     * So JSON output is orderable even after frames have been dropped upstream.
     */
    guint64 frame_number;

    /**
     * @brief Wall-clock microseconds spent in inference for this frame.
     *
     * Needed for the latency half of the benchmark; measuring it anywhere else
     * would miss the queueing that is the interesting part.
     */
    guint64 inference_us;
};

G_BEGIN_DECLS

/**
 * @brief Registers (once) and returns the metadata API GType.
 * @return The GType for GstRfdFaceMetaAPI.
 */
GType gst_rfd_face_meta_api_get_type(void);

/// The metadata API GType, for gst_buffer_get_meta().
#define GST_RFD_FACE_META_API_TYPE (gst_rfd_face_meta_api_get_type())

/**
 * @brief Registers (once) and returns the metadata implementation info.
 * @return The GstMetaInfo describing GstRfdFaceMeta.
 */
const GstMetaInfo* gst_rfd_face_meta_get_info(void);

/**
 * @brief Fetches this element's metadata from a buffer.
 * @param b The GstBuffer to read.
 * @return The attached GstRfdFaceMeta, or NULL if the buffer carries none.
 */
#define gst_buffer_get_rfd_face_meta(b) \
    ((GstRfdFaceMeta*)gst_buffer_get_meta((b), GST_RFD_FACE_META_API_TYPE))

/**
 * @brief Adds (or replaces) the metadata on a buffer.
 * @param buffer A writable GstBuffer.
 * @return The new metadata, or NULL if it could not be attached.
 */
GstRfdFaceMeta* gst_buffer_add_rfd_face_meta(GstBuffer* buffer);

G_END_DECLS

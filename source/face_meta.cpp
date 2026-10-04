/**
 * @file face_meta.cpp
 * @brief Registration and lifecycle of GstRfdFaceMeta.
 *
 * GstMeta memory is allocated by GStreamer rather than new'd, so the owned C++
 * vector is constructed in init and destroyed in free instead of relying on
 * constructors running.
 */

#include "face_meta.hpp"

#include <new>

/// RFD's vocabulary (Str, Vec, UInt, ...) is used unqualified in this file;
/// see util.hpp.
using namespace RFD;

namespace {

/**
 * @brief GstMetaInitFunction: constructs the owned vector and zeroes the counters.
 * @param meta The metadata being initialised.
 * @return TRUE on success, FALSE if the vector could not be allocated.
 */
gboolean rfd_face_meta_init(GstMeta* meta, gpointer /*params*/, GstBuffer* /*buffer*/) {
    auto* self = reinterpret_cast<GstRfdFaceMeta*>(meta);

    // GstMeta memory is allocated by GStreamer, not new'd, so the vector has to
    // be constructed explicitly rather than relying on a constructor running.
    self->faces = new (std::nothrow) Vec<RFDxGST::FaceResult>();
    self->frame_number = 0;
    self->inference_us = 0;

    return self->faces != nullptr;
}

/**
 * @brief GstMetaFreeFunction: destroys the owned vector.
 * @param meta The metadata being freed.
 */
void rfd_face_meta_free(GstMeta* meta, GstBuffer* /*buffer*/) {
    auto* self = reinterpret_cast<GstRfdFaceMeta*>(meta);
    delete self->faces;
    self->faces = nullptr;
}

/**
 * @brief GstMetaTransformFunction: copies the metadata onto another buffer.
 * @param dest The destination buffer.
 * @param meta The source metadata.
 * @param type The transform being attempted.
 * @return TRUE if the metadata was copied, FALSE for any transform but a
 *         straight copy.
 */
gboolean rfd_face_meta_transform(GstBuffer* dest, GstMeta* meta, GstBuffer* /*buffer*/, GQuark type,
                                 gpointer /*data*/) {
    const auto* source = reinterpret_cast<GstRfdFaceMeta*>(meta);

    // Only a straight copy is meaningful. Under a scale or crop the boxes would
    // need remapping, and silently copying stale coordinates is worse than
    // refusing - the caller then knows the metadata did not survive.
    if (!GST_META_TRANSFORM_IS_COPY(type)) {
        return FALSE;
    }

    GstRfdFaceMeta* copy = gst_buffer_add_rfd_face_meta(dest);
    if (copy == nullptr) {
        return FALSE;
    }
    *copy->faces = *source->faces;
    copy->frame_number = source->frame_number;
    copy->inference_us = source->inference_us;
    return TRUE;
}

}  // namespace

GType gst_rfd_face_meta_api_get_type(void) {
    static GType type = 0;
    static const gchar* tags[] = {GST_META_TAG_VIDEO_STR, nullptr};

    if (g_once_init_enter(&type)) {
        const GType registered = gst_meta_api_type_register("GstRfdFaceMetaAPI", tags);
        g_once_init_leave(&type, registered);
    }
    return type;
}

const GstMetaInfo* gst_rfd_face_meta_get_info(void) {
    static const GstMetaInfo* info = nullptr;

    if (g_once_init_enter(&info)) {
        const GstMetaInfo* registered =
            gst_meta_register(GST_RFD_FACE_META_API_TYPE, "GstRfdFaceMeta", sizeof(GstRfdFaceMeta),
                              rfd_face_meta_init, rfd_face_meta_free, rfd_face_meta_transform);
        g_once_init_leave(&info, registered);
    }
    return info;
}

GstRfdFaceMeta* gst_buffer_add_rfd_face_meta(GstBuffer* buffer) {
    g_return_val_if_fail(GST_IS_BUFFER(buffer), nullptr);
    g_return_val_if_fail(gst_buffer_is_writable(buffer), nullptr);

    return reinterpret_cast<GstRfdFaceMeta*>(
        gst_buffer_add_meta(buffer, gst_rfd_face_meta_get_info(), nullptr));
}

/**
 * @file embedder.hpp
 * @brief SFace embeddings.
 *
 */

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect/face.hpp>

#include "util.hpp"

namespace RFDxFACE {

/// RFD's vocabulary (Str, Vec, UInt, ...) is used unqualified throughout this
/// module; see util.hpp.
using namespace RFD;

/**
 * @brief Turns a detected face into a unit-length 128-D SFace embedding.
 */
class Embedder {
   public:
    /**
     * @brief Loads the vendored SFace weights.
     * @throws RuntimeError if the weights cannot be loaded.
     */
    Embedder();

    /**
     * @brief Aligns and embeds one detected face.
     *
     * Normalising here, rather than relying on FaceRecognizerSF::match() to do
     * it internally, buys two things: cosine similarity becomes a plain dot
     * product, and averaging several embeddings into one gallery template is
     * well defined (mean of unit vectors, renormalised).
     *
     * @param frame         The full-resolution frame the detection came from.
     * @param detection_row The raw Detection::row from the Detector.
     * @return A 1x128 CV_32F row, L2-normalised.
     * @throws RuntimeError if SFace returns an unexpected feature size.
     */
    [[nodiscard]] cv::Mat embed(const cv::Mat& frame, const cv::Mat& detection_row) const;

    /// Length of an SFace embedding.
    static constexpr int DIMENSIONS = 128;

    /// OpenCV's documented SFace cosine operating point, kept for reference.
    static constexpr float UPSTREAM_COSINE_THRESHOLD = 0.363F;

    /// OpenCV's documented SFace L2 operating point, kept for reference.
    static constexpr float UPSTREAM_L2_THRESHOLD = 1.128F;

    /**
     * @brief The threshold this project actually uses, measured rather than
     *        inherited.
     *
     * tools/validate_data.py matches every probe clip against the gallery twice:
     * once normally, and once with that clip's own subject removed. Over the five
     * clips that gives a genuine-score floor of 0.420 and an impostor-score
     * ceiling of 0.355, so a threshold in 0.355..0.420 separates them. 0.388 is
     * the midpoint, balancing false accepts against false rejects rather than
     * hugging one edge.
     *
     * Upstream's 0.363 does work on this data but sits only 0.008 above the
     * impostor ceiling - one blurrier frame from a false accept.
     *
     * The window is only 0.065 wide, which is worth knowing: this gallery is five
     * older white men, so inter-class similarity is high and there is less room
     * than a more varied gallery would give.
     *
     * @note This number is a property of *this gallery and footage*, not of
     *       SFace. Re-run the validator after changing either; it prints the
     *       window and the midpoint.
     */
    static constexpr float COSINE_THRESHOLD = 0.388F;

    /**
     * @brief sha256 of the weights these embeddings come from.
     *
     * Baked in by CMake from external/models/SHA256SUMS. Written into every
     * gallery file so a gallery built with different weights can be rejected
     * instead of silently mismatching - embeddings are only comparable within
     * one model.
     */
    static constexpr const char* MODEL_DIGEST = RFD_SFACE_SHA256;

   private:
    cv::Ptr<cv::FaceRecognizerSF> impl_;
};

}  // namespace RFDxFACE

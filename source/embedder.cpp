/**
 * @file embedder.cpp
 * @brief Implementation of RFDxFACE::Embedder - align with the detector's own
 *        landmarks, then embed. See embedder.hpp for why those two steps are
 *        inseparable.
 */

#include "embedder.hpp"

namespace RFDxFACE {

Embedder::Embedder() : impl_(cv::FaceRecognizerSF::create(RFD_SFACE_MODEL, "")) {
    if (impl_ == nullptr) {
        throw RuntimeError("failed to load SFace weights: " RFD_SFACE_MODEL);
    }
}

cv::Mat Embedder::embed(const cv::Mat& frame, const cv::Mat& detection_row) const {
    cv::Mat aligned;
    impl_->alignCrop(frame, detection_row, aligned);

    cv::Mat feature;
    impl_->feature(aligned, feature);
    if (feature.empty() || feature.total() != static_cast<UInt>(DIMENSIONS)) {
        throw RuntimeError("SFace returned an unexpected feature size");
    }

    // feature() hands back an unnormalised row; see the header for why we
    // normalise eagerly. clone() because reshape() is a view and `feature` is
    // about to go out of scope.
    cv::Mat normalised;
    cv::normalize(feature, normalised, 1.0, 0.0, cv::NORM_L2);
    return normalised.reshape(1, 1).clone();
}

}  // namespace RFDxFACE

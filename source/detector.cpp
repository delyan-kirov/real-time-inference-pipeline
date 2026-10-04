/**
 * @file detector.cpp
 * @brief Implementation of RFDxFACE::Detector - see detector.hpp for why the
 *        input size is re-declared and why frames are downscaled.
 */

#include "detector.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <utility>

namespace RFDxFACE {

namespace {
/// YuNet row layout: [x, y, w, h, (lx, ly) * 5, score] - 15 floats. Everything
/// before the score is a coordinate and therefore scales with the input.
constexpr int LANDMARK_OFFSET = 4;
constexpr int SCORE_OFFSET = 14;
}  // namespace

Detector::Detector(float score_threshold, float nms_threshold, int top_k, int max_input_edge)
    // The size passed here is a placeholder - detect() replaces it with the real
    // input size. It cannot be 0x0, because create() rejects that outright.
    : impl_(cv::FaceDetectorYN::create(RFD_YUNET_MODEL, "", cv::Size(320, 320), score_threshold,
                                       nms_threshold, top_k)),
      max_input_edge_(max_input_edge) {
    if (impl_ == nullptr) {
        throw RuntimeError("failed to load YuNet weights: " RFD_YUNET_MODEL);
    }
}

Vec<Detection> Detector::detect(const cv::Mat& frame) {
    if (frame.empty()) {
        return {};
    }
    if (frame.type() != CV_8UC3) {
        throw InvalidArgument("Detector::detect expects an 8-bit BGR frame");
    }

    // See the header: oversized faces are invisible to YuNet, so cap the input.
    double scale = 1.0;
    const int long_edge = std::max(frame.cols, frame.rows);
    if (max_input_edge_ > 0 && long_edge > max_input_edge_) {
        scale = static_cast<double>(max_input_edge_) / static_cast<double>(long_edge);
    }

    cv::Mat input;
    if (scale < 1.0) {
        // INTER_AREA is the right filter for shrinking; it averages rather than
        // point-samples, so it does not alias away the detail YuNet needs.
        cv::resize(frame, input, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        input = frame;
    }

    // The fixed-shape export must be told the exact size it is about to see.
    if (input.size() != input_size_) {
        impl_->setInputSize(input.size());
        input_size_ = input.size();
    }

    cv::Mat faces;
    impl_->detect(input, faces);
    if (faces.empty()) {
        return {};
    }

    // Undo the downscale so callers - and alignCrop, which runs against the
    // full-resolution frame - see original-frame coordinates.
    const float inverse = static_cast<float>(1.0 / scale);

    Vec<Detection> detections;
    detections.reserve(static_cast<UInt>(faces.rows));

    for (int i = 0; i < faces.rows; ++i) {
        cv::Mat row = faces.row(i).clone();

        if (scale < 1.0) {
            // Coordinates only: the score at SCORE_OFFSET must not be touched.
            for (int c = 0; c < SCORE_OFFSET; ++c) {
                row.at<float>(0, c) *= inverse;
            }
        }

        const float* v = row.ptr<float>(0);

        Detection detection;
        detection.box = cv::Rect2f(v[0], v[1], v[2], v[3]);
        for (UInt k = 0; k < LANDMARK_COUNT; ++k) {
            const int base = LANDMARK_OFFSET + (2 * static_cast<int>(k));
            detection.landmarks[k] = cv::Point2f(v[base], v[base + 1]);
        }
        detection.score = v[SCORE_OFFSET];
        detection.row = std::move(row);

        detections.push_back(std::move(detection));
    }
    return detections;
}

}  // namespace RFDxFACE

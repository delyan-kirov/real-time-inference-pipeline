/**
 * @file detector.hpp
 * @brief YuNet face detection.
 *
 * A wrapper over cv::FaceDetectorYN carrying the two pieces of real logic that
 * using YuNet correctly requires.
 *
 * 1. Input size tracking. The 2023mar export has a *fixed* input shape, and
 *    OpenCV 4.x infers at whatever size we declare through setInputSize().
 *    Declaring a size that does not match the frame does not fail - it silently
 *    returns coordinates on the wrong basis. So the size is re-declared whenever
 *    the input dimensions change, which is what makes a mid-stream resolution
 *    change (an RTSP reconnect negotiating different caps) safe.
 *
 * 2. Downscaling large frames. YuNet was trained to find faces roughly 10x10 to
 *    300x300 pixels, and that range is a property of the training scheme, not a
 *    soft preference. Feed it a 1280px portrait where the head is 600px tall and
 *    it finds *nothing* - no error, no warning, just an empty result that looks
 *    exactly like "there is no face here". Measured on our own gallery stills,
 *    detection went from 10/36 at native resolution to 28/36 when the long edge
 *    was capped at 640.
 *
 *    So frames are downscaled to `max_input_edge` before detection, and every
 *    coordinate is scaled back to the original frame afterwards. Callers always
 *    see results in original-frame coordinates and never need to know this
 *    happened.
 *
 *    The trade-off runs the other way for small faces: downscaling a wide shot
 *    of a crowd can push distant faces under the 10px floor, and we measured
 *    exactly that on one group photograph (4 faces at native, 0 at 640). A
 *    close-up gallery still and a wide surveillance shot genuinely want
 *    different caps, which is why this is a constructor parameter rather than a
 *    constant.
 */

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect/face.hpp>

#include "util.hpp"

/**
 * @namespace RFDxFACE
 * @brief Detection, embedding and gallery matching, independent of how frames
 *        arrive.
 */
namespace RFDxFACE {

/// RFD's vocabulary (Str, Vec, UInt, ...) is used unqualified throughout this
/// module; see util.hpp.
using namespace RFD;

/// Number of landmarks YuNet emits per face, and SFace's alignment consumes.
constexpr UInt LANDMARK_COUNT = 5;

/**
 * @brief One face found in a frame, in that frame's own pixel coordinates.
 */
struct Detection {
    /// Bounding box in original-frame coordinates.
    cv::Rect2f box;

    /// Right eye, left eye, nose tip, right mouth corner, left mouth corner.
    Arr<cv::Point2f, LANDMARK_COUNT> landmarks{};

    /// Detector confidence, in 0..1.
    float score = 0.0F;

    /**
     * @brief The raw 15-value YuNet output row.
     *
     * cv::FaceRecognizerSF::alignCrop takes this verbatim, so it is kept rather
     * than reconstructed from the fields above - reconstructing it is where
     * subtle alignment bugs come from.
     */
    cv::Mat row;
};

/**
 * @brief Runs YuNet over frames and reports faces in original-frame coordinates.
 *
 * Not thread-safe: detect() mutates the network's declared input size, so one
 * instance belongs to one streaming thread.
 */
class Detector {
   public:
    /**
     * @brief Loads the vendored YuNet weights.
     *
     * Thresholds follow OpenCV's own YuNet samples.
     *
     * @param score_threshold Minimum confidence for a detection to be reported.
     * @param nms_threshold   IoU threshold for non-maximum suppression.
     * @param top_k           Maximum number of candidate boxes kept before NMS.
     * @param max_input_edge  Caps the longest side of the detector input; 0
     *                        disables downscaling entirely.
     * @throws RuntimeError if the weights cannot be loaded.
     */
    explicit Detector(float score_threshold = 0.9F, float nms_threshold = 0.3F, int top_k = 5000,
                      int max_input_edge = DEFAULT_MAX_INPUT_EDGE);

    /**
     * @brief Detects every face in @p frame.
     *
     * All returned coordinates are in @p frame's own pixel space, whatever
     * downscaling happened internally.
     *
     * @param frame 8-bit 3-channel BGR image, which is what the DNN backend wants.
     * @return The faces found, empty if there are none or the frame is empty.
     * @throws InvalidArgument if @p frame is not 8-bit BGR, rather than
     *         returning nonsense.
     */
    Vec<Detection> detect(const cv::Mat& frame);

    /**
     * @brief The size the network is currently configured for.
     *
     * This is the *downscaled* size, not the frame's. Exposed so tests and logs
     * can confirm a caps change was picked up.
     */
    [[nodiscard]] cv::Size input_size() const {
        return input_size_;
    }

    /**
     * @brief Default cap on the detector input's longest side.
     *
     * Chosen from the measurement in the file comment: large enough to keep wide
     * shots usable, small enough that a close-up head lands in YuNet's trained
     * size range.
     */
    static constexpr int DEFAULT_MAX_INPUT_EDGE = 640;

   private:
    cv::Ptr<cv::FaceDetectorYN> impl_;
    cv::Size input_size_{0, 0};
    int max_input_edge_;
};

}  // namespace RFDxFACE

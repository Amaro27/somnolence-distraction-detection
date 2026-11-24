#pragma once

#include "Types.hpp"

#include <chrono>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>

/**
 * @brief Full-frame ONNX CNN yawning detector with temporal accumulation.
 *
 * Operates independently of Dlib face landmark detection:
 * 1. Preprocesses full BGR camera frame -> RGB 112x112 (INTER_AREA), scaled to [0, 1] in NHWC layout.
 * 2. Executes ONNX CNN inference via OpenCV DNN to obtain per-frame yawning probability.
 * 3. Evaluates sustained yawning (prob >= 0.80 for >= 1.0s) as a Light fatigue cue
 *    with linear accumulator decay and alarm cooldown.
 */
class YawnDetector {
public:
    explicit YawnDetector(YawnConfig config = YawnConfig{});

    // Non-copyable, movable
    YawnDetector(const YawnDetector&) = delete;
    YawnDetector& operator=(const YawnDetector&) = delete;
    YawnDetector(YawnDetector&&) noexcept = default;
    YawnDetector& operator=(YawnDetector&&) noexcept = default;

    /**
     * @brief Loads the yawning ONNX model and performs a warm-up forward pass.
     * @return true if the model loaded successfully, false otherwise.
     */
    bool initialize();

    /**
     * @brief Resets yawn duration tracking and cooldown timers.
     */
    void reset() noexcept;

    /**
     * @brief Runs full-frame CNN inference and updates the temporal yawn accumulator.
     * @param bgr_frame Raw camera frame (CV_8UC3 BGR).
     * @param now Current timestamp.
     * @return YawnAssessment containing CNN probability, yawn duration, yawning flag, and optional cue.
     */
    YawnAssessment process(
        const cv::Mat& bgr_frame,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    const YawnConfig& config() const noexcept { return config_; }

private:
    void pack_nhwc_blob(const cv::Mat& bgr_frame);

    YawnConfig config_;
    cv::dnn::Net net_;

    // Pre-allocated reusable preprocessing buffers (zero hot-path heap allocations)
    cv::Mat rgb_frame_;
    cv::Mat resized_rgb_;
    cv::Mat input_blob_;

    // Temporal yawn state
    bool in_yawn_{false};
    bool yawn_cue_emitted_{false};
    double yawn_accum_sec_{0.0};
    bool has_last_time_{false};
    std::chrono::steady_clock::time_point last_time_;
    std::chrono::steady_clock::time_point last_confirmed_yawn_;
};

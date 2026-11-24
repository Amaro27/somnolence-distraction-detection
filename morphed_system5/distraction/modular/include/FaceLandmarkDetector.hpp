#pragma once

#include "Types.hpp"

#include <chrono>
#include <optional>
#include <string>
#include <opencv2/core/mat.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

/**
 * @brief High-performance Dlib 68-landmark detector.
 *
 * Runs full face detection and 68-point shape prediction in a single pass.
 * Completely immune to ghost/hallucinated faces: shape_predictor is only invoked
 * if a real face is verified by the frontal face detector.
 * Tracks missing face durations and triggers lost-face recovery if absent for >= 3s.
 */
class FaceLandmarkDetector {
public:
    explicit FaceLandmarkDetector(FaceLandmarkConfig config = FaceLandmarkConfig{});

    // Non-copyable, movable
    FaceLandmarkDetector(const FaceLandmarkDetector&) = delete;
    FaceLandmarkDetector& operator=(const FaceLandmarkDetector&) = delete;
    FaceLandmarkDetector(FaceLandmarkDetector&&) noexcept = default;
    FaceLandmarkDetector& operator=(FaceLandmarkDetector&&) noexcept = default;

    /**
     * @brief Initializes Dlib frontal face detector and shape predictor.
     * Searches standard candidate paths if default model path is not directly found.
     * @return true if initialized successfully, false otherwise.
     */
    bool initialize();

    /**
     * @brief Extracts 68 landmarks from a BGR or single-channel Grayscale frame.
     * @param frame Raw camera frame (CV_8UC3 or CV_8UC1).
     * @param now Timestamp of the current frame.
     * @return std::optional<FacialLandmarks68> with 68 coordinates if found, std::nullopt otherwise.
     */
    std::optional<FacialLandmarks68> detect(
        const cv::Mat& frame,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    /**
     * @brief Resets face loss counters and timers.
     */
    void reset();

    bool is_face_missing() const { return face_missing_; }
    const FaceLandmarkConfig& config() const { return config_; }

private:
    std::string resolve_predictor_path() const;

    FaceLandmarkConfig config_;
    dlib::frontal_face_detector detector_;
    dlib::shape_predictor predictor_;
    cv::Mat gray_buf_;

    bool initialized_{false};
    bool face_missing_{false};
    bool warned_3s_{false};
    std::chrono::steady_clock::time_point face_lost_start_;
};

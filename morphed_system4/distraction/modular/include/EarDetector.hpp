#pragma once

#include "Types.hpp"

#include <chrono>
#include <deque>
#include <vector>

/**
 * @brief High-precision physiological Eye Aspect Ratio (EAR) detector.
 *
 * Implements:
 * 1. Geometric 6-point EAR calculation for left and right eyes.
 * 2. Baseline threshold application (70% descent, 20% closure) from calibration.
 * 3. Cascaded adaptive filtering (median filter + moving average baseline drift).
 * 4. Micro-sleep detection (>500ms eye closure) emitting Strong cues.
 * 5. Rolling 30-second PERCLOS tracking:
 *    - High PERCLOS (>=15.0%) -> Strong cue
 *    - Medium PERCLOS (7.5% - 15.0%) -> Light cue
 */
class EarDetector {
public:
    explicit EarDetector(EarConfig config = EarConfig{});
    ~EarDetector() = default;

    // Non-copyable, non-movable
    EarDetector(const EarDetector&) = delete;
    EarDetector& operator=(const EarDetector&) = delete;

    /**
     * @brief Computes geometric 6-point average EAR (left + right eyes) from 68 landmarks.
     */
    static double compute_ear(const FacialLandmarks68& landmarks);

    /**
     * @brief Applies calibrated open-eye and closed-eye EAR baselines and computes
     * dynamic descent (70%) and closure (20%) thresholds.
     */
    void apply_baseline(
        double ear_open,
        double ear_closed,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    /**
     * @brief Resets calibration status and all fatigue tracking buffers.
     */
    void reset();

    /**
     * @brief Processes facial landmarks for the current frame.
     * Performs active cascaded filtering and fatigue evaluation once calibrated.
     * @param landmarks 68 extracted facial landmarks.
     * @param now Timestamp of the current frame.
     * @return EarAssessment containing smoothed EAR, PERCLOS %, blink metrics, and cue events.
     */
    EarAssessment process(
        const FacialLandmarks68& landmarks,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    bool is_calibrated() const { return calibrated_; }
    const EarConfig& config() const { return config_; }

private:
    static double calculate_single_eye_ear(const cv::Point2f& p0, const cv::Point2f& p1,
                                           const cv::Point2f& p2, const cv::Point2f& p3,
                                           const cv::Point2f& p4, const cv::Point2f& p5);

    EarConfig config_;
    bool calibrated_{false};

    // Baseline and dynamic thresholds
    double ear_base_open_{0.30};
    double ear_base_closed_{0.15};
    double thresh_descent_{0.25};
    double thresh_closure_{0.18};
    double floor_closure_{0.18};

    // Cascaded adaptive filter buffers
    std::deque<double> buffer_raw_;
    std::deque<double> buffer_median_;
    double current_ear_open_{0.30};

    // Blink duration & micro-sleep tracking state
    bool in_blink_cycle_{false};
    bool reached_effective_closure_{false};
    std::chrono::steady_clock::time_point t_start_closure_;
    std::chrono::steady_clock::time_point t_last_closed_;
    double last_blink_duration_ms_{0.0};
    bool microsleep_alarm_active_{false};
    std::chrono::steady_clock::time_point last_microsleep_cue_time_;

    // Rolling 30-second PERCLOS tracking state
    std::deque<std::pair<std::chrono::steady_clock::time_point, int>> perclos_buffer_;
    std::chrono::steady_clock::time_point perclos_start_time_;
    std::chrono::steady_clock::time_point last_perclos_cue_time_;
    double perclos_pct_{0.0};
};

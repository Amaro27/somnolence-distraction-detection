#pragma once

#include "CalibrationAudioGuide.hpp"
#include "Types.hpp"

#include <chrono>
#include <deque>
#include <mutex>
#include <vector>

/**
 * @brief High-precision physiological Eye Aspect Ratio (EAR) detector.
 *
 * Implements:
 * 1. Geometric 6-point EAR calculation for left and right eyes.
 * 2. Two-phase physiological calibration coordinated with acoustic guidance feedback.
 * 3. Cascaded adaptive filtering (median filter + moving average baseline drift).
 * 4. Micro-sleep detection (>500ms eye closure) emitting Strong cues.
 * 5. Rolling 30-second PERCLOS tracking:
 *    - High PERCLOS (>=15.0%) -> Strong cue
 *    - Medium PERCLOS (7.5% - 15.0%) -> Light cue
 */
class EarDetector {
public:
    explicit EarDetector(EarConfig config = EarConfig{});
    ~EarDetector();

    // Non-copyable, non-movable
    EarDetector(const EarDetector&) = delete;
    EarDetector& operator=(const EarDetector&) = delete;

    /**
     * @brief Initiates two-phase calibration sequence with acoustic guidance.
     * Transitions state from Pending to CalibOpen.
     * @return true if calibration successfully started, false if already in progress.
     */
    bool start_calibration(std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

    /**
     * @brief Resets calibration state machine and all fatigue tracking buffers.
     */
    void reset();

    /**
     * @brief Processes facial landmarks for the current frame.
     * Advances calibration FSM or performs active cascaded filtering and fatigue evaluation.
     * @param landmarks 68 extracted facial landmarks.
     * @param now Timestamp of the current frame.
     * @return EarAssessment containing smoothed EAR, PERCLOS %, blink metrics, and cue events.
     */
    EarAssessment process(
        const FacialLandmarks68& landmarks,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    CalibrationPhase phase() const { return audio_guide_.phase(); }
    bool is_calibrated() const { return audio_guide_.phase() == CalibrationPhase::Completed; }
    bool is_measuring_open() const { return audio_guide_.is_measuring_open(); }
    bool is_measuring_closed() const { return audio_guide_.is_measuring_closed(); }
    bool is_measuring() const { return audio_guide_.is_measuring(); }

    const EarConfig& config() const { return config_; }

private:
    double calculate_single_eye_ear(const cv::Point2f& p0, const cv::Point2f& p1,
                                    const cv::Point2f& p2, const cv::Point2f& p3,
                                    const cv::Point2f& p4, const cv::Point2f& p5) const;

    void compute_calibrated_thresholds();

    EarConfig config_;
    CalibrationAudioGuide audio_guide_;
    mutable std::mutex sample_mutex_;

    // Calibration accumulator buffers
    std::vector<double> open_samples_;
    std::vector<double> closed_samples_;
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

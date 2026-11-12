#pragma once

#include "Types.hpp"

#include <chrono>
#include <optional>
#include <vector>

/**
 * @brief Anthropometric head orientation pitch and droop estimator.
 *
 * Implements pitch calculation invariant to head roll in the image plane using
 * eye-to-chin vertical projections (rho).
 * Roll calculations and thresholds have been completely removed.
 * Evaluates time-based head droop (pitch <= -20 deg for >= 1.5s) as a Light fatigue cue.
 */
class HeadOrientationDetector {
public:
    explicit HeadOrientationDetector(HeadOrientationConfig config = HeadOrientationConfig{});

    /**
     * @brief Begins accumulating neutral head orientation samples for calibration.
     */
    void start_calibration();

    /**
     * @brief Accumulates a single frame's neutral facial proportions (rho).
     */
    void accumulate_calibration(const FacialLandmarks68& landmarks);

    /**
     * @brief Computes the median rho baseline and marks orientation as calibrated.
     */
    void finalize_calibration();

    /**
     * @brief Resets calibration baseline and temporal droop timers.
     */
    void reset();

    /**
     * @brief Processes facial landmarks and evaluates pitch angle and droop fatigue cue.
     * @param landmarks 68 extracted landmark coordinates.
     * @param current_phase Active system calibration phase.
     * @param now Current timestamp.
     * @return HeadOrientationAssessment containing pitch, rho, phase, and any triggered cue.
     */
    HeadOrientationAssessment process(
        const FacialLandmarks68& landmarks,
        CalibrationPhase current_phase,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    bool is_calibrated() const { return calibrated_; }
    double baseline_rho() const { return baseline_rho_; }
    const HeadOrientationConfig& config() const { return config_; }

private:
    double compute_rho(const FacialLandmarks68& landmarks) const;

    HeadOrientationConfig config_;
    std::vector<double> calib_rho_buf_;
    double baseline_rho_{0.0};
    bool calibrated_{false};

    // Time-based droop detection state (1.5 seconds)
    bool in_droop_{false};
    bool droop_cue_emitted_{false};
    std::chrono::steady_clock::time_point droop_start_time_;
};

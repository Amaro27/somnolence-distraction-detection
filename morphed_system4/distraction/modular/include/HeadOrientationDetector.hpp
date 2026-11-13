#pragma once

#include "Types.hpp"

#include <chrono>
#include <optional>

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
     * @brief Computes roll-invariant vertical eye-to-nose/face proportion (rho).
     */
    static double compute_rho(const FacialLandmarks68& landmarks);

    /**
     * @brief Sets the calibrated neutral head pitch proportion (rho).
     */
    void apply_baseline(double neutral_rho);

    /**
     * @brief Resets calibration baseline and temporal droop timers.
     */
    void reset();

    /**
     * @brief Processes facial landmarks and evaluates pitch angle and droop fatigue cue.
     * @param landmarks 68 extracted landmark coordinates.
     * @param now Current timestamp.
     * @return HeadOrientationAssessment containing pitch, rho, and any triggered cue.
     */
    HeadOrientationAssessment process(
        const FacialLandmarks68& landmarks,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    bool is_calibrated() const { return calibrated_; }
    double baseline_rho() const { return baseline_rho_; }
    const HeadOrientationConfig& config() const { return config_; }

private:
    HeadOrientationConfig config_;
    double baseline_rho_{0.0};
    bool calibrated_{false};

    // Time-based droop detection state (1.5 seconds)
    bool in_droop_{false};
    bool droop_cue_emitted_{false};
    std::chrono::steady_clock::time_point droop_start_time_;
};

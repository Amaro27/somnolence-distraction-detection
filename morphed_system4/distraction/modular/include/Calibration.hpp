#pragma once

#include "AudioManager.hpp"
#include "FaceLandmarkDetector.hpp"
#include "ThreadSafeMailbox.hpp"
#include "Types.hpp"

#include <atomic>
#include <optional>
#include <opencv2/core/mat.hpp>

/**
 * @brief Non-owning context of dependencies required during calibration.
 */
struct CalibrationContext {
    ThreadSafeMailbox<cv::Mat>& mailbox;
    FaceLandmarkDetector&       face;
    AudioManager&               audio;
    const std::atomic<bool>&    shutdown; // Only exit path ('q' / SIGINT)
};

/**
 * @brief Computed physiological baselines from a completed calibration run.
 */
struct CalibrationResult {
    double ear_open;
    double ear_closed;
    double neutral_rho;
};

/**
 * @brief Runs the complete two-phase acoustic-guided physiological calibration sequence.
 *
 * Executes top-to-bottom on the caller thread without spawning extra threads.
 * Never aborts on missing face or movement requests: missing-face frames are skipped,
 * and measurement windows repeat their cadence beeps until at least
 * cfg.min_samples_per_window valid samples are collected.
 * Only exits early (returning std::nullopt) if ctx.shutdown becomes true.
 */
std::optional<CalibrationResult> run_calibration(
    CalibrationContext& ctx,
    const CalibrationAudioConfig& cfg = CalibrationAudioConfig{}
);

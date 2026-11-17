#pragma once

#include "Types.hpp"

#include <chrono>
#include <optional>

/**
 * @brief Mouth Aspect Ratio (MAR) and sustained yawn detector.
 *
 * Implements inner-lip MAR computation from 68 Dlib facial landmarks (indices 60-67).
 * Independent of vehicle calibration: operates immediately on active landmarks.
 * Evaluates sustained mouth opening (MAR > 0.55 for >= 4.0s) as a Light fatigue cue.
 * Employs a cooldown mechanism to prevent duplicate alarms for a single sustained yawn.
 */
class MarDetector {
public:
    explicit MarDetector(MarConfig config = MarConfig{});

    /**
     * @brief Resets yawn duration tracking and cooldown timers.
     */
    void reset();

    /**
     * @brief Processes facial landmarks and evaluates Mouth Aspect Ratio and yawn events.
     * @param landmarks 68 extracted landmark coordinates.
     * @param now Current timestamp.
     * @return MarAssessment containing raw MAR, yawn duration, yawning flag, and optional cue.
     */
    MarAssessment process(
        const FacialLandmarks68& landmarks,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    /**
     * @brief Updates yawn accumulator during missing-face frames (holds state across brief
     * face dropouts up to 1.5s before decaying, preventing false resets during deep yawns).
     */
    MarAssessment on_face_lost(
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
    );

    const MarConfig& config() const { return config_; }

private:
    MarConfig config_;
    bool in_yawn_{false};
    bool yawn_cue_emitted_{false};
    double yawn_accum_sec_{0.0};
    double closed_mouth_sec_{0.0};
    double missing_face_sec_{0.0};
    double last_mar_raw_{0.0};
    bool has_last_time_{false};
    std::chrono::steady_clock::time_point last_time_;
    std::chrono::steady_clock::time_point last_confirmed_yawn_;
};

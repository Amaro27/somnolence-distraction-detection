#pragma once

#include "Types.hpp"

#include <deque>
#include <string>

namespace AlertLogic {

struct DistractionAssessment {
    bool is_distracted{false};
    int not_looking_votes{0};
    int looking_votes{0};
    int total_votes{0};
    float avg_looking_prob{1.0f};
};

/**
 * @brief Evaluates the sliding window of gaze classification frames.
 * Pure, stateless function (no locks, no I/O).
 */
DistractionAssessment evaluate_distraction_mode(
    const std::deque<DistractionDetectionResult>& window,
    ThresholdEvaluationPolicy policy,
    float min_confidence) noexcept;

struct SomnolenceAssessment {
    bool trigger_high_alert{false};
    std::string trigger_reason;
};

/**
 * @brief Evaluates the 2-minute sliding window of light somnolence cues.
 * Pure, stateless function (no locks, no I/O).
 */
SomnolenceAssessment evaluate_somnolence_cues(
    const std::deque<CueEvent>& cue_window);

const char* risk_to_string(RiskLevel risk) noexcept;
const char* cue_type_to_string(CueType type) noexcept;

} // namespace AlertLogic

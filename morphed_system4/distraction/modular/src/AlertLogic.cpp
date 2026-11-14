#include "AlertLogic.hpp"

namespace AlertLogic {

DistractionAssessment evaluate_distraction_mode(
    const std::deque<DistractionDetectionResult>& window,
    ThresholdEvaluationPolicy policy,
    float min_confidence) noexcept {
    DistractionAssessment assessment;
    if (window.empty()) {
        return assessment;
    }

    int not_looking_count = 0;
    int looking_count = 0;
    float sum_looking_prob = 0.0f;

    for (const auto& item : window) {
        if (policy == ThresholdEvaluationPolicy::BinarizeThreshold) {
            // Interpretation A: mode of binarized detections
            if (item.gaze_class == GazeClass::NotLookingRoad) {
                ++not_looking_count;
            } else {
                ++looking_count;
            }
            sum_looking_prob += item.looking_road_probability;
        } else {
            // Interpretation B: filter out frames below confidence
            if (item.looking_road_probability >= min_confidence) {
                ++looking_count;
                sum_looking_prob += item.looking_road_probability;
            } else if ((1.0f - item.looking_road_probability) >= min_confidence) {
                ++not_looking_count;
            }
        }
    }

    const int total_votes = not_looking_count + looking_count;
    if (total_votes == 0) {
        return assessment;
    }

    assessment.not_looking_votes = not_looking_count;
    assessment.looking_votes = looking_count;
    assessment.total_votes = total_votes;
    assessment.avg_looking_prob = sum_looking_prob / static_cast<float>(total_votes);
    assessment.is_distracted = (not_looking_count > looking_count);
    return assessment;
}

SomnolenceAssessment evaluate_somnolence_cues(const std::deque<CueEvent>& cue_window) {
    SomnolenceAssessment assessment;
    // Two LIGHT cues within the sliding window trigger drowsiness alert
    if (cue_window.size() >= 2) {
        assessment.trigger_high_alert = true;
        assessment.trigger_reason = "Drowsiness (2 Light Cues in 2 Minutes)";
    }
    return assessment;
}

const char* risk_to_string(RiskLevel risk) noexcept {
    switch (risk) {
        case RiskLevel::None:   return "NONE";
        case RiskLevel::Low:    return "LOW";
        case RiskLevel::Medium: return "MEDIUM";
        case RiskLevel::High:   return "HIGH";
    }
    return "UNKNOWN";
}

const char* cue_type_to_string(CueType type) noexcept {
    switch (type) {
        case CueType::OffRoadGaze:   return "OffRoadGaze";
        case CueType::Perclos:       return "PERCLOS";
        case CueType::BlinkDuration: return "BlinkDuration";
        case CueType::HeadDroop:     return "HeadDroop";
        case CueType::Yawn:          return "Yawn";
    }
    return "UnknownCue";
}

} // namespace AlertLogic

#include "AlertManager.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

namespace {
const char* risk_to_string(RiskLevel risk) {
    switch (risk) {
        case RiskLevel::None:   return "NONE";
        case RiskLevel::Low:    return "LOW";
        case RiskLevel::Medium: return "MEDIUM";
        case RiskLevel::High:   return "HIGH";
    }
    return "UNKNOWN";
}

const char* cue_type_to_string(CueType type) {
    switch (type) {
        case CueType::OffRoadGaze:   return "OffRoadGaze";
        case CueType::Perclos:       return "PERCLOS";
        case CueType::BlinkDuration: return "BlinkDuration";
        case CueType::HeadDroop:     return "HeadDroop";
        case CueType::Yawn:          return "Yawn";
    }
    return "UnknownCue";
}
} // namespace

AlertManager::AlertManager(AlertConfig config)
    : config_(config),
      window_start_time_(std::chrono::steady_clock::now()),
      last_alert_sound_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

void AlertManager::update_gaze(const DistractionDetectionResult& detection) {
    auto now = detection.timestamp;
    gaze_window_.push_back(detection);

    // Prune entries older than the distraction window duration (default 2000 ms)
    auto cutoff = now - config_.distraction_window.value;
    while (!gaze_window_.empty() && gaze_window_.front().timestamp < cutoff) {
        gaze_window_.pop_front();
    }

    // Check if system has accumulated enough data to span the warm-up window
    if (!has_warmed_up_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - window_start_time_);
        if (elapsed >= config_.distraction_window.value) {
            has_warmed_up_ = true;
            std::cout << "[+] AlertManager: Initial "
                      << std::chrono::duration_cast<std::chrono::seconds>(config_.distraction_window.value).count()
                      << "s temporal window filled. Active mode evaluation started.\n";
        } else {
            return; // Maintain default Safe state during warm-up
        }
    }

    evaluate_distraction_mode(now);
}

void AlertManager::evaluate_distraction_mode(std::chrono::steady_clock::time_point /*now*/) {
    if (gaze_window_.empty()) {
        return;
    }

    // Do not downgrade a high-risk drowsiness state
    if (current_state_ == DriverState::Drowsy) {
        return;
    }

    int not_looking_count = 0;
    int looking_count = 0;
    float sum_looking_prob = 0.0f;

    for (const auto& item : gaze_window_) {
        if (config_.policy == ThresholdEvaluationPolicy::BinarizeThreshold) {
            // Interpretation A: mode of binarized detections
            if (item.gaze_class == GazeClass::NotLookingRoad) {
                not_looking_count++;
            } else {
                looking_count++;
            }
            sum_looking_prob += item.looking_road_probability;
        } else {
            // Interpretation B: filter out frames below confidence
            if (item.looking_road_probability >= config_.min_confidence.value) {
                looking_count++;
                sum_looking_prob += item.looking_road_probability;
            } else if ((1.0f - item.looking_road_probability) >= config_.min_confidence.value) {
                not_looking_count++;
            }
        }
    }

    int total_votes = not_looking_count + looking_count;
    if (total_votes == 0) return;

    float avg_looking_prob = sum_looking_prob / static_cast<float>(total_votes);
    bool is_mode_distracted = (not_looking_count > looking_count);

    if (is_mode_distracted) {
        if (current_state_ != DriverState::Distracted) {
            current_state_ = DriverState::Distracted;
            current_risk_ = RiskLevel::Medium;
            std::cout << "\n[ALERT] Distraction detected: Driver looking away from road!\n"
                      << "        Mode: NotLookingRoad (" << not_looking_count << "/" << total_votes << " votes in 2.0s)\n"
                      << "        Avg Looking Probability: " << std::fixed << std::setprecision(1)
                      << (avg_looking_prob * 100.0f) << "%\n";
        }
        emit_auditory_alert(RiskLevel::Medium, "Driver looking away from road");
    } else {
        if (current_state_ == DriverState::Distracted) {
            current_state_ = DriverState::Safe;
            current_risk_ = RiskLevel::None;
            std::cout << "\n[INFO] Alert cleared: Driver attention returned to road.\n"
                      << "       Mode: LookingRoad (" << looking_count << "/" << total_votes << " votes in 2.0s)\n";
        }
    }
}

void AlertManager::register_cue(CueType type, CueStrength strength, const std::string& details) {
    auto now = std::chrono::steady_clock::now();

    if (strength == CueStrength::Strong) {
        // Condition 1: Immediate alarm on any STRONG cue
        current_state_ = DriverState::Drowsy;
        current_risk_ = RiskLevel::High;
        std::cout << "\n[ALERT] CRITICAL DROWSINESS (STRONG CUE): " << cue_type_to_string(type)
                  << (details.empty() ? "" : " - " + details) << "\n"
                  << "        Immediate alert triggered!\n";
        emit_auditory_alert(RiskLevel::High, "Critical Drowsiness (Strong Cue: " + std::string(cue_type_to_string(type)) + ")");
        return;
    }

    // Condition 2: Accumulate LIGHT cues over 2-minute sliding window
    cue_window_.push_back({type, strength, now, details});
    evaluate_somnolence_cues(now);
}

void AlertManager::evaluate_somnolence_cues(std::chrono::steady_clock::time_point now) {
    // Prune cues older than light_cue_window (default 120 seconds / 2 minutes)
    auto cutoff = now - config_.light_cue_window;
    while (!cue_window_.empty() && cue_window_.front().timestamp < cutoff) {
        cue_window_.pop_front();
    }

    // Two LIGHT cues within 2 minutes trigger drowsiness alert
    if (cue_window_.size() >= 2) {
        current_state_ = DriverState::Drowsy;
        current_risk_ = RiskLevel::High;
        std::cout << "\n[ALERT] DROWSINESS DETECTED: " << cue_window_.size()
                  << " light fatigue cues accumulated within 2 minutes!\n";
        for (size_t i = 0; i < cue_window_.size(); ++i) {
            std::cout << "        Cue #" << (i + 1) << ": " << cue_type_to_string(cue_window_[i].type)
                      << " (" << cue_window_[i].details << ")\n";
        }
        emit_auditory_alert(RiskLevel::High, "Drowsiness (2 Light Cues in 2 Minutes)");
    }
}

void AlertManager::emit_auditory_alert(RiskLevel risk, const std::string& reason) {
    auto now = std::chrono::steady_clock::now();
    double min_interval_sec = 1.0;
    if (config_.audio_config.repetition_rate_hz > 0.0) {
        min_interval_sec = 1.0 / config_.audio_config.repetition_rate_hz;
    }

    auto elapsed = std::chrono::duration<double>(now - last_alert_sound_time_).count();
    if (elapsed >= min_interval_sec) {
        last_alert_sound_time_ = now;

        float volume = config_.audio_config.min_volume;
        if (risk == RiskLevel::High) {
            volume = config_.audio_config.max_volume;
        } else if (risk == RiskLevel::Medium) {
            volume = (config_.audio_config.min_volume + config_.audio_config.max_volume) * 0.5f;
        }

        std::cout << "[AUDITORY ALERT] Tone pulse emitted | Risk: " << risk_to_string(risk)
                  << " | Vol: " << std::fixed << std::setprecision(2) << volume
                  << " | Reason: " << reason << "\n";
    }
}

void AlertManager::reset() {
    gaze_window_.clear();
    cue_window_.clear();
    current_state_ = DriverState::Safe;
    current_risk_ = RiskLevel::None;
    window_start_time_ = std::chrono::steady_clock::now();
    has_warmed_up_ = false;
    std::cout << "[*] AlertManager state reset to Safe.\n";
}

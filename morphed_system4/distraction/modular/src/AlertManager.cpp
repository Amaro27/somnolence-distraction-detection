#include "AlertManager.hpp"
#include "AlertLogic.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

AlertManager::AlertManager(AlertConfig config, AudioManager& audio)
    : config_(std::move(config)),
      audio_(audio),
      window_start_time_(std::chrono::steady_clock::now()),
      last_alert_sound_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

void AlertManager::update_gaze(const DistractionDetectionResult& detection) {
    bool warmed_up_now = false;
    long long warmup_secs = 0;
    bool transitioned_to_distracted = false;
    bool transitioned_to_safe = false;
    bool emit_audio = false;
    float audio_volume = 0.0f;
    AlertLogic::DistractionAssessment assessment;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = detection.timestamp;
        gaze_window_.push_back(detection);

        // Prune entries older than the distraction window duration (default 2000 ms)
        const auto cutoff = now - config_.distraction_window.value;
        while (!gaze_window_.empty() && gaze_window_.front().timestamp < cutoff) {
            gaze_window_.pop_front();
        }

        // Check if system has accumulated enough data to span the warm-up window
        if (!has_warmed_up_) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - window_start_time_);
            if (elapsed >= config_.distraction_window.value) {
                has_warmed_up_ = true;
                warmed_up_now = true;
                warmup_secs = std::chrono::duration_cast<std::chrono::seconds>(config_.distraction_window.value).count();
            } else {
                return; // Maintain default Safe state during warm-up
            }
        }

        // Do not downgrade a high-risk drowsiness state
        if (!gaze_window_.empty() && current_state_ != DriverState::Drowsy) {
            assessment = AlertLogic::evaluate_distraction_mode(
                gaze_window_,
                config_.policy,
                config_.min_confidence.value
            );

            if (assessment.total_votes > 0) {
                if (assessment.is_distracted) {
                    if (current_state_ != DriverState::Distracted) {
                        current_state_ = DriverState::Distracted;
                        current_risk_ = RiskLevel::Medium;
                        transitioned_to_distracted = true;
                    }
                    emit_audio = try_consume_audio_cooldown(RiskLevel::Medium, audio_volume);
                } else {
                    if (current_state_ == DriverState::Distracted) {
                        current_state_ = DriverState::Safe;
                        current_risk_ = RiskLevel::None;
                        transitioned_to_safe = true;
                    }
                }
            }
        }
    }

    if (warmed_up_now) {
        std::cout << "[+] AlertManager: Initial " << warmup_secs
                  << "s temporal window filled. Active mode evaluation started.\n";
    }

    if (transitioned_to_distracted) {
        std::cout << "\n[ALERT] Distraction detected: Driver looking away from road!\n"
                  << "        Mode: NotLookingRoad (" << assessment.not_looking_votes << "/"
                  << assessment.total_votes << " votes in 2.0s)\n"
                  << "        Avg Looking Probability: " << std::fixed << std::setprecision(1)
                  << (assessment.avg_looking_prob * 100.0f) << "%\n";
    } else if (transitioned_to_safe) {
        std::cout << "\n[INFO] Alert cleared: Driver attention returned to road.\n"
                  << "       Mode: LookingRoad (" << assessment.looking_votes << "/"
                  << assessment.total_votes << " votes in 2.0s)\n";
    }

    if (emit_audio) {
        dispatch_auditory_alert(RiskLevel::Medium, audio_volume, "Driver looking away from road");
    }
}

void AlertManager::register_cue(CueType type, CueStrength strength, const std::string& details) {
    bool is_strong_trigger = false;
    bool trigger_high_alert = false;
    bool emit_audio = false;
    float audio_volume = 0.0f;
    std::string trigger_reason;
    std::vector<CueEvent> cues_snapshot;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();

        if (strength == CueStrength::Strong) {
            // Condition 1: Immediate alarm on any STRONG cue
            current_state_ = DriverState::Drowsy;
            current_risk_ = RiskLevel::High;
            is_strong_trigger = true;
            emit_audio = try_consume_audio_cooldown(RiskLevel::High, audio_volume);
        } else {
            // Condition 2: Accumulate LIGHT cues over 2-minute sliding window
            cue_window_.push_back({type, strength, now, details});

            const auto cutoff = now - config_.light_cue_window;
            while (!cue_window_.empty() && cue_window_.front().timestamp < cutoff) {
                cue_window_.pop_front();
            }

            const AlertLogic::SomnolenceAssessment assessment = AlertLogic::evaluate_somnolence_cues(cue_window_);
            if (assessment.trigger_high_alert) {
                current_state_ = DriverState::Drowsy;
                current_risk_ = RiskLevel::High;
                trigger_high_alert = true;
                trigger_reason = assessment.trigger_reason;
                cues_snapshot.assign(cue_window_.begin(), cue_window_.end());
                emit_audio = try_consume_audio_cooldown(RiskLevel::High, audio_volume);
            }
        }
    }

    if (is_strong_trigger) {
        const char* cue_name = AlertLogic::cue_type_to_string(type);
        std::cout << "\n[ALERT] CRITICAL DROWSINESS (STRONG CUE): " << cue_name
                  << (details.empty() ? "" : " - " + details) << "\n"
                  << "        Immediate alert triggered!\n";
        if (emit_audio) {
            dispatch_auditory_alert(
                RiskLevel::High,
                audio_volume,
                std::string("Critical Drowsiness (Strong Cue: ") + cue_name + ")"
            );
        }
        return;
    }

    if (trigger_high_alert) {
        std::cout << "\n[ALERT] DROWSINESS DETECTED: " << cues_snapshot.size()
                  << " light fatigue cues accumulated within 2 minutes!\n";
        for (size_t i = 0; i < cues_snapshot.size(); ++i) {
            std::cout << "        Cue #" << (i + 1) << ": "
                      << AlertLogic::cue_type_to_string(cues_snapshot[i].type)
                      << " (" << cues_snapshot[i].details << ")\n";
        }
        if (emit_audio) {
            dispatch_auditory_alert(RiskLevel::High, audio_volume, trigger_reason);
        }
    }
}

void AlertManager::emit_auditory_alert(RiskLevel risk, const std::string& reason) {
    float volume = 0.0f;
    bool emit_audio = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        emit_audio = try_consume_audio_cooldown(risk, volume);
    }

    if (emit_audio) {
        dispatch_auditory_alert(risk, volume, reason);
    }
}

bool AlertManager::try_consume_audio_cooldown(RiskLevel risk, float& out_volume) noexcept {
    const auto now = std::chrono::steady_clock::now();
    double min_interval_sec = 1.0;
    if (config_.audio_config.repetition_rate_hz > 0.0) {
        min_interval_sec = 1.0 / config_.audio_config.repetition_rate_hz;
    }

    const double elapsed = std::chrono::duration<double>(now - last_alert_sound_time_).count();
    if (elapsed < min_interval_sec) {
        return false;
    }

    last_alert_sound_time_ = now;
    out_volume = config_.audio_config.min_volume;
    if (risk == RiskLevel::High) {
        out_volume = config_.audio_config.max_volume;
    } else if (risk == RiskLevel::Medium) {
        out_volume = (config_.audio_config.min_volume + config_.audio_config.max_volume) * 0.5f;
    }
    return true;
}

void AlertManager::dispatch_auditory_alert(RiskLevel risk, float volume, const std::string& reason) {
    std::cout << "[AUDITORY ALERT] Tone pulse emitted | Risk: " << AlertLogic::risk_to_string(risk)
              << " | Vol: " << std::fixed << std::setprecision(2) << volume
              << " | Reason: " << reason << "\n";
    audio_.play_alert(risk);
}

void AlertManager::reset() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        gaze_window_.clear();
        cue_window_.clear();
        current_state_ = DriverState::Safe;
        current_risk_ = RiskLevel::None;
        window_start_time_ = std::chrono::steady_clock::now();
        has_warmed_up_ = false;
    }
    std::cout << "[*] AlertManager state reset to Safe.\n";
}

DriverState AlertManager::current_driver_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_state_;
}

RiskLevel AlertManager::current_risk() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_risk_;
}

AlertConfig AlertManager::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

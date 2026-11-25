#include "AlertManager.hpp"
#include "AlertLogic.hpp"
#include "TelegramNotifier.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

namespace {

std::string log_somnolence_triggers_and_reason(
    const std::vector<CueEvent>& strong_cues_triggered,
    bool trigger_high_alert,
    const std::string& trigger_reason,
    const std::vector<CueEvent>& cues_snapshot
) {
#ifdef DEBUG_VIEWS
    for (const auto& cue : strong_cues_triggered) {
        const char* cue_name = AlertLogic::cue_type_to_string(cue.type);
        std::cout << "\n[ALERT] CRITICAL DROWSINESS (STRONG CUE): " << cue_name
                  << (cue.details.empty() ? "" : " - " + cue.details) << "\n"
                  << "        Immediate alert triggered!\n";
    }

    if (trigger_high_alert) {
        std::cout << "\n[ALERT] DROWSINESS DETECTED: " << cues_snapshot.size()
                  << " light fatigue cues accumulated within 2 minutes!\n";
        for (size_t i = 0; i < cues_snapshot.size(); ++i) {
            std::cout << "        Cue #" << (i + 1) << ": "
                      << AlertLogic::cue_type_to_string(cues_snapshot[i].type)
                      << " (" << cues_snapshot[i].details << ")\n";
        }
    }
#else
    (void)cues_snapshot;
#endif

    if (!strong_cues_triggered.empty()) {
        return std::string("Critical Drowsiness (Strong Cue: ") +
               AlertLogic::cue_type_to_string(strong_cues_triggered.back().type) + ")";
    }
    if (trigger_high_alert) {
        return trigger_reason;
    }
    return "Sustained Drowsiness";
}

bool consume_rate_cooldown(
    double rate_hz,
    double fallback_interval_sec,
    std::chrono::steady_clock::time_point& last_alert_time
) noexcept {
    const auto now = std::chrono::steady_clock::now();
    const double min_interval_sec = (rate_hz > 0.0) ? (1.0 / rate_hz) : fallback_interval_sec;
    const double elapsed = std::chrono::duration<double>(now - last_alert_time).count();
    if (elapsed < min_interval_sec) {
        return false;
    }
    last_alert_time = now;
    return true;
}

void log_auditory_alert(float gain_db, const std::string& reason) {
#ifdef DEBUG_VIEWS
    std::cout << "[AUDITORY ALERT] Tone pulse emitted"
              << " | Gain: " << std::fixed << std::setprecision(1) << gain_db << " dB"
              << " | Reason: " << reason << "\n";
#else
    (void)gain_db;
    (void)reason;
#endif
}

} // namespace

AlertManager::AlertManager(AlertConfig config, AudioManager& audio)
    : config_(std::move(config)),
      audio_(audio),
      last_alert_sound_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

void AlertManager::update_gaze(const DistractionDetectionResult& detection) {
    bool warmed_up_now = false;
    size_t mode_frames = 0;
    long long sustained_window_ms = 0;
    bool trigger_distraction_alert = false;
    bool transitioned_to_safe = false;
    AlertLogic::DistractionAssessment assessment;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = detection.timestamp;
        mode_frames = std::max<size_t>(1, config_.mode_window_frames);
        sustained_window_ms = config_.distraction_window.count();

        gaze_window_.push_back(detection);
        while (gaze_window_.size() > mode_frames) {
            gaze_window_.pop_front();
        }

        // Wait until the sliding mode window has accumulated mode_frames predictions
        if (gaze_window_.size() < mode_frames) {
            return;
        }
        if (!initial_warmup_logged_) {
            initial_warmup_logged_ = true;
            warmed_up_now = true;
        }

        // Do not trigger or evaluate distraction if a drowsiness alert is active or reproducing
        if (is_drowsiness_alert_reproducing_locked() || current_state_ == DriverState::Drowsy) {
            distracted_since_.reset();
            return;
        }

        assessment = AlertLogic::evaluate_distraction_mode(
            gaze_window_,
            config_.policy,
            config_.min_confidence
        );

        if (assessment.total_votes == 0) {
            distracted_since_.reset();
            if (current_state_ == DriverState::Distracted) {
                current_state_ = DriverState::Safe;
                transitioned_to_safe = true;
            }
        } else if (assessment.is_distracted) {
            if (!distracted_since_.has_value()) {
                distracted_since_ = now;
            }
            const auto sustained = std::chrono::duration_cast<std::chrono::milliseconds>(now - *distracted_since_);
            if (sustained >= config_.distraction_window) {
                current_state_ = DriverState::Distracted;
                trigger_distraction_alert = true;
            }
        } else {
            distracted_since_.reset();
            if (current_state_ == DriverState::Distracted) {
                current_state_ = DriverState::Safe;
                transitioned_to_safe = true;
            }
        }
    }

#ifdef DEBUG_VIEWS
    if (warmed_up_now) {
        std::cout << "[+] AlertManager: Initial " << mode_frames
                  << "-frame mode window filled. Active distraction monitoring ("
                  << sustained_window_ms << " ms sustained) started.\n";
    }
#else
    (void)warmed_up_now;
    (void)mode_frames;
    (void)sustained_window_ms;
#endif

    if (trigger_distraction_alert) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (is_drowsiness_alert_reproducing_locked()) {
                return; // Drowsiness took priority, abort distraction alert reproduction
            }
            drowsiness_audio_active_ = false;
        }

#ifdef DEBUG_VIEWS
        std::cout << "\n[ALERT] Distraction detected: Driver looking away from road!\n"
                  << "        Mode: NotLookingRoad (" << assessment.not_looking_votes << "/"
                  << assessment.total_votes << " votes in " << mode_frames << " frames, sustained >= "
                  << sustained_window_ms << " ms)\n"
                  << "        Avg Looking Probability: " << std::fixed << std::setprecision(1)
                  << (assessment.avg_looking_prob * 100.0f) << "%\n";
#endif
        // Dispatch Telegram alert asynchronously without blocking frame processing
        std::thread([]() {
            send_telegram_alert("⚠️ Distraction Alert: Driver looking away from road!");
        }).detach();

        log_auditory_alert(config_.audio_config.distraction_gain_db, "Driver looking away from road");
        audio_.play_blocking(
            config_.audio_config.distraction_sound_path,
            config_.audio_config.distraction_gain_db
        );

        std::lock_guard<std::mutex> lock(mutex_);
        gaze_window_.clear();
        distracted_since_.reset();
        if (current_state_ == DriverState::Distracted) {
            current_state_ = DriverState::Safe;
        }
    } else if (transitioned_to_safe) {
#ifdef DEBUG_VIEWS
        std::cout << "\n[INFO] Alert cleared: Driver attention returned to road.\n"
                  << "       Mode: LookingRoad (" << assessment.looking_votes << "/"
                  << assessment.total_votes << " votes in " << mode_frames << " frames)\n";
#endif
        std::lock_guard<std::mutex> lock(mutex_);
        if (!is_drowsiness_alert_reproducing_locked()) {
            audio_.stop();
        }
    }
}

void AlertManager::register_cue_locked(
    const CueEvent& cue,
    std::vector<CueEvent>& strong_cues_triggered,
    bool& trigger_high_alert,
    std::string& trigger_reason,
    std::vector<CueEvent>& cues_snapshot
) {
    if (cue.strength == CueStrength::Strong) {
        if (current_state_ != DriverState::Drowsy) {
            current_state_ = DriverState::Drowsy;
            drowsiness_step_idx_ = 0;
            last_alert_sound_time_ = std::chrono::steady_clock::now() - std::chrono::seconds(10);
        }
        strong_cues_triggered.push_back(cue);
    } else {
        cue_window_.push_back(cue);

        const auto cutoff = cue.timestamp - config_.light_cue_window;
        while (!cue_window_.empty() && cue_window_.front().timestamp < cutoff) {
            cue_window_.pop_front();
        }

        const AlertLogic::SomnolenceAssessment assessment = AlertLogic::evaluate_somnolence_cues(cue_window_);
        if (assessment.trigger_high_alert) {
            if (current_state_ != DriverState::Drowsy) {
                current_state_ = DriverState::Drowsy;
                drowsiness_step_idx_ = 0;
                last_alert_sound_time_ = std::chrono::steady_clock::now() - std::chrono::seconds(10);
            }
            trigger_high_alert = true;
            trigger_reason = assessment.trigger_reason;
            cues_snapshot.assign(cue_window_.begin(), cue_window_.end());
        }
    }
}

void AlertManager::register_cue(CueType type, CueStrength strength, const std::string& details) {
    std::vector<CueEvent> strong_cues_triggered;
    bool trigger_high_alert = false;
    bool emit_audio = false;
    float gain_db = 0.0f;
    std::string trigger_reason;
    std::vector<CueEvent> cues_snapshot;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const CueEvent cue{type, strength, std::chrono::steady_clock::now(), details};
        register_cue_locked(cue, strong_cues_triggered, trigger_high_alert, trigger_reason, cues_snapshot);
        if (current_state_ == DriverState::Drowsy) {
            emit_audio = try_consume_drowsiness_cooldown(gain_db);
        }
    }

    const std::string reason = log_somnolence_triggers_and_reason(
        strong_cues_triggered, trigger_high_alert, trigger_reason, cues_snapshot
    );
    if (!strong_cues_triggered.empty() || trigger_high_alert) {
        std::thread([alert_text = "⚠️ Somnolence Alert: " + reason]() {
            send_telegram_alert(alert_text);
        }).detach();
    }
    if (emit_audio) {
        dispatch_auditory_alert(config_.audio_config.drowsiness_sound_path, gain_db, reason);
    }
}

void AlertManager::update_somnolence(
    const EarAssessment& ear,
    const HeadOrientationAssessment& head,
    const YawnAssessment& yawn
) {
    std::vector<CueEvent> strong_cues_triggered;
    bool trigger_high_alert = false;
    bool cleared_drowsiness = false;
    bool emit_audio = false;
    float gain_db = 0.0f;
    std::string trigger_reason;
    std::vector<CueEvent> cues_snapshot;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool is_yawning_suppression = (yawn.yawn_duration_sec >= 0.2);

        for (const auto& cue : ear.cues) {
            if (is_yawning_suppression && (cue.type == CueType::BlinkDuration || cue.type == CueType::Perclos)) {
                continue;
            }
            register_cue_locked(cue, strong_cues_triggered, trigger_high_alert, trigger_reason, cues_snapshot);
        }
        if (head.cue.has_value()) {
            register_cue_locked(*head.cue, strong_cues_triggered, trigger_high_alert, trigger_reason, cues_snapshot);
        }
        if (yawn.cue.has_value()) {
            register_cue_locked(*yawn.cue, strong_cues_triggered, trigger_high_alert, trigger_reason, cues_snapshot);
        }

        const bool blink_active = !is_yawning_suppression && ear.in_blink_cycle;
        const bool features_normal = !blink_active && !head.is_drooping && !yawn.is_yawning;
        if (current_state_ == DriverState::Drowsy) {
            if (features_normal && strong_cues_triggered.empty() && !trigger_high_alert) {
                current_state_ = DriverState::Safe;
                drowsiness_step_idx_ = 0;
                cue_window_.clear();
                last_alert_sound_time_ = std::chrono::steady_clock::now() - std::chrono::seconds(10);
                cleared_drowsiness = true;
            } else {
                emit_audio = try_consume_drowsiness_cooldown(gain_db);
            }
        }
    }

    const std::string reason = log_somnolence_triggers_and_reason(
        strong_cues_triggered, trigger_high_alert, trigger_reason, cues_snapshot
    );
    if (cleared_drowsiness) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            drowsiness_audio_active_ = false;
        }
#ifdef DEBUG_VIEWS
        std::cout << "\n[INFO] Drowsiness alert cleared: Driver physiological features returned to normal.\n";
#endif
        audio_.stop();
    } else {
        if (!strong_cues_triggered.empty() || trigger_high_alert) {
            std::thread([alert_text = "⚠️ Somnolence Alert: " + reason]() {
                send_telegram_alert(alert_text);
            }).detach();
        }
        if (emit_audio) {
            dispatch_auditory_alert(config_.audio_config.drowsiness_sound_path, gain_db, reason);
        }
    }
}

void AlertManager::tick_drowsiness_if_active() {
    bool emit_audio = false;
    float gain_db = 0.0f;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_state_ == DriverState::Drowsy) {
            emit_audio = try_consume_drowsiness_cooldown(gain_db);
        }
    }

    if (emit_audio) {
        dispatch_auditory_alert(
            config_.audio_config.drowsiness_sound_path,
            gain_db,
            "Sustained Drowsiness"
        );
    }
}

bool AlertManager::try_consume_drowsiness_cooldown(float& out_gain_db) noexcept {
    if (!consume_rate_cooldown(
            config_.audio_config.drowsiness_repetition_rate_hz,
            60.0 / 63.5,
            last_alert_sound_time_)) {
        return false;
    }

    const auto& steps = config_.audio_config.drowsiness_gain_steps_db;
    const size_t idx = std::min(drowsiness_step_idx_, steps.size() - 1);
    out_gain_db = steps[idx];
    ++drowsiness_step_idx_;
    return true;
}

void AlertManager::dispatch_auditory_alert(
    const std::string& sound_path,
    float gain_db,
    const std::string& reason
) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        drowsiness_audio_active_ = true;
    }
    // Drowsiness takes priority: interrupt any active distraction playback immediately
    audio_.stop();
    log_auditory_alert(gain_db, reason);
    audio_.play(sound_path, gain_db);
}

void AlertManager::reset() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        gaze_window_.clear();
        cue_window_.clear();
        current_state_ = DriverState::Safe;
        distracted_since_.reset();
        last_alert_sound_time_ = std::chrono::steady_clock::now() - std::chrono::seconds(10);
        drowsiness_step_idx_ = 0;
        drowsiness_audio_active_ = false;
        initial_warmup_logged_ = false;
    }
    audio_.stop();
#ifdef DEBUG_VIEWS
    std::cout << "[*] AlertManager state reset to Safe.\n";
#endif
}

bool AlertManager::is_drowsiness_alert_reproducing() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_drowsiness_alert_reproducing_locked();
}

bool AlertManager::is_drowsiness_alert_reproducing_locked() const {
    if (current_state_ == DriverState::Drowsy) {
        return true;
    }
    if (drowsiness_audio_active_ && audio_.is_playing()) {
        return true;
    }
    return false;
}

DriverState AlertManager::current_driver_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_state_;
}

AlertConfig AlertManager::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

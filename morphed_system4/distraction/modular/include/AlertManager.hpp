#pragma once

#include "AudioManager.hpp"
#include "Types.hpp"

#include <chrono>
#include <deque>
#include <mutex>
#include <string>

/**
 * @brief Unified alert coordinator and state machine.
 *
 * Responsibilities:
 * 1. Evaluates 2.0-second sliding window mode for off-road gaze distraction.
 * 2. Emits placeholder print statements for alert triggers and recovery.
 * 3. Dispatches rate-controlled, non-blocking auditory alert pulses.
 * 4. Evaluates multimodal somnolence cues (1 strong cue or 2 light cues in 2 min).
 * 5. Provides thread-safe interfaces for concurrent distraction and somnolence workers.
 */
class AlertManager {
public:
    /**
     * @brief Constructs AlertManager with specified alert thresholds, timing, and audio manager.
     */
    AlertManager(AlertConfig config, AudioManager& audio);

    /**
     * @brief Pushes a new frame gaze classification into the 2.0-second sliding window.
     * Evaluates mode and triggers/clears distraction alert accordingly.
     * @param detection Single-frame inference result.
     */
    void update_gaze(const DistractionDetectionResult& detection);

    /**
     * @brief Extension hook for future somnolence / drowsiness module.
     * Evaluates immediate strong cues or 2 light cues within 2 minutes.
     * @param type Cue type (PERCLOS, blink, droop, yawn).
     * @param strength Light (accumulates over 2 min) or Strong (immediate trigger).
     * @param details Descriptive message explaining the cue.
     */
    void register_cue(CueType type, CueStrength strength, const std::string& details = "");

    /**
     * @brief Emits a rate-controlled non-blocking auditory alert tone (placeholder print).
     * @param risk Low, Medium, or High risk level.
     * @param reason Human-readable description of alert trigger.
     */
    void emit_auditory_alert(RiskLevel risk, const std::string& reason);

    DriverState current_driver_state() const;
    RiskLevel current_risk() const;

    /**
     * @brief Resets temporal history and transitions state back to Safe.
     */
    void reset();

    AlertConfig config() const;

private:
    void evaluate_distraction_mode(std::chrono::steady_clock::time_point now);
    void evaluate_somnolence_cues(std::chrono::steady_clock::time_point now);

    mutable std::mutex mutex_;
    AlertConfig config_;
    AudioManager& audio_;
    DriverState current_state_{DriverState::Safe};
    RiskLevel current_risk_{RiskLevel::None};

    std::deque<DistractionDetectionResult> gaze_window_; // 2.0s sliding window
    std::deque<CueEvent> cue_window_;                    // 2-minute light cue history
    std::chrono::steady_clock::time_point window_start_time_;
    std::chrono::steady_clock::time_point last_alert_sound_time_;
    bool has_warmed_up_{false};
};

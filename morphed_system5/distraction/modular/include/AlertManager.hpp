#pragma once

#include "AudioManager.hpp"
#include "Types.hpp"

#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

/**
 * @brief Unified alert coordinator and state machine.
 *
 * Responsibilities:
 * 1. Maintains thread-safe temporal windows and driver state.
 * 2. Delegates stateless window evaluation to AlertLogic.
 * 3. Emits console notifications and rate-controlled auditory alerts outside mutex lock.
 * 4. Provides thread-safe interfaces for concurrent distraction and somnolence workers.
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
     * @brief Extension hook for somnolence / drowsiness module.
     * Evaluates immediate strong cues or 2 light cues within 2 minutes.
     * @param type Cue type (PERCLOS, blink, droop, yawn).
     * @param strength Light (accumulates over 2 min) or Strong (immediate trigger).
     * @param details Descriptive message explaining the cue.
     */
    void register_cue(CueType type, CueStrength strength, const std::string& details = "");

    /**
     * @brief Evaluates per-frame somnolence assessments, triggers/sustains stepped
     * drowsiness alarms at 63.5 BPM, and stops audio immediately when features recover.
     */
    void update_somnolence(
        const EarAssessment& ear,
        const HeadOrientationAssessment& head,
        const MarAssessment& mar
    );

    /**
     * @brief Sustains active drowsiness alarm cadence when face landmarks are temporarily lost.
     */
    void tick_drowsiness_if_active();

    DriverState current_driver_state() const;

    /**
     * @brief Checks if a drowsiness alert is active or actively reproducing audio.
     */
    bool is_drowsiness_alert_reproducing() const;

    /**
     * @brief Resets temporal history and transitions state back to Safe.
     */
    void reset();

    AlertConfig config() const;

private:
    bool is_drowsiness_alert_reproducing_locked() const;
    bool try_consume_drowsiness_cooldown(float& out_gain_db) noexcept;
    void register_cue_locked(
        const CueEvent& cue,
        std::vector<CueEvent>& strong_cues_triggered,
        bool& trigger_high_alert,
        std::string& trigger_reason,
        std::vector<CueEvent>& cues_snapshot
    );
    void dispatch_auditory_alert(const std::string& sound_path, float gain_db, const std::string& reason);

    mutable std::mutex mutex_;
    AlertConfig config_;
    AudioManager& audio_;
    DriverState current_state_{DriverState::Safe};

    std::deque<DistractionDetectionResult> gaze_window_; // 2.0s sliding window
    std::deque<CueEvent> cue_window_;                    // 2-minute light cue history
    std::chrono::steady_clock::time_point window_start_time_;
    std::chrono::steady_clock::time_point last_alert_sound_time_;
    size_t drowsiness_step_idx_{0};
    bool drowsiness_audio_active_{false};
    bool has_warmed_up_{false};
    bool initial_warmup_logged_{false};
};

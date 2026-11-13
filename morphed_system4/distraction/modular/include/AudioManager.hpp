#pragma once

#include "Types.hpp"

#include <atomic>
#include <mutex>
#include <string>

/**
 * @brief Unified audio-only playback and alert sound service.
 *
 * Handles non-blocking and blocking sound playback, immediate cancellation,
 * and risk-prioritized alert sound dispatch. Contains zero calibration or
 * detector domain logic.
 */
class AudioManager {
public:
    explicit AudioManager(AuditoryAlertConfig alert_cfg = AuditoryAlertConfig{});
    ~AudioManager();

    AudioManager(const AudioManager&) = delete;
    AudioManager& operator=(const AudioManager&) = delete;

    /**
     * @brief Plays a WAV file asynchronously (non-blocking).
     */
    void play(const std::string& path);

    /**
     * @brief Plays a WAV file synchronously (blocks until finished or stopped).
     * @return true if playback completed normally, false if interrupted/failed.
     */
    bool play_blocking(const std::string& path);

    /**
     * @brief Immediately halts any active audio playback.
     */
    void stop();

    /**
     * @brief Plays the configured alert sound corresponding to the given risk level.
     * No-op if a blocking prompt is currently playing.
     */
    void play_alert(RiskLevel risk);

    const AuditoryAlertConfig& alert_config() const { return alert_cfg_; }

private:
    AuditoryAlertConfig alert_cfg_;
    std::mutex mutex_;
    std::atomic<bool> blocking_active_{false};
};

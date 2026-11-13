#pragma once

#include "SoundPlayer.hpp"
#include "Types.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

/**
 * @brief Coordinates the two-phase calibration acoustic feedback sequence.
 *
 * Sequence:
 * Phase 1: Guided instructions (calib_phase_1.wav) -> 1s silence -> Posture tone (sine-415-Hz-1-second-padded.wav)
 *          -> 1s silence -> 8 cadence beeps (single-beep-350-hz.wav) over 4.0s (active open eyes measurement).
 * Transition: 1s silence (no transition tone).
 * Phase 2: Guided instructions (calib_phase_2.wav) replacing the 1s tone -> 1s silence
 *          -> 8 cadence beeps (single-beep-350-hz.wav) over 4.0s (active closed eyes floor measurement).
 * Completion: Success sound (tada-LOUD.wav) -> Completed phase.
 */
class CalibrationAudioGuide {
public:
    explicit CalibrationAudioGuide(CalibrationAudioConfig config = CalibrationAudioConfig{});
    ~CalibrationAudioGuide();

    // Non-copyable, non-movable
    CalibrationAudioGuide(const CalibrationAudioGuide&) = delete;
    CalibrationAudioGuide& operator=(const CalibrationAudioGuide&) = delete;

    /**
     * @brief Launches the acoustic calibration sequence on a background thread.
     * @param on_phase_changed Callback invoked on each phase transition.
     * @param on_complete Callback invoked before playing success sound to finalize thresholds.
     * @return true if started, false if already running.
     */
    bool start(
        std::function<void(CalibrationPhase)> on_phase_changed,
        std::function<void()> on_complete
    );

    /**
     * @brief Immediately aborts audio playback, halts the sequence, and resets to Pending.
     */
    void stop();

    CalibrationPhase phase() const { return phase_.load(); }
    bool is_measuring_open() const { return is_measuring_open_.load(); }
    bool is_measuring_closed() const { return is_measuring_closed_.load(); }
    bool is_measuring() const { return is_measuring_open_.load() || is_measuring_closed_.load(); }
    bool is_active() const { return is_active_.load(); }

    const CalibrationAudioConfig& config() const { return config_; }

private:
    void sequence_worker(
        std::function<void(CalibrationPhase)> on_phase_changed,
        std::function<void()> on_complete
    );

    bool interruptible_sleep(std::chrono::milliseconds dur);

    CalibrationAudioConfig config_;
    std::atomic<CalibrationPhase> phase_{CalibrationPhase::Pending};
    std::atomic<bool> is_measuring_open_{false};
    std::atomic<bool> is_measuring_closed_{false};
    std::atomic<bool> is_active_{false};
    std::atomic<bool> cancel_requested_{false};
    std::thread worker_thread_;
};

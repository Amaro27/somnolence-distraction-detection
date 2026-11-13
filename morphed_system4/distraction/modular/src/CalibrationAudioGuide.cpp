#include "CalibrationAudioGuide.hpp"

#include <iostream>

CalibrationAudioGuide::CalibrationAudioGuide(CalibrationAudioConfig config)
    : config_(std::move(config)) {}

CalibrationAudioGuide::~CalibrationAudioGuide() {
    stop();
}

bool CalibrationAudioGuide::interruptible_sleep(std::chrono::milliseconds dur) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < dur) {
        if (cancel_requested_.load()) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return !cancel_requested_.load();
}

bool CalibrationAudioGuide::start(
    std::function<void(CalibrationPhase)> on_phase_changed,
    std::function<void()> on_complete
) {
    if (is_active_.load()) {
        return false;
    }

    // Clean up any lingering previous thread
    stop();

    cancel_requested_.store(false);
    is_active_.store(true);
    phase_.store(CalibrationPhase::CalibOpen);

    worker_thread_ = std::thread(
        &CalibrationAudioGuide::sequence_worker,
        this,
        std::move(on_phase_changed),
        std::move(on_complete)
    );

    return true;
}

void CalibrationAudioGuide::stop() {
    cancel_requested_.store(true);
    SoundPlayer::stop();

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    is_measuring_open_.store(false);
    is_measuring_closed_.store(false);
    is_active_.store(false);
}

void CalibrationAudioGuide::sequence_worker(
    std::function<void(CalibrationPhase)> on_phase_changed,
    std::function<void()> on_complete
) {
    auto notify_phase = [&](CalibrationPhase p) {
        phase_.store(p);
        if (on_phase_changed) {
            on_phase_changed(p);
        }
    };

    try {
        // ── Phase 1: Neutral Head Pose & Open Eyes ─────────────────────────
        notify_phase(CalibrationPhase::CalibOpen);
        std::cout << "\n=======================================================\n"
                  << "[+] Calibration Phase 1: Guided Instructions\n"
                  << "    Look straight ahead at the road with eyes open.\n"
                  << "=======================================================\n" << std::flush;

        // 1A. Play Phase 1 Guided Instructions (calib_phase_1.wav)
        if (!SoundPlayer::play_sync(config_.phase1_instructions_path)) {
            if (cancel_requested_.load()) return;
        }
        if (!interruptible_sleep(config_.silence_gap)) return;

        // 1B. Play Phase 1 Preparation Tone (sine-415-Hz-1-second-padded.wav)
        std::cout << "[+] Calibration: Playing preparation tone...\n" << std::flush;
        if (!SoundPlayer::play_sync(config_.phase1_tone_path)) {
            if (cancel_requested_.load()) return;
        }
        if (!interruptible_sleep(config_.silence_gap)) return;

        // 1C. Active Measurement Window: 8 beeps over 4 seconds
        std::cout << "[+] Calibration: Active measurement window (Phase 1). Accumulating samples...\n" << std::flush;
        is_measuring_open_.store(true);
        for (size_t i = 0; i < config_.num_beeps; ++i) {
            if (cancel_requested_.load()) {
                is_measuring_open_.store(false);
                return;
            }
            SoundPlayer::play_async(config_.cadence_beep_path);
            if (!interruptible_sleep(config_.beep_interval)) {
                is_measuring_open_.store(false);
                return;
            }
        }
        is_measuring_open_.store(false);
        std::cout << "[+] Calibration: Phase 1 measurement completed.\n" << std::flush;

        // ── Transition ───────────────────────────────────────────────────────
        notify_phase(CalibrationPhase::Transition);
        std::cout << "\n=======================================================\n"
                  << "[+] Transition: CLOSE EYES NOW AND HOLD\n"
                  << "=======================================================\n" << std::flush;
        // Transition tone removed per user instruction.
        if (!interruptible_sleep(config_.transition_silence)) return;

        // ── Phase 2: Closed Eyes Floor Baseline ─────────────────────────────
        notify_phase(CalibrationPhase::CalibClosed);
        std::cout << "\n=======================================================\n"
                  << "[+] Calibration Phase 2: Guided Instructions\n"
                  << "    Keep eyes gently closed.\n"
                  << "=======================================================\n" << std::flush;

        // 2A. Play Phase 2 Guided Instructions (calib_phase_2.wav)
        // Replaces first tone in Phase 2 per user instruction.
        if (!SoundPlayer::play_sync(config_.phase2_instructions_path)) {
            if (cancel_requested_.load()) return;
        }
        if (!interruptible_sleep(config_.silence_gap)) return;

        // 2B. Active Measurement Window: 8 beeps over 4 seconds
        std::cout << "[+] Calibration: Active measurement window (Phase 2). Accumulating samples...\n" << std::flush;
        is_measuring_closed_.store(true);
        for (size_t i = 0; i < config_.num_beeps; ++i) {
            if (cancel_requested_.load()) {
                is_measuring_closed_.store(false);
                return;
            }
            SoundPlayer::play_async(config_.cadence_beep_path);
            if (!interruptible_sleep(config_.beep_interval)) {
                is_measuring_closed_.store(false);
                return;
            }
        }
        is_measuring_closed_.store(false);
        std::cout << "[+] Calibration: Phase 2 measurement completed.\n" << std::flush;

        // ── Completion ───────────────────────────────────────────────────────
        if (on_complete) {
            on_complete();
        }

        std::cout << "[+] Calibration: Playing success chime...\n" << std::flush;
        SoundPlayer::play_sync(config_.success_sound_path);

        notify_phase(CalibrationPhase::Completed);
        is_active_.store(false);
        std::cout << "\n=======================================================\n"
                  << "[+] Calibration sequence fully completed!\n"
                  << "=======================================================\n" << std::flush;

    } catch (const std::exception& e) {
        std::cerr << "[!] CalibrationAudioGuide Exception: " << e.what() << "\n" << std::flush;
        is_measuring_open_.store(false);
        is_measuring_closed_.store(false);
        is_active_.store(false);
    }
}

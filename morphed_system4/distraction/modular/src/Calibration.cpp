#include "Calibration.hpp"
#include "FaceGeometry.hpp"
#include "MathUtils.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

namespace {

bool pause_interruptible(std::chrono::milliseconds dur, const std::atomic<bool>& shutdown) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < dur) {
        if (shutdown.load()) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return !shutdown.load();
}

bool measure_window(
    CalibrationContext& ctx,
    const CalibrationAudioConfig& cfg,
    const std::function<void(const FacialLandmarks68&)>& on_landmarks
) {
    cv::Mat frame;
    for (size_t i = 0; i < cfg.num_beeps; ++i) {
        if (ctx.shutdown.load()) {
            return false;
        }

        auto beep_start = std::chrono::steady_clock::now();
        ctx.audio.play(cfg.cadence_beep_path);

        while (true) {
            if (ctx.shutdown.load()) {
                return false;
            }
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - beep_start);
            if (elapsed >= cfg.beep_interval) {
                break;
            }

            auto remaining = cfg.beep_interval - elapsed;
            auto wait_dur = std::min(remaining, std::chrono::milliseconds(50));

            if (ctx.mailbox.get_latest(frame, wait_dur)) {
                auto lm_opt = ctx.face.detect(frame, std::chrono::steady_clock::now());
                if (lm_opt.has_value()) {
                    on_landmarks(*lm_opt);
                }
            } else if (ctx.mailbox.is_stopped()) {
                return false;
            }
        }
    }
    return !ctx.shutdown.load();
}

} // namespace

std::optional<CalibrationResult> run_calibration(
    CalibrationContext& ctx,
    const CalibrationAudioConfig& cfg
) {
    // ── Phase 1: Neutral Head Pose & Open Eyes ─────────────────────────────
    std::cout << "\n=======================================================\n"
              << "[+] Calibration Phase 1: Guided Instructions\n"
              << "    Look straight ahead at the road with eyes open.\n"
              << "=======================================================\n" << std::flush;

    ctx.audio.play_blocking(cfg.phase1_instructions_path);
    if (ctx.shutdown.load()) return std::nullopt;
    if (!pause_interruptible(cfg.silence_gap, ctx.shutdown)) return std::nullopt;

    std::cout << "[+] Calibration: Playing preparation tone...\n" << std::flush;
    ctx.audio.play_blocking(cfg.phase1_tone_path);
    if (ctx.shutdown.load()) return std::nullopt;
    if (!pause_interruptible(cfg.silence_gap, ctx.shutdown)) return std::nullopt;

    std::cout << "[+] Calibration: Active measurement window (Phase 1). Accumulating samples...\n" << std::flush;
    std::vector<double> open_samples;
    std::vector<double> rho_samples;
    bool first_attempt_p1 = true;

    while (open_samples.size() < cfg.min_samples_per_window) {
        if (ctx.shutdown.load()) return std::nullopt;
        if (!first_attempt_p1) {
            std::cout << "[CALIBRATION] Phase 1: Insufficient face samples ("
                      << open_samples.size() << "/" << cfg.min_samples_per_window
                      << "). Repeating measurement beeps...\n" << std::flush;
        }
        first_attempt_p1 = false;

        if (!measure_window(ctx, cfg, [&](const FacialLandmarks68& lm) {
                double ear = FaceGeometry::compute_ear(lm);
                double rho = FaceGeometry::compute_rho(lm);
                if (ear > 0.0) open_samples.push_back(ear);
                if (rho > 0.0) rho_samples.push_back(rho);
            })) {
            return std::nullopt;
        }
    }
    std::cout << "[+] Calibration: Phase 1 measurement completed (" << open_samples.size() << " samples).\n" << std::flush;

    // ── Transition ─────────────────────────────────────────────────────────
    std::cout << "\n=======================================================\n"
              << "[+] Transition: CLOSE EYES NOW AND HOLD\n"
              << "=======================================================\n" << std::flush;
    if (!pause_interruptible(cfg.transition_silence, ctx.shutdown)) return std::nullopt;

    // ── Phase 2: Closed Eyes Floor Baseline ────────────────────────────────
    std::cout << "\n=======================================================\n"
              << "[+] Calibration Phase 2: Guided Instructions\n"
              << "    Keep eyes gently closed.\n"
              << "=======================================================\n" << std::flush;

    ctx.audio.play_blocking(cfg.phase2_instructions_path);
    if (ctx.shutdown.load()) return std::nullopt;
    if (!pause_interruptible(cfg.silence_gap, ctx.shutdown)) return std::nullopt;

    std::cout << "[+] Calibration: Active measurement window (Phase 2). Accumulating samples...\n" << std::flush;
    std::vector<double> closed_samples;
    bool first_attempt_p2 = true;

    while (closed_samples.size() < cfg.min_samples_per_window) {
        if (ctx.shutdown.load()) return std::nullopt;
        if (!first_attempt_p2) {
            std::cout << "[CALIBRATION] Phase 2: Insufficient face samples ("
                      << closed_samples.size() << "/" << cfg.min_samples_per_window
                      << "). Repeating measurement beeps...\n" << std::flush;
        }
        first_attempt_p2 = false;

        if (!measure_window(ctx, cfg, [&](const FacialLandmarks68& lm) {
                double ear = FaceGeometry::compute_ear(lm);
                if (ear > 0.0) closed_samples.push_back(ear);
            })) {
            return std::nullopt;
        }
    }
    std::cout << "[+] Calibration: Phase 2 measurement completed (" << closed_samples.size() << " samples).\n" << std::flush;

    // ── Completion ─────────────────────────────────────────────────────────
    if (ctx.shutdown.load()) return std::nullopt;

    std::cout << "[+] Calibration: Playing success chime...\n" << std::flush;
    ctx.audio.play(cfg.success_sound_path);

    return CalibrationResult{
        MathUtils::compute_mean(open_samples),
        MathUtils::compute_mean(closed_samples),
        MathUtils::compute_median(rho_samples)
    };
}

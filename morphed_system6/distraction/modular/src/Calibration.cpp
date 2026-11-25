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
    const std::string& phase_label,
    size_t initial_count,
    const std::function<size_t(const FacialLandmarks68&)>& on_landmarks
) {
    cv::Mat frame;
    size_t collected = initial_count;
    const auto window_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < cfg.num_beeps; ++i) {
        if (ctx.shutdown.load()) {
            return false;
        }

        ctx.audio.play(cfg.cadence_beep_path, 3.0f);

        const auto next_beep_deadline =
            window_start + cfg.beep_interval * static_cast<long long>(i + 1);

        while (true) {
            if (ctx.shutdown.load()) {
                return false;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_beep_deadline) {
                break;
            }

            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(next_beep_deadline - now);
            // Avoid starting ~21ms Dlib face detection when <=30ms remain before the next beep
            if (remaining <= std::chrono::milliseconds(30)) {
                std::this_thread::sleep_for(std::min(remaining, std::chrono::milliseconds(10)));
                continue;
            }

            const auto wait_dur = std::min(remaining - std::chrono::milliseconds(30), std::chrono::milliseconds(50));
            if (ctx.mailbox.get_latest(frame, wait_dur)) {
                auto lm_opt = ctx.face.detect(frame, std::chrono::steady_clock::now());
                if (lm_opt.has_value()) {
                    collected = on_landmarks(*lm_opt);
                }
                if (ctx.visualizer) {
                    ctx.visualizer->update_calibration(phase_label, collected, cfg.min_samples_per_window, lm_opt);
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
#ifdef DEBUG_VIEWS
    std::cout << "\n=======================================================\n"
              << "[+] Calibration Phase 1: Guided Instructions\n"
              << "    Look straight ahead at the road with eyes open.\n"
              << "=======================================================\n" << std::flush;
#endif
    if (ctx.visualizer) {
        ctx.visualizer->update_calibration("Phase 1: Look straight ahead (eyes open)", 0, cfg.min_samples_per_window);
    }

    ctx.audio.play_blocking(cfg.phase1_instructions_path, 6.0f); // +6dB gain for voice instructions
    if (ctx.shutdown.load()) return std::nullopt;
    if (!pause_interruptible(cfg.silence_gap, ctx.shutdown)) return std::nullopt;

#ifdef DEBUG_VIEWS
    std::cout << "[+] Calibration: Playing preparation tone...\n" << std::flush;
#endif
    ctx.audio.play_blocking(cfg.phase1_tone_path);
    if (ctx.shutdown.load()) return std::nullopt;
    if (!pause_interruptible(cfg.silence_gap, ctx.shutdown)) return std::nullopt;

#ifdef DEBUG_VIEWS
    std::cout << "[+] Calibration: Active measurement window (Phase 1). Accumulating samples...\n" << std::flush;
#endif
    if (ctx.visualizer) {
        ctx.visualizer->update_calibration("Phase 1: Measuring open eyes & pitch", 0, cfg.min_samples_per_window);
    }
    std::vector<double> open_samples;
    std::vector<double> rho_samples;
    bool first_attempt_p1 = true;

    while (open_samples.size() < cfg.min_samples_per_window) {
        if (ctx.shutdown.load()) return std::nullopt;
        if (!first_attempt_p1) {
#ifdef DEBUG_VIEWS
            std::cout << "[CALIBRATION] Phase 1: Insufficient face samples ("
                      << open_samples.size() << "/" << cfg.min_samples_per_window
                      << "). Repeating measurement beeps...\n" << std::flush;
#endif
        }
        first_attempt_p1 = false;

        if (!measure_window(ctx, cfg, "Phase 1: Measuring open eyes & pitch", open_samples.size(), [&](const FacialLandmarks68& lm) {
                double ear = FaceGeometry::compute_ear(lm);
                double rho = FaceGeometry::compute_rho(lm);
                if (ear > 0.0) open_samples.push_back(ear);
                if (rho > 0.0) rho_samples.push_back(rho);
                return open_samples.size();
            })) {
            return std::nullopt;
        }
    }
#ifdef DEBUG_VIEWS
    std::cout << "[+] Calibration: Phase 1 measurement completed (" << open_samples.size() << " samples).\n" << std::flush;

    // ── Transition ─────────────────────────────────────────────────────────
    std::cout << "\n=======================================================\n"
              << "[+] Transition: CLOSE EYES NOW AND HOLD\n"
              << "=======================================================\n" << std::flush;
#endif
    if (ctx.visualizer) {
        ctx.visualizer->update_calibration("Transition: CLOSE EYES NOW AND HOLD", open_samples.size(), cfg.min_samples_per_window);
    }
    if (!pause_interruptible(cfg.transition_silence, ctx.shutdown)) return std::nullopt;

    // ── Phase 2: Closed Eyes Floor Baseline ────────────────────────────────
#ifdef DEBUG_VIEWS
    std::cout << "\n=======================================================\n"
              << "[+] Calibration Phase 2: Guided Instructions\n"
              << "    Keep eyes gently closed.\n"
              << "=======================================================\n" << std::flush;
#endif
    if (ctx.visualizer) {
        ctx.visualizer->update_calibration("Phase 2: Keep eyes gently closed", 0, cfg.min_samples_per_window);
    }

    ctx.audio.play_blocking(cfg.phase2_instructions_path, 6.0f);
    if (ctx.shutdown.load()) return std::nullopt;
    if (!pause_interruptible(cfg.silence_gap, ctx.shutdown)) return std::nullopt;

#ifdef DEBUG_VIEWS
    std::cout << "[+] Calibration: Playing preparation tone...\n" << std::flush;
#endif
    ctx.audio.play_blocking(cfg.phase2_tone_path);
    if (ctx.shutdown.load()) return std::nullopt;
    if (!pause_interruptible(cfg.silence_gap, ctx.shutdown)) return std::nullopt;

#ifdef DEBUG_VIEWS
    std::cout << "[+] Calibration: Active measurement window (Phase 2). Accumulating samples...\n" << std::flush;
#endif
    if (ctx.visualizer) {
        ctx.visualizer->update_calibration("Phase 2: Measuring closed eyes floor", 0, cfg.min_samples_per_window);
    }
    std::vector<double> closed_samples;
    bool first_attempt_p2 = true;

    while (closed_samples.size() < cfg.min_samples_per_window) {
        if (ctx.shutdown.load()) return std::nullopt;
        if (!first_attempt_p2) {
#ifdef DEBUG_VIEWS
            std::cout << "[CALIBRATION] Phase 2: Insufficient face samples ("
                      << closed_samples.size() << "/" << cfg.min_samples_per_window
                      << "). Repeating measurement beeps...\n" << std::flush;
#endif
        }
        first_attempt_p2 = false;

        if (!measure_window(ctx, cfg, "Phase 2: Measuring closed eyes floor", closed_samples.size(), [&](const FacialLandmarks68& lm) {
                double ear = FaceGeometry::compute_ear(lm);
                if (ear > 0.0) closed_samples.push_back(ear);
                return closed_samples.size();
            })) {
            return std::nullopt;
        }
    }
#ifdef DEBUG_VIEWS
    std::cout << "[+] Calibration: Phase 2 measurement completed (" << closed_samples.size() << " samples).\n" << std::flush;
#endif

    // ── Completion ─────────────────────────────────────────────────────────
    if (ctx.shutdown.load()) return std::nullopt;

#ifdef DEBUG_VIEWS
    std::cout << "[+] Calibration: Playing success chime...\n" << std::flush;
#endif
    if (ctx.visualizer) {
        ctx.visualizer->update_calibration("Calibration complete!", closed_samples.size(), cfg.min_samples_per_window);
    }
    ctx.audio.play(cfg.success_sound_path);

    return CalibrationResult{
        MathUtils::compute_mean(open_samples),
        MathUtils::compute_mean(closed_samples),
        MathUtils::compute_median(rho_samples)
    };
}

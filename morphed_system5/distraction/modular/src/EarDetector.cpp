#include "EarDetector.hpp"
#include "FaceGeometry.hpp"
#include "MathUtils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

EarDetector::EarDetector(EarConfig config)
    : config_(std::move(config)),
      last_time_(std::chrono::steady_clock::now()),
      last_microsleep_cue_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)),
      last_perclos_cue_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

void EarDetector::clear_perclos() noexcept {
    perclos_buffer_.clear();
    closed_samples_count_ = 0;
    has_perclos_start_ = false;
    perclos_pct_ = 0.0;
}

void EarDetector::reset() {
    calibrated_ = false;
    buffer_raw_.clear();
    buffer_median_.clear();
    clear_perclos();
    in_blink_cycle_ = false;
    reached_effective_closure_ = false;
    closure_accum_sec_ = 0.0;
    last_ear_raw_ = 0.0;
    has_last_time_ = false;
    microsleep_alarm_active_ = false;
    last_blink_duration_ms_ = 0.0;
}

double EarDetector::compute_ear(const FacialLandmarks68& landmarks) {
    return FaceGeometry::compute_ear(landmarks);
}

void EarDetector::apply_baseline(
    double ear_open,
    double ear_closed,
    std::chrono::steady_clock::time_point now
) {
    ear_base_open_ = (ear_open > 0.0) ? ear_open : 0.30;
    ear_base_closed_ = (ear_closed > 0.0) ? ear_closed : 0.15;

    double dynamic_range = std::max(config_.min_dynamic_range, ear_base_open_ - ear_base_closed_);

    thresh_descent_ = ear_base_closed_ + (config_.factor_descenso * dynamic_range);
    thresh_closure_ = ear_base_closed_ + (config_.sensibilidad_cierre * dynamic_range);
    floor_closure_ = thresh_closure_;

    current_ear_open_ = ear_base_open_;

    buffer_raw_.clear();
    buffer_median_.clear();
    for (size_t i = 0; i < config_.median_window; ++i) buffer_raw_.push_back(ear_base_open_);
    for (size_t i = 0; i < config_.mov_avg_window; ++i) buffer_median_.push_back(ear_base_open_);

    clear_perclos();
    in_blink_cycle_ = false;
    reached_effective_closure_ = false;
    closure_accum_sec_ = 0.0;
    last_ear_raw_ = ear_base_open_;
    has_last_time_ = false;
    microsleep_alarm_active_ = false;
    last_blink_duration_ms_ = 0.0;
    (void)now;
    calibrated_ = true;

#ifdef DEBUG_VIEWS
    std::cout << "\n=======================================================\n"
              << "[+] EarDetector: Calibration Complete!\n"
              << "    Open Baseline    : " << std::fixed << std::setprecision(3) << ear_base_open_ << "\n"
              << "    Closed Floor     : " << ear_base_closed_ << "\n"
              << "    Descent Thresh   : " << thresh_descent_ << " (70% aperture)\n"
              << "    Closure Thresh   : " << thresh_closure_ << " (20% aperture)\n"
              << "=======================================================\n" << std::flush;
#endif
}

EarAssessment EarDetector::process(
    const FacialLandmarks68& landmarks,
    std::chrono::steady_clock::time_point now,
    bool suppress_closure,
    bool record_perclos
) {
    // 1. Calculate Eye Aspect Ratio (EAR) for both eyes
    double ear_raw = compute_ear(landmarks);
    last_ear_raw_ = ear_raw;
    std::vector<CueEvent> cues;

    // 2. Calculate dynamic delta time dt
    const double dt = MathUtils::compute_clamped_dt(now, last_time_, has_last_time_);

    // 3. Active Monitoring: Cascaded Adaptive Filtering
    if (!in_blink_cycle_ && ear_raw > thresh_descent_) {
        buffer_raw_.push_back(ear_raw);
        if (buffer_raw_.size() > config_.median_window) {
            buffer_raw_.pop_front();
        }

        if (buffer_raw_.size() == config_.median_window) {
            double ear_med = MathUtils::compute_median(buffer_raw_);
            buffer_median_.push_back(ear_med);
            if (buffer_median_.size() > config_.mov_avg_window) {
                buffer_median_.pop_front();
            }

            if (buffer_median_.size() >= config_.min_samples_for_update) {
                current_ear_open_ = MathUtils::compute_mean(buffer_median_);

                double dynamic_range = std::max(config_.min_dynamic_range, current_ear_open_ - ear_base_closed_);
                thresh_descent_ = ear_base_closed_ + (config_.factor_descenso * dynamic_range);
                thresh_closure_ = ear_base_closed_ + (config_.sensibilidad_cierre * dynamic_range);
            }
        }
    }

    // Safety floor: adaptive closure threshold cannot fall below calibrated baseline floor
    if (thresh_closure_ < floor_closure_) {
        thresh_closure_ = floor_closure_;
    }

    double ear_median_val = buffer_raw_.empty() ? ear_raw : MathUtils::compute_median(buffer_raw_);
    double ear_mov_avg_val = buffer_median_.empty() ? ear_raw : MathUtils::compute_mean(buffer_median_);

    // 4. Blink Cycle & Micro-Sleep Tracking (>500ms -> Strong cue)
    if (suppress_closure) {
        in_blink_cycle_ = false;
        reached_effective_closure_ = false;
        microsleep_alarm_active_ = false;
        closure_accum_sec_ = 0.0;
    } else if (ear_raw < thresh_descent_) {
        if (!in_blink_cycle_) {
            in_blink_cycle_ = true;
            reached_effective_closure_ = false;
            closure_accum_sec_ = 0.0;
        }

        if (ear_raw <= thresh_closure_) {
            if (!reached_effective_closure_) {
                reached_effective_closure_ = true;
                closure_accum_sec_ = dt;
            } else {
                closure_accum_sec_ += dt;
            }

            const double microsleep_thresh_sec = std::chrono::duration<double>(config_.microsleep_thresh).count();
            if (closure_accum_sec_ >= microsleep_thresh_sec) {
                if (!microsleep_alarm_active_) {
                    microsleep_alarm_active_ = true;
                    last_microsleep_cue_time_ = now;
                    if (calibrated_) {
                        cues.push_back(CueEvent{
                            CueType::BlinkDuration,
                            CueStrength::Strong,
                            now,
                            "Micro-sleep: eye closure held shut for >= " +
                                std::to_string(config_.microsleep_thresh.count()) + " ms"
                        });
                    }
                }
            }
        } else {
            // Between thresh_closure_ and thresh_descent_: decay slightly rather than instant wipe
            closure_accum_sec_ = std::max(0.0, closure_accum_sec_ - 1.5 * dt);
        }
    } else {
        if (in_blink_cycle_) {
            if (reached_effective_closure_) {
                last_blink_duration_ms_ = std::max(0.0, closure_accum_sec_ * 1000.0);
            }
            in_blink_cycle_ = false;
            reached_effective_closure_ = false;
            microsleep_alarm_active_ = false;
            closure_accum_sec_ = 0.0;
        }
    }

    // 5. True Rolling 30-Second PERCLOS Tracking (O(1) running count)
    // PERCLOS is ONLY recorded if calibrated AND record_perclos is true (e.g. moving vehicle)
    if (calibrated_ && record_perclos) {
        int is_closed_sample = (!suppress_closure && ear_raw <= thresh_closure_) ? 1 : 0;
        perclos_buffer_.push_back({now, is_closed_sample});
        if (is_closed_sample == 1) {
            ++closed_samples_count_;
        }

        // Prune samples older than perclos_window (default 30 seconds)
        auto perclos_cutoff = now - config_.perclos_window;
        while (!perclos_buffer_.empty() && perclos_buffer_.front().first < perclos_cutoff) {
            if (perclos_buffer_.front().second == 1) {
                assert(closed_samples_count_ > 0);
                --closed_samples_count_;
            }
            perclos_buffer_.pop_front();
        }

        if (!has_perclos_start_) {
            perclos_start_time_ = now;
            has_perclos_start_ = true;
        }

        const auto elapsed = now - perclos_start_time_;
        const bool perclos_warmed_up = (elapsed >= config_.perclos_window);
        const double warmup_remaining_sec = perclos_warmed_up ? 0.0 :
            std::max(0.0, std::chrono::duration<double>(config_.perclos_window - elapsed).count());

        if (!perclos_buffer_.empty()) {
            perclos_pct_ = (static_cast<double>(closed_samples_count_) / static_cast<double>(perclos_buffer_.size())) * 100.0;
        } else {
            perclos_pct_ = 0.0;
        }

        const double reported_perclos_pct = perclos_pct_;
        const bool reported_warmed_up = perclos_warmed_up;
        const double reported_warmup_remaining = warmup_remaining_sec;

        if (perclos_warmed_up && !suppress_closure && in_blink_cycle_) {
            auto elapsed_since_cue = std::chrono::duration_cast<std::chrono::seconds>(now - last_perclos_cue_time_);
            if (elapsed_since_cue >= std::chrono::seconds(5) && perclos_pct_ >= config_.perclos_high_thresh) {
                last_perclos_cue_time_ = now;
                cues.push_back(CueEvent{
                    CueType::Perclos,
                    CueStrength::Strong,
                    now,
                    "High PERCLOS: " + std::to_string(static_cast<int>(perclos_pct_)) + "% (>= 15%)"
                });
                clear_perclos();
            }
        }

        return EarAssessment{
            ear_raw,
            ear_median_val,
            ear_mov_avg_val,
            reported_perclos_pct,
            reported_warmed_up,
            reported_warmup_remaining,
            last_blink_duration_ms_,
            in_blink_cycle_,
            cues
        };
    } else {
        // When uncalibrated or stationary: ensure PERCLOS buffer is completely cleared
        clear_perclos();

        return EarAssessment{
            ear_raw,
            ear_median_val,
            ear_mov_avg_val,
            0.0,
            false,
            std::chrono::duration<double>(config_.perclos_window).count(),
            last_blink_duration_ms_,
            in_blink_cycle_,
            cues
        };
    }
}

EarAssessment EarDetector::on_face_lost(
    std::chrono::steady_clock::time_point now
) {
    in_blink_cycle_ = false;
    reached_effective_closure_ = false;
    microsleep_alarm_active_ = false;
    closure_accum_sec_ = 0.0;
    has_last_time_ = false;

    if (!calibrated_) {
        clear_perclos();
        double ear_median_val = buffer_raw_.empty() ? last_ear_raw_ : MathUtils::compute_median(buffer_raw_);
        double ear_mov_avg_val = buffer_median_.empty() ? last_ear_raw_ : MathUtils::compute_mean(buffer_median_);
        return EarAssessment{
            last_ear_raw_,
            ear_median_val,
            ear_mov_avg_val,
            0.0,
            false,
            std::chrono::duration<double>(config_.perclos_window).count(),
            last_blink_duration_ms_,
            false,
            {}
        };
    }

    // Prune samples older than perclos_window (default 30 seconds)
    auto perclos_cutoff = now - config_.perclos_window;
    while (!perclos_buffer_.empty() && perclos_buffer_.front().first < perclos_cutoff) {
        if (perclos_buffer_.front().second == 1) {
            if (closed_samples_count_ > 0) {
                --closed_samples_count_;
            }
        }
        perclos_buffer_.pop_front();
    }

    if (!perclos_buffer_.empty()) {
        perclos_pct_ = (static_cast<double>(closed_samples_count_) / static_cast<double>(perclos_buffer_.size())) * 100.0;
    } else {
        perclos_pct_ = 0.0;
        has_perclos_start_ = false;
    }

    const auto elapsed = has_perclos_start_
        ? (now - perclos_start_time_)
        : std::chrono::steady_clock::duration::zero();
    const bool perclos_warmed_up = has_perclos_start_ && (elapsed >= config_.perclos_window);
    const double warmup_remaining_sec = perclos_warmed_up ? 0.0 :
        std::max(0.0, std::chrono::duration<double>(config_.perclos_window - elapsed).count());

    double ear_median_val = buffer_raw_.empty() ? last_ear_raw_ : MathUtils::compute_median(buffer_raw_);
    double ear_mov_avg_val = buffer_median_.empty() ? last_ear_raw_ : MathUtils::compute_mean(buffer_median_);

    return EarAssessment{
        last_ear_raw_,
        ear_median_val,
        ear_mov_avg_val,
        perclos_pct_,
        perclos_warmed_up,
        warmup_remaining_sec,
        last_blink_duration_ms_,
        false,
        {}
    };
}

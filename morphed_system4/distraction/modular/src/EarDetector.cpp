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
      last_microsleep_cue_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)),
      last_perclos_cue_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

void EarDetector::reset() {
    calibrated_ = false;
    buffer_raw_.clear();
    buffer_median_.clear();
    perclos_buffer_.clear();
    closed_samples_count_ = 0;
    in_blink_cycle_ = false;
    reached_effective_closure_ = false;
    microsleep_alarm_active_ = false;
    perclos_pct_ = 0.0;
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

    perclos_buffer_.clear();
    closed_samples_count_ = 0;
    in_blink_cycle_ = false;
    reached_effective_closure_ = false;
    microsleep_alarm_active_ = false;
    perclos_pct_ = 0.0;
    last_blink_duration_ms_ = 0.0;
    perclos_start_time_ = now;
    calibrated_ = true;

    std::cout << "\n=======================================================\n"
              << "[+] EarDetector: Calibration Complete!\n"
              << "    Open Baseline    : " << std::fixed << std::setprecision(3) << ear_base_open_ << "\n"
              << "    Closed Floor     : " << ear_base_closed_ << "\n"
              << "    Descent Thresh   : " << thresh_descent_ << " (70% aperture)\n"
              << "    Closure Thresh   : " << thresh_closure_ << " (20% aperture)\n"
              << "=======================================================\n" << std::flush;
}

EarAssessment EarDetector::process(
    const FacialLandmarks68& landmarks,
    std::chrono::steady_clock::time_point now
) {
    // 1. Calculate Eye Aspect Ratio (EAR) for both eyes
    double ear_raw = compute_ear(landmarks);
    std::vector<CueEvent> cues;

    // 2. Do not evaluate fatigue cues until calibrated
    if (!calibrated_) {
        return EarAssessment{
            ear_raw,
            ear_raw,
            ear_raw,
            0.0,
            0.0,
            false,
            cues
        };
    }

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
    if (ear_raw < thresh_descent_) {
        if (!in_blink_cycle_) {
            in_blink_cycle_ = true;
            reached_effective_closure_ = false;
        }

        if (ear_raw <= thresh_closure_) {
            if (!reached_effective_closure_) {
                reached_effective_closure_ = true;
                t_start_closure_ = now;
            }
            t_last_closed_ = now;

            auto held_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start_closure_);
            if (held_duration >= config_.microsleep_thresh) {
                if (!microsleep_alarm_active_) {
                    microsleep_alarm_active_ = true;
                    last_microsleep_cue_time_ = now;
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
        if (in_blink_cycle_) {
            if (reached_effective_closure_) {
                auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t_last_closed_ - t_start_closure_);
                last_blink_duration_ms_ = std::max(0.0, static_cast<double>(dur.count()));

                if (dur >= config_.microsleep_thresh && !microsleep_alarm_active_) {
                    cues.push_back(CueEvent{
                        CueType::BlinkDuration,
                        CueStrength::Strong,
                        now,
                        "Micro-sleep: blink duration was " + std::to_string(dur.count()) + " ms"
                    });
                }
            }
            in_blink_cycle_ = false;
            reached_effective_closure_ = false;
            microsleep_alarm_active_ = false;
        }
    }

    // 5. True Rolling 30-Second PERCLOS Tracking (O(1) running count)
    int is_closed_sample = (ear_raw <= thresh_closure_) ? 1 : 0;
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

    auto perclos_monitoring_time = std::chrono::duration_cast<std::chrono::seconds>(now - perclos_start_time_);
    bool perclos_warmed_up = perclos_monitoring_time >= config_.perclos_window;

    if (perclos_warmed_up && !perclos_buffer_.empty()) {
        perclos_pct_ = (static_cast<double>(closed_samples_count_) / static_cast<double>(perclos_buffer_.size())) * 100.0;

        auto elapsed_since_cue = std::chrono::duration_cast<std::chrono::seconds>(now - last_perclos_cue_time_);
        if (elapsed_since_cue >= std::chrono::seconds(5)) {
            if (perclos_pct_ >= config_.perclos_high_thresh) {
                last_perclos_cue_time_ = now;
                cues.push_back(CueEvent{
                    CueType::Perclos,
                    CueStrength::Strong,
                    now,
                    "High PERCLOS: " + std::to_string(static_cast<int>(perclos_pct_)) + "% (>= 15%)"
                });
            } else if (perclos_pct_ >= config_.perclos_medium_thresh) {
                last_perclos_cue_time_ = now;
                cues.push_back(CueEvent{
                    CueType::Perclos,
                    CueStrength::Light,
                    now,
                    "Medium PERCLOS: " + std::to_string(static_cast<int>(perclos_pct_)) + "% (7.5% - 15%)"
                });
            }
        }
    } else {
        perclos_pct_ = 0.0;
    }

    return EarAssessment{
        ear_raw,
        ear_median_val,
        ear_mov_avg_val,
        perclos_pct_,
        last_blink_duration_ms_,
        in_blink_cycle_,
        cues
    };
}

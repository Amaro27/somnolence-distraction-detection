#include "EarDetector.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {
double compute_median(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t n = vals.size();
    return (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) * 0.5;
}

double compute_mean(const std::vector<double>& vals) {
    if (vals.empty()) return 0.0;
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    return sum / static_cast<double>(vals.size());
}
} // namespace

EarDetector::EarDetector(EarConfig config)
    : config_(config),
      phase_start_(std::chrono::steady_clock::now()),
      last_microsleep_cue_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)),
      last_perclos_cue_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

bool EarDetector::start_calibration(std::chrono::steady_clock::time_point now) {
    if (phase_ != CalibrationPhase::Pending && phase_ != CalibrationPhase::Completed) {
        std::cout << "[EarDetector] Calibration is already in progress.\n";
        return false;
    }

    phase_ = CalibrationPhase::CalibOpen;
    phase_start_ = now;
    open_samples_.clear();
    closed_samples_.clear();
    buffer_raw_.clear();
    buffer_median_.clear();
    perclos_buffer_.clear();
    in_blink_cycle_ = false;
    microsleep_alarm_active_ = false;

    std::cout << "\n[+] EarDetector: Started Phase 1: Calibrating Open Eyes (~"
              << config_.phase_duration.count() << "s). Look forward with eyes comfortably open.\n";
    return true;
}

void EarDetector::reset() {
    phase_ = CalibrationPhase::Pending;
    open_samples_.clear();
    closed_samples_.clear();
    buffer_raw_.clear();
    buffer_median_.clear();
    perclos_buffer_.clear();
    in_blink_cycle_ = false;
    reached_effective_closure_ = false;
    microsleep_alarm_active_ = false;
    perclos_pct_ = 0.0;
    last_blink_duration_ms_ = 0.0;
}

double EarDetector::calculate_single_eye_ear(
    const cv::Point2f& p0, const cv::Point2f& p1,
    const cv::Point2f& p2, const cv::Point2f& p3,
    const cv::Point2f& p4, const cv::Point2f& p5
) const {
    double dist_a = cv::norm(p1 - p5);
    double dist_b = cv::norm(p2 - p4);
    double dist_c = cv::norm(p0 - p3);

    if (dist_c < 1e-6) {
        return 0.0;
    }

    return (dist_a + dist_b) / (2.0 * dist_c);
}

void EarDetector::update_calibration_fsm(double ear_raw, std::chrono::steady_clock::time_point now) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - phase_start_);

    if (phase_ == CalibrationPhase::CalibOpen) {
        if (ear_raw > 0.0) {
            open_samples_.push_back(ear_raw);
        }
        if (elapsed >= config_.phase_duration) {
            phase_ = CalibrationPhase::Transition;
            phase_start_ = now;
            std::cout << "\n[+] EarDetector: Phase 1 complete. TRANSITION: CLOSE EYES NOW AND HOLD (~"
                      << config_.transition_duration.count() << "s)...\n";
        }
    } else if (phase_ == CalibrationPhase::Transition) {
        if (elapsed >= config_.transition_duration) {
            phase_ = CalibrationPhase::CalibClosed;
            phase_start_ = now;
            std::cout << "\n[+] EarDetector: Started Phase 2: Calibrating Closed Eyes (~"
                      << config_.phase_duration.count() << "s). Keep eyes gently closed.\n";
        }
    } else if (phase_ == CalibrationPhase::CalibClosed) {
        if (ear_raw > 0.0) {
            closed_samples_.push_back(ear_raw);
        }
        if (elapsed >= config_.phase_duration) {
            compute_calibrated_thresholds();
            phase_ = CalibrationPhase::Completed;
            perclos_start_time_ = now;
        }
    }
}

void EarDetector::compute_calibrated_thresholds() {
    ear_base_open_ = open_samples_.empty() ? 0.30 : compute_mean(open_samples_);
    ear_base_closed_ = closed_samples_.empty() ? 0.15 : compute_mean(closed_samples_);

    double dynamic_range = std::max(config_.min_dynamic_range, ear_base_open_ - ear_base_closed_);

    thresh_descent_ = ear_base_closed_ + (config_.factor_descenso * dynamic_range);
    thresh_closure_ = ear_base_closed_ + (config_.sensibilidad_cierre * dynamic_range);
    floor_closure_ = thresh_closure_;

    current_ear_open_ = ear_base_open_;

    buffer_raw_.clear();
    buffer_median_.clear();
    for (size_t i = 0; i < config_.median_window; ++i) buffer_raw_.push_back(ear_base_open_);
    for (size_t i = 0; i < config_.mov_avg_window; ++i) buffer_median_.push_back(ear_base_open_);

    std::cout << "\n=======================================================\n"
              << "[+] EarDetector: Calibration Complete!\n"
              << "    Open Baseline    : " << std::fixed << std::setprecision(3) << ear_base_open_ << "\n"
              << "    Closed Floor     : " << ear_base_closed_ << "\n"
              << "    Descent Thresh   : " << thresh_descent_ << " (70% aperture)\n"
              << "    Closure Thresh   : " << thresh_closure_ << " (20% aperture)\n"
              << "=======================================================\n";
}

EarAssessment EarDetector::process(
    const FacialLandmarks68& landmarks,
    std::chrono::steady_clock::time_point now
) {
    // 1. Calculate Eye Aspect Ratio (EAR) for both eyes
    double ear_r = calculate_single_eye_ear(
        landmarks[36], landmarks[37], landmarks[38],
        landmarks[39], landmarks[40], landmarks[41]
    );

    double ear_l = calculate_single_eye_ear(
        landmarks[42], landmarks[43], landmarks[44],
        landmarks[45], landmarks[46], landmarks[47]
    );

    double ear_raw = (ear_r + ear_l) * 0.5;
    std::vector<CueEvent> cues;

    // 2. If calibration is active, advance state machine
    if (phase_ != CalibrationPhase::Completed) {
        if (phase_ != CalibrationPhase::Pending) {
            update_calibration_fsm(ear_raw, now);
        }

        return EarAssessment{
            ear_raw,
            ear_raw,
            ear_raw,
            0.0,
            0.0,
            false,
            phase_,
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
            std::vector<double> raw_vec(buffer_raw_.begin(), buffer_raw_.end());
            double ear_med = compute_median(raw_vec);

            buffer_median_.push_back(ear_med);
            if (buffer_median_.size() > config_.mov_avg_window) {
                buffer_median_.pop_front();
            }

            if (buffer_median_.size() >= config_.min_samples_for_update) {
                std::vector<double> med_vec(buffer_median_.begin(), buffer_median_.end());
                current_ear_open_ = compute_mean(med_vec);

                double dynamic_range = std::max(config_.min_dynamic_range, current_ear_open_ - ear_base_closed_);
                thresh_closure_ = ear_base_closed_ + (config_.sensibilidad_cierre * dynamic_range);
                thresh_descent_ = ear_base_closed_ + (config_.factor_descenso * dynamic_range);
            }
        }
    }

    // Safety floor: adaptive closure threshold cannot fall below calibrated baseline floor
    if (thresh_closure_ < floor_closure_) {
        thresh_closure_ = floor_closure_;
    }

    std::vector<double> raw_copy(buffer_raw_.begin(), buffer_raw_.end());
    double ear_median_val = raw_copy.empty() ? ear_raw : compute_median(raw_copy);
    std::vector<double> med_copy(buffer_median_.begin(), buffer_median_.end());
    double ear_mov_avg_val = med_copy.empty() ? ear_raw : compute_mean(med_copy);

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

    // 5. Rolling 30-Second PERCLOS Tracker
    int is_closed = (ear_raw <= thresh_closure_) ? 1 : 0;
    perclos_buffer_.emplace_back(now, is_closed);

    // Evict samples older than 30s
    auto cutoff = now - config_.perclos_window;
    while (!perclos_buffer_.empty() && perclos_buffer_.front().first < cutoff) {
        perclos_buffer_.pop_front();
    }

    auto warmup_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - perclos_start_time_);
    if (warmup_elapsed >= config_.perclos_window && !perclos_buffer_.empty()) {
        int closed_count = 0;
        for (const auto& sample : perclos_buffer_) {
            closed_count += sample.second;
        }
        perclos_pct_ = (static_cast<double>(closed_count) / static_cast<double>(perclos_buffer_.size())) * 100.0;

        // Rate-limit PERCLOS cue emission to at most once every 5 seconds
        auto time_since_last_perclos = std::chrono::duration<double>(now - last_perclos_cue_time_).count();
        if (time_since_last_perclos >= 5.0) {
            if (perclos_pct_ >= config_.perclos_high_thresh) {
                last_perclos_cue_time_ = now;
                std::stringstream ss;
                ss << std::fixed << std::setprecision(1) << perclos_pct_;
                cues.push_back(CueEvent{
                    CueType::Perclos,
                    CueStrength::Strong,
                    now,
                    "High PERCLOS: " + ss.str() + "% eye closure over 30s (>= 15%)"
                });
            } else if (perclos_pct_ >= config_.perclos_medium_thresh) {
                last_perclos_cue_time_ = now;
                std::stringstream ss;
                ss << std::fixed << std::setprecision(1) << perclos_pct_;
                cues.push_back(CueEvent{
                    CueType::Perclos,
                    CueStrength::Light,
                    now,
                    "Medium PERCLOS: " + ss.str() + "% eye closure over 30s (7.5% - 15%)"
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
        phase_,
        cues
    };
}

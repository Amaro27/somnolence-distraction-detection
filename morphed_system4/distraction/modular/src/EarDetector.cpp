#include "EarDetector.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {
double compute_median(std::vector<double>& vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t mid = vals.size() / 2;
    if (vals.size() % 2 == 0) {
        return (vals[mid - 1] + vals[mid]) * 0.5;
    }
    return vals[mid];
}

double compute_mean(const std::vector<double>& vals) {
    if (vals.empty()) return 0.0;
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    return sum / static_cast<double>(vals.size());
}
} // namespace

EarDetector::EarDetector(EarConfig config)
    : config_(std::move(config)),
      last_microsleep_cue_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)),
      last_perclos_cue_time_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

void EarDetector::reset() {
    calibrated_ = false;
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
) {
    double dist_a = cv::norm(p1 - p5);
    double dist_b = cv::norm(p2 - p4);
    double dist_c = cv::norm(p0 - p3);

    if (dist_c < 1e-6) {
        return 0.0;
    }

    return (dist_a + dist_b) / (2.0 * dist_c);
}

double EarDetector::compute_ear(const FacialLandmarks68& landmarks) {
    double ear_r = calculate_single_eye_ear(
        landmarks[36], landmarks[37], landmarks[38],
        landmarks[39], landmarks[40], landmarks[41]
    );

    double ear_l = calculate_single_eye_ear(
        landmarks[42], landmarks[43], landmarks[44],
        landmarks[45], landmarks[46], landmarks[47]
    );

    return (ear_r + ear_l) * 0.5;
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
                thresh_descent_ = ear_base_closed_ + (config_.factor_descenso * dynamic_range);
                thresh_closure_ = ear_base_closed_ + (config_.sensibilidad_cierre * dynamic_range);
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

    // 5. True Rolling 30-Second PERCLOS Tracking
    int is_closed_sample = (ear_raw <= thresh_closure_) ? 1 : 0;
    perclos_buffer_.push_back({now, is_closed_sample});

    // Prune samples older than perclos_window (default 30 seconds)
    auto perclos_cutoff = now - config_.perclos_window;
    while (!perclos_buffer_.empty() && perclos_buffer_.front().first < perclos_cutoff) {
        perclos_buffer_.pop_front();
    }

    auto perclos_monitoring_time = std::chrono::duration_cast<std::chrono::seconds>(now - perclos_start_time_);
    bool perclos_warmed_up = perclos_monitoring_time >= config_.perclos_window;

    if (perclos_warmed_up && !perclos_buffer_.empty()) {
        size_t closed_count = 0;
        for (const auto& entry : perclos_buffer_) {
            if (entry.second == 1) {
                closed_count++;
            }
        }
        perclos_pct_ = (static_cast<double>(closed_count) / static_cast<double>(perclos_buffer_.size())) * 100.0;

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

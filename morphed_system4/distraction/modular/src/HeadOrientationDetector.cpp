#include "HeadOrientationDetector.hpp"
#include "FaceGeometry.hpp"
#include "MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

HeadOrientationDetector::HeadOrientationDetector(HeadOrientationConfig config)
    : config_(config),
      last_time_(std::chrono::steady_clock::now()) {}

void HeadOrientationDetector::apply_baseline(double neutral_rho) {
    baseline_rho_ = (neutral_rho > 0.0) ? neutral_rho : 0.50;
    calibrated_ = true;
    in_droop_ = false;
    droop_cue_emitted_ = false;
    droop_accum_sec_ = 0.0;
    last_pitch_deg_ = 0.0;
    last_rho_ = baseline_rho_;
    has_last_time_ = false;
#ifdef DEBUG_VIEWS
    std::cout << "[HeadOrientationDetector] Calibration complete. Baseline rho: " << baseline_rho_ << "\n";
#endif
}

void HeadOrientationDetector::reset() {
    baseline_rho_ = 0.0;
    calibrated_ = false;
    in_droop_ = false;
    droop_cue_emitted_ = false;
    droop_accum_sec_ = 0.0;
    last_pitch_deg_ = 0.0;
    last_rho_ = 0.0;
    has_last_time_ = false;
}

double HeadOrientationDetector::compute_rho(const FacialLandmarks68& landmarks) {
    return FaceGeometry::compute_rho(landmarks);
}

HeadOrientationAssessment HeadOrientationDetector::process(
    const FacialLandmarks68& landmarks,
    std::chrono::steady_clock::time_point now,
    bool suppress_droop
) {
    const double dt = MathUtils::compute_clamped_dt(now, last_time_, has_last_time_);
    double rho = compute_rho(landmarks);
    double pitch_deg = 0.0;
    std::optional<CueEvent> cue = std::nullopt;

    last_rho_ = rho;

    if (calibrated_) {
        double delta_rho = rho - baseline_rho_;
        pitch_deg = std::atan2(-delta_rho, config_.kp_pitch) * (180.0 / CV_PI);
        last_pitch_deg_ = pitch_deg;

        // Time-based droop detection (pitch <= -20 deg sustained for >= 1.5s)
        if (!suppress_droop && pitch_deg <= config_.pitch_droop_thresh) {
            in_droop_ = true;
            const double target_sec = std::chrono::duration<double>(config_.droop_sustained_duration).count();
            droop_accum_sec_ = std::min(target_sec + 0.5, droop_accum_sec_ + dt);

            if (droop_accum_sec_ >= target_sec && !droop_cue_emitted_) {
                droop_cue_emitted_ = true;
                cue = CueEvent{
                    CueType::HeadDroop,
                    CueStrength::Light,
                    now,
                    "Head droop (" + std::to_string(static_cast<int>(pitch_deg)) + " deg) sustained for >= 1.5s"
                };
            }
        } else {
            droop_accum_sec_ = std::max(0.0, droop_accum_sec_ - 2.0 * dt);
            if (droop_accum_sec_ <= 0.0) {
                in_droop_ = false;
                droop_cue_emitted_ = false;
            }
        }
    } else {
        last_pitch_deg_ = 0.0;
        in_droop_ = false;
        droop_cue_emitted_ = false;
        droop_accum_sec_ = 0.0;
    }

    return HeadOrientationAssessment{
        pitch_deg,
        rho,
        in_droop_,
        cue
    };
}

HeadOrientationAssessment HeadOrientationDetector::on_face_lost(
    std::chrono::steady_clock::time_point now
) {
    const double dt = MathUtils::compute_clamped_dt(now, last_time_, has_last_time_);
    if (droop_accum_sec_ > 0.0) {
        droop_accum_sec_ = std::max(0.0, droop_accum_sec_ - 1.0 * dt);
        if (droop_accum_sec_ <= 0.0) {
            in_droop_ = false;
            droop_cue_emitted_ = false;
        }
    }

    return HeadOrientationAssessment{
        last_pitch_deg_,
        last_rho_,
        in_droop_,
        std::nullopt
    };
}

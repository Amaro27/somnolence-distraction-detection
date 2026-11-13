#include "HeadOrientationDetector.hpp"

#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>

HeadOrientationDetector::HeadOrientationDetector(HeadOrientationConfig config)
    : config_(config),
      droop_start_time_(std::chrono::steady_clock::now()) {}

void HeadOrientationDetector::apply_baseline(double neutral_rho) {
    baseline_rho_ = (neutral_rho > 0.0) ? neutral_rho : 0.50;
    calibrated_ = true;
    in_droop_ = false;
    droop_cue_emitted_ = false;
    std::cout << "[HeadOrientationDetector] Calibration complete. Baseline rho: " << baseline_rho_ << "\n";
}

void HeadOrientationDetector::reset() {
    baseline_rho_ = 0.0;
    calibrated_ = false;
    in_droop_ = false;
    droop_cue_emitted_ = false;
}

double HeadOrientationDetector::compute_rho(const FacialLandmarks68& landmarks) {
    cv::Point2d p36(landmarks[36].x, landmarks[36].y);
    cv::Point2d p45(landmarks[45].x, landmarks[45].y);

    cv::Point2d e_mid = (p36 + p45) * 0.5;
    cv::Point2d v_eyes = p45 - p36;
    double w_eyes = cv::norm(v_eyes);

    cv::Point2d u_x;
    cv::Point2d u_y;

    if (w_eyes < 1e-6) {
        u_x = cv::Point2d(1.0, 0.0);
        u_y = cv::Point2d(0.0, 1.0);
    } else {
        u_x = v_eyes * (1.0 / w_eyes);
        u_y = cv::Point2d(-u_x.y, u_x.x); // Downward perpendicular
    }

    cv::Point2d p33(landmarks[33].x, landmarks[33].y);
    cv::Point2d p8(landmarks[8].x, landmarks[8].y);

    double d_top = (p33 - e_mid).dot(u_y);
    double d_bottom = (p8 - p33).dot(u_y);
    double h_face = std::max(d_top + d_bottom, 1e-6);

    return d_top / h_face;
}

HeadOrientationAssessment HeadOrientationDetector::process(
    const FacialLandmarks68& landmarks,
    std::chrono::steady_clock::time_point now
) {
    double rho = compute_rho(landmarks);
    double pitch_deg = 0.0;
    std::optional<CueEvent> cue = std::nullopt;

    if (calibrated_) {
        double delta_rho = rho - baseline_rho_;
        pitch_deg = std::atan2(-delta_rho, config_.kp_pitch) * (180.0 / CV_PI);

        // Time-based droop detection (pitch <= -20 deg sustained for >= 1.5s)
        if (pitch_deg <= config_.pitch_droop_thresh.degrees) {
            if (!in_droop_) {
                in_droop_ = true;
                droop_cue_emitted_ = false;
                droop_start_time_ = now;
            } else {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - droop_start_time_);
                if (elapsed >= config_.droop_sustained_duration.value && !droop_cue_emitted_) {
                    droop_cue_emitted_ = true;
                    cue = CueEvent{
                        CueType::HeadDroop,
                        CueStrength::Light,
                        now,
                        "Head droop (" + std::to_string(static_cast<int>(pitch_deg)) + " deg) sustained for >= 1.5s"
                    };
                }
            }
        } else {
            in_droop_ = false;
            droop_cue_emitted_ = false;
        }
    } else {
        in_droop_ = false;
        droop_cue_emitted_ = false;
    }

    return HeadOrientationAssessment{
        pitch_deg,
        rho,
        cue
    };
}

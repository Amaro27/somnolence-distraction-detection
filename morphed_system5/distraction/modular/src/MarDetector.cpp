#include "MarDetector.hpp"
#include "FaceGeometry.hpp"
#include "MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

MarDetector::MarDetector(MarConfig config)
    : config_(std::move(config)),
      last_time_(std::chrono::steady_clock::now()),
      last_confirmed_yawn_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

void MarDetector::reset() {
    in_yawn_ = false;
    yawn_cue_emitted_ = false;
    yawn_accum_sec_ = 0.0;
    closed_mouth_sec_ = 0.0;
    missing_face_sec_ = 0.0;
    last_mar_raw_ = 0.0;
    has_last_time_ = false;
    last_time_ = std::chrono::steady_clock::now();
    last_confirmed_yawn_ = std::chrono::steady_clock::now() - std::chrono::seconds(10);
}

MarAssessment MarDetector::process(
    const FacialLandmarks68& landmarks,
    std::chrono::steady_clock::time_point now
) {
    const double dt = MathUtils::compute_clamped_dt(now, last_time_, has_last_time_);
    missing_face_sec_ = 0.0;

    double mar_raw = FaceGeometry::compute_mar(landmarks);
    last_mar_raw_ = mar_raw;
    std::optional<CueEvent> cue = std::nullopt;
    const double target_sec = static_cast<double>(config_.sustained_duration.count());

    if (mar_raw > config_.mar_threshold) {
        closed_mouth_sec_ = 0.0;
        in_yawn_ = true;
        yawn_accum_sec_ = std::min(target_sec + 0.6, yawn_accum_sec_ + dt);

        if (yawn_accum_sec_ >= target_sec) {
            auto cooldown_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_confirmed_yawn_);
            if (!yawn_cue_emitted_ && cooldown_elapsed >= config_.alarm_cooldown) {
                yawn_cue_emitted_ = true;
                last_confirmed_yawn_ = now;

                std::ostringstream ss;
                ss << "Sustained yawn >=" << config_.sustained_duration.count()
                   << "s (duration: " << std::fixed << std::setprecision(1) << yawn_accum_sec_
                   << "s, MAR: " << std::setprecision(2) << mar_raw << ")";

                cue = CueEvent{
                    CueType::Yawn,
                    CueStrength::Light,
                    now,
                    ss.str()
                };
            }
        }
    } else {
        closed_mouth_sec_ += dt;
        if (closed_mouth_sec_ <= 0.30) {
            // Brief mouth-closed noise/jitter tolerance
            yawn_accum_sec_ = std::max(0.0, yawn_accum_sec_ - 0.5 * dt);
        } else {
            // Confirmed mouth closure: cap to 0.50s to provide a natural post-yawn eye-recovery window
            yawn_accum_sec_ = std::min(yawn_accum_sec_, 0.50);
            yawn_accum_sec_ = std::max(0.0, yawn_accum_sec_ - 1.0 * dt);
        }

        if (yawn_accum_sec_ <= 0.0) {
            in_yawn_ = false;
            yawn_cue_emitted_ = false;
        }
    }

    return MarAssessment{
        mar_raw,
        yawn_accum_sec_,
        in_yawn_,
        cue
    };
}

MarAssessment MarDetector::on_face_lost(
    std::chrono::steady_clock::time_point now
) {
    const double dt = MathUtils::compute_clamped_dt(now, last_time_, has_last_time_);
    missing_face_sec_ += dt;

    if (yawn_accum_sec_ > 0.0) {
        const double decay_rate = (missing_face_sec_ <= 1.5) ? 0.15 : 1.5;
        yawn_accum_sec_ = std::max(0.0, yawn_accum_sec_ - decay_rate * dt);
        if (yawn_accum_sec_ <= 0.0) {
            yawn_accum_sec_ = 0.0;
            in_yawn_ = false;
            yawn_cue_emitted_ = false;
        }
    }

    return MarAssessment{
        last_mar_raw_,
        yawn_accum_sec_,
        in_yawn_,
        std::nullopt
    };
}

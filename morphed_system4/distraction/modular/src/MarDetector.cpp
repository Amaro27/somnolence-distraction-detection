#include "MarDetector.hpp"
#include "FaceGeometry.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

MarDetector::MarDetector(MarConfig config)
    : config_(std::move(config)),
      yawn_start_(std::chrono::steady_clock::now()),
      last_confirmed_yawn_(std::chrono::steady_clock::time_point::min()) {}

void MarDetector::reset() {
    in_yawn_ = false;
    yawn_cue_emitted_ = false;
    yawn_start_ = std::chrono::steady_clock::now();
    last_confirmed_yawn_ = std::chrono::steady_clock::time_point::min();
}

MarAssessment MarDetector::process(
    const FacialLandmarks68& landmarks,
    std::chrono::steady_clock::time_point now
) {
    double mar_raw = FaceGeometry::compute_mar(landmarks);
    double yawn_duration_sec = 0.0;
    std::optional<CueEvent> cue = std::nullopt;

    if (mar_raw > config_.mar_threshold.value) {
        if (!in_yawn_) {
            in_yawn_ = true;
            yawn_start_ = now;
            yawn_cue_emitted_ = false;
        }

        yawn_duration_sec = std::chrono::duration<double>(now - yawn_start_).count();

        if (yawn_duration_sec >= static_cast<double>(config_.sustained_duration.count())) {
            auto cooldown_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_confirmed_yawn_);
            if (!yawn_cue_emitted_ && cooldown_elapsed >= config_.alarm_cooldown) {
                yawn_cue_emitted_ = true;
                last_confirmed_yawn_ = now;

                std::ostringstream ss;
                ss << "Sustained yawn >=" << config_.sustained_duration.count()
                   << "s (duration: " << std::fixed << std::setprecision(1) << yawn_duration_sec
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
        in_yawn_ = false;
        yawn_cue_emitted_ = false;
    }

    return MarAssessment{
        mar_raw,
        yawn_duration_sec,
        in_yawn_,
        cue
    };
}

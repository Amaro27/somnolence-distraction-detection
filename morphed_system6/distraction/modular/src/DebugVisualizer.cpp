#include "DebugVisualizer.hpp"

#ifdef DEBUG_VIEWS

#include "FaceGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr const char* WINDOW_NAME = "Morphed System 6 - Real-Time ADAS Monitor";

const char* system_state_label(SystemState state) noexcept {
    switch (state) {
        case SystemState::StationaryUncalibrated: return "Stationary (Uncal)";
        case SystemState::MovingUncalibrated:     return "Moving (Uncal)";
        case SystemState::Calibrating:            return "CALIBRATING";
        case SystemState::StationaryCalibrated:   return "Stationary (Cal)";
        case SystemState::MovingCalibrated:       return "Moving (Active)";
    }
    return "Unknown";
}

const char* driver_state_label(DriverState state) noexcept {
    switch (state) {
        case DriverState::Safe:       return "SAFE";
        case DriverState::Distracted: return "DISTRACTED";
        case DriverState::Drowsy:     return "DROWSY";
    }
    return "UNKNOWN";
}

cv::Scalar driver_state_color(DriverState state) noexcept {
    switch (state) {
        case DriverState::Safe:       return cv::Scalar(0, 220, 0);
        case DriverState::Distracted: return cv::Scalar(0, 165, 255);
        case DriverState::Drowsy:     return cv::Scalar(0, 0, 255);
    }
    return cv::Scalar(255, 255, 255);
}

void draw_landmark_loop(
    cv::Mat& canvas,
    const FacialLandmarks68& lm,
    size_t start_idx,
    size_t end_idx,
    const cv::Scalar& color,
    bool draw_points
) {
    for (size_t i = start_idx; i <= end_idx; ++i) {
        const size_t next = (i == end_idx) ? start_idx : (i + 1);
        cv::line(
            canvas,
            cv::Point(static_cast<int>(std::lround(lm[i].x)), static_cast<int>(std::lround(lm[i].y))),
            cv::Point(static_cast<int>(std::lround(lm[next].x)), static_cast<int>(std::lround(lm[next].y))),
            color,
            1,
            cv::LINE_AA
        );
        if (draw_points) {
            cv::circle(
                canvas,
                cv::Point(static_cast<int>(std::lround(lm[i].x)), static_cast<int>(std::lround(lm[i].y))),
                2,
                color,
                -1,
                cv::LINE_AA
            );
        }
    }
}

double latency_to_fps(double ms) noexcept {
    return (ms > 1e-3) ? (1000.0 / ms) : 0.0;
}

} // namespace

DebugVisualizer::DebugVisualizer()
    : mailbox_(std::make_shared<ThreadSafeMailbox<cv::Mat>>()) {}

DebugVisualizer::~DebugVisualizer() {
    stop();
}

void DebugVisualizer::update_gaze(
    const DistractionDetectionResult& res,
    size_t buf_size,
    size_t seq_len,
    float min_conf,
    double latency_ms
) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.has_gaze = true;
    snapshot_.gaze = res;
    snapshot_.lstm_buf_size = buf_size;
    snapshot_.lstm_seq_len = seq_len;
    snapshot_.gaze_min_conf = min_conf;
    snapshot_.gaze_latency_ms = latency_ms;
}

void DebugVisualizer::update_somnolence(
    const std::optional<FacialLandmarks68>& landmarks,
    const EarAssessment& ear,
    double thresh_descent,
    double thresh_closure,
    double perclos_thresh,
    const HeadOrientationAssessment& head,
    const YawnAssessment& yawn,
    float yawn_prob_thresh,
    double latency_ms,
    bool face_missing,
    bool calibrated
) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.has_somno = true;
    snapshot_.landmarks = landmarks;
    snapshot_.ear = ear;
    snapshot_.ear_thresh_descent = thresh_descent;
    snapshot_.ear_thresh_closure = thresh_closure;
    snapshot_.perclos_high_thresh = perclos_thresh;
    snapshot_.head = head;
    snapshot_.yawn = yawn;
    snapshot_.yawn_prob_thresh = yawn_prob_thresh;
    snapshot_.somno_latency_ms = latency_ms;
    snapshot_.face_missing = face_missing;
    snapshot_.calibrated = calibrated;
}

void DebugVisualizer::update_calibration(
    const std::string& phase_text,
    size_t collected_samples,
    size_t target_samples,
    const std::optional<FacialLandmarks68>& landmarks
) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.calib_phase_text = phase_text;
    snapshot_.calib_collected = collected_samples;
    snapshot_.calib_target = target_samples;
    if (landmarks.has_value()) {
        snapshot_.landmarks = landmarks;
        snapshot_.face_missing = false;
    }
}

void DebugVisualizer::reset() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.has_gaze = false;
    snapshot_.lstm_buf_size = 0;
    snapshot_.landmarks.reset();
    snapshot_.ear = EarAssessment{};
    snapshot_.head = HeadOrientationAssessment{};
    snapshot_.yawn = YawnAssessment{};
    snapshot_.face_missing = false;
    snapshot_.calib_phase_text.clear();
    snapshot_.calib_collected = 0;
}

void DebugVisualizer::stop() noexcept {
    stop_recording();
    if (mailbox_) {
        mailbox_->stop();
    }
    if (window_open_) {
        cv::destroyAllWindows();
        window_open_ = false;
    }
}

void DebugVisualizer::start_recording(const std::string& path, double fps) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (video_writer_.isOpened()) {
        video_writer_.release();
    }
    recording_path_ = path;
    recording_fps_ = fps;
    is_recording_.store(true);
}

void DebugVisualizer::stop_recording() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (video_writer_.isOpened()) {
        video_writer_.release();
    }
    is_recording_.store(false);
}

void DebugVisualizer::toggle_recording() {
    if (is_recording()) {
        const std::string saved_name = recording_path_;
        stop_recording();
        std::cout << "\n[*] Real-time recording stopped. Saved to: " << saved_name << "\n" << std::flush;
    } else {
        auto now = std::chrono::system_clock::now();
        auto t_c = std::chrono::system_clock::to_time_t(now);
        char buf[96];
        std::strftime(buf, sizeof(buf), "rasp_recording_%Y%m%d_%H%M%S.mp4", std::localtime(&t_c));
        start_recording(buf);
        std::cout << "\n[+] Real-time recording started: " << buf << "\n" << std::flush;
    }
}

ConsoleCommand DebugVisualizer::render_step(SystemState sys_state, DriverState drv_state) {
    cv::Mat frame;
    if (mailbox_ && mailbox_->get_latest(frame, std::chrono::milliseconds(30))) {
        if (!frame.empty()) {
            last_frame_ = frame;
        }
    }

    if (last_frame_.empty()) {
        return ConsoleCommand::None;
    }

    Snapshot snap;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snap = snapshot_;
    }

    cv::Mat canvas = last_frame_.clone();
    const cv::Scalar state_color = driver_state_color(drv_state);

    // 1. Draw facial landmarks, bounding box, 2D pitch axes, and active tags
    if (snap.landmarks.has_value()) {
        const auto& lm = *snap.landmarks;

        // Right Eye (36..41) & Left Eye (42..47) in cyan
        const cv::Scalar cyan(255, 255, 0);
        // draw_landmark_loop(canvas, lm, FaceGeometry::Index::RIGHT_EYE_P1, FaceGeometry::Index::RIGHT_EYE_P6, cyan, true);
        // draw_landmark_loop(canvas, lm, FaceGeometry::Index::LEFT_EYE_P1, FaceGeometry::Index::LEFT_EYE_P6, cyan, true);

        // Outer Lip (48..59) in orange-yellow & Inner Lip (60..67) in bright yellow
        // draw_landmark_loop(canvas, lm, 48, 59, cv::Scalar(0, 180, 255), false);
        // draw_landmark_loop(canvas, lm, FaceGeometry::Index::INNER_LIP_CORNER_LEFT, FaceGeometry::Index::INNER_LIP_BOT_LEFT, cv::Scalar(0, 255, 255), true);

        // Head Pitch reference landmarks (36, 45, 33, 8) in magenta
        const cv::Scalar magenta(255, 0, 255);
        constexpr size_t pitch_refs[] = {
            FaceGeometry::Index::RIGHT_EYE_P1,
            FaceGeometry::Index::LEFT_EYE_P4,
            FaceGeometry::Index::NOSE_TIP,
            FaceGeometry::Index::CHIN
        };
        for (size_t idx : pitch_refs) {
            // cv::circle(canvas, cv::Point(static_cast<int>(std::lround(lm[idx].x)), static_cast<int>(std::lround(lm[idx].y))), 3, magenta, -1, cv::LINE_AA);
        }

        // 2D Roll-Invariant Facial Coordinate Axes
        const cv::Point2d p36(lm[FaceGeometry::Index::RIGHT_EYE_P1].x, lm[FaceGeometry::Index::RIGHT_EYE_P1].y);
        const cv::Point2d p45(lm[FaceGeometry::Index::LEFT_EYE_P4].x, lm[FaceGeometry::Index::LEFT_EYE_P4].y);
        const cv::Point2d e_mid = (p36 + p45) * 0.5;
        const cv::Point2d v_eyes = p45 - p36;
        const double w_eyes = cv::norm(v_eyes);

        cv::Point2d u_x(1.0, 0.0);
        cv::Point2d u_y(0.0, 1.0);
        if (w_eyes >= 1e-6) {
            u_x = v_eyes * (1.0 / w_eyes);
            u_y = cv::Point2d(-u_x.y, u_x.x);
        }

        const double axis_len = std::max(w_eyes, 40.0);
        const cv::Point origin(static_cast<int>(std::lround(e_mid.x)), static_cast<int>(std::lround(e_mid.y)));
        const cv::Point pt_x(
            static_cast<int>(std::lround(e_mid.x + u_x.x * (axis_len * 0.5))),
            static_cast<int>(std::lround(e_mid.y + u_x.y * (axis_len * 0.5)))
        );
        const cv::Point pt_y(
            static_cast<int>(std::lround(e_mid.x + u_y.x * (axis_len * 0.75))),
            static_cast<int>(std::lround(e_mid.y + u_y.y * (axis_len * 0.75)))
        );

        // cv::arrowedLine(canvas, origin, pt_x, cyan, 2, cv::LINE_AA, 0, 0.2);
        const cv::Scalar pitch_axis_color = snap.head.is_drooping ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        // cv::arrowedLine(canvas, origin, pt_y, pitch_axis_color, 2, cv::LINE_AA, 0, 0.2);

        // Tight face bounding box padded by 12px
        float min_x = lm[0].x, max_x = lm[0].x;
        float min_y = lm[0].y, max_y = lm[0].y;
        for (size_t i = 1; i < lm.size(); ++i) {
            min_x = std::min(min_x, lm[i].x);
            max_x = std::max(max_x, lm[i].x);
            min_y = std::min(min_y, lm[i].y);
            max_y = std::max(max_y, lm[i].y);
        }
        const int bx1 = std::max(0, static_cast<int>(std::floor(min_x)) - 12);
        const int by1 = std::max(0, static_cast<int>(std::floor(min_y)) - 12);
        const int bx2 = std::min(canvas.cols - 1, static_cast<int>(std::ceil(max_x)) + 12);
        const int by2 = std::min(canvas.rows - 1, static_cast<int>(std::ceil(max_y)) + 12);
        cv::rectangle(canvas, cv::Point(bx1, by1), cv::Point(bx2, by2), state_color, 2, cv::LINE_AA);

        // Active warning tags above bounding box
        int tag_y = std::max(22, by1 - 8);
        auto draw_tag = [&](const char* text, const cv::Scalar& col) {
            cv::putText(canvas, text, cv::Point(bx1, tag_y), cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv::LINE_AA);
            tag_y -= 22;
        };

        if (snap.ear.in_blink_cycle && snap.ear.ear_raw <= snap.ear_thresh_closure) {
            draw_tag("EYES CLOSED / MICROSLEEP", cv::Scalar(0, 0, 255));
        }
        if (snap.head.is_drooping) {
            draw_tag("HEAD DROOP", cv::Scalar(0, 165, 255));
        }
        if (snap.yawn.is_yawning) {
            draw_tag("YAWNING", cv::Scalar(0, 255, 255));
        }
        if (snap.calibrated && sys_state == SystemState::MovingCalibrated &&
            snap.ear.perclos_warmed_up && snap.ear.perclos_pct >= snap.perclos_high_thresh) {
            draw_tag("HIGH PERCLOS", cv::Scalar(0, 0, 255));
        }
    }

    // 2. Top-Left Telemetry HUD Panel
    const bool is_calibrating = (sys_state == SystemState::Calibrating);
    const int hud_w = std::min(canvas.cols - 20, 580);
    const int hud_h = is_calibrating ? 215 : 185;
    if (hud_w > 50 && canvas.rows > hud_h + 20) {
        cv::rectangle(canvas, cv::Rect(10, 10, hud_w, hud_h), cv::Scalar(20, 20, 20), -1);
        cv::rectangle(canvas, cv::Rect(10, 10, hud_w, hud_h), state_color, 2);

        char line_buf[192];
        int y_cursor = 32;

        // Line 1: System & Driver State
        std::snprintf(
            line_buf,
            sizeof(line_buf),
            "SYSTEM: %s | DRIVER: %s",
            system_state_label(sys_state),
            driver_state_label(drv_state)
        );
        cv::putText(canvas, line_buf, cv::Point(20, y_cursor), cv::FONT_HERSHEY_SIMPLEX, 0.52, state_color, 2, cv::LINE_AA);
        y_cursor += 24;

        // Calibration guidance & progress bar when calibrating
        if (is_calibrating) {
            const std::string& phase_txt = snap.calib_phase_text.empty() ? "Calibrating..." : snap.calib_phase_text;
            std::snprintf(
                line_buf,
                sizeof(line_buf),
                "CALIB: %s (%zu/%zu)",
                phase_txt.c_str(),
                snap.calib_collected,
                snap.calib_target
            );
            cv::putText(canvas, line_buf, cv::Point(20, y_cursor), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
            y_cursor += 10;

            const int bar_x = 20;
            const int bar_w = std::min(hud_w - 24, 340);
            const int bar_h = 10;
            const double frac = (snap.calib_target > 0)
                                    ? std::min(1.0, static_cast<double>(snap.calib_collected) / static_cast<double>(snap.calib_target))
                                    : 0.0;
            cv::rectangle(canvas, cv::Rect(bar_x, y_cursor, bar_w, bar_h), cv::Scalar(120, 120, 120), 1);
            const int fill_w = static_cast<int>(bar_w * frac);
            if (fill_w > 1) {
                cv::rectangle(canvas, cv::Rect(bar_x + 1, y_cursor + 1, fill_w - 1, bar_h - 1), cv::Scalar(0, 220, 255), -1);
            }
            y_cursor += 22;
        }

        // Gaze LSTM telemetry + probability bar
        const char* gaze_str = !snap.has_gaze
                                   ? "PAUSED"
                                   : (snap.gaze.gaze_class == GazeClass::LookingRoad ? "LOOKING ROAD" : "NOT LOOKING ROAD");
        const cv::Scalar gaze_color = !snap.has_gaze
                                          ? cv::Scalar(180, 180, 180)
                                          : (snap.gaze.gaze_class == GazeClass::LookingRoad
                                                 ? cv::Scalar(0, 255, 0)
                                                 : cv::Scalar(0, 0, 255));
        const float p_road = snap.has_gaze ? snap.gaze.looking_road_probability : 0.0f;
        std::snprintf(
            line_buf,
            sizeof(line_buf),
            "LSTM Gaze: %s | P(Road): %.1f%% (Thresh: %.0f%%)",
            gaze_str,
            p_road * 100.0f,
            snap.gaze_min_conf * 100.0f
        );
        cv::putText(canvas, line_buf, cv::Point(20, y_cursor), cv::FONT_HERSHEY_SIMPLEX, 0.48, gaze_color, 1, cv::LINE_AA);
        y_cursor += 8;

        // Probability bar + threshold marker
        {
            const int bar_x = 20;
            const int bar_w = std::min(hud_w - 24, 340);
            const int bar_h = 8;
            const float clamped_p = std::clamp(p_road, 0.0f, 1.0f);
            const float clamped_t = std::clamp(snap.gaze_min_conf, 0.0f, 1.0f);
            cv::rectangle(canvas, cv::Rect(bar_x, y_cursor, bar_w, bar_h), cv::Scalar(110, 110, 110), 1);
            const int fill_w = static_cast<int>(bar_w * clamped_p);
            if (fill_w > 1) {
                cv::rectangle(canvas, cv::Rect(bar_x + 1, y_cursor + 1, fill_w - 1, bar_h - 1), gaze_color, -1);
            }
            const int thresh_x = bar_x + static_cast<int>(bar_w * clamped_t);
            cv::line(canvas, cv::Point(thresh_x, y_cursor - 2), cv::Point(thresh_x, y_cursor + bar_h + 2), cv::Scalar(255, 255, 255), 1);
        }
        y_cursor += 22;

        std::snprintf(
            line_buf,
            sizeof(line_buf),
            "LSTM Buf: %zu/%zuf | Gaze Worker: %.1f ms (%.1f FPS)",
            snap.lstm_buf_size,
            snap.lstm_seq_len,
            snap.gaze_latency_ms,
            latency_to_fps(snap.gaze_latency_ms)
        );
        cv::putText(canvas, line_buf, cv::Point(20, y_cursor), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        y_cursor += 22;

        // Somnolence / Dlib telemetry
        std::snprintf(
            line_buf,
            sizeof(line_buf),
            "Head Pitch: %+5.1f deg (rho: %.3f) [%s]",
            snap.head.pitch_deg,
            snap.head.rho,
            snap.head.is_drooping ? "DROOPING" : "NORMAL"
        );
        cv::putText(
            canvas,
            line_buf,
            cv::Point(20, y_cursor),
            cv::FONT_HERSHEY_SIMPLEX,
            0.46,
            snap.head.is_drooping ? cv::Scalar(0, 165, 255) : cv::Scalar(220, 255, 220),
            1,
            cv::LINE_AA
        );
        y_cursor += 20;

        std::snprintf(
            line_buf,
            sizeof(line_buf),
            "EAR: %.3f (Avg: %.3f | Desc: %.3f | Clos: %.3f) | Blink: %.0f ms",
            snap.ear.ear_raw,
            snap.ear.ear_mov_avg,
            snap.ear_thresh_descent,
            snap.ear_thresh_closure,
            snap.ear.last_blink_duration_ms
        );
        cv::putText(canvas, line_buf, cv::Point(20, y_cursor), cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(255, 255, 200), 1, cv::LINE_AA);
        y_cursor += 20;

        cv::Scalar perclos_color = cv::Scalar(200, 230, 255);
        if (!snap.calibrated) {
            std::snprintf(
                line_buf,
                sizeof(line_buf),
                "PERCLOS (30s): -- (Uncalibrated) | Yawn Prob: %.2f (Thresh: %.2f) | Yawn: %.1fs",
                snap.yawn.yawn_prob,
                snap.yawn_prob_thresh,
                snap.yawn.yawn_duration_sec
            );
            perclos_color = cv::Scalar(160, 160, 160);
        } else if (sys_state != SystemState::MovingCalibrated) {
            std::snprintf(
                line_buf,
                sizeof(line_buf),
                "PERCLOS (30s): -- (Stopped) | Yawn Prob: %.2f (Thresh: %.2f) | Yawn: %.1fs",
                snap.yawn.yawn_prob,
                snap.yawn_prob_thresh,
                snap.yawn.yawn_duration_sec
            );
            perclos_color = cv::Scalar(160, 160, 160);
        } else if (!snap.ear.perclos_warmed_up) {
            std::snprintf(
                line_buf,
                sizeof(line_buf),
                "PERCLOS (30s): %.1f%% [Warmup: %.0fs] | Yawn Prob: %.2f (Thresh: %.2f) | Yawn: %.1fs",
                snap.ear.perclos_pct,
                snap.ear.perclos_warmup_remaining_sec,
                snap.yawn.yawn_prob,
                snap.yawn_prob_thresh,
                snap.yawn.yawn_duration_sec
            );
            perclos_color = cv::Scalar(200, 230, 255);
        } else if (snap.ear.perclos_pct >= snap.perclos_high_thresh) {
            std::snprintf(
                line_buf,
                sizeof(line_buf),
                "PERCLOS (30s): %.1f%% [FATIGUE >=%.0f%%] | Yawn Prob: %.2f | Yawn: %.1fs",
                snap.ear.perclos_pct,
                snap.perclos_high_thresh,
                snap.yawn.yawn_prob,
                snap.yawn.yawn_duration_sec
            );
            perclos_color = cv::Scalar(0, 0, 255);
        } else {
            std::snprintf(
                line_buf,
                sizeof(line_buf),
                "PERCLOS (30s): %.1f%% | Yawn Prob: %.2f (Thresh: %.2f) | Yawn: %.1fs",
                snap.ear.perclos_pct,
                snap.yawn.yawn_prob,
                snap.yawn_prob_thresh,
                snap.yawn.yawn_duration_sec
            );
            perclos_color = cv::Scalar(200, 230, 255);
        }
        cv::putText(canvas, line_buf, cv::Point(20, y_cursor), cv::FONT_HERSHEY_SIMPLEX, 0.44, perclos_color, 1, cv::LINE_AA);
        y_cursor += 20;

        const char* face_str = !snap.has_somno ? "IDLE" : (snap.face_missing ? "MISSING" : "TRACKING");
        std::snprintf(
            line_buf,
            sizeof(line_buf),
            "Somno Worker: %.1f ms (%.1f FPS) | Face: %s",
            snap.somno_latency_ms,
            latency_to_fps(snap.somno_latency_ms),
            face_str
        );
        cv::putText(
            canvas,
            line_buf,
            cv::Point(20, y_cursor),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            snap.face_missing ? cv::Scalar(0, 100, 255) : cv::Scalar(220, 220, 220),
            1,
            cv::LINE_AA
        );
    }

    if (!snap.landmarks.has_value() && snap.yawn.is_yawning && canvas.rows > hud_h + 60) {
        cv::putText(
            canvas,
            "YAWNING",
            cv::Point(16, 10 + hud_h + 26),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(0, 255, 255),
            2,
            cv::LINE_AA
        );
    }

    // 3. Bottom Control Bar
    if (canvas.rows >= 32) {
        cv::rectangle(canvas, cv::Rect(0, canvas.rows - 26, canvas.cols, 26), cv::Scalar(20, 20, 20), -1);
        cv::putText(
            canvas,
            "[Controls] 'm': Toggle Move | 'c': Calibrate | 'r': Record Screen | 'q': Quit",
            cv::Point(12, canvas.rows - 8),
            cv::FONT_HERSHEY_SIMPLEX,
            0.46,
            cv::Scalar(200, 200, 200),
            1,
            cv::LINE_AA
        );
    }

    if (is_recording_.load()) {
        // Red recording indicator in top-right corner
        cv::circle(canvas, cv::Point(canvas.cols - 110, 30), 7, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        cv::putText(canvas, "REC", cv::Point(canvas.cols - 96, 36), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        if (!video_writer_.isOpened()) {
            int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
            video_writer_.open(recording_path_, fourcc, recording_fps_, cv::Size(canvas.cols, canvas.rows), true);
            if (!video_writer_.isOpened()) {
                // Fallback to mp4v codec
                fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                video_writer_.open(recording_path_, fourcc, recording_fps_, cv::Size(canvas.cols, canvas.rows), true);
            }
        }
        if (video_writer_.isOpened()) {
            video_writer_.write(canvas);
        }
    }

    cv::imshow(WINDOW_NAME, canvas);
    window_open_ = true;

    const int key = cv::waitKey(1) & 0xFF;
    if (key == 'm' || key == 'M') {
        return ConsoleCommand::ToggleMovement;
    }
    if (key == 'c' || key == 'C') {
        return ConsoleCommand::Calibrate;
    }
    if (key == 'r' || key == 'R') {
        return ConsoleCommand::ToggleRecording;
    }
    if (key == 'q' || key == 'Q' || key == 27) {
        return ConsoleCommand::Quit;
    }
    return ConsoleCommand::None;
}

#endif

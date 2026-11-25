#pragma once

#include "ConsoleController.hpp"
#include "ThreadSafeMailbox.hpp"
#include "Types.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <opencv2/core/mat.hpp>

#ifdef DEBUG_VIEWS
#include <mutex>
#include <opencv2/videoio.hpp>

/**
 * @brief Real-time OpenCV debug visualizer displaying Dlib landmarks, head pitch axes,
 * EAR/MAR/PERCLOS telemetry, and LSTM gaze predictions when DEBUG_VIEWS is defined.
 */
class DebugVisualizer {
public:
    DebugVisualizer();
    ~DebugVisualizer();

    DebugVisualizer(const DebugVisualizer&) = delete;
    DebugVisualizer& operator=(const DebugVisualizer&) = delete;

    std::shared_ptr<ThreadSafeMailbox<cv::Mat>> mailbox() const noexcept { return mailbox_; }

    void update_gaze(
        const DistractionDetectionResult& res,
        size_t buf_size,
        size_t seq_len,
        float min_conf,
        double latency_ms
    );

    void update_somnolence(
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
    );

    void update_calibration(
        const std::string& phase_text,
        size_t collected_samples,
        size_t target_samples,
        const std::optional<FacialLandmarks68>& landmarks = std::nullopt
    );

    void reset() noexcept;
    void stop() noexcept;

    void start_recording(const std::string& path, double fps = 15.0);
    void stop_recording();
    void toggle_recording();
    bool is_recording() const noexcept { return is_recording_.load(); }

    /**
     * @brief Pulls latest camera frame, overlays debug HUD/landmarks, displays via cv::imshow,
     * writes frame to video recorder if active, and returns any console command triggered via cv::waitKey(1).
     */
    ConsoleCommand render_step(SystemState sys_state, DriverState drv_state);

private:
    struct Snapshot {
        bool has_gaze{false};
        DistractionDetectionResult gaze;
        size_t lstm_buf_size{0};
        size_t lstm_seq_len{10};
        float gaze_min_conf{0.54f};
        double gaze_latency_ms{0.0};

        bool has_somno{false};
        std::optional<FacialLandmarks68> landmarks;
        EarAssessment ear;
        double ear_thresh_descent{0.25};
        double ear_thresh_closure{0.18};
        double perclos_high_thresh{15.0};
        HeadOrientationAssessment head;
        YawnAssessment yawn;
        float yawn_prob_thresh{0.80f};
        double somno_latency_ms{0.0};
        bool face_missing{false};
        bool calibrated{false};

        std::string calib_phase_text;
        size_t calib_collected{0};
        size_t calib_target{15};
    };

    std::shared_ptr<ThreadSafeMailbox<cv::Mat>> mailbox_;
    mutable std::mutex mutex_;
    Snapshot snapshot_;
    cv::Mat last_frame_;
    bool window_open_{false};

    cv::VideoWriter video_writer_;
    std::string recording_path_;
    std::atomic<bool> is_recording_{false};
    double recording_fps_{15.0};
};

#else

/**
 * @brief Zero-cost stub for Release builds when DEBUG_VIEWS is disabled.
 */
class DebugVisualizer {
public:
    DebugVisualizer() = default;
    ~DebugVisualizer() = default;

    DebugVisualizer(const DebugVisualizer&) = delete;
    DebugVisualizer& operator=(const DebugVisualizer&) = delete;

    std::shared_ptr<ThreadSafeMailbox<cv::Mat>> mailbox() const noexcept { return nullptr; }

    void update_gaze(
        const DistractionDetectionResult&,
        size_t,
        size_t,
        float,
        double
    ) noexcept {}

    void update_somnolence(
        const std::optional<FacialLandmarks68>&,
        const EarAssessment&,
        double,
        double,
        double,
        const HeadOrientationAssessment&,
        const YawnAssessment&,
        float,
        double,
        bool,
        bool
    ) noexcept {}

    void update_calibration(
        const std::string&,
        size_t,
        size_t,
        const std::optional<FacialLandmarks68>& = std::nullopt
    ) noexcept {}

    void reset() noexcept {}
    void stop() noexcept {}

    void start_recording(const std::string&, double = 15.0) noexcept {}
    void stop_recording() noexcept {}
    void toggle_recording() noexcept {}
    bool is_recording() const noexcept { return false; }

    ConsoleCommand render_step(SystemState, DriverState) noexcept {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return ConsoleCommand::None;
    }
};

#endif

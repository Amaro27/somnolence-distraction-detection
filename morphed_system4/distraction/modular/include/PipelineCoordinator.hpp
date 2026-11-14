#pragma once

#include "AlertManager.hpp"
#include "AudioManager.hpp"
#include "CameraCapture.hpp"
#include "ConsoleController.hpp"
#include "EarDetector.hpp"
#include "FaceLandmarkDetector.hpp"
#include "GazeDetector.hpp"
#include "HeadOrientationDetector.hpp"
#include "MarDetector.hpp"
#include "Preprocessor.hpp"
#include "ThreadSafeMailbox.hpp"
#include "Types.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <opencv2/core/mat.hpp>

/**
 * @brief RAII coordinator managing all pipeline components, worker threads,
 * console command dispatch, and lifecycle transitions.
 */
class PipelineCoordinator {
public:
    struct Options {
        std::string source{"0"};
        std::string backbone_path{};
        std::string head_path{};
        std::string predictor_path{};
        float conf_threshold{0.70f};
        int window_ms{2000};
        size_t seq_len{20};
        ThresholdEvaluationPolicy policy{ThresholdEvaluationPolicy::BinarizeThreshold};
        bool start_moving{false};
    };

    explicit PipelineCoordinator(Options options);
    ~PipelineCoordinator();

    // RAII non-copyable, non-movable
    PipelineCoordinator(const PipelineCoordinator&) = delete;
    PipelineCoordinator& operator=(const PipelineCoordinator&) = delete;

    /**
     * @brief Initializes neural models and Dlib shape predictor.
     * @return true if all detectors initialized successfully.
     */
    bool initialize();

    /**
     * @brief Starts camera capture, worker threads, and runs supervisory loop until shutdown or stream EOF.
     * @param external_shutdown Flag set by OS signal handlers (SIGINT/SIGTERM).
     * @return Process exit code (0 on clean shutdown, 1 on startup failure).
     */
    int run(const std::atomic<bool>& external_shutdown);

    /**
     * @brief Signals shutdown and interrupts blocking audio/mailbox waits.
     */
    void request_shutdown() noexcept;

    /**
     * @brief Stops all threads, capture, audio, and console cleanly (idempotent).
     */
    void stop();

private:
    static LSTMInferenceConfig build_model_config(const Options& options);
    static AlertConfig build_alert_config(const Options& options);
    static FaceLandmarkConfig build_face_config(const Options& options);

    void handle_console_command(ConsoleCommand cmd);
    void gaze_worker_loop();
    void somnolence_worker_loop();

    Options options_;
    LSTMInferenceConfig model_config_;
    AlertConfig alert_config_;
    CalibrationAudioConfig calib_audio_config_;

    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> stopped_{false};
    std::atomic<SystemState> system_state_;
    std::atomic<bool> gaze_reset_requested_{false}; // In-band reset signal for gaze_worker_
    bool initialized_{false};

    // Frame handoff mailboxes
    std::shared_ptr<ThreadSafeMailbox<cv::Mat>> mailbox_gaze_;
    std::shared_ptr<ThreadSafeMailbox<cv::Mat>> mailbox_somno_;

    // Core pipeline components
    AudioManager audio_manager_;
    AlertManager alert_manager_;
    GazeDetector gaze_detector_; // Strictly thread-confined to gaze_worker_ after initialize()
    Preprocessor preprocessor_;
    FaceLandmarkDetector face_detector_;
    HeadOrientationDetector head_detector_;
    EarDetector ear_detector_;
    MarDetector mar_detector_;
    CameraCapture camera_capture_;
    ConsoleController console_controller_;

    // Worker threads
    std::thread gaze_worker_;
    std::thread somnolence_worker_;
};

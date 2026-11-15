#include "PipelineCoordinator.hpp"
#include "Calibration.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>

namespace {
#ifdef DEBUG_VIEWS
void print_benchmark_stats(const char* name, std::vector<double>& samples_ms) {
    if (samples_ms.empty()) {
        std::cout << "  " << name << ": (0 samples - no active frames processed)\n";
        return;
    }

    std::sort(samples_ms.begin(), samples_ms.end());
    const size_t n = samples_ms.size();

    const double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    const double mean_ms = sum / static_cast<double>(n);

    double sq_diff_sum = 0.0;
    for (double v : samples_ms) {
        const double d = v - mean_ms;
        sq_diff_sum += d * d;
    }
    const double stddev_ms = (n > 1) ? std::sqrt(sq_diff_sum / static_cast<double>(n - 1)) : 0.0;

    const double median_ms = (n % 2 != 0)
                                 ? samples_ms[n / 2]
                                 : 0.5 * (samples_ms[n / 2 - 1] + samples_ms[n / 2]);
    const size_t p95_idx = (n == 1)
                               ? 0
                               : std::min(n - 1, static_cast<size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1);
    const double p95_ms = samples_ms[p95_idx];
    const double min_ms = samples_ms.front();
    const double max_ms = samples_ms.back();

    auto to_fps = [](double ms) -> double { return (ms > 1e-6) ? (1000.0 / ms) : 0.0; };

    const std::ios_base::fmtflags old_flags = std::cout.flags();
    const std::streamsize old_prec = std::cout.precision();

    std::cout << std::fixed << std::setprecision(2)
              << "  " << name << " (N=" << n << ")\n"
              << "    Latency (ms) : Mean=" << mean_ms
              << " | Std=" << stddev_ms
              << " | Median=" << median_ms
              << " | P95=" << p95_ms
              << " | Min=" << min_ms
              << " | Max=" << max_ms << "\n"
              << "    Throughput   : Mean=" << to_fps(mean_ms) << " FPS"
              << " | Median=" << to_fps(median_ms) << " FPS"
              << " | P5=" << to_fps(p95_ms) << " FPS"
              << " | Min=" << to_fps(max_ms) << " FPS"
              << " | Max=" << to_fps(min_ms) << " FPS\n";

    std::cout.flags(old_flags);
    std::cout.precision(old_prec);
}
#endif
} // namespace

LSTMInferenceConfig PipelineCoordinator::build_model_config(const Options& options) {
    LSTMInferenceConfig cfg;
    if (!options.backbone_path.empty()) {
        cfg.backbone_path = options.backbone_path;
    }
    if (!options.head_path.empty()) {
        cfg.head_path = options.head_path;
    }
    cfg.input_dimension = cv::Size(224, 224);
    cfg.seq_len = options.seq_len;
    cfg.feature_dim = 512;
    return cfg;
}

AlertConfig PipelineCoordinator::build_alert_config(const Options& options) {
    AlertConfig cfg;
    cfg.min_confidence = options.conf_threshold;
    cfg.distraction_window = std::chrono::milliseconds(options.window_ms);
    cfg.policy = options.policy;
    cfg.light_cue_window = std::chrono::seconds(120);
    return cfg;
}

FaceLandmarkConfig PipelineCoordinator::build_face_config(const Options& options) {
    FaceLandmarkConfig cfg;
    if (!options.predictor_path.empty()) {
        cfg.predictor_path = options.predictor_path;
    }
    return cfg;
}

PipelineCoordinator::PipelineCoordinator(Options options)
    : options_(std::move(options)),
      model_config_(build_model_config(options_)),
      alert_config_(build_alert_config(options_)),
      system_state_(options_.start_moving ? SystemState::MovingUncalibrated
                                          : SystemState::StationaryUncalibrated),
      mailbox_gaze_(std::make_shared<ThreadSafeMailbox<cv::Mat>>()),
      mailbox_somno_(std::make_shared<ThreadSafeMailbox<cv::Mat>>()),
      audio_manager_(),
      alert_manager_(alert_config_, audio_manager_),
      gaze_detector_(model_config_),
      preprocessor_(model_config_.input_dimension),
      face_detector_(build_face_config(options_)),
      debug_visualizer_(),
      camera_capture_(options_.source, {mailbox_gaze_, mailbox_somno_, debug_visualizer_.mailbox()}) {}

PipelineCoordinator::~PipelineCoordinator() {
    stop();
}

bool PipelineCoordinator::initialize() {
    if (initialized_) {
        return true;
    }

    if (!gaze_detector_.initialize()) {
#ifdef DEBUG_VIEWS
        std::cerr << "[!] Fatal: Could not initialize ONNX models.\n";
#endif
        return false;
    }

    if (!face_detector_.initialize()) {
#ifdef DEBUG_VIEWS
        std::cerr << "[!] Fatal: Could not initialize Dlib shape predictor.\n";
#endif
        return false;
    }

    initialized_ = true;
    return true;
}

void PipelineCoordinator::request_shutdown() noexcept {
    shutdown_requested_.store(true, std::memory_order_release);
    audio_manager_.stop();
    if (mailbox_gaze_) mailbox_gaze_->stop();
    if (mailbox_somno_) mailbox_somno_->stop();
    debug_visualizer_.stop();
}

void PipelineCoordinator::stop() {
    if (stopped_.exchange(true)) {
        return;
    }

#ifdef DEBUG_VIEWS
    const bool had_active_runtime =
        gaze_worker_.joinable() ||
        somnolence_worker_.joinable() ||
        camera_capture_.is_running() ||
        console_controller_.is_running();

    if (had_active_runtime) {
        std::cout << "[*] Stopping pipeline components...\n" << std::flush;
    }
#endif

    request_shutdown();
    camera_capture_.stop();

    if (gaze_worker_.joinable()) {
        gaze_worker_.join();
    }
    if (somnolence_worker_.joinable()) {
        somnolence_worker_.join();
    }

    console_controller_.stop();

#ifdef DEBUG_VIEWS
    if (had_active_runtime) {
        std::cout << "\n==================== PIPELINE BENCHMARK SUMMARY ====================\n";
        print_benchmark_stats("Distraction Worker [Compute]   ", gaze_work_ms_);
        print_benchmark_stats("Distraction Worker [Full Loop] ", gaze_cycle_ms_);
        print_benchmark_stats("Somnolence Worker  [Compute]   ", somno_work_ms_);
        print_benchmark_stats("Somnolence Worker  [Full Loop] ", somno_cycle_ms_);
        print_benchmark_stats("Main Supervisory Loop          ", main_loop_ms_);
        std::cout << "====================================================================\n\n" << std::flush;
        std::cout << "[+] Clean shutdown completed successfully.\n" << std::flush;
    }
#endif
}

int PipelineCoordinator::run(const std::atomic<bool>& external_shutdown) {
    if (!initialized_ && !initialize()) {
        return 1;
    }

    // Spawn worker threads
    gaze_worker_ = std::thread(&PipelineCoordinator::gaze_worker_loop, this);
    somnolence_worker_ = std::thread(&PipelineCoordinator::somnolence_worker_loop, this);

    // Fail-fast camera initialization (single device open)
    if (!camera_capture_.start()) {
#ifdef DEBUG_VIEWS
        std::cerr << "[!] Fatal: Could not initialize camera capture worker.\n" << std::flush;
#endif
        stop();
        return 1;
    }

    console_controller_.start();

#ifdef DEBUG_VIEWS
    const SystemState startup_state = system_state_.load();
    const bool is_moving_startup = (startup_state == SystemState::MovingUncalibrated ||
                                    startup_state == SystemState::MovingCalibrated);
    std::cout << "[+] Pipeline active.\n"
              << "    Vehicle movement state: "
              << (is_moving_startup ? "MOVING (Uncalibrated - detections paused)" : "STATIONARY (Paused)")
              << "\n"
              << "    Confidence threshold  : " << (options_.conf_threshold * 100.0f) << "%\n"
              << "    Mode window duration  : " << options_.window_ms << " ms\n"
              << "    LSTM sequence length  : " << options_.seq_len << " frames\n\n" << std::flush;
#endif

    // Supervisory control loop
    main_loop_ms_.reserve(18000);
    while (!shutdown_requested_.load() && !external_shutdown.load()) {
        const auto t_iter_start = std::chrono::steady_clock::now();

        const ConsoleCommand cmd = console_controller_.poll_command();
        if (cmd != ConsoleCommand::None) {
            handle_console_command(cmd);
            if (shutdown_requested_.load()) {
                const auto t_iter_end = std::chrono::steady_clock::now();
                main_loop_ms_.push_back(std::chrono::duration<double, std::milli>(t_iter_end - t_iter_start).count());
                break;
            }
        }

        if (!camera_capture_.is_running() && mailbox_gaze_->is_stopped() && mailbox_somno_->is_stopped()) {
            const auto t_iter_end = std::chrono::steady_clock::now();
            main_loop_ms_.push_back(std::chrono::duration<double, std::milli>(t_iter_end - t_iter_start).count());
            break;
        }

        const ConsoleCommand gui_cmd = debug_visualizer_.render_step(
            system_state_.load(),
            alert_manager_.current_driver_state()
        );
        if (gui_cmd != ConsoleCommand::None) {
            handle_console_command(gui_cmd);
            if (shutdown_requested_.load()) {
                const auto t_iter_end = std::chrono::steady_clock::now();
                main_loop_ms_.push_back(std::chrono::duration<double, std::milli>(t_iter_end - t_iter_start).count());
                break;
            }
        }

        const auto t_iter_end = std::chrono::steady_clock::now();
        main_loop_ms_.push_back(std::chrono::duration<double, std::milli>(t_iter_end - t_iter_start).count());
    }

    stop();
    return 0;
}

void PipelineCoordinator::handle_console_command(ConsoleCommand cmd) {
    switch (cmd) {
        case ConsoleCommand::ToggleMovement: {
            switch (system_state_.load()) {
                case SystemState::Calibrating:
#ifdef DEBUG_VIEWS
                    std::cout << "\n[VEHICLE] Ignored: Cannot change movement state while calibration is in progress.\n" << std::flush;
#endif
                    return;
                case SystemState::MovingUncalibrated:
                    system_state_.store(SystemState::StationaryUncalibrated);
                    alert_manager_.reset();
                    gaze_reset_requested_.store(true, std::memory_order_release);
                    debug_visualizer_.reset();
#ifdef DEBUG_VIEWS
                    std::cout << "\n[VEHICLE] State changed: STATIONARY (Detections paused)\n" << std::flush;
#endif
                    return;
                case SystemState::MovingCalibrated:
                    system_state_.store(SystemState::StationaryCalibrated);
                    alert_manager_.reset();
                    gaze_reset_requested_.store(true, std::memory_order_release);
                    debug_visualizer_.reset();
#ifdef DEBUG_VIEWS
                    std::cout << "\n[VEHICLE] State changed: STATIONARY (Detections paused)\n" << std::flush;
#endif
                    return;
                case SystemState::StationaryUncalibrated:
                    system_state_.store(SystemState::MovingUncalibrated);
#ifdef DEBUG_VIEWS
                    std::cout << "\n[VEHICLE] State changed: MOVING\n"
                              << "[!] Not calibrated: detections disabled. Stop ('m') and calibrate ('c').\n" << std::flush;
#endif
                    return;
                case SystemState::StationaryCalibrated:
                    system_state_.store(SystemState::MovingCalibrated);
#ifdef DEBUG_VIEWS
                    std::cout << "\n[VEHICLE] State changed: MOVING\n" << std::flush;
#endif
                    return;
            }
            return;
        }
        case ConsoleCommand::Calibrate: {
            const SystemState s = system_state_.load();
            if (s == SystemState::MovingUncalibrated || s == SystemState::MovingCalibrated) {
#ifdef DEBUG_VIEWS
                std::cout << "\n[CALIBRATION] Ignored: Calibration is only permitted while vehicle is stationary.\n" << std::flush;
#endif
                return;
            }
            if (s == SystemState::Calibrating) {
#ifdef DEBUG_VIEWS
                std::cout << "\n[CALIBRATION] Ignored: Calibration already in progress.\n" << std::flush;
#endif
                return;
            }

#ifdef DEBUG_VIEWS
            std::cout << "\n[CALIBRATION] Starting coordinated calibration sequence...\n"
                      << "              Follow acoustic guidance instructions.\n" << std::flush;
#endif
            system_state_.store(SystemState::Calibrating);
            return;
        }
        case ConsoleCommand::Quit:
#ifdef DEBUG_VIEWS
            std::cout << "\n[*] Quit requested via console. Initiating clean shutdown...\n" << std::flush;
#endif
            request_shutdown();
            return;
        case ConsoleCommand::None:
            return;
    }
}

void PipelineCoordinator::gaze_worker_loop() {
    try {
        const int blob_dims[4] = {1, 3, model_config_.input_dimension.height, model_config_.input_dimension.width};
        cv::Mat input_blob(4, blob_dims, CV_32F);
        cv::Mat frame;

        gaze_work_ms_.reserve(18000);
        gaze_cycle_ms_.reserve(18000);

        while (!shutdown_requested_.load()) {
            const auto t_iter_start = std::chrono::steady_clock::now();

            // In-band thread-confined reset (zero cross-thread data race on GazeDetector)
            if (gaze_reset_requested_.exchange(false, std::memory_order_acq_rel)) {
                gaze_detector_.reset();
            }

            bool got_frame = mailbox_gaze_->get_latest(frame, std::chrono::milliseconds(100));
            if (!got_frame) {
                if (mailbox_gaze_->is_stopped()) {
                    break;
                }
                continue;
            }

            if (gaze_reset_requested_.exchange(false, std::memory_order_acq_rel)) {
                gaze_detector_.reset();
            }

            // Skip gaze inference unless moving and calibrated
            if (system_state_.load() != SystemState::MovingCalibrated) {
                continue;
            }

            const auto t_work_start = std::chrono::steady_clock::now();
            preprocessor_.process(frame, input_blob);

            DistractionDetectionResult result = gaze_detector_.infer_frame(
                input_blob,
                alert_config_.min_confidence,
                alert_config_.policy
            );

            const auto t_work_end = std::chrono::steady_clock::now();
            alert_manager_.update_gaze(result);
            const auto t_end = std::chrono::steady_clock::now();

            const double work_ms = std::chrono::duration<double, std::milli>(t_work_end - t_work_start).count();
            debug_visualizer_.update_gaze(
                result,
                gaze_detector_.buffer_size(),
                model_config_.seq_len,
                alert_config_.min_confidence,
                work_ms
            );

            gaze_work_ms_.push_back(work_ms);
            gaze_cycle_ms_.push_back(std::chrono::duration<double, std::milli>(t_end - t_iter_start).count());
        }
    } catch (const std::exception& e) {
#ifdef DEBUG_VIEWS
        std::cerr << "[!] GazeWorker Exception: " << e.what() << "\n" << std::flush;
#else
        (void)e;
#endif
    }
}

void PipelineCoordinator::somnolence_worker_loop() {
    try {
        cv::Mat frame;
        bool was_calibrated = false;
        auto last_face_probe_time = std::chrono::steady_clock::now() - std::chrono::seconds(10);

        somno_work_ms_.reserve(18000);
        somno_cycle_ms_.reserve(18000);

        while (!shutdown_requested_.load()) {
            const auto t_iter_start = std::chrono::steady_clock::now();

            if (system_state_.load() == SystemState::Calibrating) {
                CalibrationContext calib_ctx{
                    *mailbox_somno_,
                    face_detector_,
                    audio_manager_,
                    shutdown_requested_,
                    &debug_visualizer_
                };
                auto result = run_calibration(calib_ctx, calib_audio_config_);
                if (result.has_value()) {
                    auto now_calib = std::chrono::steady_clock::now();
                    ear_detector_.apply_baseline(result->ear_open, result->ear_closed, now_calib);
                    head_detector_.apply_baseline(result->neutral_rho);
                    mar_detector_.reset();
                    face_detector_.reset();
                    was_calibrated = true;
                    system_state_.store(SystemState::StationaryCalibrated);
#ifdef DEBUG_VIEWS
                    std::cout << "[+] System calibrated. Press 'm' to switch to MOVING and enable monitoring.\n" << std::flush;
#endif
                } else {
                    system_state_.store(was_calibrated ? SystemState::StationaryCalibrated
                                                       : SystemState::StationaryUncalibrated);
                }
                continue;
            }

            bool got_frame = mailbox_somno_->get_latest(frame, std::chrono::milliseconds(100));
            if (!got_frame) {
                if (mailbox_somno_->is_stopped()) {
                    break;
                }
                continue;
            }

            // Skip all somnolence detections unless moving and calibrated
            if (system_state_.load() != SystemState::MovingCalibrated) {
                continue;
            }

            const auto t_work_start = std::chrono::steady_clock::now();

            // When face is missing, throttle face search to lost_face_probe_interval (default 500ms)
            // to prioritize CPU throughput for the distraction worker.
            const bool face_missing = face_detector_.is_face_missing();
            const auto probe_interval = face_detector_.config().lost_face_probe_interval;
            const bool should_probe = !face_missing ||
                (std::chrono::duration_cast<std::chrono::milliseconds>(t_work_start - last_face_probe_time) >= probe_interval);

            std::optional<FacialLandmarks68> landmarks_opt;
            if (should_probe) {
                last_face_probe_time = t_work_start;
                landmarks_opt = face_detector_.detect(frame, t_work_start);
            }
            EarAssessment ear_res{};
            HeadOrientationAssessment head_res{};
            MarAssessment mar_res{};
            if (landmarks_opt.has_value()) {
                const auto& landmarks = landmarks_opt.value();

                ear_res = ear_detector_.process(landmarks, t_work_start);
                head_res = head_detector_.process(landmarks, t_work_start);
                mar_res = mar_detector_.process(landmarks, t_work_start);

                alert_manager_.update_somnolence(ear_res, head_res, mar_res);
            } else {
                alert_manager_.tick_drowsiness_if_active();
            }

            const auto t_end = std::chrono::steady_clock::now();
            const double work_ms = std::chrono::duration<double, std::milli>(t_end - t_work_start).count();
            debug_visualizer_.update_somnolence(
                landmarks_opt,
                ear_res,
                ear_detector_.thresh_descent(),
                ear_detector_.thresh_closure(),
                head_res,
                mar_res,
                mar_detector_.config().mar_threshold,
                work_ms,
                face_detector_.is_face_missing(),
                ear_detector_.is_calibrated()
            );

            somno_work_ms_.push_back(work_ms);
            somno_cycle_ms_.push_back(std::chrono::duration<double, std::milli>(t_end - t_iter_start).count());
        }
    } catch (const std::exception& e) {
#ifdef DEBUG_VIEWS
        std::cerr << "[!] SomnolenceWorker Exception: " << e.what() << "\n" << std::flush;
#else
        (void)e;
#endif
    }
}

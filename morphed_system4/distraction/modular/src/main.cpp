#include "AlertManager.hpp"
#include "AudioManager.hpp"
#include "Calibration.hpp"
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
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shutdown_requested.store(true);
    }
}
} // namespace

int main(int argc, char** argv) {
    std::string source = "0";
    std::string backbone_path = "";
    std::string head_path = "";
    std::string predictor_path = "";
    float conf_threshold_val = 0.70f;
    int window_ms = 2000;
    size_t seq_len_val = 20;
    ThresholdEvaluationPolicy policy = ThresholdEvaluationPolicy::BinarizeThreshold;
    VehicleMovementState initial_movement = VehicleMovementState::Stationary;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--source" && i + 1 < argc) {
            source = argv[++i];
        } else if ((arg == "--backbone-path" || arg == "--backbone") && i + 1 < argc) {
            backbone_path = argv[++i];
        } else if ((arg == "--head-path" || arg == "--head") && i + 1 < argc) {
            head_path = argv[++i];
        } else if ((arg == "--predictor-path" || arg == "--predictor") && i + 1 < argc) {
            predictor_path = argv[++i];
        } else if ((arg == "--conf-threshold" || arg == "--conf") && i + 1 < argc) {
            conf_threshold_val = std::stof(argv[++i]);
        } else if ((arg == "--window-ms" || arg == "--window") && i + 1 < argc) {
            window_ms = std::stoi(argv[++i]);
        } else if ((arg == "--seq-len" || arg == "--seq_len") && i + 1 < argc) {
            seq_len_val = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--policy" && i + 1 < argc) {
            std::string pol_arg = argv[++i];
            if (pol_arg == "filter") {
                policy = ThresholdEvaluationPolicy::FilterUncertain;
            } else {
                policy = ThresholdEvaluationPolicy::BinarizeThreshold;
            }
        } else if (arg == "--moving") {
            initial_movement = VehicleMovementState::Moving;
        } else if (arg == "--stationary") {
            initial_movement = VehicleMovementState::Stationary;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: modular_distraction [OPTIONS]\n\n"
                      << "Options:\n"
                      << "  --source <0|path>       Webcam index (default: 0) or video file path\n"
                      << "  --backbone-path <path>  Path to resnet18_lstm_backbone.onnx\n"
                      << "  --head-path <path>      Path to resnet18_lstm_head.onnx\n"
                      << "  --predictor-path <path> Path to shape_predictor_68_face_landmarks.dat\n"
                      << "  --conf-threshold <0.70> Confidence threshold for gaze assessment (default: 0.70)\n"
                      << "  --window-ms <2000>      Sliding mode evaluation duration in ms (default: 2000)\n"
                      << "  --seq-len <20>          LSTM sequence memory window in frames (default: 20)\n"
                      << "  --policy <binarize|filter> Threshold policy (default: binarize)\n"
                      << "  --moving                Start in moving state (default: stationary)\n"
                      << "  --help, -h              Display this help message\n";
            return 0;
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "========================================================\n"
              << "[*] Real-Time Modular Driver Distraction & Somnolence System\n"
              << "[*] Parallel Dual-Worker Pipeline | Decoupled Alert Engine\n"
              << "========================================================\n";

    // 1. Initialize Thread-Safe Mailboxes for zero-lag frame handoff to both workers
    auto mailbox_gaze = std::make_shared<ThreadSafeMailbox<cv::Mat>>();
    auto mailbox_somno = std::make_shared<ThreadSafeMailbox<cv::Mat>>();

    // 2. Configure and Initialize 2-Stage ONNX Gaze Detector (Worker 1)
    LSTMInferenceConfig model_config;
    if (!backbone_path.empty()) {
        model_config.backbone_path = backbone_path;
    }
    if (!head_path.empty()) {
        model_config.head_path = head_path;
    }
    model_config.input_dimension = ImageDimension2D(224, 224);
    model_config.seq_len = SequenceLengthInFrames(seq_len_val);
    model_config.feature_dim = 512;

    GazeDetector gaze_detector(model_config);
    if (!gaze_detector.initialize()) {
        std::cerr << "[!] Fatal: Could not initialize ONNX models.\n";
        return 1;
    }

    Preprocessor preprocessor(model_config.input_dimension);

    // 3. Configure and Initialize Dlib Face Landmark Detector (Worker 2)
    FaceLandmarkConfig face_config;
    if (!predictor_path.empty()) {
        face_config.predictor_path = predictor_path;
    }
    FaceLandmarkDetector face_detector(face_config);
    if (!face_detector.initialize()) {
        std::cerr << "[!] Fatal: Could not initialize Dlib shape predictor.\n";
        return 1;
    }

    // 4. Configure Physiological & Postural Detectors (Worker 2)
    HeadOrientationDetector head_detector;
    EarDetector ear_detector;
    MarDetector mar_detector;

    // 5. Unified AudioManager & Thread-Safe AlertManager
    AlertConfig alert_config;
    alert_config.min_confidence = ConfidenceThreshold(conf_threshold_val);
    alert_config.distraction_window = DetectionWindowDuration(std::chrono::milliseconds(window_ms));
    alert_config.policy = policy;
    alert_config.light_cue_window = std::chrono::seconds(120);
    alert_config.audio_config.repetition_rate_hz = 1.0;
    alert_config.audio_config.min_volume = 0.3f;
    alert_config.audio_config.max_volume = 1.0f;

    AudioManager audio_manager(alert_config.audio_config);
    AlertManager alert_manager(alert_config, audio_manager);

    // 6. Shared System Coordination State (Single Atomic State Machine)
    std::atomic<SystemState> system_state{
        initial_movement == VehicleMovementState::Moving
            ? SystemState::MovingUncalibrated
            : SystemState::StationaryUncalibrated};
    CalibrationAudioConfig calib_audio_config;

    // 7. Interactive Non-Blocking Console Controller ('m', 'c', 'q')
    ConsoleController console(
        [&system_state, &alert_manager, &gaze_detector]() {
            switch (system_state.load()) {
                case SystemState::Calibrating:
                    std::cout << "\n[VEHICLE] Ignored: Cannot change movement state while calibration is in progress.\n" << std::flush;
                    return;
                case SystemState::MovingUncalibrated:
                    system_state.store(SystemState::StationaryUncalibrated);
                    alert_manager.reset();
                    gaze_detector.reset();
                    std::cout << "\n[VEHICLE] State changed: STATIONARY (Detections paused)\n" << std::flush;
                    return;
                case SystemState::MovingCalibrated:
                    system_state.store(SystemState::StationaryCalibrated);
                    alert_manager.reset();
                    gaze_detector.reset();
                    std::cout << "\n[VEHICLE] State changed: STATIONARY (Detections paused)\n" << std::flush;
                    return;
                case SystemState::StationaryUncalibrated:
                    system_state.store(SystemState::MovingUncalibrated);
                    std::cout << "\n[VEHICLE] State changed: MOVING\n"
                              << "[!] Not calibrated: detections disabled. Stop ('m') and calibrate ('c').\n" << std::flush;
                    return;
                case SystemState::StationaryCalibrated:
                    system_state.store(SystemState::MovingCalibrated);
                    std::cout << "\n[VEHICLE] State changed: MOVING\n" << std::flush;
                    return;
            }
        },
        [&system_state]() {
            const SystemState s = system_state.load();
            // Stationary Gate: Calibration is only permitted while stationary
            if (s == SystemState::MovingUncalibrated || s == SystemState::MovingCalibrated) {
                std::cout << "\n[CALIBRATION] Ignored: Calibration is only permitted while vehicle is stationary.\n" << std::flush;
                return;
            }

            // In-flight locking: Prevent re-triggering while calibration is already in progress
            if (s == SystemState::Calibrating) {
                std::cout << "\n[CALIBRATION] Ignored: Calibration already in progress.\n" << std::flush;
                return;
            }

            std::cout << "\n[CALIBRATION] Starting coordinated calibration sequence...\n"
                      << "              Follow acoustic guidance instructions.\n" << std::flush;
            system_state.store(SystemState::Calibrating);
        },
        [&audio_manager, &mailbox_gaze, &mailbox_somno]() {
            std::cout << "\n[*] Quit requested via console. Initiating clean shutdown...\n" << std::flush;
            g_shutdown_requested.store(true);
            audio_manager.stop();
            mailbox_gaze->stop();
            mailbox_somno->stop();
        }
    );

    // 8. Launch Parallel Worker Threads

    // ── Thread 1: Gaze Distraction Worker (~45 ms) ─────────────────────────
    std::thread gaze_worker([&]() {
        try {
            const int blob_dims[4] = {1, 3, model_config.input_dimension.height, model_config.input_dimension.width};
            cv::Mat input_blob(4, blob_dims, CV_32F);
            cv::Mat frame;

            while (!g_shutdown_requested.load()) {
                bool got_frame = mailbox_gaze->get_latest(frame, std::chrono::milliseconds(100));
                if (!got_frame) {
                    if (mailbox_gaze->is_stopped()) {
                        break;
                    }
                    continue;
                }

                // Skip gaze inference unless moving and calibrated
                if (system_state.load() != SystemState::MovingCalibrated) {
                    continue;
                }

                // Preprocess frame into normalized NCHW blob (CLAHE + ImageNet scaling)
                preprocessor.process(frame, input_blob);

                // Run 2-stage CNN-LSTM inference
                DistractionDetectionResult result = gaze_detector.infer_frame(
                    input_blob,
                    alert_config.min_confidence,
                    alert_config.policy
                );

                // Feed detection into sliding mode AlertManager
                alert_manager.update_gaze(result);
            }
        } catch (const std::exception& e) {
            std::cerr << "[!] GazeWorker Exception: " << e.what() << "\n" << std::flush;
        }
    });

    // ── Thread 2: Somnolence Worker (~50 ms) ────────────────────────────────
    std::thread somnolence_worker([&]() {
        try {
            cv::Mat frame;
            bool was_calibrated = false;

            while (!g_shutdown_requested.load()) {
                if (system_state.load() == SystemState::Calibrating) {
                    CalibrationContext calib_ctx{
                        *mailbox_somno,
                        face_detector,
                        audio_manager,
                        g_shutdown_requested
                    };
                    auto result = run_calibration(calib_ctx, calib_audio_config);
                    if (result.has_value()) {
                        auto now_calib = std::chrono::steady_clock::now();
                        ear_detector.apply_baseline(result->ear_open, result->ear_closed, now_calib);
                        head_detector.apply_baseline(result->neutral_rho);
                        mar_detector.reset();
                        face_detector.reset();
                        was_calibrated = true;
                        system_state.store(SystemState::StationaryCalibrated);
                        std::cout << "[+] System calibrated. Press 'm' to switch to MOVING and enable monitoring.\n" << std::flush;
                    } else {
                        system_state.store(was_calibrated ? SystemState::StationaryCalibrated
                                                          : SystemState::StationaryUncalibrated);
                    }
                    continue;
                }

                bool got_frame = mailbox_somno->get_latest(frame, std::chrono::milliseconds(100));
                if (!got_frame) {
                    if (mailbox_somno->is_stopped()) {
                        break;
                    }
                    continue;
                }

                // Skip all somnolence detections unless moving and calibrated
                if (system_state.load() != SystemState::MovingCalibrated) {
                    continue;
                }

                auto now = std::chrono::steady_clock::now();

                // Single-pass Dlib 68-point landmark detection (zero bounding box caching)
                auto landmarks_opt = face_detector.detect(frame, now);
                if (!landmarks_opt.has_value()) {
                    continue;
                }

                const auto& landmarks = landmarks_opt.value();

                // Process EAR, Head Orientation, and Mouth Aspect Ratio
                EarAssessment ear_res = ear_detector.process(landmarks, now);
                HeadOrientationAssessment head_res = head_detector.process(landmarks, now);
                MarAssessment mar_res = mar_detector.process(landmarks, now);

                // Register fatigue cues (all gated on MovingCalibrated)
                for (const auto& cue : ear_res.cues) {
                    alert_manager.register_cue(cue.type, cue.strength, cue.details);
                }
                if (head_res.cue.has_value()) {
                    alert_manager.register_cue(head_res.cue->type, head_res.cue->strength, head_res.cue->details);
                }
                if (mar_res.cue.has_value()) {
                    alert_manager.register_cue(mar_res.cue->type, mar_res.cue->strength, mar_res.cue->details);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[!] SomnolenceWorker Exception: " << e.what() << "\n" << std::flush;
        }
    });

    // 9. Start Dedicated Background Camera Capture Thread
    CameraCapture capture(source, {mailbox_gaze, mailbox_somno});
    if (!capture.start()) {
        std::cerr << "[!] Fatal: Could not initialize camera capture worker.\n" << std::flush;
        g_shutdown_requested.store(true);
        mailbox_gaze->stop();
        mailbox_somno->stop();
        if (gaze_worker.joinable()) gaze_worker.join();
        if (somnolence_worker.joinable()) somnolence_worker.join();
        return 1;
    }

    // 10. Start Console Controller
    console.start();

    const SystemState startup_state = system_state.load();
    const bool is_moving_startup = (startup_state == SystemState::MovingUncalibrated ||
                                    startup_state == SystemState::MovingCalibrated);
    std::cout << "[+] Pipeline active.\n"
              << "    Vehicle movement state: "
              << (is_moving_startup ? "MOVING (Uncalibrated - detections paused)" : "STATIONARY (Paused)")
              << "\n"
              << "    Confidence threshold  : " << conf_threshold_val * 100.0f << "%\n"
              << "    Mode window duration  : " << window_ms << " ms\n"
              << "    LSTM sequence length  : " << seq_len_val << " frames\n\n" << std::flush;

    // 11. Main Supervisory Wait Loop
    while (!g_shutdown_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!capture.is_running() && mailbox_gaze->is_stopped() && mailbox_somno->is_stopped()) {
            break;
        }
    }

    std::cout << "[*] Stopping pipeline components...\n" << std::flush;
    g_shutdown_requested.store(true);
    audio_manager.stop();
    capture.stop();
    mailbox_gaze->stop();
    mailbox_somno->stop();

    if (gaze_worker.joinable()) {
        gaze_worker.join();
    }
    if (somnolence_worker.joinable()) {
        somnolence_worker.join();
    }

    console.stop();
    std::cout << "[+] Clean shutdown completed successfully.\n" << std::flush;

    return 0;
}

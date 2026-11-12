#include "AlertManager.hpp"
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
    VehicleMovementState initial_movement = VehicleMovementState::Moving;

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
                      << "  --stationary            Start in stationary state (detections paused)\n"
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

    // 5. Unified Thread-Safe AlertManager
    AlertConfig alert_config;
    alert_config.min_confidence = ConfidenceThreshold(conf_threshold_val);
    alert_config.distraction_window = DetectionWindowDuration(std::chrono::milliseconds(window_ms));
    alert_config.policy = policy;
    alert_config.light_cue_window = std::chrono::seconds(120);
    alert_config.audio_config.repetition_rate_hz = 1.0;
    alert_config.audio_config.min_volume = 0.3f;
    alert_config.audio_config.max_volume = 1.0f;

    AlertManager alert_manager(alert_config);

    // 6. Shared Vehicle Movement State (Atomic)
    std::atomic<VehicleMovementState> vehicle_state{initial_movement};

    // 7. Interactive Non-Blocking Console Controller ('m', 'c', 'q')
    ConsoleController console(
        [&vehicle_state, &alert_manager, &gaze_detector, &ear_detector, &head_detector]() {
            if (vehicle_state.load() == VehicleMovementState::Moving) {
                vehicle_state.store(VehicleMovementState::Stationary);
                alert_manager.reset();
                gaze_detector.reset();
                std::cout << "\n[VEHICLE] State changed: STATIONARY (Distraction inference paused)\n" << std::flush;
            } else {
                vehicle_state.store(VehicleMovementState::Moving);
                // Safeguard: If vehicle starts moving during calibration, abort to Pending
                CalibrationPhase current_phase = ear_detector.phase();
                if (current_phase != CalibrationPhase::Pending && current_phase != CalibrationPhase::Completed) {
                    ear_detector.reset();
                    head_detector.reset();
                    std::cout << "\n[CALIBRATION] Aborted: Vehicle started moving during calibration. Reverted to Pending.\n" << std::flush;
                }
                std::cout << "\n[VEHICLE] State changed: MOVING (Active monitoring enabled)\n" << std::flush;
            }
        },
        [&vehicle_state, &ear_detector, &head_detector]() {
            // Stationary Gate: Calibration is only permitted while stationary
            if (vehicle_state.load() == VehicleMovementState::Moving) {
                std::cout << "\n[CALIBRATION] Ignored: Calibration is only permitted while vehicle is stationary.\n" << std::flush;
                return;
            }

            // In-flight locking: Prevent re-triggering while calibration is already in progress
            CalibrationPhase phase = ear_detector.phase();
            if (phase == CalibrationPhase::CalibOpen || phase == CalibrationPhase::Transition || phase == CalibrationPhase::CalibClosed) {
                std::cout << "\n[CALIBRATION] Ignored: Calibration already in progress.\n" << std::flush;
                return;
            }

            std::cout << "\n[CALIBRATION] Starting coordinated calibration sequence...\n"
                      << "             Phase 1: Neutral Head Pose + Open Eyes (~4s). Look forward at the road.\n" << std::flush;
            ear_detector.start_calibration();
            head_detector.start_calibration();
        },
        [&mailbox_gaze, &mailbox_somno]() {
            std::cout << "\n[*] Quit requested via console. Initiating clean shutdown...\n" << std::flush;
            g_shutdown_requested.store(true);
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

                // Skip gaze inference when stationary
                if (vehicle_state.load() == VehicleMovementState::Stationary) {
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

            while (!g_shutdown_requested.load()) {
                bool got_frame = mailbox_somno->get_latest(frame, std::chrono::milliseconds(100));
                if (!got_frame) {
                    if (mailbox_somno->is_stopped()) {
                        break;
                    }
                    continue;
                }

                auto now = std::chrono::steady_clock::now();
                bool is_moving = (vehicle_state.load() == VehicleMovementState::Moving);
                CalibrationPhase phase = ear_detector.phase();
                bool is_calibrating = (phase != CalibrationPhase::Pending && phase != CalibrationPhase::Completed);

                // Skip processing if vehicle is stationary AND not actively calibrating
                if (!is_moving && !is_calibrating) {
                    continue;
                }

                // Single-pass Dlib 68-point landmark detection (zero bounding box caching)
                auto landmarks_opt = face_detector.detect(frame, now);

                if (!landmarks_opt.has_value()) {
                    if (is_calibrating && face_detector.is_face_missing()) {
                        // Face missing during calibration aborts back to Pending
                        ear_detector.reset();
                        head_detector.reset();
                        std::cout << "\n[CALIBRATION] Aborted: Driver face lost during calibration. Reverted to Pending.\n" << std::flush;
                    }
                    continue;
                }

                const auto& landmarks = landmarks_opt.value();

                // Calibration lifecycle coordination
                if (phase == CalibrationPhase::CalibOpen) {
                    head_detector.accumulate_calibration(landmarks);
                }

                // Process EAR detector (advances calibration FSM or monitors blinks/PERCLOS)
                EarAssessment ear_res = ear_detector.process(landmarks, now);

                // Transition checkpoint: finalize head neutral baseline when moving into transition
                if (ear_detector.phase() == CalibrationPhase::Transition && !head_detector.is_calibrated()) {
                    head_detector.finalize_calibration();
                }

                // Process Head Orientation (pitch droop)
                HeadOrientationAssessment head_res = head_detector.process(landmarks, ear_detector.phase(), now);

                // Process Mouth Aspect Ratio (yawn)
                MarAssessment mar_res = mar_detector.process(landmarks, now);

                // Active Fatigue Alert Evaluation when vehicle is moving
                if (is_moving) {
                    if (ear_detector.is_calibrated()) {
                        for (const auto& cue : ear_res.cues) {
                            alert_manager.register_cue(cue.type, cue.strength, cue.details);
                        }
                        if (head_res.cue.has_value()) {
                            alert_manager.register_cue(head_res.cue->type, head_res.cue->strength, head_res.cue->details);
                        }
                    }
                    if (mar_res.cue.has_value()) {
                        alert_manager.register_cue(mar_res.cue->type, mar_res.cue->strength, mar_res.cue->details);
                    }
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

    std::cout << "[+] Pipeline active.\n"
              << "    Vehicle movement state: "
              << (vehicle_state.load() == VehicleMovementState::Moving ? "MOVING (Active)" : "STATIONARY (Paused)")
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

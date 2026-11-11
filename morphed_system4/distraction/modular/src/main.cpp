#include "AlertManager.hpp"
#include "CameraCapture.hpp"
#include "ConsoleController.hpp"
#include "GazeDetector.hpp"
#include "Preprocessor.hpp"
#include "ThreadSafeMailbox.hpp"
#include "Types.hpp"

#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>

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
                      << "  --conf-threshold <0.70> Confidence threshold for gaze assessment (default: 0.70)\n"
                      << "  --window-ms <2000>      Sliding mode evaluation duration in ms (default: 2000)\n"
                      << "  --seq-len <10>          LSTM sequence memory window in frames (default: 10)\n"
                      << "  --policy <binarize|filter> Threshold policy (default: binarize)\n"
                      << "  --stationary            Start in stationary state (detections paused)\n"
                      << "  --help, -h              Display this help message\n";
            return 0;
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "========================================================\n"
              << "[*] Real-Time Modular Driver Distraction Detection System\n"
              << "[*] Multithreaded Capture | Decoupled Alert Engine\n"
              << "========================================================\n";

    // 1. Initialize Thread-Safe Mailbox for zero-lag frame handoff
    auto mailbox = std::make_shared<ThreadSafeMailbox<cv::Mat>>();

    // 2. Start Dedicated Background Camera Capture Thread
    CameraCapture capture(source, mailbox);
    if (!capture.start()) {
        std::cerr << "[!] Fatal: Could not initialize camera capture worker.\n";
        return 1;
    }

    // 3. Configure and Initialize 2-Stage ONNX Gaze Detector
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

    GazeDetector detector(model_config);
    if (!detector.initialize()) {
        std::cerr << "[!] Fatal: Could not initialize ONNX models.\n";
        capture.stop();
        return 1;
    }

    // 4. Configure Preprocessor and AlertManager
    Preprocessor preprocessor(model_config.input_dimension);

    AlertConfig alert_config;
    alert_config.min_confidence = ConfidenceThreshold(conf_threshold_val);
    alert_config.distraction_window = DetectionWindowDuration(std::chrono::milliseconds(window_ms));
    alert_config.policy = policy;
    alert_config.light_cue_window = std::chrono::seconds(120);
    alert_config.audio_config.repetition_rate_hz = 1.0;
    alert_config.audio_config.min_volume = 0.3f;
    alert_config.audio_config.max_volume = 1.0f;

    AlertManager alert_manager(alert_config);

    // 5. Shared Vehicle Movement State (Atomic)
    std::atomic<VehicleMovementState> vehicle_state{initial_movement};

    // 6. Interactive Non-Blocking Console Controller ('m', 'c', 'q')
    ConsoleController console(
        [&vehicle_state, &alert_manager, &detector]() {
            if (vehicle_state.load() == VehicleMovementState::Moving) {
                vehicle_state.store(VehicleMovementState::Stationary);
                alert_manager.reset();
                detector.reset();
                std::cout << "\n[VEHICLE] State changed: STATIONARY (Detections paused)\n";
            } else {
                vehicle_state.store(VehicleMovementState::Moving);
                std::cout << "\n[VEHICLE] State changed: MOVING (Detections enabled)\n";
            }
        },
        []() {
            std::cout << "\n[CALIBRATION] Calibration triggered (Placeholder for future drowsiness baseline).\n";
        },
        [&mailbox]() {
            std::cout << "\n[*] Quit requested via console. Initiating clean shutdown...\n";
            g_shutdown_requested.store(true);
            mailbox->stop();
        }
    );
    console.start();

    std::cout << "[+] Pipeline active.\n"
              << "    Vehicle movement state: "
              << (vehicle_state.load() == VehicleMovementState::Moving ? "MOVING (Active)" : "STATIONARY (Paused)")
              << "\n"
              << "    Confidence threshold : " << conf_threshold_val * 100.0f << "%\n"
              << "    Mode window duration : " << window_ms << " ms\n"
              << "    LSTM sequence length : " << seq_len_val << " frames\n\n";

    // Pre-allocate destination blob (1 x 3 x 224 x 224)
    const int blob_dims[4] = {1, 3, model_config.input_dimension.height, model_config.input_dimension.width};
    cv::Mat input_blob(4, blob_dims, CV_32F);
    cv::Mat frame;

    // 7. Main Pipeline Loop (Decoupled from Camera Capture Thread)
    while (!g_shutdown_requested.load()) {
        // Fetch freshest frame from background capture thread (drop-oldest pattern)
        bool got_frame = mailbox->get_latest(frame, std::chrono::milliseconds(100));
        if (!got_frame) {
            if (mailbox->is_stopped() || !capture.is_running()) {
                std::cout << "[*] Capture stream ended. Exiting main processing loop.\n";
                break;
            }
            continue;
        }

        // If vehicle is stationary (simulating zero IMU speed), skip inference
        if (vehicle_state.load() == VehicleMovementState::Stationary) {
            continue;
        }

        // Preprocess frame into normalized NCHW blob (CLAHE + ImageNet scaling)
        preprocessor.process(frame, input_blob);

        // Run 2-stage CNN-LSTM inference
        DistractionDetectionResult result = detector.infer_frame(
            input_blob,
            alert_config.min_confidence,
            alert_config.policy
        );

        // Feed detection into 2.0-second sliding mode AlertManager
        alert_manager.update_gaze(result);
    }

    std::cout << "[*] Stopping pipeline components...\n";
    console.stop();
    capture.stop();
    std::cout << "[+] Clean shutdown completed successfully.\n";

    return 0;
}

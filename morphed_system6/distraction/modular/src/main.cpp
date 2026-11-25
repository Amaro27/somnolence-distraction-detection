#include "PipelineCoordinator.hpp"
#include "TelegramNotifier.hpp"
#include "Types.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <iostream>
#include <string>

namespace {
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shutdown_requested.store(true);
    }
}

void print_help() {
#ifdef DEBUG_VIEWS
    std::cout << "Usage: modular_distraction [OPTIONS]\n\n"
              << "Options:\n"
              << "  --source <0|path>          Webcam index (default: 0) or video file path\n"
              << "  --model-type <preopt|unified> LSTM preset: preopt (1-layer, 10f) or unified (2-layer, 20f) (default: preopt)\n"
              << "  --backbone-path <path>     Custom path to ResNet-18 backbone ONNX\n"
              << "  --head-path <path>         Custom path to LSTM head ONNX\n"
              << "  --predictor-path <path>    Path to shape_predictor_68_face_landmarks.dat\n"
              << "  --yawn-model-path <path>   Path to cnn_yawning.onnx\n"
              << "  --conf-threshold <0.54>    Confidence threshold for gaze assessment (default: 0.54)\n"
              << "  --mode-frames <N>          Sliding mode window size in frames (default: 10 preopt / 20 unified)\n"
              << "  --window-ms <2000>         Sustained distraction duration in ms (default: 2000)\n"
              << "  --seq-len <N>              LSTM sequence memory window in frames (default: 10 preopt / 20 unified)\n"
              << "  --policy <binarize|filter|argmax> Threshold policy (default: binarize)\n"
              << "  --record <path.mp4>        Record real-time HUD & video output to MP4 file (debug builds)\n"
              << "  --moving                   Start in moving state (default: stationary)\n"
              << "  --no-imu                   Disable hardware IMU (manual movement control with 'm')\n"
              << "  --imu-bus <path>           I2C bus path for IMU (default: /dev/i2c-1)\n"
              << "  --imu-addr <hex>           I2C address in hex (default: 0x68)\n"
              << "  --imu-thresh-high <val>    Acceleration threshold to declare moving (default: 0.80)\n"
              << "  --imu-thresh-low <val>     Acceleration threshold to declare parked (default: 0.35)\n"
              << "  --help, -h                 Display this help message\n";
#endif
}
} // namespace

int main(int argc, char** argv) {
    struct TelegramLifecycle {
        TelegramLifecycle() { telegram_init(); }
        ~TelegramLifecycle() { telegram_cleanup(); }
    } tg_lifecycle;

    PipelineCoordinator::Options options;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--source" && i + 1 < argc) {
            options.source = argv[++i];
        } else if (arg == "--model-type" && i + 1 < argc) {
            const std::string mt = argv[++i];
            options.use_preopt = (mt != "unified");
        } else if ((arg == "--backbone-path" || arg == "--backbone") && i + 1 < argc) {
            options.backbone_path = argv[++i];
        } else if ((arg == "--head-path" || arg == "--head") && i + 1 < argc) {
            options.head_path = argv[++i];
        } else if ((arg == "--predictor-path" || arg == "--predictor") && i + 1 < argc) {
            options.predictor_path = argv[++i];
        } else if ((arg == "--yawn-model-path" || arg == "--yawn-model") && i + 1 < argc) {
            options.yawn_model_path = argv[++i];
        } else if ((arg == "--conf-threshold" || arg == "--conf") && i + 1 < argc) {
            options.conf_threshold = std::stof(argv[++i]);
        } else if ((arg == "--mode-frames" || arg == "--mode_frames") && i + 1 < argc) {
            options.mode_frames = std::max<size_t>(1, static_cast<size_t>(std::stoul(argv[++i])));
        } else if ((arg == "--window-ms" || arg == "--window") && i + 1 < argc) {
            options.window_ms = std::stoi(argv[++i]);
        } else if ((arg == "--seq-len" || arg == "--seq_len") && i + 1 < argc) {
            options.seq_len = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--policy" && i + 1 < argc) {
            const std::string pol_arg = argv[++i];
            if (pol_arg == "argmax") {
                options.policy = ThresholdEvaluationPolicy::Argmax;
            } else if (pol_arg == "filter") {
                options.policy = ThresholdEvaluationPolicy::FilterUncertain;
            } else {
                options.policy = ThresholdEvaluationPolicy::BinarizeThreshold;
            }
        } else if (arg == "--record" && i + 1 < argc) {
            options.record_path = argv[++i];
        } else if (arg == "--moving") {
            options.start_moving = true;
        } else if (arg == "--stationary") {
            options.start_moving = false;
        } else if (arg == "--no-imu") {
            options.use_imu = false;
        } else if (arg == "--imu-bus" && i + 1 < argc) {
            options.imu_bus = argv[++i];
        } else if (arg == "--imu-addr" && i + 1 < argc) {
            options.imu_address = static_cast<uint8_t>(std::stoul(argv[++i], nullptr, 16));
        } else if (arg == "--imu-thresh-high" && i + 1 < argc) {
            options.imu_threshold_high = std::stof(argv[++i]);
        } else if (arg == "--imu-thresh-low" && i + 1 < argc) {
            options.imu_threshold_low = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

#ifdef DEBUG_VIEWS
    std::cout << "========================================================\n"
              << "[*] Real-Time Modular Driver Distraction & Somnolence System\n"
              << "[*] Parallel Dual-Worker Pipeline | Decoupled Alert Engine\n"
              << "========================================================\n";
#endif

    PipelineCoordinator coordinator(std::move(options));
    if (!coordinator.initialize()) {
        return 1;
    }

    return coordinator.run(g_shutdown_requested);
}

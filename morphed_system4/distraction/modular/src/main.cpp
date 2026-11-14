#include "PipelineCoordinator.hpp"
#include "Types.hpp"

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
}
} // namespace

int main(int argc, char** argv) {
    PipelineCoordinator::Options options;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--source" && i + 1 < argc) {
            options.source = argv[++i];
        } else if ((arg == "--backbone-path" || arg == "--backbone") && i + 1 < argc) {
            options.backbone_path = argv[++i];
        } else if ((arg == "--head-path" || arg == "--head") && i + 1 < argc) {
            options.head_path = argv[++i];
        } else if ((arg == "--predictor-path" || arg == "--predictor") && i + 1 < argc) {
            options.predictor_path = argv[++i];
        } else if ((arg == "--conf-threshold" || arg == "--conf") && i + 1 < argc) {
            options.conf_threshold = std::stof(argv[++i]);
        } else if ((arg == "--window-ms" || arg == "--window") && i + 1 < argc) {
            options.window_ms = std::stoi(argv[++i]);
        } else if ((arg == "--seq-len" || arg == "--seq_len") && i + 1 < argc) {
            options.seq_len = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--policy" && i + 1 < argc) {
            const std::string pol_arg = argv[++i];
            options.policy = (pol_arg == "filter")
                                 ? ThresholdEvaluationPolicy::FilterUncertain
                                 : ThresholdEvaluationPolicy::BinarizeThreshold;
        } else if (arg == "--moving") {
            options.start_moving = true;
        } else if (arg == "--stationary") {
            options.start_moving = false;
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "========================================================\n"
              << "[*] Real-Time Modular Driver Distraction & Somnolence System\n"
              << "[*] Parallel Dual-Worker Pipeline | Decoupled Alert Engine\n"
              << "========================================================\n";

    PipelineCoordinator coordinator(std::move(options));
    if (!coordinator.initialize()) {
        return 1;
    }

    return coordinator.run(g_shutdown_requested);
}

/**
 * Real-Time Gaze Classification (LightGazeCNN) — C++ Dedicated Performance Benchmark Harness.
 *
 * Implements 1:1 functional and telemetry parity with `benchmark_RTdetection.py`:
 *   - Depth 1 & Depth 2 scopes measured via high-resolution RAII guards (`std::chrono::high_resolution_clock`).
 *   - Zero-allocation per-frame image preprocessing with SIMD-friendly HWC -> NCHW ImageNet normalization.
 *   - Native C++ ONNX inference powered by OpenCV DNN (`cv::dnn::readNetFromONNX`).
 *   - Auto-terminates at 1000 frames by default (configurable via --max-frames).
 *   - Exports Event-Log CSV (handles ragged data) and Summary Statistics CSV.
 *   - Supports webcam ("0") or pre-recorded video files (.avi, .mp4, .mov).
 *   - Supports --no-gui for pure computational throughput benchmarking.
 *
 * Controls:
 *   q — Quit and export CSVs
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// ── ImageNet Normalization & Model Constants ────────────────────────────────

constexpr int DEFAULT_IMG_SIZE = 112;

constexpr float MEAN_R = 0.485f;
constexpr float MEAN_G = 0.456f;
constexpr float MEAN_B = 0.406f;

constexpr float STD_R = 0.229f;
constexpr float STD_G = 0.224f;
constexpr float STD_B = 0.225f;

// Precomputed affine constants: (pixel / 255.0f - mean) / std = pixel * SCALE + SHIFT
constexpr float SCALE_R = 1.0f / (255.0f * STD_R);
constexpr float SCALE_G = 1.0f / (255.0f * STD_G);
constexpr float SCALE_B = 1.0f / (255.0f * STD_B);

constexpr float SHIFT_R = -MEAN_R / STD_R;
constexpr float SHIFT_G = -MEAN_G / STD_G;
constexpr float SHIFT_B = -MEAN_B / STD_B;

const std::array<std::string, 2> CLASS_NAMES = {"not_looking_road", "looking_road"};

// ── Benchmark Collector ──────────────────────────────────────────────────────

struct BenchmarkEvent {
    std::string timestamp;
    int frame_id;
    int face_id;
    std::string scope_name;
    int depth;
    double elapsed_ms;
    std::string context;
};

struct SummaryStats {
    std::string scope_name;
    int depth;
    std::string context;
    int count;
    double mean_ms;
    double std_ms;
    double median_ms;
    double p95_ms;
    double min_ms;
    double max_ms;
    double total_ms;
    double pct_of_total_frame;
};

class BenchmarkCollector {
public:
    explicit BenchmarkCollector(std::string output_dir = ".")
        : output_dir_(std::move(output_dir)), current_frame_(0) {
        records_.reserve(20000);
    }

    int current_frame() const { return current_frame_; }
    void increment_frame() { current_frame_++; }

    void record(const std::string& scope_name, int depth, int face_id, double elapsed_ms, const std::string& context) {
        records_.push_back({
            get_current_timestamp_iso(),
            current_frame_,
            face_id,
            scope_name,
            depth,
            elapsed_ms,
            context
        });
    }

    void export_csvs() {
        if (records_.empty()) {
            std::cout << "[BENCHMARK] No measurements recorded.\n";
            return;
        }

        std::string ts = get_current_timestamp_compact();
        std::string events_path = output_dir_ + "/benchmark_events_" + ts + ".csv";
        std::string summary_path = output_dir_ + "/benchmark_summary_" + ts + ".csv";

        // 1. Export Raw Events CSV (Tidy format)
        std::ofstream ef(events_path);
        if (ef.is_open()) {
            ef << "timestamp,frame_id,face_id,scope_name,depth,elapsed_ms,context\n";
            for (const auto& r : records_) {
                ef << r.timestamp << "," << r.frame_id << "," << r.face_id << ","
                   << r.scope_name << "," << r.depth << "," << std::fixed << std::setprecision(3)
                   << r.elapsed_ms << "," << r.context << "\n";
            }
            ef.close();
            std::cout << "\n[BENCHMARK] Events log saved to: " << events_path << "\n";
        }

        // 2. Compute Summary Statistics by (scope_name, depth, context)
        std::map<std::tuple<std::string, int, std::string>, std::vector<double>> grouped;
        double total_frame_cycle_ms = 0.0;

        for (const auto& r : records_) {
            grouped[{r.scope_name, r.depth, r.context}].push_back(r.elapsed_ms);
            if (r.scope_name == "total_frame_cycle") {
                total_frame_cycle_ms += r.elapsed_ms;
            }
        }

        std::vector<SummaryStats> summary_rows;
        summary_rows.reserve(grouped.size());

        for (auto& pair : grouped) {
            const auto& key = pair.first;
            auto& vals = pair.second;
            int n = static_cast<int>(vals.size());
            if (n == 0) continue;

            double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
            double mean_v = sum / n;
            double var_v = 0.0;
            if (n > 1) {
                for (double x : vals) var_v += (x - mean_v) * (x - mean_v);
                var_v /= (n - 1);
            }
            double std_v = std::sqrt(var_v);

            std::sort(vals.begin(), vals.end());
            double med_v = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) * 0.5;
            int p95_idx = std::min(static_cast<int>(n * 0.95), n - 1);
            double p95_v = vals[p95_idx];
            double min_v = vals.front();
            double max_v = vals.back();
            double pct_frame = (total_frame_cycle_ms > 0.0) ? (sum / total_frame_cycle_ms * 100.0) : 0.0;

            summary_rows.push_back({
                std::get<0>(key),
                std::get<1>(key),
                std::get<2>(key),
                n,
                mean_v,
                std_v,
                med_v,
                p95_v,
                min_v,
                max_v,
                sum,
                pct_frame
            });
        }

        // Sort: depth ascending, then total_ms descending
        std::sort(summary_rows.begin(), summary_rows.end(), [](const SummaryStats& a, const SummaryStats& b) {
            if (a.depth != b.depth) return a.depth < b.depth;
            return a.total_ms > b.total_ms;
        });

        // Write Summary CSV
        std::ofstream sf(summary_path);
        if (sf.is_open()) {
            sf << "scope_name,depth,context,count,mean_ms,std_ms,median_ms,p95_ms,min_ms,max_ms,total_ms,pct_of_total_frame\n";
            for (const auto& s : summary_rows) {
                sf << s.scope_name << "," << s.depth << "," << s.context << ","
                   << s.count << "," << std::fixed << std::setprecision(3)
                   << s.mean_ms << "," << s.std_ms << "," << s.median_ms << ","
                   << s.p95_ms << "," << s.min_ms << "," << s.max_ms << ","
                   << s.total_ms << "," << std::setprecision(2) << s.pct_of_total_frame << "\n";
            }
            sf.close();
            std::cout << "[BENCHMARK] Summary stats saved to: " << summary_path << "\n";
        }

        print_terminal_summary(summary_rows);
    }

private:
    std::string output_dir_;
    int current_frame_;
    std::vector<BenchmarkEvent> records_;

    static std::string get_current_timestamp_iso() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
#if defined(_WIN32)
        localtime_s(&tm_buf, &t);
#else
        localtime_r(&t, &tm_buf);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
        char full[80];
        std::snprintf(full, sizeof(full), "%s.%03d", buf, static_cast<int>(ms.count()));
        return std::string(full);
    }

    static std::string get_current_timestamp_compact() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
#if defined(_WIN32)
        localtime_s(&tm_buf, &t);
#else
        localtime_r(&t, &tm_buf);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
        return std::string(buf);
    }

    static void print_terminal_summary(const std::vector<SummaryStats>& summary_rows) {
        std::cout << "\n" << std::string(98, '=') << "\n";
        std::cout << "                        C++ BENCHMARK PERFORMANCE SUMMARY\n";
        std::cout << std::string(98, '=') << "\n";
        std::cout << std::left << std::setw(24) << "Scope Name" << " | "
                  << std::setw(5) << "Depth" << " | "
                  << std::setw(13) << "Context" << " | "
                  << std::setw(6) << "Count" << " | "
                  << std::setw(8) << "Mean(ms)" << " | "
                  << std::setw(8) << "p95(ms)" << " | "
                  << std::setw(8) << "Total(s)" << " | "
                  << std::setw(7) << "% Frame" << "\n";
        std::cout << std::string(98, '-') << "\n";

        for (const auto& r : summary_rows) {
            double tot_s = r.total_ms / 1000.0;
            char pct_buf[32];
            if (r.scope_name == "total_frame_cycle") {
                std::snprintf(pct_buf, sizeof(pct_buf), "100.0%%");
            } else {
                std::snprintf(pct_buf, sizeof(pct_buf), "%.1f%%", r.pct_of_total_frame);
            }
            std::cout << std::left << std::setw(24) << r.scope_name << " | "
                      << std::setw(5) << r.depth << " | "
                      << std::setw(13) << r.context << " | "
                      << std::setw(6) << r.count << " | "
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << r.mean_ms << " | "
                      << std::setw(8) << r.p95_ms << " | "
                      << std::setw(8) << tot_s << " | "
                      << std::setw(7) << pct_buf << "\n";
        }
        std::cout << std::string(98, '=') << "\n\n";
    }
};

// RAII Scope Measurement Guard
class MeasureScope {
public:
    MeasureScope(BenchmarkCollector& col, std::string scope_name, int depth, int face_id = -1, std::string context = "")
        : col_(col), scope_name_(std::move(scope_name)), depth_(depth), face_id_(face_id), context_(std::move(context)),
          t0_(std::chrono::high_resolution_clock::now()) {}

    ~MeasureScope() {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();
        col_.record(scope_name_, depth_, face_id_, ms, context_);
    }

private:
    BenchmarkCollector& col_;
    std::string scope_name_;
    int depth_;
    int face_id_;
    std::string context_;
    std::chrono::high_resolution_clock::time_point t0_;
};

// ── File & Model Path Resolution ────────────────────────────────────────────

inline bool file_exists(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream f(path.c_str());
    return f.good();
}

inline std::string resolve_onnx_model_path(const std::string& user_path, const std::string& argv0) {
    if (!user_path.empty()) {
        // If user passed a .pth file, check if a sibling .onnx exists
        if (user_path.size() > 4 && user_path.substr(user_path.size() - 4) == ".pth") {
            std::string onnx_sibling = user_path.substr(0, user_path.size() - 4) + ".onnx";
            if (file_exists(onnx_sibling)) {
                std::cout << "[INFO] Automatically resolved '.pth' argument to ONNX file: " << onnx_sibling << "\n";
                return onnx_sibling;
            }
        }
        if (file_exists(user_path)) {
            return user_path;
        }
        return "";
    }

    const std::vector<std::string> rel_candidates = {
        "../../../../../../Modelos/light_cnn_session_full_model.onnx",
        "../../../../../Modelos/light_cnn_session_full_model.onnx",
        "../../../../Modelos/light_cnn_session_full_model.onnx",
        "../../../Modelos/light_cnn_session_full_model.onnx",
        "../../Modelos/light_cnn_session_full_model.onnx",
        "../Modelos/light_cnn_session_full_model.onnx",
        "CNN_Gaze/Modelos/light_cnn_session_full_model.onnx",
        "Modelos/light_cnn_session_full_model.onnx",
        "light_cnn_session_full_model.onnx"
    };

    // 1. Check relative to executable directory
    try {
        std::filesystem::path exe_dir = std::filesystem::absolute(std::filesystem::path(argv0)).parent_path();
        for (const auto& rel : rel_candidates) {
            std::filesystem::path p = (exe_dir / rel).lexically_normal();
            if (file_exists(p.string())) {
                return p.string();
            }
        }
    } catch (...) {}

    // 2. Check relative to current working directory
    for (const auto& candidate : rel_candidates) {
        if (file_exists(candidate)) {
            return candidate;
        }
    }
    return "";
}

// ── Fast HWC (RGB uint8) -> NCHW (float32 normalized) Tensor Packing ────────

inline void pack_normalized_nchw_blob(const cv::Mat& resized_rgb, cv::Mat& input_blob) {
    const int height = resized_rgb.rows;
    const int width = resized_rgb.cols;
    const int hw = height * width;

    float* c0 = input_blob.ptr<float>(0, 0); // R plane
    float* c1 = input_blob.ptr<float>(0, 1); // G plane
    float* c2 = input_blob.ptr<float>(0, 2); // B plane

    if (resized_rgb.isContinuous()) {
        const uint8_t* src = resized_rgb.ptr<uint8_t>(0);
        for (int i = 0; i < hw; ++i) {
            c0[i] = static_cast<float>(src[i * 3 + 0]) * SCALE_R + SHIFT_R;
            c1[i] = static_cast<float>(src[i * 3 + 1]) * SCALE_G + SHIFT_G;
            c2[i] = static_cast<float>(src[i * 3 + 2]) * SCALE_B + SHIFT_B;
        }
    } else {
        int idx = 0;
        for (int y = 0; y < height; ++y) {
            const uint8_t* row_ptr = resized_rgb.ptr<uint8_t>(y);
            for (int x = 0; x < width; ++x, ++idx) {
                c0[idx] = static_cast<float>(row_ptr[x * 3 + 0]) * SCALE_R + SHIFT_R;
                c1[idx] = static_cast<float>(row_ptr[x * 3 + 1]) * SCALE_G + SHIFT_G;
                c2[idx] = static_cast<float>(row_ptr[x * 3 + 2]) * SCALE_B + SHIFT_B;
            }
        }
    }
}

// ── Main Benchmark Entrypoint ───────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string source = "0";
    int max_frames = 1000;
    bool no_gui = false;
    std::string output_dir = ".";
    std::string model_path_arg = "";
    int img_size = DEFAULT_IMG_SIZE;

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--source" && i + 1 < argc) {
            source = argv[++i];
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--no-gui") {
            no_gui = true;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path_arg = argv[++i];
        } else if (arg == "--img-size" && i + 1 < argc) {
            img_size = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_RTdetection [--source <0|path>] [--max-frames <1000>] "
                      << "[--no-gui] [--output-dir <path>] [--model-path <onnx_path>] [--img-size <112>]\n";
            return 0;
        }
    }

    std::string resolved_model_path = resolve_onnx_model_path(model_path_arg, argc > 0 ? argv[0] : "");
    if (resolved_model_path.empty()) {
        std::cerr << "[!] Error: Could not locate ONNX model file"
                  << (model_path_arg.empty() ? " ('light_cnn_session_full_model.onnx')" : (" ('" + model_path_arg + "')"))
                  << ".\n"
                  << "    Please convert the PyTorch .pth model first by running:\n"
                  << "        python export_to_onnx.py\n"
                  << "    or specify a valid path via --model-path.\n";
        return 1;
    }

    std::cout << "[*] Running C++ OpenCV DNN inference on CPU (AVX2 optimized)\n";

    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(resolved_model_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[+] Successfully loaded ONNX model from '" << resolved_model_path << "'\n";
    } catch (const cv::Exception& e) {
        std::cerr << "[!] Error loading ONNX model: " << e.what() << "\n";
        return 1;
    }

    // Pre-allocate 4D NCHW input tensor (1 x 3 x img_size x img_size)
    const int blob_dims[4] = {1, 3, img_size, img_size};
    cv::Mat input_blob(4, blob_dims, CV_32F, cv::Scalar(0.0f));

    // Pre-warm network to avoid cold-start allocation overhead on frame 0
    try {
        net.setInput(input_blob);
        cv::Mat warmup_out = net.forward();
        (void)warmup_out;
    } catch (const cv::Exception& e) {
        std::cerr << "[!] Warning during model pre-warm: " << e.what() << "\n";
    }

    // Determine video source (camera index or video file path)
    cv::VideoCapture cap;
    std::string source_label;
    bool is_camera_index = !source.empty() && std::all_of(source.begin(), source.end(), [](unsigned char c) { return std::isdigit(c); });

    if (is_camera_index) {
        cap.open(std::stoi(source));
        source_label = "Webcam (" + source + ")";
    } else {
        if (!file_exists(source)) {
            std::cerr << "[!] Error: Video file not found: " << source << "\n";
            return 1;
        }
        cap.open(source);
        source_label = "Video (" + source + ")";
    }

    if (!cap.isOpened()) {
        std::cerr << "[!] Error: Could not open video source: " << source_label << "\n";
        return 1;
    }

    BenchmarkCollector bench(output_dir);

    std::string max_frames_label = (max_frames > 0) ? (std::to_string(max_frames) + " (auto-quit)") : "Unlimited";
    std::cout << "[INFO] Starting dedicated C++ benchmark on: " << source_label << "\n";
    std::cout << "[INFO] GUI enabled: " << (!no_gui ? "True" : "False") << " | Max frames: " << max_frames_label << "\n";
    std::cout << "[INFO] Controls: Press 'q' in video window to exit early and export CSVs.\n";

    // Pre-allocate reusable frame buffers
    cv::Mat frame;
    cv::Mat rgb_frame;
    cv::Mat resized_rgb(img_size, img_size, CV_8UC3);
    cv::Mat outputs;

    auto prev_time = std::chrono::high_resolution_clock::now();
    double fps = 0.0;

    try {
        while (true) {
            if (max_frames > 0 && bench.current_frame() >= max_frames) {
                std::cout << "\n[INFO] Auto-quitting: Reached target limit of " << max_frames << " frames.\n";
                break;
            }

            char key = -1;
            {
                MeasureScope frame_scope(bench, "total_frame_cycle", 1);

                // 1. Camera / Video Frame Read
                {
                    MeasureScope read_scope(bench, "camera_read", 1);
                    if (!cap.read(frame) || frame.empty()) {
                        std::cout << "[INFO] End of video stream reached or failed to grab frame.\n";
                        break;
                    }
                }

                // Calculate instantaneous display FPS
                auto curr_time = std::chrono::high_resolution_clock::now();
                double time_diff = std::chrono::duration<double>(curr_time - prev_time).count();
                if (time_diff > 0.0) {
                    fps = 1.0 / time_diff;
                }
                prev_time = curr_time;

                // 2. Preprocessing (Depth 1 with Depth 2 breakdown matching Python scopes)
                {
                    MeasureScope prep_scope(bench, "frame_preprocess", 1);
                    {
                        MeasureScope color_scope(bench, "color_and_pil_convert", 2);
                        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
                    }
                    {
                        MeasureScope tensor_scope(bench, "transform_and_tensor", 2);
                        cv::resize(rgb_frame, resized_rgb, cv::Size(img_size, img_size), 0.0, 0.0, cv::INTER_LINEAR);
                        pack_normalized_nchw_blob(resized_rgb, input_blob);
                    }
                }

                // 3. Model Inference (Depth 1)
                {
                    MeasureScope inf_scope(bench, "model_inference", 1);
                    net.setInput(input_blob);
                    outputs = net.forward();
                }

                // 4. Postprocessing (Softmax & Argmax)
                std::string label;
                double conf_score = 0.0;
                {
                    MeasureScope post_scope(bench, "postprocessing", 1);
                    const float* logits = outputs.ptr<float>(0);
                    int num_classes = outputs.cols;
                    float max_logit = logits[0];
                    for (int c = 1; c < num_classes; ++c) {
                        if (logits[c] > max_logit) max_logit = logits[c];
                    }

                    double sum_exp = 0.0;
                    int predicted_idx = 0;
                    double best_exp = 0.0;
                    for (int c = 0; c < num_classes; ++c) {
                        double e = std::exp(static_cast<double>(logits[c] - max_logit));
                        sum_exp += e;
                        if (c == 0 || e > best_exp) {
                            best_exp = e;
                            predicted_idx = c;
                        }
                    }

                    double confidence = (sum_exp > 0.0) ? (best_exp / sum_exp) : 0.0;
                    int safe_idx = std::clamp(predicted_idx, 0, static_cast<int>(CLASS_NAMES.size()) - 1);
                    label = CLASS_NAMES[safe_idx];
                    conf_score = confidence * 100.0;
                }

                // 5. UI Rendering & Drawing (Optional if not --no-gui)
                if (!no_gui) {
                    {
                        MeasureScope overlay_scope(bench, "draw_overlay", 1);
                        cv::Scalar color = (label == "looking_road") ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

                        char fps_buf[64];
                        std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);

                        char text_buf[128];
                        std::snprintf(text_buf, sizeof(text_buf), "%s (%.1f%%)", label.c_str(), conf_score);

                        cv::putText(frame, fps_buf, cv::Point(20, 40),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
                        cv::putText(frame, text_buf, cv::Point(20, 80),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
                    }

                    {
                        MeasureScope show_scope(bench, "gui_imshow", 1);
                        cv::imshow("Real-Time Gaze Classification", frame);
                    }

                    {
                        MeasureScope wait_scope(bench, "gui_waitkey", 1);
                        key = static_cast<char>(cv::waitKey(1) & 0xFF);
                    }
                }
            }

            bench.increment_frame();

            if (key == 'q') {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] Error during benchmark loop: " << e.what() << "\n";
    }

    cap.release();
    if (!no_gui) {
        cv::destroyAllWindows();
    }
    bench.export_csvs();

    return 0;
}

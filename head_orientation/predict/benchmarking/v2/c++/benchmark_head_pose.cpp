/**
 * Head Pose Estimation — C++ Dedicated Performance Benchmark Harness (v2).
 *
 * Implements 1:1 functional parity with the Python v2 benchmark harness:
 *   - Depth 1 & Depth 2 scopes measured via high-resolution RAII guards.
 *   - Zero-copy mapping from cv::Mat to dlib::cv_image.
 *   - Auto-terminates at 1000 frames by default (configurable via --max-frames).
 *   - Exports Event-Log CSV (handles ragged data) and Summary Statistics CSV.
 *   - Supports webcam ("0") or pre-recorded video files.
 *   - Supports --no-gui for pure computational throughput benchmarking.
 *
 * Controls:
 *   q — Quit and export CSVs
 *   r — Re-calibrate baseline
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

// ── Anthropometric & Classification Constants ───────────────────────────────

constexpr double KP_PITCH = 0.18;
constexpr double PITCH_THRESH_MILD = -8.0;
constexpr double PITCH_THRESH_WARNING = -15.0;
constexpr double ROLL_THRESH_MILD = 8.0;
constexpr double ROLL_THRESH_WARNING = 18.0;
constexpr int SUSTAINED_FRAMES = 20;
constexpr int CALIBRATION_FRAMES = 60;

const std::array<int, 4> KEY_LANDMARKS = {36, 45, 33, 8};

// ── Models & Geometrical Calculations ───────────────────────────────────────

struct FaceFrame {
    cv::Point2d e_mid;
    cv::Point2d u_x;
    cv::Point2d u_y;
    double w_eyes;
    double roll_deg;
};

struct PoseMetrics {
    double pitch_deg;
    double roll_deg;
    double delta_roll;
    double rho;
    cv::Point2d e_mid;
    cv::Point2d u_x;
    cv::Point2d u_y;
    double w_eyes;
};

struct Baseline {
    double rho = 0.0;
    double roll = 0.0;
    bool valid = false;
};

inline FaceFrame build_face_frame(const std::array<cv::Point, 68>& puntos) {
    cv::Point2d p36(puntos[36].x, puntos[36].y);
    cv::Point2d p45(puntos[45].x, puntos[45].y);

    cv::Point2d e_mid = (p36 + p45) * 0.5;
    cv::Point2d v_eyes = p45 - p36;
    double w_eyes = cv::norm(v_eyes);

    cv::Point2d u_x;
    cv::Point2d u_y;
    double roll_deg = 0.0;

    if (w_eyes < 1e-6) {
        u_x = cv::Point2d(1.0, 0.0);
        u_y = cv::Point2d(0.0, 1.0);
        roll_deg = 0.0;
    } else {
        u_x = v_eyes * (1.0 / w_eyes);
        u_y = cv::Point2d(-u_x.y, u_x.x);
        roll_deg = -std::atan2(v_eyes.y, v_eyes.x) * (180.0 / CV_PI);
    }

    return {e_mid, u_x, u_y, w_eyes, roll_deg};
}

inline PoseMetrics estimate_head_pose(const std::array<cv::Point, 68>& puntos, const Baseline* baseline = nullptr) {
    FaceFrame ff = build_face_frame(puntos);

    cv::Point2d p33(puntos[33].x, puntos[33].y);
    cv::Point2d p8(puntos[8].x, puntos[8].y);

    double d_top = (p33 - ff.e_mid).dot(ff.u_y);
    double d_bottom = (p8 - p33).dot(ff.u_y);
    double h_face = std::max(d_top + d_bottom, 1e-6);

    double rho = d_top / h_face;
    double pitch_deg = 0.0;
    double delta_roll = 0.0;

    if (baseline && baseline->valid) {
        double delta_rho = rho - baseline->rho;
        pitch_deg = std::atan2(-delta_rho, KP_PITCH) * (180.0 / CV_PI);
        delta_roll = ff.roll_deg - baseline->roll;
    }

    return {pitch_deg, ff.roll_deg, delta_roll, rho, ff.e_mid, ff.u_x, ff.u_y, ff.w_eyes};
}

// ── Calibration & Fatigue Classification ────────────────────────────────────

class Calibrator {
public:
    explicit Calibrator(int n_frames = CALIBRATION_FRAMES) : n_frames_(n_frames) { reset(); }

    void reset() {
        rho_buf_.clear();
        roll_buf_.clear();
        baseline_ = Baseline();
    }

    bool is_calibrated() const { return baseline_.valid; }
    double progress() const { return std::min(static_cast<double>(rho_buf_.size()) / n_frames_, 1.0); }
    const Baseline& baseline() const { return baseline_; }

    void accumulate(const PoseMetrics& metrics) {
        rho_buf_.push_back(metrics.rho);
        roll_buf_.push_back(metrics.roll_deg);

        if (static_cast<int>(rho_buf_.size()) >= n_frames_) {
            baseline_.rho = compute_median(rho_buf_);
            baseline_.roll = compute_median(roll_buf_);
            baseline_.valid = true;
        }
    }

private:
    int n_frames_;
    std::vector<double> rho_buf_;
    std::vector<double> roll_buf_;
    Baseline baseline_;

    static double compute_median(std::vector<double> vals) {
        if (vals.empty()) return 0.0;
        size_t n = vals.size();
        std::sort(vals.begin(), vals.end());
        return (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) * 0.5;
    }
};

class FatigueClassifier {
public:
    explicit FatigueClassifier(int sustained = SUSTAINED_FRAMES)
        : sustained_(sustained), pitch_counter_(0), roll_counter_(0) {}

    void reset() {
        pitch_counter_ = 0;
        roll_counter_ = 0;
    }

    std::pair<std::string, std::string> classify(double pitch_deg, double roll_deg, const Baseline& baseline) {
        double delta_roll = roll_deg - baseline.roll;
        std::string pitch_level;
        std::string roll_level;

        // Pitch droop classification
        if (pitch_deg > PITCH_THRESH_MILD) {
            pitch_counter_ = 0;
            pitch_level = "NORMAL";
        } else if (pitch_deg > PITCH_THRESH_WARNING) {
            pitch_counter_++;
            pitch_level = (pitch_counter_ > sustained_) ? "MILD_DROOP" : "NORMAL";
        } else {
            pitch_counter_++;
            pitch_level = (pitch_counter_ > sustained_) ? "FATIGUE_WARNING" : "MILD_DROOP";
        }

        // Roll tilt classification
        double abs_droll = std::abs(delta_roll);
        if (abs_droll < ROLL_THRESH_MILD) {
            roll_counter_ = 0;
            roll_level = "NORMAL";
        } else if (abs_droll < ROLL_THRESH_WARNING) {
            roll_counter_++;
            roll_level = (roll_counter_ > sustained_) ? "MILD_TILT" : "NORMAL";
        } else {
            roll_counter_++;
            roll_level = (roll_counter_ > sustained_) ? "SIGNIFICANT_TILT" : "MILD_TILT";
        }

        return {pitch_level, roll_level};
    }

private:
    int sustained_;
    int pitch_counter_;
    int roll_counter_;
};

// ── UI & Overlays ───────────────────────────────────────────────────────────

inline cv::Scalar get_status_color(const std::string& level) {
    if (level == "NORMAL") return cv::Scalar(0, 255, 0);
    if (level == "MILD_DROOP" || level == "MILD_TILT") return cv::Scalar(0, 200, 255);
    if (level == "FATIGUE_WARNING" || level == "SIGNIFICANT_TILT") return cv::Scalar(0, 0, 255);
    return cv::Scalar(255, 255, 255);
}

inline void draw_landmarks(cv::Mat& frame, const std::array<cv::Point, 68>& puntos) {
    for (int idx : KEY_LANDMARKS) {
        cv::circle(frame, puntos[idx], 3, cv::Scalar(0, 255, 255), -1);
    }
}

inline void draw_face_axes(cv::Mat& frame, const PoseMetrics& metrics, const std::string& pitch_level) {
    cv::Point e_mid(static_cast<int>(metrics.e_mid.x), static_cast<int>(metrics.e_mid.y));
    double w = metrics.w_eyes * 0.5;

    // Eye line horizontal axis (cyan)
    cv::Point pt_x(static_cast<int>(metrics.e_mid.x + metrics.u_x.x * w),
                   static_cast<int>(metrics.e_mid.y + metrics.u_x.y * w));
    cv::arrowedLine(frame, e_mid, pt_x, cv::Scalar(255, 255, 0), 2, 8, 0, 0.2);

    // Midline vertical axis (color by pitch level)
    cv::Scalar color = get_status_color(pitch_level);
    cv::Point pt_y(static_cast<int>(metrics.e_mid.x + metrics.u_y.x * w * 1.5),
                   static_cast<int>(metrics.e_mid.y + metrics.u_y.y * w * 1.5));
    cv::arrowedLine(frame, e_mid, pt_y, color, 2, 8, 0, 0.2);
}

inline void draw_calibration_overlay(cv::Mat& frame, double progress) {
    int bar_x = 20, bar_y = 25, bar_w = 180, bar_h = 14;
    std::string text = "Calibrating: " + std::to_string(static_cast<int>(progress * 100)) + "%";
    cv::putText(frame, text, cv::Point(bar_x, bar_y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255, 255, 255), 1);
    cv::rectangle(frame, cv::Rect(bar_x, bar_y, bar_w, bar_h), cv::Scalar(120, 120, 120), 1);
    int fill_w = static_cast<int>(bar_w * progress);
    cv::rectangle(frame, cv::Rect(bar_x + 1, bar_y + 1, fill_w, bar_h - 1), cv::Scalar(0, 230, 0), -1);
}

inline void draw_hud(cv::Mat& frame, const PoseMetrics& metrics, const std::string& pitch_lvl, const std::string& roll_lvl) {
    int w = frame.cols;
    char pitch_buf[64];
    char roll_buf[64];
    std::snprintf(pitch_buf, sizeof(pitch_buf), "Pitch: %+5.1f deg  [%s]", metrics.pitch_deg, pitch_lvl.c_str());
    std::snprintf(roll_buf, sizeof(roll_buf), "Roll : %+5.1f deg  [%s]", metrics.roll_deg, roll_lvl.c_str());

    cv::putText(frame, pitch_buf, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.55, get_status_color(pitch_lvl), 2);
    cv::putText(frame, roll_buf, cv::Point(20, 55), cv::FONT_HERSHEY_SIMPLEX, 0.55, get_status_color(roll_lvl), 2);

    if (pitch_lvl == "FATIGUE_WARNING") {
        cv::rectangle(frame, cv::Rect(0, 0, w, 36), cv::Scalar(0, 0, 255), -1);
        cv::putText(frame, "FATIGUE DETECTED: HEAD DROOPING!", cv::Point(w / 2 - 190, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.70, cv::Scalar(255, 255, 255), 2);
    }
}

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

        // 1. Export Raw Events CSV
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

        // 2. Compute Summary Statistics
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

        // Write summary CSV
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
        std::cout << "                      C++ BENCHMARK PERFORMANCE SUMMARY (v2)\n";
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
            std::string pct_str = (r.scope_name != "total_frame_cycle")
                                      ? (std::to_string(static_cast<int>(r.pct_of_total_frame)) + "%")
                                      : "100.0%";
            std::cout << std::left << std::setw(24) << r.scope_name << " | "
                      << std::setw(5) << r.depth << " | "
                      << std::setw(13) << r.context << " | "
                      << std::setw(6) << r.count << " | "
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << r.mean_ms << " | "
                      << std::setw(8) << r.p95_ms << " | "
                      << std::setw(8) << tot_s << " | "
                      << std::setw(7) << pct_str << "\n";
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

// ── Model Path Resolution ───────────────────────────────────────────────────

inline std::string find_shape_predictor_path() {
    const std::vector<std::string> candidates = {
        "../../../../input/shape_predictor_68_face_landmarks.dat",
        "../../../input/shape_predictor_68_face_landmarks.dat",
        "../../input/shape_predictor_68_face_landmarks.dat",
        "../input/shape_predictor_68_face_landmarks.dat",
        "input/shape_predictor_68_face_landmarks.dat",
        "head_orientation/predict/input/shape_predictor_68_face_landmarks.dat"
    };

    for (const auto& path : candidates) {
        std::ifstream f(path.c_str());
        if (f.good()) return path;
    }
    return "";
}

// ── Main Benchmark Entrypoint ───────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string source = "0";
    int max_frames = 1000;
    bool no_gui = false;
    std::string output_dir = ".";

    // Parse command line arguments
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
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_head_pose [--source <0|path>] [--max-frames <1000>] [--no-gui] [--output-dir <path>]\n";
            return 0;
        }
    }

    std::string predictor_path = find_shape_predictor_path();
    if (predictor_path.empty()) {
        std::cerr << "[ERROR] Could not locate 'shape_predictor_68_face_landmarks.dat'. Please check input directory.\n";
        return 1;
    }

    // Initialize Dlib detector and shape predictor
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    try {
        dlib::deserialize(predictor_path) >> predictor;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to deserialize shape predictor: " << e.what() << "\n";
        return 1;
    }

    // Open video stream
    cv::VideoCapture cap;
    if (source.size() == 1 && std::isdigit(source[0])) {
        cap.open(std::stoi(source));
    } else {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Failed to open video source: " << source << "\n";
        return 1;
    }

    Calibrator calibrator;
    FatigueClassifier classifier;
    BenchmarkCollector bench(output_dir);

    std::cout << "[INFO] Starting C++ dedicated benchmark on source: " << source << "\n";
    std::cout << "[INFO] GUI enabled: " << (!no_gui ? "True" : "False")
              << " | Max frames: " << (max_frames > 0 ? (std::to_string(max_frames) + " (auto-quit)") : "Unlimited") << "\n";
    std::cout << "[INFO] Controls: Press 'q' in window to exit early, 'r' to recalibrate.\n";

    cv::Mat frame;
    cv::Mat gray;

    try {
        while (true) {
            if (max_frames > 0 && bench.current_frame() >= max_frames) {
                std::cout << "\n[INFO] Auto-quitting: Reached target limit of " << max_frames << " frames.\n";
                break;
            }

            {
                MeasureScope frame_scope(bench, "total_frame_cycle", 1);

                // 1. Camera / Video Read
                {
                    MeasureScope read_scope(bench, "camera_read", 1);
                    if (!cap.read(frame) || frame.empty()) {
                        std::cout << "[INFO] End of video stream reached.\n";
                        break;
                    }
                }

                // 2. Frame Preprocessing
                {
                    MeasureScope prep_scope(bench, "frame_preprocess", 1);
                    cv::flip(frame, frame, 1);
                    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                }

                // 3. Face Detection (Dlib HOG zero-copy view on gray mat)
                dlib::cv_image<unsigned char> cimg(gray);
                std::vector<dlib::rectangle> rostros;
                {
                    MeasureScope det_scope(bench, "face_detector", 1);
                    rostros = detector(cimg, 0);
                }

                // 4. Face Processing Loop (Depth 2)
                for (size_t face_id = 0; face_id < rostros.size(); ++face_id) {
                    dlib::full_object_detection shape;
                    {
                        MeasureScope pred_scope(bench, "shape_predictor", 2, static_cast<int>(face_id));
                        shape = predictor(cimg, rostros[face_id]);
                    }

                    std::array<cv::Point, 68> puntos;
                    {
                        MeasureScope pts_scope(bench, "landmarks_array_init", 2, static_cast<int>(face_id));
                        for (int i = 0; i < 68; ++i) {
                            puntos[i] = cv::Point(shape.part(i).x(), shape.part(i).y());
                        }
                    }

                    {
                        MeasureScope draw_pts_scope(bench, "draw_landmarks", 2, static_cast<int>(face_id));
                        draw_landmarks(frame, puntos);
                    }

                    if (!calibrator.is_calibrated()) {
                        PoseMetrics metrics;
                        {
                            MeasureScope est_scope(bench, "estimate_head_pose", 2, static_cast<int>(face_id), "uncalibrated");
                            metrics = estimate_head_pose(puntos, nullptr);
                        }
                        {
                            MeasureScope acc_scope(bench, "calibrator_accumulate", 2, static_cast<int>(face_id));
                            calibrator.accumulate(metrics);
                        }
                        {
                            MeasureScope ax_scope(bench, "draw_face_axes", 2, static_cast<int>(face_id), "calibrating");
                            draw_face_axes(frame, metrics, "NORMAL");
                        }
                        {
                            MeasureScope bar_scope(bench, "draw_calibration_bar", 2, static_cast<int>(face_id));
                            draw_calibration_overlay(frame, calibrator.progress());
                        }
                    } else {
                        PoseMetrics metrics;
                        {
                            MeasureScope est_scope(bench, "estimate_head_pose", 2, static_cast<int>(face_id), "calibrated");
                            metrics = estimate_head_pose(puntos, &calibrator.baseline());
                        }
                        std::pair<std::string, std::string> lvls;
                        {
                            MeasureScope clf_scope(bench, "fatigue_classify", 2, static_cast<int>(face_id));
                            lvls = classifier.classify(metrics.pitch_deg, metrics.roll_deg, calibrator.baseline());
                        }
                        {
                            MeasureScope ax_scope(bench, "draw_face_axes", 2, static_cast<int>(face_id), "tracking");
                            draw_face_axes(frame, metrics, lvls.first);
                        }
                        {
                            MeasureScope hud_scope(bench, "draw_hud", 2, static_cast<int>(face_id));
                            draw_hud(frame, metrics, lvls.first, lvls.second);
                        }
                    }
                }

                // 5. Optional GUI Rendering
                char key = -1;
                if (!no_gui) {
                    {
                        MeasureScope show_scope(bench, "gui_imshow", 1);
                        cv::imshow("C++ Head Pose Benchmark (Pitch & Roll)", frame);
                    }
                    {
                        MeasureScope wait_scope(bench, "gui_waitkey", 1);
                        key = static_cast<char>(cv::waitKey(1) & 0xFF);
                    }
                }

                bench.increment_frame();

                if (key == 'q') {
                    break;
                } else if (key == 'r') {
                    calibrator.reset();
                    classifier.reset();
                }
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

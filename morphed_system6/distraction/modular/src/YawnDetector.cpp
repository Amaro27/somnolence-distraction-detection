#include "YawnDetector.hpp"
#include "MathUtils.hpp"
#include "PathUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <opencv2/imgproc.hpp>

namespace {
bool file_exists(const std::string& path) {
    if (path.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

constexpr float INV_255 = 1.0f / 255.0f;
} // namespace

YawnDetector::YawnDetector(YawnConfig config)
    : config_(std::move(config)),
      last_time_(std::chrono::steady_clock::now()),
      last_confirmed_yawn_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

bool YawnDetector::initialize() {
    config_.model_path = PathUtils::resolve_asset_path(config_.model_path);

    if (!file_exists(config_.model_path)) {
#ifdef DEBUG_VIEWS
        std::cerr << "[!] YawnDetector Error: ONNX model not found at '" << config_.model_path << "'\n";
#endif
        return false;
    }

    try {
        net_ = cv::dnn::readNetFromONNX(config_.model_path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#ifdef DEBUG_VIEWS
        std::cout << "[+] Loaded Yawning CNN ONNX model from: " << config_.model_path << "\n";
#endif
    } catch (const cv::Exception& e) {
#ifdef DEBUG_VIEWS
        std::cerr << "[!] YawnDetector OpenCV DNN Exception: " << e.what() << "\n";
#else
        (void)e;
#endif
        return false;
    }

    // Pre-allocate NHWC input tensor (1 x H x W x 3) and reusable resize buffer
    resized_rgb_.create(config_.input_dimension.height, config_.input_dimension.width, CV_8UC3);
    const int blob_dims[4] = {1, config_.input_dimension.height, config_.input_dimension.width, 3};
    input_blob_.create(4, blob_dims, CV_32F);
    input_blob_.setTo(cv::Scalar(0.0f));

    // Pre-warm network to avoid cold-start allocation overhead on frame 0
    try {
        net_.setInput(input_blob_);
        cv::Mat warmup_out = net_.forward();
        (void)warmup_out;
#ifdef DEBUG_VIEWS
        std::cout << "[+] YawnDetector neural network pre-warmed successfully.\n";
#endif
    } catch (const cv::Exception& e) {
#ifdef DEBUG_VIEWS
        std::cerr << "[!] YawnDetector warning during pre-warm: " << e.what() << "\n";
#else
        (void)e;
#endif
    }

    reset();
    return true;
}

void YawnDetector::reset() noexcept {
    in_yawn_ = false;
    yawn_cue_emitted_ = false;
    yawn_accum_sec_ = 0.0;
    has_last_time_ = false;
    last_time_ = std::chrono::steady_clock::now();
    last_confirmed_yawn_ = std::chrono::steady_clock::now() - std::chrono::seconds(10);
}

void YawnDetector::pack_nhwc_blob(const cv::Mat& bgr_frame) {
    cv::cvtColor(bgr_frame, rgb_frame_, cv::COLOR_BGR2RGB);
    cv::resize(rgb_frame_, resized_rgb_, config_.input_dimension, 0.0, 0.0, cv::INTER_AREA);

    const int total_elements = resized_rgb_.rows * resized_rgb_.cols * 3;
    float* dst = input_blob_.ptr<float>(0);

    if (resized_rgb_.isContinuous()) {
        const uint8_t* src = resized_rgb_.ptr<uint8_t>(0);
        for (int i = 0; i < total_elements; ++i) {
            dst[i] = static_cast<float>(src[i]) * INV_255;
        }
    } else {
        int idx = 0;
        for (int y = 0; y < resized_rgb_.rows; ++y) {
            const uint8_t* row = resized_rgb_.ptr<uint8_t>(y);
            for (int x = 0; x < resized_rgb_.cols * 3; ++x, ++idx) {
                dst[idx] = static_cast<float>(row[x]) * INV_255;
            }
        }
    }
}

YawnAssessment YawnDetector::process(
    const cv::Mat& bgr_frame,
    std::chrono::steady_clock::time_point now
) {
    if (bgr_frame.empty()) {
        return YawnAssessment{0.0f, yawn_accum_sec_, in_yawn_, std::nullopt};
    }

    const double dt = MathUtils::compute_clamped_dt(now, last_time_, has_last_time_);

    // Preprocess full frame and run CNN forward pass
    pack_nhwc_blob(bgr_frame);
    net_.setInput(input_blob_);
    cv::Mat outputs = net_.forward();
    const float yawn_prob = outputs.ptr<float>(0)[0];

    std::optional<CueEvent> cue = std::nullopt;
    const double target_sec = std::chrono::duration<double>(config_.sustained_duration).count();

    if (yawn_prob >= config_.prob_threshold) {
        yawn_accum_sec_ = std::min(target_sec, yawn_accum_sec_ + dt);
        in_yawn_ = true;

        if (yawn_accum_sec_ >= target_sec) {
            auto cooldown_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_confirmed_yawn_);
            if (!yawn_cue_emitted_ && cooldown_elapsed >= config_.alarm_cooldown) {
                yawn_cue_emitted_ = true;
                last_confirmed_yawn_ = now;

                std::ostringstream ss;
                ss << "Sustained yawn >=" << std::fixed << std::setprecision(1) << target_sec
                   << "s (duration: " << std::setprecision(1) << yawn_accum_sec_
                   << "s, prob: " << std::setprecision(2) << yawn_prob << ")";

                cue = CueEvent{
                    CueType::Yawn,
                    CueStrength::Light,
                    now,
                    ss.str()
                };
            }
        }
    } else {
        if (yawn_accum_sec_ > 0.0) {
            yawn_accum_sec_ = std::max(0.0, yawn_accum_sec_ - dt);
        }
        if (yawn_accum_sec_ <= 0.0) {
            yawn_accum_sec_ = 0.0;
            in_yawn_ = false;
            yawn_cue_emitted_ = false;
        }
    }

    return YawnAssessment{
        yawn_prob,
        yawn_accum_sec_,
        in_yawn_,
        cue
    };
}

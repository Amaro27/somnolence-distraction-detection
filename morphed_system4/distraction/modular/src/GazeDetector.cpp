#include "GazeDetector.hpp"
#include "PathUtils.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace {
bool file_exists(const std::string& path) {
    if (path.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}
} // namespace

GazeDetector::GazeDetector(LSTMInferenceConfig config)
    : config_(std::move(config)) {}

bool GazeDetector::initialize() {
    config_.backbone_path = PathUtils::resolve_asset_path(config_.backbone_path);
    config_.head_path = PathUtils::resolve_asset_path(config_.head_path);

    if (!file_exists(config_.backbone_path)) {
        std::cerr << "[!] GazeDetector Error: Backbone model not found at '" << config_.backbone_path << "'\n";
        return false;
    }
    if (!file_exists(config_.head_path)) {
        std::cerr << "[!] GazeDetector Error: LSTM head model not found at '" << config_.head_path << "'\n";
        return false;
    }

    try {
        net_backbone_ = cv::dnn::readNetFromONNX(config_.backbone_path);
        net_backbone_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_backbone_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[+] Loaded Spatial ResNet-18 Backbone from: " << config_.backbone_path << "\n";

        net_head_ = cv::dnn::readNetFromONNX(config_.head_path);
        net_head_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_head_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[+] Loaded Temporal LSTM Head from: " << config_.head_path << "\n";
    } catch (const cv::Exception& e) {
        std::cerr << "[!] GazeDetector OpenCV DNN Exception: " << e.what() << "\n";
        return false;
    }

    // Allocate 3D sequence blob and flat ring buffer once (zero hot-path allocations)
    const int seq_dims[3] = {1, static_cast<int>(config_.seq_len.value), static_cast<int>(config_.feature_dim)};
    seq_blob_.create(3, seq_dims, CV_32F);
    seq_blob_.setTo(cv::Scalar(0.0f));
    ring_buffer_.assign(config_.seq_len.value * config_.feature_dim, 0.0f);

    // Pre-warm both networks with dummy forward pass to eliminate first-frame latency
    try {
        const int img_dims[4] = {1, 3, config_.input_dimension.height, config_.input_dimension.width};
        cv::Mat dummy_img(4, img_dims, CV_32F, cv::Scalar(0.0f));

        net_backbone_.setInput(dummy_img);
        cv::Mat warm_feat = net_backbone_.forward();

        net_head_.setInput(seq_blob_);
        cv::Mat warm_out = net_head_.forward();

        (void)warm_feat;
        (void)warm_out;
        std::cout << "[+] GazeDetector neural networks pre-warmed successfully.\n";
    } catch (const cv::Exception& e) {
        std::cerr << "[!] GazeDetector warning during pre-warm: " << e.what() << "\n";
    }

    reset();
    return true;
}

void GazeDetector::reset() noexcept {
    ring_head_ = 0;
    ring_count_ = 0;
}

void GazeDetector::pack_sequence_blob() noexcept {
    float* dst = seq_blob_.ptr<float>(0);
    const size_t seq_len = config_.seq_len.value;
    const size_t feature_dim = config_.feature_dim;
    const size_t buf_len = ring_count_;
    if (buf_len == 0) return;

    const size_t oldest_idx = (buf_len < seq_len) ? 0 : ring_head_;
    const size_t pad_count = seq_len - buf_len;

    // During warmup frames (buf_len < seq_len), pad leading slots with earliest feature
    const float* oldest_ptr = ring_buffer_.data() + oldest_idx * feature_dim;
    for (size_t t = 0; t < pad_count; ++t) {
        std::memcpy(dst + t * feature_dim, oldest_ptr, feature_dim * sizeof(float));
    }
    for (size_t t = 0; t < buf_len; ++t) {
        const size_t slot = (oldest_idx + t) % seq_len;
        const float* src_ptr = ring_buffer_.data() + slot * feature_dim;
        std::memcpy(dst + (pad_count + t) * feature_dim, src_ptr, feature_dim * sizeof(float));
    }
}

DistractionDetectionResult GazeDetector::infer_frame(
    const cv::Mat& input_blob,
    ConfidenceThreshold conf_threshold,
    ThresholdEvaluationPolicy policy
) {
    // 1. Stage 1: Spatial ResNet-18 Backbone
    net_backbone_.setInput(input_blob);
    cv::Mat feat_mat = net_backbone_.forward();

    // 2. Write 512-d feature directly into pre-allocated flat ring buffer
    const size_t seq_len = config_.seq_len.value;
    const size_t feature_dim = config_.feature_dim;
    float* slot_ptr = ring_buffer_.data() + ring_head_ * feature_dim;
    std::memcpy(slot_ptr, feat_mat.ptr<float>(0), feature_dim * sizeof(float));

    ring_head_ = (ring_head_ + 1) % seq_len;
    if (ring_count_ < seq_len) {
        ++ring_count_;
    }

    // 3. Stage 2: Temporal LSTM Head
    pack_sequence_blob();
    net_head_.setInput(seq_blob_);
    cv::Mat outputs = net_head_.forward();

    // 4. Compute Softmax Probabilities
    const float* logits = outputs.ptr<float>(0);
    float max_logit = std::max(logits[0], logits[1]);
    double exp0 = std::exp(static_cast<double>(logits[0] - max_logit)); // not_looking_road
    double exp1 = std::exp(static_cast<double>(logits[1] - max_logit)); // looking_road
    double sum_exp = exp0 + exp1;

    float p_not_looking = static_cast<float>(sum_exp > 0.0 ? (exp0 / sum_exp) : 0.5);
    float p_looking = static_cast<float>(sum_exp > 0.0 ? (exp1 / sum_exp) : 0.5);

    // 5. Evaluate GazeClass based on policy
    GazeClass assigned_class = GazeClass::LookingRoad;

    if (policy == ThresholdEvaluationPolicy::BinarizeThreshold) {
        // Interpretation A: P(looking_road) >= threshold -> LookingRoad; else NotLookingRoad
        if (p_looking >= conf_threshold.value) {
            assigned_class = GazeClass::LookingRoad;
        } else {
            assigned_class = GazeClass::NotLookingRoad;
        }
    } else {
        // Interpretation B: Argmax with confidence filtering
        if (p_looking >= p_not_looking) {
            assigned_class = GazeClass::LookingRoad;
        } else {
            assigned_class = GazeClass::NotLookingRoad;
        }
    }

    DistractionDetectionResult result;
    result.gaze_class = assigned_class;
    result.looking_road_probability = p_looking;
    result.timestamp = std::chrono::steady_clock::now();

    return result;
}

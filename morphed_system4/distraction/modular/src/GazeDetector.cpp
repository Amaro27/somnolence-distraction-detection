#include "GazeDetector.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
bool file_exists(const std::string& path) {
    if (path.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

std::string find_model_path(const std::string& path) {
    if (file_exists(path)) return path;
    // When executing from build directory (out/build/<preset>/)
    if (file_exists("../../../" + path)) return "../../../" + path;
    // When executing from workspace root
    if (file_exists("morphed_system4/distraction/modular/" + path)) {
        return "morphed_system4/distraction/modular/" + path;
    }
    return path;
}
} // namespace

GazeDetector::GazeDetector(LSTMInferenceConfig config)
    : config_(std::move(config)) {}

bool GazeDetector::initialize() {
    config_.backbone_path = find_model_path(config_.backbone_path);
    config_.head_path = find_model_path(config_.head_path);

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

    // Allocate 3D sequence blob for LSTM head: (1 x seq_len x feature_dim)
    const int seq_dims[3] = {1, static_cast<int>(config_.seq_len.value), static_cast<int>(config_.feature_dim)};
    seq_blob_.create(3, seq_dims, CV_32F);
    seq_blob_.setTo(cv::Scalar(0.0f));

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

void GazeDetector::reset() {
    feature_buffer_.clear();
}

void GazeDetector::pack_sequence_blob() {
    float* dst = seq_blob_.ptr<float>(0);
    const int seq_len = static_cast<int>(config_.seq_len.value);
    const int feature_dim = static_cast<int>(config_.feature_dim);
    const int buf_len = static_cast<int>(feature_buffer_.size());
    const int pad_count = seq_len - buf_len;

    // During warmup frames (buf_len < seq_len), pad leading slots with earliest feature
    for (int t = 0; t < pad_count; ++t) {
        std::memcpy(dst + t * feature_dim, feature_buffer_.front().data(), feature_dim * sizeof(float));
    }
    for (int t = 0; t < buf_len; ++t) {
        std::memcpy(dst + (pad_count + t) * feature_dim, feature_buffer_[t].data(), feature_dim * sizeof(float));
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

    // 2. Extract 512-d feature vector and maintain circular buffer
    std::vector<float> feat_vec(config_.feature_dim);
    std::memcpy(feat_vec.data(), feat_mat.ptr<float>(0), config_.feature_dim * sizeof(float));

    if (feature_buffer_.size() >= config_.seq_len.value) {
        feature_buffer_.pop_front();
    }
    feature_buffer_.push_back(std::move(feat_vec));

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

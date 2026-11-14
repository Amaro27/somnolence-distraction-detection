#pragma once

#include "Types.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>

/**
 * @brief Two-stage ONNX inference engine for spatial-temporal driver gaze classification.
 *
 * Stage 1: Spatial ResNet-18 Backbone ONNX (extracts 512-d feature vector per frame).
 * Stage 2: Temporal LSTM Head ONNX (evaluates sequence window of 512-d embeddings).
 */
class GazeDetector {
public:
    /**
     * @brief Constructs GazeDetector with specified inference configuration.
     * @param config Neural network parameters, sequence lengths, and ONNX file paths.
     */
    explicit GazeDetector(LSTMInferenceConfig config = LSTMInferenceConfig{});

    // Non-copyable, movable
    GazeDetector(const GazeDetector&) = delete;
    GazeDetector& operator=(const GazeDetector&) = delete;
    GazeDetector(GazeDetector&&) noexcept = default;
    GazeDetector& operator=(GazeDetector&&) noexcept = default;

    /**
     * @brief Loads neural networks and performs warm-up forward pass.
     * @return true if both models loaded and warmed up successfully, false otherwise.
     */
    bool initialize();

    /**
     * @brief Runs spatial feature extraction and temporal LSTM sequence inference on preprocessed blob.
     * @param input_blob 4D normalized tensor (1 x 3 x H x W).
     * @param conf_threshold Minimum confidence threshold applied per the evaluation policy.
     * @param policy BinarizeThreshold (Interpretation A) or FilterUncertain (Interpretation B).
     * @return DistractionDetectionResult containing class, probabilities, and timestamp.
     */
    DistractionDetectionResult infer_frame(
        const cv::Mat& input_blob,
        ConfidenceThreshold conf_threshold = ConfidenceThreshold(0.70f),
        ThresholdEvaluationPolicy policy = ThresholdEvaluationPolicy::BinarizeThreshold
    );

    /**
     * @brief Resets the temporal feature circular buffer.
     */
    void reset() noexcept;

    /**
     * @brief Returns the number of accumulated frames currently in the LSTM circular buffer.
     */
    size_t buffer_size() const noexcept { return ring_count_; }

    const LSTMInferenceConfig& config() const { return config_; }

private:
    void pack_sequence_blob() noexcept;

    LSTMInferenceConfig config_;
    cv::dnn::Net net_backbone_;
    cv::dnn::Net net_head_;

    // 3D sequence blob for LSTM head: (1 x seq_len x feature_dim)
    cv::Mat seq_blob_;

    // Flat pre-allocated circular buffer (Zero heap allocations on hot path)
    std::vector<float> ring_buffer_;
    size_t ring_head_{0};
    size_t ring_count_{0};
};

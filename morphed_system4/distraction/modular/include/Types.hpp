#pragma once

#include <chrono>
#include <cstddef>
#include <string>

// ── 1. Domain Enums ──────────────────────────────────────────────────────────

/**
 * @brief Classification classes trained by the GazeResNetLSTM model.
 * 0: not_looking_road (distracted)
 * 1: looking_road (attentive)
 */
enum class GazeClass : int {
    NotLookingRoad = 0,
    LookingRoad    = 1
};

/**
 * @brief Simulates vehicle movement state (typically provided by IMU or CAN bus).
 * When Stationary, detection inference is paused.
 * When Moving, detection inference and alerts are active.
 */
enum class VehicleMovementState {
    Stationary,
    Moving
};

/**
 * @brief Overall driver alertness state evaluated by AlertManager.
 */
enum class DriverState {
    Safe,
    Distracted,
    Drowsy
};

/**
 * @brief Prioritized risk level governing auditory alert tone patterns.
 */
enum class RiskLevel {
    None,
    Low,
    Medium,
    High
};

/**
 * @brief Drowsiness cue intensity for multimodal alert evaluation.
 */
enum class CueStrength {
    Light,
    Strong
};

/**
 * @brief Types of driver fatigue and distraction cues.
 */
enum class CueType {
    OffRoadGaze,
    Perclos,
    BlinkDuration,
    HeadDroop,
    Yawn
};

/**
 * @brief Strategy for applying confidence threshold to detection window:
 * - BinarizeThreshold (Interpretation A): P(looking_road) >= threshold -> 1; otherwise -> 0.
 * - FilterUncertain (Interpretation B): Ignore detections with confidence < threshold.
 */
enum class ThresholdEvaluationPolicy {
    BinarizeThreshold,
    FilterUncertain
};

// ── 2. Explicit Strongly-Typed Wrappers ──────────────────────────────────────

/**
 * @brief Confidence threshold [0.0f, 1.0f] for gaze classification validity.
 */
struct ConfidenceThreshold {
    explicit constexpr ConfidenceThreshold(float v = 0.70f) : value(v) {}
    float value;
};

/**
 * @brief Time window span for sliding mode calculation (default: 2000 ms).
 */
struct DetectionWindowDuration {
    explicit constexpr DetectionWindowDuration(std::chrono::milliseconds dur = std::chrono::milliseconds(2000))
        : value(dur) {}
    std::chrono::milliseconds value;
};

/**
 * @brief Temporal sequence length in frames for LSTM recurrence (default: 20, matching ONNX model).
 */
struct SequenceLengthInFrames {
    explicit constexpr SequenceLengthInFrames(size_t len = 20) : value(len) {}
    size_t value;
};

/**
 * @brief 2D input dimensions (width x height) for neural network preprocessing.
 */
struct ImageDimension2D {
    explicit constexpr ImageDimension2D(int w = 224, int h = 224) : width(w), height(h) {}
    int width;
    int height;
};

// ── 3. Data Transfer Objects ─────────────────────────────────────────────────

/**
 * @brief Result of single-frame spatial-temporal inference.
 */
struct DistractionDetectionResult {
    GazeClass gaze_class{GazeClass::LookingRoad};
    float looking_road_probability{1.0f};
    std::chrono::steady_clock::time_point timestamp{std::chrono::steady_clock::now()};
};

// ── 4. Component Configurations ─────────────────────────────────────────────

/**
 * @brief Configuration parameters for GazeResNetLSTM ONNX inference.
 */
struct LSTMInferenceConfig {
    std::string backbone_path{"input/resnet18_lstm_backbone.onnx"};
    std::string head_path{"input/resnet18_lstm_head.onnx"};
    ImageDimension2D input_dimension{224, 224};
    SequenceLengthInFrames seq_len{20};
    size_t feature_dim{512};
};

/**
 * @brief Configuration for auditory alert tone pulses.
 */
struct AuditoryAlertConfig {
    double repetition_rate_hz{1.0}; // Pulses per second
    float min_volume{0.2f};
    float max_volume{1.0f};
};

/**
 * @brief Configuration for AlertManager state machine & temporal window.
 */
struct AlertConfig {
    ConfidenceThreshold min_confidence{0.70f};
    DetectionWindowDuration distraction_window{std::chrono::milliseconds(2000)};
    ThresholdEvaluationPolicy policy{ThresholdEvaluationPolicy::BinarizeThreshold};
    std::chrono::seconds light_cue_window{120}; // 2 minutes
    AuditoryAlertConfig audio_config;
};

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>
#include <opencv2/core/types.hpp>

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

/**
 * @brief Coordinated physiological calibration phase state machine.
 */
enum class CalibrationPhase {
    Pending,      // Uncalibrated / awaiting user trigger while stationary
    CalibOpen,    // Phase 1: Neutral head orientation + open eyes (~4s)
    Transition,   // User prompt to close eyes (~2s)
    CalibClosed,  // Phase 2: Closed eyes floor (~4s)
    Completed     // Calibration complete, active monitoring
};

// ── 2. Explicit Strongly-Typed Wrappers & Aliases ────────────────────────────

using FacialLandmarks68 = std::array<cv::Point2f, 68>;

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

/**
 * @brief Head orientation pitch threshold (degrees) for droop detection.
 */
struct PitchThreshold {
    explicit constexpr PitchThreshold(double d = -20.0) : degrees(d) {}
    double degrees;
};

/**
 * @brief Mouth Aspect Ratio (MAR) threshold for yawn detection.
 */
struct MarThreshold {
    explicit constexpr MarThreshold(double t = 0.55) : value(t) {}
    double value;
};

/**
 * @brief Time duration for persistence checks (e.g., sustained head droop or yawn).
 */
struct SustainedDuration {
    explicit constexpr SustainedDuration(std::chrono::milliseconds dur = std::chrono::milliseconds(1500))
        : value(dur) {}
    std::chrono::milliseconds value;
};

// ── 3. Data Transfer Objects ─────────────────────────────────────────────────

/**
 * @brief Recorded fatigue/drowsiness cue event for multimodal fusion.
 */
struct CueEvent {
    CueType type;
    CueStrength strength;
    std::chrono::steady_clock::time_point timestamp;
    std::string details;
};

/**
 * @brief Result of single-frame spatial-temporal inference.
 */
struct DistractionDetectionResult {
    GazeClass gaze_class{GazeClass::LookingRoad};
    float looking_road_probability{1.0f};
    std::chrono::steady_clock::time_point timestamp{std::chrono::steady_clock::now()};
};

/**
 * @brief Result of single-frame head orientation estimation.
 */
struct HeadOrientationAssessment {
    double pitch_deg{0.0};
    double rho{0.0};
    CalibrationPhase phase{CalibrationPhase::Pending};
    std::optional<CueEvent> cue; // Light: Droop (pitch <= -20 deg for >= 1.5s)
};

/**
 * @brief Result of single-frame Eye Aspect Ratio and blink/PERCLOS assessment.
 */
struct EarAssessment {
    double ear_raw{0.0};
    double ear_median{0.0};
    double ear_mov_avg{0.0};
    double perclos_pct{0.0};
    double last_blink_duration_ms{0.0};
    bool in_blink_cycle{false};
    CalibrationPhase phase{CalibrationPhase::Pending};
    std::vector<CueEvent> cues;
};

/**
 * @brief Result of single-frame Mouth Aspect Ratio and yawn assessment.
 */
struct MarAssessment {
    double mar_raw{0.0};
    double yawn_duration_sec{0.0};
    bool is_yawning{false};
    std::optional<CueEvent> cue; // Light: Yawn (>4.0s sustained)
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

/**
 * @brief Configuration parameters for Dlib 68-landmark detection.
 */
struct FaceLandmarkConfig {
    std::string predictor_path{"input/shape_predictor_68_face_landmarks.dat"};
    std::chrono::seconds lost_face_timeout{3}; // 3-second recovery timeout
};

/**
 * @brief Configuration parameters for head orientation pitch estimation.
 */
struct HeadOrientationConfig {
    double kp_pitch{0.18};
    PitchThreshold pitch_droop_thresh{PitchThreshold(-20.0)};
    SustainedDuration droop_sustained_duration{std::chrono::milliseconds(1500)};
};

/**
 * @brief Configuration parameters for Eye Aspect Ratio (EAR), blink, and PERCLOS.
 */
struct EarConfig {
    std::chrono::seconds phase_duration{4};       // 4s open, 4s closed
    std::chrono::seconds transition_duration{2};  // 2s pause
    double factor_descenso{0.70};
    double sensibilidad_cierre{0.20};
    double min_dynamic_range{0.05};
    size_t median_window{5};
    size_t mov_avg_window{40};
    size_t min_samples_for_update{15};
    std::chrono::milliseconds microsleep_thresh{500};
    std::chrono::seconds perclos_window{30};
    double perclos_medium_thresh{7.5};
    double perclos_high_thresh{15.0};
};

/**
 * @brief Configuration parameters for Mouth Aspect Ratio (MAR) and yawn tracking.
 */
struct MarConfig {
    MarThreshold mar_threshold{MarThreshold(0.55)};
    std::chrono::seconds sustained_duration{4};
    std::chrono::seconds alarm_cooldown{2};
};

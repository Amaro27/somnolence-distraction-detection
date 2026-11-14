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
 * @brief Unified 5-state lifecycle state machine.
 */
enum class SystemState {
    StationaryUncalibrated, // Default startup, stationary, detections disabled
    MovingUncalibrated,     // Vehicle moving without calibration; detections disabled (warns user)
    Calibrating,            // Stationary acoustic calibration actively running
    StationaryCalibrated,   // Calibrated, vehicle stationary, detections paused
    MovingCalibrated        // Calibrated, vehicle moving, full multimodal detection active
};

// ── 2. Explicit Strongly-Typed Wrappers & Aliases ────────────────────────────

using FacialLandmarks68 = std::array<cv::Point2f, 68>;

/**
 * @brief Confidence threshold [0.0f, 1.0f] for gaze classification validity.
 * Transitional conversion shim until Phase 4 migration completes.
 */
struct ConfidenceThreshold {
    float value{0.70f};
    constexpr ConfidenceThreshold(float v = 0.70f) noexcept : value(v) {}
    constexpr operator float() const noexcept { return value; }
};

/**
 * @brief Time window span for sliding mode calculation (default: 2000 ms).
 * Transitional conversion shim until Phase 4 migration completes.
 */
struct DetectionWindowDuration {
    std::chrono::milliseconds value{2000};
    constexpr DetectionWindowDuration(std::chrono::milliseconds dur = std::chrono::milliseconds(2000)) noexcept
        : value(dur) {}
    constexpr operator std::chrono::milliseconds() const noexcept { return value; }
};

/**
 * @brief Temporal sequence length in frames for LSTM recurrence (default: 20, matching ONNX model).
 * Transitional conversion shim until Phase 4 migration completes.
 */
struct SequenceLengthInFrames {
    size_t value{20};
    constexpr SequenceLengthInFrames(size_t len = 20) noexcept : value(len) {}
    constexpr operator size_t() const noexcept { return value; }
};

/**
 * @brief 2D input dimensions (width x height) for neural network preprocessing.
 * Transitional conversion shim until Phase 4 migration completes.
 */
struct ImageDimension2D {
    int width{224};
    int height{224};
    constexpr ImageDimension2D(int w = 224, int h = 224) noexcept : width(w), height(h) {}
    ImageDimension2D(cv::Size sz) noexcept : width(sz.width), height(sz.height) {}
    operator cv::Size() const noexcept { return cv::Size(width, height); }
};

/**
 * @brief Head orientation pitch threshold (degrees) for droop detection.
 * Transitional conversion shim until Phase 4 migration completes.
 */
struct PitchThreshold {
    double degrees{-20.0};
    constexpr PitchThreshold(double d = -20.0) noexcept : degrees(d) {}
    constexpr operator double() const noexcept { return degrees; }
};

/**
 * @brief Mouth Aspect Ratio (MAR) threshold for yawn detection.
 * Transitional conversion shim until Phase 4 migration completes.
 */
struct MarThreshold {
    double value{0.55};
    constexpr MarThreshold(double t = 0.55) noexcept : value(t) {}
    constexpr operator double() const noexcept { return value; }
};

/**
 * @brief Time duration for persistence checks (e.g., sustained head droop or yawn).
 * Transitional conversion shim until Phase 4 migration completes.
 */
struct SustainedDuration {
    std::chrono::milliseconds value{1500};
    constexpr SustainedDuration(std::chrono::milliseconds dur = std::chrono::milliseconds(1500)) noexcept
        : value(dur) {}
    constexpr operator std::chrono::milliseconds() const noexcept { return value; }
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
    std::string medium_risk_sound_path{"input/sounds/single-beep-350-hz.wav"};
    std::string high_risk_sound_path{"input/sounds/Ring06-LOUD.wav"};
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
 * @brief Configuration parameters for calibration acoustic guidance feedback.
 *
 * Sound file paths can be modified here or overridden programmatically.
 */
struct CalibrationAudioConfig {
    std::string phase1_instructions_path{"input/sounds/calib_phase_1.wav"};
    std::string phase1_tone_path{"input/sounds/sine-415-Hz-1-second-padded.wav"};
    std::string cadence_beep_path{"input/sounds/single-beep-350-hz.wav"};
    std::string phase2_instructions_path{"input/sounds/calib_phase_2.wav"};
    std::string success_sound_path{"input/sounds/tada-LOUD.wav"};

    std::chrono::milliseconds silence_gap{1000};          // 1.0s silence gap between voice/tone and beeps
    std::chrono::milliseconds beep_interval{500};         // 500 ms between cadence beeps
    size_t num_beeps{8};                                  // 8 beeps total (8 * 500ms = 4.0s active measurement)
    std::chrono::milliseconds transition_silence{1000};   // 1.0s silence during transition
    size_t min_samples_per_window{15};                    // Minimum valid landmark samples required per window
};

/**
 * @brief Configuration parameters for Eye Aspect Ratio (EAR), blink, and PERCLOS.
 */
struct EarConfig {
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

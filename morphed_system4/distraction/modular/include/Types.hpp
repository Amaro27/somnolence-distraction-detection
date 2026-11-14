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

// ── 2. Type Aliases ──────────────────────────────────────────────────────────

using FacialLandmarks68 = std::array<cv::Point2f, 68>;

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
    cv::Size input_dimension{224, 224};
    size_t seq_len{20};
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
    float min_confidence{0.70f};
    std::chrono::milliseconds distraction_window{2000};
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
    double pitch_droop_thresh{-20.0};
    std::chrono::milliseconds droop_sustained_duration{1500};
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
    std::string phase2_tone_path{"input/sounds/sine-415-Hz-1-second-padded.wav"};
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
    double mar_threshold{0.55};
    std::chrono::seconds sustained_duration{4};
    std::chrono::seconds alarm_cooldown{2};
};

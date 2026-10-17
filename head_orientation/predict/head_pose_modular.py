"""
Head Pose Estimation — Modular, Component-Aware Architecture.

Estimates head pitch (droop / somnolence) and roll (lateral tilt) relative to
a calibrated neutral baseline using vertical facial landmark vectors.

Architectural Highlights:
    - Domain-scoped immutable dataclasses for configurations (KISS & SRP).
    - Discrete categorical states modeled via StrEnum.
    - Strongly typed pipeline transfer objects (PoseMetrics, BaselinePose, FatigueAssessment).
    - Decoupled components with constructor dependency injection.
    - Pure visualization renderer isolated from estimation logic.

Controls:
    q — Quit
    r — Re-calibrate baseline
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Tuple

import cv2
import dlib
import numpy as np

# StrEnum support (Python 3.11+ native, fallback for 3.10)
try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Domain Enums & Landmark Indices
# ─────────────────────────────────────────────────────────────────────────────

class PitchLevel(StrEnum):
    """Categorical alertness state based on pitch droop."""
    NORMAL = "NORMAL"
    MILD_DROOP = "MILD_DROOP"
    FATIGUE_WARNING = "FATIGUE_WARNING"


class RollLevel(StrEnum):
    """Categorical head tilt state based on absolute roll angle."""
    NORMAL = "NORMAL"
    MILD_TILT = "MILD_TILT"
    SIGNIFICANT_TILT = "SIGNIFICANT_TILT"


@dataclass(frozen=True)
class LandmarkIndices:
    """dlib 68-landmark indices used for face frame and pose projection."""
    right_eye_outer: int = 36  # Outer corner of right eye (image-left)
    left_eye_outer: int = 45   # Outer corner of left eye (image-right)
    nose_tip: int = 33         # Subnasale / tip of nose
    chin: int = 8              # Gnathion / lowest point of chin

    @property
    def key_indices(self) -> Tuple[int, ...]:
        """Subset of landmarks drawn on the preview HUD."""
        return (self.right_eye_outer, self.left_eye_outer, self.nose_tip, self.chin)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Configuration Dataclasses (Scoped by Domain)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PoseEstimatorConfig:
    """Anthropometric and numerical stability parameters for pose geometry."""
    kp_pitch: float = 0.18        # Anthropometric sensitivity constant for pitch
    epsilon: float = 1e-6         # Division-by-zero guard for vector norms & height
    landmarks: LandmarkIndices = field(default_factory=LandmarkIndices)


@dataclass(frozen=True)
class CalibrationConfig:
    """Settings for neutral baseline accumulation."""
    required_frames: int = 60     # Frames needed to establish neutral baseline (~2s at 30 FPS)


@dataclass(frozen=True)
class ClassifierConfig:
    """Thresholds and temporal hysteresis limits for fatigue detection."""
    # Pitch thresholds (degrees): negative values represent drooping downward
    pitch_thresh_mild: float = -8.0
    pitch_thresh_warning: float = -15.0

    # Roll thresholds (degrees): absolute lateral tilt deviation from baseline
    roll_thresh_mild: float = 8.0
    roll_thresh_warning: float = 18.0

    # Temporal persistence filter
    sustained_frames: int = 20    # Frames a condition must persist before escalating


@dataclass(frozen=True)
class VisualizerConfig:
    """Visual style, color mappings, and HUD layout settings."""
    level_colors: dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        PitchLevel.NORMAL: (0, 255, 0),              # Green
        PitchLevel.MILD_DROOP: (0, 200, 255),        # Yellow / Orange
        PitchLevel.FATIGUE_WARNING: (0, 0, 255),     # Red
        RollLevel.NORMAL: (0, 255, 0),               # Green
        RollLevel.MILD_TILT: (0, 200, 255),          # Yellow / Orange
        RollLevel.SIGNIFICANT_TILT: (0, 0, 255),     # Red
    })
    landmark_color: Tuple[int, int, int] = (0, 255, 255)  # Cyan
    axis_x_color: Tuple[int, int, int] = (255, 255, 0)    # Cyan / Yellow
    progress_bar_color: Tuple[int, int, int] = (0, 230, 0)
    warning_banner_bg: Tuple[int, int, int] = (0, 0, 255)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pipeline Data Transfer Objects (Immutable Typed Results)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BaselinePose:
    """Calibrated neutral pose baseline values."""
    rho: float
    roll: float


@dataclass(frozen=True)
class PoseMetrics:
    """Computed 2D facial frame geometry and head orientation metrics."""
    pitch_deg: float
    roll_deg: float
    delta_roll: float
    rho: float
    e_mid: np.ndarray
    u_x: np.ndarray
    u_y: np.ndarray
    w_eyes: float


@dataclass(frozen=True)
class FatigueAssessment:
    """Classification outcome for pitch droop and roll tilt."""
    pitch_level: PitchLevel
    roll_level: RollLevel
    pitch_counter: int
    roll_counter: int

    @property
    def is_warning(self) -> bool:
        return self.pitch_level == PitchLevel.FATIGUE_WARNING


# ─────────────────────────────────────────────────────────────────────────────
# 4. Decoupled Pipeline Components
# ─────────────────────────────────────────────────────────────────────────────

class FaceLandmarkDetector:
    """Encapsulates dlib frontal face detection and 68-landmark shape prediction."""

    DEFAULT_CANDIDATE_PATHS = (
        os.path.join("head_orientation", "predict", "input", "shape_predictor_68_face_landmarks.dat"),
        os.path.join("head_orientation", "input", "shape_predictor_68_face_landmarks.dat"),
        os.path.join(os.path.dirname(__file__), "input", "shape_predictor_68_face_landmarks.dat"),
        os.path.join("input", "shape_predictor_68_face_landmarks.dat"),
    )

    def __init__(self, predictor_path: Optional[str] = None):
        if predictor_path is None:
            predictor_path = self.find_predictor_file()

        if not os.path.isfile(predictor_path):
            raise FileNotFoundError(
                f"Could not locate facial landmark model at: {predictor_path}. "
                "Please verify shape_predictor_68_face_landmarks.dat exists in the input folder."
            )

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    @classmethod
    def find_predictor_file(cls, candidates: Sequence[str] = DEFAULT_CANDIDATE_PATHS) -> str:
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def detect_landmarks(self, gray_frame: np.ndarray) -> list[np.ndarray]:
        """Detects faces and extracts (68, 2) landmark matrices for each face."""
        faces = self.detector(gray_frame, 0)
        landmarks_list = []

        for face in faces:
            shape = self.predictor(gray_frame, face)
            points = np.zeros((68, 2), dtype=int)
            for i in range(68):
                points[i] = (shape.part(i).x, shape.part(i).y)
            landmarks_list.append(points)

        return landmarks_list


class HeadPoseEstimator:
    """Computes pitch and roll angles using 2D facial landmark geometry."""

    def __init__(self, config: Optional[PoseEstimatorConfig] = None):
        self.config = config or PoseEstimatorConfig()

    def build_face_frame(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Constructs 2D face coordinates (origin, unit vectors, eye distance, roll)."""
        idx = self.config.landmarks
        p_right = points[idx.right_eye_outer].astype(np.float64)
        p_left = points[idx.left_eye_outer].astype(np.float64)

        e_mid = (p_right + p_left) / 2.0
        v_eyes = p_left - p_right
        w_eyes = float(np.linalg.norm(v_eyes))

        if w_eyes < self.config.epsilon:
            u_x = np.array([1.0, 0.0])
            u_y = np.array([0.0, 1.0])
            roll_deg = 0.0
        else:
            u_x = v_eyes / w_eyes
            u_y = np.array([-u_x[1], u_x[0]])  # Downward perpendicular
            roll_deg = float(-np.degrees(np.arctan2(v_eyes[1], v_eyes[0])))

        return e_mid, u_x, u_y, w_eyes, roll_deg

    def estimate(
        self, points: np.ndarray, baseline: Optional[BaselinePose] = None
    ) -> PoseMetrics:
        """Estimates pose metrics relative to an optional calibrated neutral baseline."""
        e_mid, u_x, u_y, w_eyes, roll_deg = self.build_face_frame(points)

        idx = self.config.landmarks
        p_nose = points[idx.nose_tip].astype(np.float64)
        p_chin = points[idx.chin].astype(np.float64)

        # Project vertical segments onto midline vector u_y
        d_top = float(np.dot(p_nose - e_mid, u_y))
        d_bottom = float(np.dot(p_chin - p_nose, u_y))
        h_face = max(d_top + d_bottom, self.config.epsilon)

        rho = d_top / h_face

        if baseline is not None:
            delta_rho = rho - baseline.rho
            pitch_deg = float(np.degrees(np.arctan2(-delta_rho, self.config.kp_pitch)))
            delta_roll = roll_deg - baseline.roll
        else:
            pitch_deg = 0.0
            delta_roll = 0.0

        return PoseMetrics(
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            delta_roll=delta_roll,
            rho=rho,
            e_mid=e_mid,
            u_x=u_x,
            u_y=u_y,
            w_eyes=w_eyes,
        )


class PoseCalibrator:
    """Accumulates neutral pose measurements over N frames to determine baseline."""

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.reset()

    def reset(self) -> None:
        self._rho_buffer: list[float] = []
        self._roll_buffer: list[float] = []
        self._baseline: Optional[BaselinePose] = None

    @property
    def is_calibrated(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> Optional[BaselinePose]:
        return self._baseline

    @property
    def progress(self) -> float:
        return min(len(self._rho_buffer) / self.config.required_frames, 1.0)

    def accumulate(self, metrics: PoseMetrics) -> Optional[BaselinePose]:
        """Accumulates neutral sample and computes median baseline once full."""
        self._rho_buffer.append(metrics.rho)
        self._roll_buffer.append(metrics.roll_deg)

        if len(self._rho_buffer) >= self.config.required_frames:
            self._baseline = BaselinePose(
                rho=float(np.median(self._rho_buffer)),
                roll=float(np.median(self._roll_buffer)),
            )
            return self._baseline
        return None


class FatigueClassifier:
    """Classifies pitch droop (somnolence) and roll tilt with temporal debounce."""

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self.reset()

    def reset(self) -> None:
        self.pitch_counter: int = 0
        self.roll_counter: int = 0

    def classify(self, metrics: PoseMetrics, baseline: BaselinePose) -> FatigueAssessment:
        cfg = self.config
        delta_roll = metrics.roll_deg - baseline.roll

        # Pitch droop classification (negative pitch = looking downward)
        if metrics.pitch_deg > cfg.pitch_thresh_mild:
            self.pitch_counter = 0
            pitch_level = PitchLevel.NORMAL
        elif metrics.pitch_deg > cfg.pitch_thresh_warning:
            self.pitch_counter += 1
            pitch_level = (
                PitchLevel.MILD_DROOP
                if self.pitch_counter > cfg.sustained_frames
                else PitchLevel.NORMAL
            )
        else:
            self.pitch_counter += 1
            pitch_level = (
                PitchLevel.FATIGUE_WARNING
                if self.pitch_counter > cfg.sustained_frames
                else PitchLevel.MILD_DROOP
            )

        # Roll tilt classification
        abs_droll = abs(delta_roll)
        if abs_droll < cfg.roll_thresh_mild:
            self.roll_counter = 0
            roll_level = RollLevel.NORMAL
        elif abs_droll < cfg.roll_thresh_warning:
            self.roll_counter += 1
            roll_level = (
                RollLevel.MILD_TILT
                if self.roll_counter > cfg.sustained_frames
                else RollLevel.NORMAL
            )
        else:
            self.roll_counter += 1
            roll_level = (
                RollLevel.SIGNIFICANT_TILT
                if self.roll_counter > cfg.sustained_frames
                else RollLevel.MILD_TILT
            )

        return FatigueAssessment(
            pitch_level=pitch_level,
            roll_level=roll_level,
            pitch_counter=self.pitch_counter,
            roll_counter=self.roll_counter,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. UI and Visualization Renderer
# ─────────────────────────────────────────────────────────────────────────────

class PoseVisualizer:
    """Renders overlays, coordinate axes, HUD metrics, and warning banners."""

    def __init__(
        self,
        config: Optional[VisualizerConfig] = None,
        landmark_indices: Optional[LandmarkIndices] = None,
    ):
        self.config = config or VisualizerConfig()
        self.landmarks = landmark_indices or LandmarkIndices()

    def draw_landmarks(self, frame: np.ndarray, points: np.ndarray) -> None:
        """Draws key landmark dots on the face."""
        for idx in self.landmarks.key_indices:
            cv2.circle(frame, tuple(points[idx]), 3, self.config.landmark_color, -1)

    def draw_face_axes(
        self, frame: np.ndarray, metrics: PoseMetrics, pitch_level: PitchLevel
    ) -> None:
        """Draws perpendicular facial coordinate axes with dynamic color coding."""
        e_mid = metrics.e_mid.astype(int)
        u_x = metrics.u_x
        u_y = metrics.u_y
        w = int(metrics.w_eyes * 0.5)

        # Horizontal eye line axis (cyan/yellow)
        pt_x = (int(e_mid[0] + u_x[0] * w), int(e_mid[1] + u_x[1] * w))
        cv2.arrowedLine(frame, tuple(e_mid), pt_x, self.config.axis_x_color, 2, tipLength=0.2)

        # Vertical midline axis (color-coded by pitch severity)
        vert_color = self.config.level_colors.get(pitch_level, (0, 255, 0))
        pt_y = (int(e_mid[0] + u_y[0] * w * 1.5), int(e_mid[1] + u_y[1] * w * 1.5))
        cv2.arrowedLine(frame, tuple(e_mid), pt_y, vert_color, 2, tipLength=0.2)

    def draw_calibration_overlay(self, frame: np.ndarray, progress: float) -> None:
        """Renders calibration progress bar."""
        bar_x, bar_y, bar_w, bar_h = 20, 25, 180, 14
        cv2.putText(
            frame,
            f"Calibrating: {int(progress * 100)}%",
            (bar_x, bar_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(
            frame,
            (bar_x + 1, bar_y + 1),
            (bar_x + fill_w, bar_y + bar_h - 1),
            self.config.progress_bar_color,
            -1,
        )

    def draw_hud(
        self, frame: np.ndarray, metrics: PoseMetrics, assessment: FatigueAssessment
    ) -> None:
        """Renders live telemetry readout and alert banner when fatigue is detected."""
        h, w = frame.shape[:2]
        pitch_color = self.config.level_colors.get(assessment.pitch_level, (255, 255, 255))
        roll_color = self.config.level_colors.get(assessment.roll_level, (255, 255, 255))

        cv2.putText(
            frame,
            f"Pitch: {metrics.pitch_deg:+5.1f} deg  [{assessment.pitch_level}]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            pitch_color,
            2,
        )
        cv2.putText(
            frame,
            f"Roll : {metrics.roll_deg:+5.1f} deg  [{assessment.roll_level}]",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            roll_color,
            2,
        )

        if assessment.is_warning:
            cv2.rectangle(frame, (0, 0), (w, 36), self.config.warning_banner_bg, -1)
            cv2.putText(
                frame,
                "FATIGUE DETECTED: HEAD DROOPING!",
                (w // 2 - 190, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class HeadPosePipeline:
    """Coordinates detector, estimator, calibrator, classifier, and renderer."""

    def __init__(
        self,
        landmarker: Optional[FaceLandmarkDetector] = None,
        estimator: Optional[HeadPoseEstimator] = None,
        calibrator: Optional[PoseCalibrator] = None,
        classifier: Optional[FatigueClassifier] = None,
        visualizer: Optional[PoseVisualizer] = None,
    ):
        self.landmarker = landmarker or FaceLandmarkDetector()
        self.estimator = estimator or HeadPoseEstimator()
        self.calibrator = calibrator or PoseCalibrator()
        self.classifier = classifier or FatigueClassifier()
        self.visualizer = visualizer or PoseVisualizer()

    def reset_calibration(self) -> None:
        """Clears calibration and resets debounced fatigue counters."""
        self.calibrator.reset()
        self.classifier.reset()
        print("[INFO] Calibration reset. Please look straight ahead to recalibrate.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Executes full pipeline across all detected faces in a single video frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_landmarks = self.landmarker.detect_landmarks(gray)

        for points in face_landmarks:
            self.visualizer.draw_landmarks(frame, points)

            if not self.calibrator.is_calibrated:
                metrics = self.estimator.estimate(points, baseline=None)
                self.calibrator.accumulate(metrics)
                self.visualizer.draw_face_axes(frame, metrics, PitchLevel.NORMAL)
                self.visualizer.draw_calibration_overlay(frame, self.calibrator.progress)
            else:
                baseline = self.calibrator.baseline
                metrics = self.estimator.estimate(points, baseline=baseline)
                assessment = self.classifier.classify(metrics, baseline=baseline)
                self.visualizer.draw_face_axes(frame, metrics, assessment.pitch_level)
                self.visualizer.draw_hud(frame, metrics, assessment)

        return frame

    def run(self, camera_index: int = 0) -> None:
        """Main acquisition and processing loop."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video capture device at index {camera_index}.")

        print("[INFO] Starting modular head pose estimation.")
        print("[CONTROLS] 'q' = Quit | 'r' = Recalibrate")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                annotated = self.process_frame(frame)

                cv2.imshow("Head Pose Estimation (Modular)", annotated)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.reset_calibration()
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    pipeline = HeadPosePipeline()
    pipeline.run(camera_index=0)


if __name__ == "__main__":
    main()

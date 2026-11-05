"""
Pipeline orchestrator and runtime loop for modular EAR/MAR fatigue detection.

Coordinates camera capture, facial landmark extraction, two-phase calibration,
adaptive cascaded filtering, multi-metric fatigue classification, and HUD presentation.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple
import cv2
import dlib
import numpy as np

try:
    from .analyzers import EarMarAnalyzer
    from .calibration import TwoPhaseEarCalibrator
    from .configuration import (
        AdaptiveFilterConfig,
        AlertSeverity,
        BlinkConfig,
        CalibrationConfig,
        CalibrationPhase,
        EarMarMetrics,
        EarMarStateAssessment,
        PerclosConfig,
        VisualizerConfig,
        YawnConfig,
    )
    from .filters import CascadedAdaptiveFilter
    from .geometry import calculate_ear, calculate_mar, extract_facial_regions
    from .visualizer import EarMarVisualizer
except ImportError:
    from analyzers import EarMarAnalyzer
    from calibration import TwoPhaseEarCalibrator
    from configuration import (
        AdaptiveFilterConfig,
        AlertSeverity,
        BlinkConfig,
        CalibrationConfig,
        CalibrationPhase,
        EarMarMetrics,
        EarMarStateAssessment,
        PerclosConfig,
        VisualizerConfig,
        YawnConfig,
    )
    from filters import CascadedAdaptiveFilter
    from geometry import calculate_ear, calculate_mar, extract_facial_regions
    from visualizer import EarMarVisualizer


class EarMarPipeline:
    """End-to-end processing pipeline for physiological blink, PERCLOS, and yawn detection."""

    DEFAULT_PREDICTOR_PATHS = (
        "shape_predictor_68_face_landmarks.dat",
        os.path.join(os.path.dirname(__file__), "..", "..", "shape_predictor_68_face_landmarks.dat"),
        os.path.join("somnolence-distraction-detection-ear_mar_adaptive_filter", "somnolence-distraction-detection-ear_mar_adaptive_filter", "shape_predictor_68_face_landmarks.dat"),
        os.path.join("somnolence-distraction-detection-head-pose-estimation", "somnolence-distraction-detection-head-pose-estimation", "head_orientation", "predict", "input", "shape_predictor_68_face_landmarks.dat"),
        os.path.join("shape_predictor_68_face_landmarks", "shape_predictor_68_face_landmarks.dat"),
    )

    def __init__(
        self,
        predictor_path: Optional[str] = None,
        calib_config: Optional[CalibrationConfig] = None,
        filter_config: Optional[AdaptiveFilterConfig] = None,
        blink_config: Optional[BlinkConfig] = None,
        perclos_config: Optional[PerclosConfig] = None,
        yawn_config: Optional[YawnConfig] = None,
        visualizer_config: Optional[VisualizerConfig] = None,
    ):
        model_path = predictor_path or self.find_predictor_file()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Could not locate 68-face landmarks model at: {model_path}")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

        self.calibrator = TwoPhaseEarCalibrator(calib_config)
        self.adaptive_filter = CascadedAdaptiveFilter(filter_config, calib_config)
        self.analyzer = EarMarAnalyzer(blink_config, perclos_config, yawn_config)
        self.visualizer = EarMarVisualizer(visualizer_config)

        self.telemetry_history: List[EarMarStateAssessment] = []

    @classmethod
    def find_predictor_file(cls) -> str:
        for p in cls.DEFAULT_PREDICTOR_PATHS:
            if os.path.isfile(p):
                return p
        return cls.DEFAULT_PREDICTOR_PATHS[0]

    def reset_calibration(self) -> None:
        """Clears calibration and re-enters IDLE prompt state."""
        self.calibrator.reset()
        self.analyzer.reset()
        print("[INFO] Calibration reset. Press 'c' to start new calibration.")

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        current_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[EarMarStateAssessment]]:
        """Processes a single BGR image frame and overlays visual annotations."""
        t = current_time if current_time is not None else time.time()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        if not faces:
            # Render search banner
            cv2.putText(frame_bgr, "Searching for face...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 165, 255), 2, cv2.LINE_AA)
            return frame_bgr, None

        # Process primary detected face
        face = faces[0]
        shape = self.predictor(gray, face)
        pts = np.zeros((68, 2), dtype=int)
        for i in range(68):
            pts[i] = (shape.part(i).x, shape.part(i).y)

        face_bbox = (face.left(), face.top(), face.width(), face.height())

        # 1. Compute Geometric EAR & MAR
        eye_r, eye_l, mouth = extract_facial_regions(pts)
        ear_r = calculate_ear(eye_r)
        ear_l = calculate_ear(eye_l)
        ear_raw = (ear_r + ear_l) / 2.0
        mar_raw = calculate_mar(mouth)

        # 2. Advance Calibration State Machine
        phase = self.calibrator.phase
        if phase != CalibrationPhase.ACTIVE:
            prev_phase = phase
            phase = self.calibrator.update(ear_raw, current_time=t)

            # If calibration just transitioned to ACTIVE, initialize filters and monitoring
            if prev_phase != CalibrationPhase.ACTIVE and phase == CalibrationPhase.ACTIVE:
                thresh = self.calibrator.thresholds
                if thresh is not None:
                    self.adaptive_filter.initialize_from_calibration(thresh)
                    self.analyzer.start_monitoring(t)

            guidance = self.calibrator.get_guidance_prompt(t)
            metrics = EarMarMetrics(
                ear_left=ear_l,
                ear_right=ear_r,
                ear_raw=ear_raw,
                ear_median=ear_raw,
                ear_mov_avg=ear_raw,
                mar_raw=mar_raw,
                threshold_descent=0.25,
                threshold_closure=0.18,
            )
            assessment = self.analyzer.analyze_frame(
                phase=phase,
                metrics=metrics,
                current_time=t,
                calib_progress=self.calibrator.get_phase_progress(t),
            )
            annotated = self.visualizer.draw_hud(frame_bgr, assessment, guidance, face_bbox, pts)
            return annotated, assessment

        # 3. Active Monitoring: Cascaded Adaptive Filtering
        is_blinking = self.analyzer.blink_tracker.in_cycle
        ear_med, ear_avg, t_desc, t_close = self.adaptive_filter.filter_and_adapt(
            ear_raw=ear_raw,
            in_blink_cycle=is_blinking,
        )

        metrics = EarMarMetrics(
            ear_left=ear_l,
            ear_right=ear_r,
            ear_raw=ear_raw,
            ear_median=ear_med,
            ear_mov_avg=ear_avg,
            mar_raw=mar_raw,
            threshold_descent=t_desc,
            threshold_closure=t_close,
        )

        # 4. Multi-Metric Fatigue Evaluation
        assessment = self.analyzer.analyze_frame(
            phase=CalibrationPhase.ACTIVE,
            metrics=metrics,
            current_time=t,
        )
        self.telemetry_history.append(assessment)

        # 5. Render HUD
        annotated = self.visualizer.draw_hud(
            frame=frame_bgr,
            assessment=assessment,
            guidance_text="",
            face_bbox=face_bbox,
            landmarks=pts,
        )
        return annotated, assessment

    def run(self, camera_idx: int = 0) -> None:
        """Main webcam acquisition and presentation loop."""
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video capture device at index {camera_idx}.")

        print("=" * 65)
        print("[*] Starting Modular EAR/MAR Fatigue & Somnolence Pipeline")
        print("[*] Controls: 'c' = Start Calibration | 'r' = Reset | 'q' = Quit")
        print("=" * 65)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                annotated, _ = self.process_frame(frame)

                cv2.imshow("EAR/MAR Fatigue Detection (Modular)", annotated)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("c"):
                    self.calibrator.start_calibration()
                elif key == ord("r"):
                    self.reset_calibration()
        finally:
            cap.release()
            cv2.destroyAllWindows()

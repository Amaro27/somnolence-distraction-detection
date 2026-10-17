"""
Pipeline orchestrator and runtime loop for modular head pose estimation.

Coordinates camera capture, face detection, baseline calibration, pose estimation,
fatigue classification, and UI rendering in a unified processing loop.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from .calibration import PoseCalibrator
    from .classification import FatigueClassifier
    from .configuration import (
        CalibrationConfig,
        ClassifierConfig,
        PitchLevel,
        PoseEstimatorConfig,
        VisualizerConfig,
    )
    from .estimation import FaceLandmarkDetector, HeadPoseEstimator
    from .visualizer import PoseVisualizer
except ImportError:
    from calibration import PoseCalibrator
    from classification import FatigueClassifier
    from configuration import (
        CalibrationConfig,
        ClassifierConfig,
        PitchLevel,
        PoseEstimatorConfig,
        VisualizerConfig,
    )
    from estimation import FaceLandmarkDetector, HeadPoseEstimator
    from visualizer import PoseVisualizer


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

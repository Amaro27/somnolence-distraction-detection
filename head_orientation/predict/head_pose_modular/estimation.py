"""
Landmark extraction and geometric pose estimation components.

Translates video frames into 68 facial landmarks and projects vertical facial
segments to calculate scale-, roll-, and yaw-invariant pitch and roll metrics.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import dlib
import numpy as np

try:
    from .configuration import (
        BaselinePose,
        LandmarkIndices,
        PoseEstimatorConfig,
        PoseMetrics,
    )
except ImportError:
    from configuration import (
        BaselinePose,
        LandmarkIndices,
        PoseEstimatorConfig,
        PoseMetrics,
    )


class FaceLandmarkDetector:
    """Encapsulates dlib frontal face detection and 68-landmark shape prediction."""

    DEFAULT_CANDIDATE_PATHS = (
        os.path.join("head_orientation", "predict", "input", "shape_predictor_68_face_landmarks.dat"),
        os.path.join("head_orientation", "input", "shape_predictor_68_face_landmarks.dat"),
        os.path.join(os.path.dirname(__file__), "..", "input", "shape_predictor_68_face_landmarks.dat"),
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

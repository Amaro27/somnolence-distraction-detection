"""
Geometric aspect ratio calculations for Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).

Provides pure NumPy vector math without OpenCV, dlib, or webcam dependencies.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

try:
    from .configuration import EarMarLandmarkIndices
except ImportError:
    from configuration import EarMarLandmarkIndices


def euclidean_dist(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Calculates 2D Euclidean distance between two coordinate pairs."""
    return float(np.linalg.norm(pt1 - pt2))


def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Computes Eye Aspect Ratio (EAR) from 6 canonical eye landmarks.

    Parameters
    ----------
    eye_points : np.ndarray, shape (6, 2)
        Landmarks ordered from outer corner to inner corner:
        Index 0: Outer corner (P36 or P42)
        Index 1, 2: Upper eyelid (P37, P38 or P43, P44)
        Index 3: Inner corner (P39 or P45)
        Index 4, 5: Lower eyelid (P40, P41 or P46, P47)

    Returns
    -------
    float
        EAR ratio representing vertical aperture over horizontal eye width.
    """
    if len(eye_points) < 6:
        return 0.0

    # Vertical distances across eyelid
    dist_a = euclidean_dist(eye_points[1], eye_points[5])
    dist_b = euclidean_dist(eye_points[2], eye_points[4])

    # Horizontal distance across eye corners
    dist_c = euclidean_dist(eye_points[0], eye_points[3])

    if dist_c < 1e-6:
        return 0.0

    return (dist_a + dist_b) / (2.0 * dist_c)


def calculate_mar(mouth_points: np.ndarray) -> float:
    """
    Computes Mouth Aspect Ratio (MAR) from 20 mouth landmarks (48:68).

    Evaluates inner lip vertical separation relative to inner horizontal corner distance.

    Parameters
    ----------
    mouth_points : np.ndarray, shape (20, 2)
        Slice of points 48 to 67 in dlib 68-point model.
        Inner lips correspond to indices 12..19 (points 60..67).

    Returns
    -------
    float
        MAR ratio representing vertical mouth opening over horizontal width.
    """
    if len(mouth_points) < 20:
        return 0.0

    # Vertical distances across inner lips
    dist_a = euclidean_dist(mouth_points[13], mouth_points[19])  # P61 - P67
    dist_b = euclidean_dist(mouth_points[14], mouth_points[18])  # P62 - P66
    dist_c = euclidean_dist(mouth_points[15], mouth_points[17])  # P63 - P65

    # Horizontal distance across inner corners
    dist_d = euclidean_dist(mouth_points[12], mouth_points[16])  # P60 - P64

    if dist_d < 1e-6:
        return 0.0

    return (dist_a + dist_b + dist_c) / (2.0 * dist_d)


def extract_facial_regions(
    landmarks_68: np.ndarray,
    indices: EarMarLandmarkIndices = EarMarLandmarkIndices(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts right eye, left eye, and mouth coordinate arrays from 68 landmarks."""
    eye_right = landmarks_68[indices.right_eye_start:indices.right_eye_end]
    eye_left = landmarks_68[indices.left_eye_start:indices.left_eye_end]
    mouth = landmarks_68[indices.mouth_start:indices.mouth_end]
    return eye_right, eye_left, mouth

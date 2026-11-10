"""
Modular Head Pose Estimation Package.

Provides component-aware modules for 2D facial landmark-based pitch and roll estimation,
neutral baseline calibration, temporal fatigue classification, and visualization.
"""

from .calibration import PoseCalibrator
from .classification import FatigueClassifier
from .configuration import (
    BaselinePose,
    CalibrationConfig,
    ClassifierConfig,
    FatigueAssessment,
    LandmarkIndices,
    PitchLevel,
    PoseEstimatorConfig,
    PoseMetrics,
    RollLevel,
    VisualizerConfig,
)
from .estimation import FaceLandmarkDetector, HeadPoseEstimator
from .pipeline import HeadPosePipeline, main
from .visualizer import PoseVisualizer

__all__ = [
    # Enums & Indices
    "PitchLevel",
    "RollLevel",
    "LandmarkIndices",
    # Configs
    "PoseEstimatorConfig",
    "CalibrationConfig",
    "ClassifierConfig",
    "VisualizerConfig",
    # DTOs
    "BaselinePose",
    "PoseMetrics",
    "FatigueAssessment",
    # Components
    "FaceLandmarkDetector",
    "HeadPoseEstimator",
    "PoseCalibrator",
    "FatigueClassifier",
    "PoseVisualizer",
    "HeadPosePipeline",
    "main",
]

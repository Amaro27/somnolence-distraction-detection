"""
Morphed Unified ADAS Monitoring Package.

Combines:
- Task 1: Head Pose Estimation & Pitch Droop
- Task 2: Driver Gaze Distraction (LightGazeCNN)
- Task 3: Facial Dynamics via EAR & MAR (Micro-sleep, 30s PERCLOS, Yawning)
- Task 4: Multimodal Alert Fusion Engine
"""

from .calibration import UnifiedSystemCalibrator
from .configuration import (
    BackendType,
    BaselineValues,
    CalibrationPhase,
    DistractionConfig,
    DistractionOutput,
    DriverRiskLevel,
    EarMarConfig,
    EarMarOutput,
    GazeStatus,
    HeadPoseConfig,
    PitchLevel,
    PoseOutput,
    UnifiedDriverAssessment,
    VisualizerConfig,
)
from .estimation_ear_mar import EarMarSubsystem, calc_ear, calc_mar
from .estimation_head_pose import HeadPoseEstimator
from .fusion import MultimodalAlertEngine
from .models_distraction import (
    DistractionInferenceEngine,
    GazeResNetLSTM,
    LightGazeCNN,
    load_clstm,
    load_light_cnn,
)
from .pipeline import UnifiedADASPipeline
from .realtime_benchmark import LiveSessionBenchmarkTracker
from .visualizer import UnifiedADASVisualizer

__all__ = [
    "BackendType",
    "BaselineValues",
    "CalibrationPhase",
    "DistractionConfig",
    "DistractionInferenceEngine",
    "DistractionOutput",
    "DriverRiskLevel",
    "EarMarConfig",
    "EarMarOutput",
    "EarMarSubsystem",
    "GazeResNetLSTM",
    "GazeStatus",
    "HeadPoseConfig",
    "HeadPoseEstimator",
    "LightGazeCNN",
    "LiveSessionBenchmarkTracker",
    "MultimodalAlertEngine",
    "PitchLevel",
    "PoseOutput",
    "UnifiedADASPipeline",
    "UnifiedADASVisualizer",
    "UnifiedDriverAssessment",
    "UnifiedSystemCalibrator",
    "VisualizerConfig",
    "calc_ear",
    "calc_mar",
    "load_clstm",
    "load_light_cnn",
]

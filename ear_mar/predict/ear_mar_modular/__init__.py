"""
Modular EAR/MAR Driver Fatigue & Somnolence Detection Package.

Exposes unified geometric functions, calibration state machine, cascaded
adaptive filter, blink/PERCLOS/yawn analyzers, visualizer, and pipeline.
"""

from .analyzers import (
    BlinkDurationTracker,
    EarMarAnalyzer,
    RollingPerclosTracker,
    YawnDurationTracker,
)
from .calibration import TwoPhaseEarCalibrator
from .configuration import (
    AdaptiveFilterConfig,
    AlertSeverity,
    BlinkAssessment,
    BlinkConfig,
    CalibratedThresholds,
    CalibrationConfig,
    CalibrationPhase,
    EarMarLandmarkIndices,
    EarMarMetrics,
    EarMarStateAssessment,
    PerclosAssessment,
    PerclosConfig,
    VisualizerConfig,
    YawnAssessment,
    YawnConfig,
)
from .filters import CascadedAdaptiveFilter
from .geometry import calculate_ear, calculate_mar, extract_facial_regions
from .pipeline import EarMarPipeline
from .visualizer import EarMarVisualizer

__all__ = [
    "AdaptiveFilterConfig",
    "AlertSeverity",
    "BlinkAssessment",
    "BlinkConfig",
    "BlinkDurationTracker",
    "CalibratedThresholds",
    "CalibrationConfig",
    "CalibrationPhase",
    "CascadedAdaptiveFilter",
    "EarMarAnalyzer",
    "EarMarLandmarkIndices",
    "EarMarMetrics",
    "EarMarPipeline",
    "EarMarStateAssessment",
    "EarMarVisualizer",
    "PerclosAssessment",
    "PerclosConfig",
    "RollingPerclosTracker",
    "TwoPhaseEarCalibrator",
    "VisualizerConfig",
    "YawnAssessment",
    "YawnConfig",
    "YawnDurationTracker",
    "calculate_ear",
    "calculate_mar",
    "extract_facial_regions",
]

"""
Modular Driver Distraction Detection Package.

Exposes unified configuration, models, preprocessing, temporal debouncers,
visualizer, and pipeline orchestrator for LightGazeCNN and C-LSTM.
"""

from .configuration import (
    BackendType,
    BenchmarkStats,
    CLSTMConfig,
    DistractionAssessment,
    DistractionStatus,
    GazeClass,
    LightGazeConfig,
    RawInferenceResult,
    VisualizerConfig,
)
from .models import (
    GazeResNetLSTM,
    LightGazeCNN,
    load_clstm,
    load_light_cnn,
)
from .pipeline import (
    CLSTMDetector,
    DistractionDetector,
    DistractionPipeline,
    LightCNNDetector,
)
from .preprocessing import CLAHETransform, FramePreprocessor
from .temporal import LSTMSequenceBuffer, ModeTemporalSmoother
from .visualizer import DistractionVisualizer

__all__ = [
    "BackendType",
    "BenchmarkStats",
    "CLSTMConfig",
    "CLSTMDetector",
    "CLAHETransform",
    "DistractionAssessment",
    "DistractionDetector",
    "DistractionPipeline",
    "DistractionStatus",
    "DistractionVisualizer",
    "FramePreprocessor",
    "GazeClass",
    "GazeResNetLSTM",
    "LSTMSequenceBuffer",
    "LightCNNDetector",
    "LightGazeCNN",
    "LightGazeConfig",
    "ModeTemporalSmoother",
    "RawInferenceResult",
    "VisualizerConfig",
    "load_clstm",
    "load_light_cnn",
]

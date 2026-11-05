"""
Temporal state aggregation and debouncing components for distraction detection.

Provides ModeTemporalSmoother (majority voting over sliding window for LightGazeCNN)
and LSTMSequenceBuffer (FIFO feature queue and recurrent classification for GazeResNetLSTM).
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional

import numpy as np
import torch

try:
    from .configuration import (
        DistractionAssessment,
        DistractionStatus,
        GazeClass,
        RawInferenceResult,
    )
    from .models import GazeResNetLSTM
except ImportError:
    from configuration import (
        DistractionAssessment,
        DistractionStatus,
        GazeClass,
        RawInferenceResult,
    )
    from models import GazeResNetLSTM


CLASS_NAMES = {
    GazeClass.NOT_LOOKING_ROAD: "not_looking_road",
    GazeClass.LOOKING_ROAD: "looking_road",
}


class ModeTemporalSmoother:
    """Computes statistical mode over sliding window of discrete predictions."""

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        """Clears sliding buffers."""
        self._pred_buffer: deque[int] = deque(maxlen=self.window_size)
        self._conf_buffer: deque[float] = deque(maxlen=self.window_size)

    @property
    def current_length(self) -> int:
        return len(self._pred_buffer)

    @property
    def capacity(self) -> int:
        return self.window_size

    def update(self, raw: RawInferenceResult) -> DistractionAssessment:
        """Pushes instantaneous prediction into FIFO buffer and calculates smoothed state."""
        self._pred_buffer.append(raw.pred_idx)
        self._conf_buffer.append(raw.confidence)

        # Statistical mode
        most_common_class = Counter(self._pred_buffer).most_common(1)[0][0]
        smoothed_conf = float(np.mean(self._conf_buffer))

        smoothed_label = CLASS_NAMES.get(most_common_class, "unknown")
        raw_label = CLASS_NAMES.get(raw.pred_idx, "unknown")

        is_distracted = (most_common_class == GazeClass.NOT_LOOKING_ROAD)
        status = DistractionStatus.DISTRACTED if is_distracted else DistractionStatus.SAFE

        return DistractionAssessment(
            status=status,
            smoothed_label=smoothed_label,
            raw_label=raw_label,
            confidence=smoothed_conf,
            is_distracted=is_distracted,
            buffer_length=len(self._pred_buffer),
            buffer_capacity=self.window_size,
            inference_latency_ms=raw.latency_ms,
        )


class LSTMSequenceBuffer:
    """Maintains FIFO queue of frame feature embeddings and evaluates recurrent LSTM."""

    def __init__(self, model: GazeResNetLSTM, seq_len: int = 20, device: Optional[torch.device] = None):
        self.model = model
        self.seq_len = seq_len
        self.device = device or next(model.parameters()).device
        self.reset()

    def reset(self) -> None:
        """Clears accumulated feature sequence."""
        self._feature_buffer: deque[torch.Tensor] = deque(maxlen=self.seq_len)

    @property
    def current_length(self) -> int:
        return len(self._feature_buffer)

    @property
    def capacity(self) -> int:
        return self.seq_len

    def update(self, frame_feature: torch.Tensor, feature_latency_ms: float) -> DistractionAssessment:
        """Appends 512-d feature vector to sequence queue and executes LSTM forward step."""
        self._feature_buffer.append(frame_feature)

        # Stack into temporal sequence tensor: (1, seq_len, 512)
        stacked = torch.cat(list(self._feature_buffer), dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model.forward_sequence_features(stacked)
            probs = torch.softmax(outputs, dim=1)
            conf_tensor, pred_tensor = torch.max(probs, 1)

        pred_idx = int(pred_tensor.item())
        confidence = float(conf_tensor.item() * 100.0)

        label = CLASS_NAMES.get(pred_idx, "unknown")
        is_distracted = (pred_idx == GazeClass.NOT_LOOKING_ROAD)
        status = DistractionStatus.DISTRACTED if is_distracted else DistractionStatus.SAFE

        return DistractionAssessment(
            status=status,
            smoothed_label=label,
            raw_label=label,
            confidence=confidence,
            is_distracted=is_distracted,
            buffer_length=len(self._feature_buffer),
            buffer_capacity=self.seq_len,
            inference_latency_ms=feature_latency_ms,
        )

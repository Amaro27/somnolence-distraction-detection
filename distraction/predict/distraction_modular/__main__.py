"""
Command-Line Interface for real-time driver distraction detection.

Supports both LightGazeCNN (with mode temporal smoothing) and C-LSTM
(GazeResNetLSTM with recurrent sequence modeling).
"""

from __future__ import annotations

import argparse
import sys

try:
    from .configuration import BackendType, CLSTMConfig, LightGazeConfig
    from .pipeline import DistractionPipeline
except ImportError:
    from configuration import BackendType, CLSTMConfig, LightGazeConfig
    from pipeline import DistractionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular Driver Distraction Detection (LightGazeCNN / C-LSTM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=[BackendType.LIGHT_CNN.value, BackendType.CLSTM.value],
        default=BackendType.LIGHT_CNN.value,
        help="Inference model backend architecture",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to trained model weights checkpoint (.pth). If empty, uses default config.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=15,
        help="Temporal mode smoothing window size (LightGazeCNN only)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=20,
        help="Temporal sequence length in frames (C-LSTM only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend == BackendType.LIGHT_CNN.value:
        weights = args.weights or LightGazeConfig.weights_path
        pipeline = DistractionPipeline.create_light_cnn(
            weights_path=weights,
            window_size=args.window_size,
        )
    else:
        weights = args.weights or CLSTMConfig.weights_path
        pipeline = DistractionPipeline.create_clstm(
            weights_path=weights,
            seq_len=args.seq_len,
        )

    pipeline.run(camera_idx=args.camera)


if __name__ == "__main__":
    main()

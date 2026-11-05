"""
CLI Launcher for the Morphed Unified ADAS System (All 3 Tasks in One Subfolder).
"""

from __future__ import annotations

import argparse
import sys

try:
    from .benchmark import run_benchmark
    from .configuration import (
        BackendType,
        DistractionConfig,
        EarMarConfig,
        HeadPoseConfig,
    )
    from .pipeline import UnifiedADASPipeline
except ImportError:
    from benchmark import run_benchmark
    from configuration import (
        BackendType,
        DistractionConfig,
        EarMarConfig,
        HeadPoseConfig,
    )
    from pipeline import UnifiedADASPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified ADAS System (Head Pose + Distraction + EAR/MAR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--video", type=str, default="", help="Path to pre-recorded video file (overrides --camera)")
    parser.add_argument(
        "--predictor",
        type=str,
        default="",
        help="Path to shape_predictor_68_face_landmarks.dat",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=[BackendType.LIGHT_CNN.value, BackendType.CLSTM.value],
        default=BackendType.LIGHT_CNN.value,
        help="Distraction model backend architecture (LightGazeCNN or C-LSTM)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to distraction model weights checkpoint (.pth)",
    )
    parser.add_argument(
        "--cnn_weights",
        type=str,
        default="",
        help="Deprecated alias for --weights",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=15,
        help="Sliding window size for LightGazeCNN temporal smoothing",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=20,
        help="Sequence buffer length for C-LSTM feature queue",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run static multi-stage hardware benchmark suite and generate report",
    )
    parser.add_argument(
        "--no_benchmark",
        action="store_true",
        help="Disable automatic real-time session telemetry logging and individual report creation",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="reports/sessions",
        help="Directory to save individual real-time session reports",
    )
    parser.add_argument(
        "--session_tag",
        type=str,
        default="",
        help="Optional descriptive label attached to report filename (e.g. 'driver1_night')",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Stop real-time run automatically after processing N frames (useful for benchmarking)",
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=None,
        help="Stop real-time run automatically after N seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.benchmark:
        run_benchmark(
            backend=args.backend,
            output_dir=args.report_dir if args.report_dir != "reports/sessions" else "reports",
            predictor_path=args.predictor or None,
        )
        return

    weights = args.weights or args.cnn_weights or ""
    if args.backend == BackendType.CLSTM.value:
        dist_cfg = DistractionConfig.for_clstm(
            weights_path=weights or "resnet18_lstm_unified.pth",
            seq_len=args.seq_len,
        )
    else:
        dist_cfg = DistractionConfig.for_light_cnn(
            weights_path=weights or "light_cnn_session_full_model.pth",
            window_size=args.window_size,
        )

    pipeline = UnifiedADASPipeline(
        predictor_path=args.predictor or None,
        distraction_config=dist_cfg,
    )

    cap_source = args.video if args.video else args.camera
    pipeline.run(
        camera_idx=cap_source,
        enable_benchmark=not args.no_benchmark,
        report_dir=args.report_dir,
        max_frames=args.max_frames,
        max_seconds=args.max_seconds,
        session_tag=args.session_tag or None,
    )


if __name__ == "__main__":
    main()

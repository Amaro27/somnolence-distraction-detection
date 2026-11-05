"""
Command-Line Interface for modular EAR/MAR driver fatigue and somnolence detection.
"""

from __future__ import annotations

import argparse
import sys

try:
    from .pipeline import EarMarPipeline
except ImportError:
    from pipeline import EarMarPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular EAR/MAR Fatigue & Somnolence Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument(
        "--predictor",
        type=str,
        default="",
        help="Path to shape_predictor_68_face_landmarks.dat file. If omitted, uses auto-discovery.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = EarMarPipeline(predictor_path=args.predictor or None)
    pipeline.run(camera_idx=args.camera)


if __name__ == "__main__":
    main()

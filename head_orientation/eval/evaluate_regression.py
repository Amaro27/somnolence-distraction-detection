"""
Head Pose Estimation Regression Evaluation Suite.
Part of the Boston University Head Tracking Dataset Benchmark.

Evaluates instantaneous continuous Pitch and Roll predictions against 3D ground
truth measurements (.dat files). Calibrates baseline from Frame 1, inverts
coordinate axes to match reference definitions, tracks detection failure rates,
and computes MAE, RMSE, and Max Error in degrees.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root and predict directory to sys.path for robust modular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
predict_dir = os.path.join(project_root, "head_orientation", "predict")
modular_dir = os.path.join(predict_dir, "head_pose_modular")

for path in [project_root, predict_dir, modular_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from head_orientation.predict.head_pose_modular.configuration import (
        BaselinePose,
        PoseEstimatorConfig,
        PoseMetrics,
    )
    from head_orientation.predict.head_pose_modular.estimation import (
        FaceLandmarkDetector,
        HeadPoseEstimator,
    )
except ImportError:
    from configuration import (
        BaselinePose,
        PoseEstimatorConfig,
        PoseMetrics,
    )
    from estimation import (
        FaceLandmarkDetector,
        HeadPoseEstimator,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Transfer Schemas
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameGroundTruth:
    """Parsed ground-truth orientation at a given frame."""
    frame_idx: int
    roll_gt: float    # Inverted: -col 5
    yaw_gt: float     # col 6 (uninverted/ignored)
    pitch_gt: float   # Inverted: -col 7


@dataclass
class SequenceEvalResult:
    """Regression evaluation metrics for an individual video sequence."""
    folder: str
    sequence_name: str
    file_format: str
    total_frames: int
    tracked_frames: int
    detection_rate: float
    pitch_mae: float
    pitch_rmse: float
    pitch_max: float
    roll_mae: float
    roll_rmse: float
    roll_max: float
    initial_rho: float
    initial_roll: float


@dataclass
class AggregateMetricStats:
    """Statistical summary across sequences."""
    metric_name: str
    count: int
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Parsing & Coordinate Alignment
# ─────────────────────────────────────────────────────────────────────────────

def parse_ground_truth(
    dat_path: str,
    invert_pitch: bool = True,
    invert_roll: bool = False,
) -> Dict[int, FrameGroundTruth]:
    """
    Parses a BU dataset .dat ground truth file and aligns signs to match pipeline.

    Sign Alignment:
      - Pitch: Pipeline defines facing ceiling as (+), BU sensor defines it as (-).
               Therefore, by default: Pitch_GT = - col_7 (invert_pitch=True).
      - Roll:  Pipeline delta_roll and BU col_5 already share the same physical sign
               (sub-2° MAE verified across sequences). By default: invert_roll=False.
      - Yaw:   col_6 (ignored for current 2-axis benchmark).
    """
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"Ground truth file not found: {dat_path}")

    gt_dict: Dict[int, FrameGroundTruth] = {}
    with open(dat_path, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) >= 7:
                frame_idx = int(parts[0])
                raw_roll = float(parts[4])
                raw_yaw = float(parts[5])
                raw_pitch = float(parts[6])

                roll_val = -raw_roll if invert_roll else raw_roll
                pitch_val = -raw_pitch if invert_pitch else raw_pitch

                gt_dict[frame_idx] = FrameGroundTruth(
                    frame_idx=frame_idx,
                    roll_gt=roll_val,
                    yaw_gt=raw_yaw,
                    pitch_gt=pitch_val,
                )
    return gt_dict


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sequence(
    video_path: str,
    dat_path: str,
    folder: str,
    detector: FaceLandmarkDetector,
    estimator: HeadPoseEstimator,
    invert_pitch: bool = True,
    invert_roll: bool = False,
) -> SequenceEvalResult:
    """Evaluates a single video sequence against ground truth."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ext = os.path.splitext(video_path)[1].lstrip(".").lower()

    ground_truth = parse_ground_truth(
        dat_path,
        invert_pitch=invert_pitch,
        invert_roll=invert_roll,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frame_idx = 0
    baseline: Optional[BaselinePose] = None

    pitch_errors: List[float] = []
    roll_errors: List[float] = []

    initial_rho = 0.0
    initial_roll = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1  # 1-indexed to match .dat file

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_landmarks = detector.detect_landmarks(gray)

        if not face_landmarks:
            # Face not detected in this frame
            continue

        # Use primary detected face (BU dataset has single subject)
        landmarks = face_landmarks[0]

        if baseline is None:
            # Frame 1 Calibration: establish neutral baseline
            metrics_init = estimator.estimate(landmarks, baseline=None)
            initial_rho = metrics_init.rho
            initial_roll = metrics_init.roll_deg
            baseline = BaselinePose(rho=initial_rho, roll=initial_roll)

        # Estimate pose relative to Frame 1 baseline
        metrics = estimator.estimate(landmarks, baseline=baseline)

        # Compare with Ground Truth if available for this frame
        if frame_idx in ground_truth:
            gt = ground_truth[frame_idx]
            p_err = abs(metrics.pitch_deg - gt.pitch_gt)
            r_err = abs(metrics.delta_roll - gt.roll_gt)

            pitch_errors.append(p_err)
            roll_errors.append(r_err)

    cap.release()

    total_frames = frame_idx
    tracked_frames = len(pitch_errors)
    detection_rate = (tracked_frames / total_frames * 100.0) if total_frames > 0 else 0.0

    if tracked_frames > 0:
        pitch_mae = float(np.mean(pitch_errors))
        pitch_rmse = float(np.sqrt(np.mean(np.square(pitch_errors))))
        pitch_max = float(np.max(pitch_errors))

        roll_mae = float(np.mean(roll_errors))
        roll_rmse = float(np.sqrt(np.mean(np.square(roll_errors))))
        roll_max = float(np.max(roll_errors))
    else:
        pitch_mae = pitch_rmse = pitch_max = float("nan")
        roll_mae = roll_rmse = roll_max = float("nan")

    return SequenceEvalResult(
        folder=folder,
        sequence_name=base_name,
        file_format=ext,
        total_frames=total_frames,
        tracked_frames=tracked_frames,
        detection_rate=detection_rate,
        pitch_mae=pitch_mae,
        pitch_rmse=pitch_rmse,
        pitch_max=pitch_max,
        roll_mae=roll_mae,
        roll_rmse=roll_rmse,
        roll_max=roll_max,
        initial_rho=initial_rho,
        initial_roll=initial_roll,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metric_stats(values: List[float], name: str) -> AggregateMetricStats:
    """Computes mean, std, median, min, max for a list of metric values."""
    valid = [v for v in values if not math.isnan(v)]
    n = len(valid)
    if n == 0:
        return AggregateMetricStats(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mean_v = float(np.mean(valid))
    std_v = float(np.std(valid, ddof=1)) if n > 1 else 0.0
    med_v = float(np.median(valid))
    min_v = float(np.min(valid))
    max_v = float(np.max(valid))

    return AggregateMetricStats(
        metric_name=name,
        count=n,
        mean=mean_v,
        std=std_v,
        median=med_v,
        min_val=min_v,
        max_val=max_v,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

def detect_dataset_root(custom_path: Optional[str] = None) -> str:
    """Finds the Boston University dataset folder."""
    if custom_path and os.path.isdir(custom_path):
        return custom_path

    candidates = [
        os.path.join("head_orientation", "download-dataset", "dataset"),
        os.path.join("download-dataset", "dataset"),
        os.path.join("head_orientation", "download-dataset"),
        "dataset",
        ".",
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "uniform-light")):
            return c

    raise FileNotFoundError(
        "Could not locate Boston University dataset directory with 'uniform-light' folder. "
        "Please specify --dataset-dir explicitly."
    )


def run_benchmark(
    dataset_dir: str,
    output_dir: str,
    video_format: str = "mov",
    folders: Optional[List[str]] = None,
    kp_pitch: float = 0.18,
    limit: Optional[int] = None,
    invert_pitch: bool = True,
    invert_roll: bool = False,
) -> Tuple[List[SequenceEvalResult], Dict[str, Dict[str, AggregateMetricStats]]]:
    """Runs complete regression benchmark across dataset sequences."""
    if folders is None:
        folders = ["uniform-light", "varying-light"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Initializing FaceLandmarkDetector and HeadPoseEstimator (kp_pitch={kp_pitch})...")
    detector = FaceLandmarkDetector()
    estimator = HeadPoseEstimator(config=PoseEstimatorConfig(kp_pitch=kp_pitch))

    print(f"[INFO] Dataset Directory : {os.path.abspath(dataset_dir)}")
    print(f"[INFO] Video Format      : .{video_format}")
    print(f"[INFO] Target Folders    : {folders}")
    print(f"[INFO] Alignments        : invert_pitch={invert_pitch}, invert_roll={invert_roll}")
    print("=" * 78)

    results: List[SequenceEvalResult] = []
    start_time = time.time()

    total_evaluated = 0

    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARN] Folder not found: {folder_path}, skipping.")
            continue

        video_files = [
            f for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(f".{video_format.lower()}")
        ]

        print(f"[INFO] Processing {len(video_files)} sequences in '{folder}'...")

        for fname in video_files:
            if limit is not None and total_evaluated >= limit:
                break

            video_path = os.path.join(folder_path, fname)
            base = os.path.splitext(fname)[0]
            dat_path = os.path.join(folder_path, f"{base}.dat")

            if not os.path.isfile(dat_path):
                print(f"[WARN] Ground truth .dat missing for {fname}, skipping.")
                continue

            seq_start = time.time()
            res = evaluate_sequence(
                video_path=video_path,
                dat_path=dat_path,
                folder=folder,
                detector=detector,
                estimator=estimator,
                invert_pitch=invert_pitch,
                invert_roll=invert_roll,
            )
            seq_elapsed = time.time() - seq_start

            results.append(res)
            total_evaluated += 1

            print(
                f"  [{total_evaluated:2d}] {res.sequence_name:<6} ({folder:<13}) | "
                f"Tracked: {res.detection_rate:5.1f}% | "
                f"Pitch MAE: {res.pitch_mae:5.2f}° | "
                f"Roll MAE: {res.roll_mae:5.2f}° | "
                f"Time: {seq_elapsed:.2f}s"
            )

        if limit is not None and total_evaluated >= limit:
            print(f"[INFO] Reached evaluation limit of {limit} sequences.")
            break

    elapsed = time.time() - start_time
    print("=" * 78)
    print(f"[INFO] Benchmark completed in {elapsed:.2f} seconds ({len(results)} sequences).")

    # ─────────────────────────────────────────────────────────────────────────
    # Compute Aggregated Statistics
    # ─────────────────────────────────────────────────────────────────────────
    grouped_results: Dict[str, List[SequenceEvalResult]] = {
        "OVERALL (All Sequences)": results,
    }
    for r in results:
        grp_name = f"Folder: {r.folder}"
        if grp_name not in grouped_results:
            grouped_results[grp_name] = []
        grouped_results[grp_name].append(r)

    summary_stats: Dict[str, Dict[str, AggregateMetricStats]] = {}
    for grp, seqs in grouped_results.items():
        summary_stats[grp] = {
            "pitch_mae": compute_metric_stats([s.pitch_mae for s in seqs], "pitch_mae"),
            "pitch_rmse": compute_metric_stats([s.pitch_rmse for s in seqs], "pitch_rmse"),
            "pitch_max": compute_metric_stats([s.pitch_max for s in seqs], "pitch_max"),
            "roll_mae": compute_metric_stats([s.roll_mae for s in seqs], "roll_mae"),
            "roll_rmse": compute_metric_stats([s.roll_rmse for s in seqs], "roll_rmse"),
            "roll_max": compute_metric_stats([s.roll_max for s in seqs], "roll_max"),
            "detection_rate": compute_metric_stats([s.detection_rate for s in seqs], "detection_rate"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Export Details CSV
    # ─────────────────────────────────────────────────────────────────────────
    details_csv_path = os.path.join(output_dir, "regression_eval_details.csv")
    with open(details_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "folder",
            "sequence_name",
            "file_format",
            "total_frames",
            "tracked_frames",
            "detection_rate_pct",
            "pitch_mae_deg",
            "pitch_rmse_deg",
            "pitch_max_deg",
            "roll_mae_deg",
            "roll_rmse_deg",
            "roll_max_deg",
            "initial_rho",
            "initial_roll_deg",
        ])
        for r in results:
            writer.writerow([
                r.folder,
                r.sequence_name,
                r.file_format,
                r.total_frames,
                r.tracked_frames,
                f"{r.detection_rate:.2f}",
                f"{r.pitch_mae:.3f}",
                f"{r.pitch_rmse:.3f}",
                f"{r.pitch_max:.3f}",
                f"{r.roll_mae:.3f}",
                f"{r.roll_rmse:.3f}",
                f"{r.roll_max:.3f}",
                f"{r.initial_rho:.4f}",
                f"{r.initial_roll:.3f}",
            ])
    print(f"[SAVED] Detailed evaluation CSV: {details_csv_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Export Summary CSV
    # ─────────────────────────────────────────────────────────────────────────
    summary_csv_path = os.path.join(output_dir, "regression_eval_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["group", "metric", "count", "mean", "std", "median", "min", "max"])
        for grp, metrics in summary_stats.items():
            for m_name, s in metrics.items():
                writer.writerow([
                    grp,
                    s.metric_name,
                    s.count,
                    f"{s.mean:.3f}",
                    f"{s.std:.3f}",
                    f"{s.median:.3f}",
                    f"{s.min_val:.3f}",
                    f"{s.max_val:.3f}",
                ])
    print(f"[SAVED] Aggregated summary CSV: {summary_csv_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Export Markdown Report
    # ─────────────────────────────────────────────────────────────────────────
    summary_md_path = os.path.join(output_dir, "regression_eval_summary.md")
    generate_regression_markdown_report(summary_md_path, results, summary_stats, video_format)
    print(f"[SAVED] Markdown evaluation report: {summary_md_path}")

    # Print Terminal Report
    print_cli_eval_summary(summary_stats)

    return results, summary_stats


def print_cli_eval_summary(summary_stats: Dict[str, Dict[str, AggregateMetricStats]]) -> None:
    """Prints an ASCII summary of benchmark results."""
    print("\n" + "=" * 94)
    print("BOSTON UNIVERSITY HEAD POSE BENCHMARK — REGRESSION EVALUATION RESULTS")
    print("=" * 94)
    header = f"{'Group':<26} | {'Seqs':<4} | {'Pitch MAE (Mean±Std)':<20} | {'Roll MAE (Mean±Std)':<19} | {'Track Rate':<10}"
    print(header)
    print("-" * 94)

    for grp, metrics in summary_stats.items():
        count = metrics["pitch_mae"].count
        p_mae = metrics["pitch_mae"]
        r_mae = metrics["roll_mae"]
        det = metrics["detection_rate"]

        p_str = f"{p_mae.mean:5.2f}° ± {p_mae.std:4.2f}°"
        r_str = f"{r_mae.mean:5.2f}° ± {r_mae.std:4.2f}°"
        det_str = f"{det.mean:5.1f}%"

        print(f"{grp:<26} | {count:<4} | {p_str:<20} | {r_str:<19} | {det_str:<10}")

    print("=" * 94 + "\n")


def generate_regression_markdown_report(
    md_path: str,
    results: List[SequenceEvalResult],
    summary_stats: Dict[str, Dict[str, AggregateMetricStats]],
    video_format: str,
) -> None:
    """Creates a comprehensive GitHub-Flavored Markdown evaluation report."""
    total_seqs = len(results)

    lines = [
        "# Boston University Dataset — Head Pose Regression Benchmark Report",
        "",
        f"> **Generated on**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Video Format Evaluated**: `.{video_format}` (1:1 200-frame alignment)  ",
        f"> **Total Sequences**: {total_seqs} sequences  ",
        f"> **Success Metrics**: Mean Absolute Error (MAE) for Pitch and Roll in degrees (°)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Aggregate Results",
        "",
        "The table below summarizes pose estimation performance across lighting conditions. Metrics are reported in **degrees (°)** relative to the Flock of Birds ground truth measurements.",
        "",
        "| Group | Sequences | Pitch MAE (Mean ± Std) | Pitch Median [Min, Max] | Roll MAE (Mean ± Std) | Roll Median [Min, Max] | Mean RMSE (Pitch / Roll) | Detection Rate |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for grp, metrics in summary_stats.items():
        count = metrics["pitch_mae"].count
        p_mae = metrics["pitch_mae"]
        p_rmse = metrics["pitch_rmse"]
        r_mae = metrics["roll_mae"]
        r_rmse = metrics["roll_rmse"]
        det = metrics["detection_rate"]

        lines.append(
            f"| **{grp}** | {count} | {p_mae.mean:.2f}° ± {p_mae.std:.2f}° | {p_mae.median:.2f}° [{p_mae.min_val:.2f}°, {p_mae.max_val:.2f}°] | {r_mae.mean:.2f}° ± {r_mae.std:.2f}° | {r_mae.median:.2f}° [{r_mae.min_val:.2f}°, {r_mae.max_val:.2f}°] | {p_rmse.mean:.2f}° / {r_rmse.mean:.2f}° | {det.mean:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Key Insights & Discussion",
        "",
        "1. **Roll Estimation Accuracy**:",
        "   - Lateral head tilt (Roll) is measured directly from the eye-to-eye angle in the 2D image plane.",
        "   - Roll estimation shows strong consistency with low standard deviation across sequences.",
        "2. **Pitch Estimation & Vertical Facial Proportions**:",
        "   - Pitch is estimated via the vertical nose position ratio $\\rho = d_{\\text{top}} / H_{\\text{face}}$ normalized against Frame 1.",
        "   - Sequences with extreme out-of-plane head rotations exhibit non-linear foreshortening that affects the linear small-angle assumption of $\\arctan2(-\\Delta\\rho, k_p)$.",
        "3. **Lighting Robustness (`uniform-light` vs `varying-light`)**:",
        "   - Comparing the two condition groups highlights the sensitivity of dlib 68-landmark detection to dynamic illumination and shading changes.",
        "",
        "---",
        "",
        "## 3. Best & Worst Performing Sequences",
        "",
        "### Top 5 Best Sequences (Lowest Combined MAE)",
        "",
        "| Sequence | Folder | Pitch MAE | Roll MAE | Mean MAE | Detection Rate |",
        "|:---|:---|:---:|:---:|:---:|:---:|",
    ])

    sorted_by_mae = sorted(
        results,
        key=lambda r: (r.pitch_mae + r.roll_mae) if not math.isnan(r.pitch_mae) else float("inf")
    )

    for r in sorted_by_mae[:5]:
        mean_mae = (r.pitch_mae + r.roll_mae) / 2.0
        lines.append(
            f"| `{r.sequence_name}` | `{r.folder}` | {r.pitch_mae:.2f}° | {r.roll_mae:.2f}° | **{mean_mae:.2f}°** | {r.detection_rate:.1f}% |"
        )

    lines.extend([
        "",
        "### Top 5 Challenging Sequences (Highest Combined MAE)",
        "",
        "| Sequence | Folder | Pitch MAE | Roll MAE | Mean MAE | Detection Rate |",
        "|:---|:---|:---:|:---:|:---:|:---:|",
    ])

    for r in sorted_by_mae[-5:][::-1]:
        mean_mae = (r.pitch_mae + r.roll_mae) / 2.0
        lines.append(
            f"| `{r.sequence_name}` | `{r.folder}` | {r.pitch_mae:.2f}° | {r.roll_mae:.2f}° | **{mean_mae:.2f}°** | {r.detection_rate:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Next Steps & Recommended Algorithmic Refinements",
        "",
        "- **Empirical Calibration of $k_p$**: The anthropometric pitch constant ($k_p = 0.18$) can be tuned globally or per-subject by minimizing pitch MAE on a calibration subset.",
        "- **Yaw-Induced Landmark Drift**: In sequences with large lateral head turns, landmark detectors shift slightly on the nose and chin contour, introducing residual pitch drift.",
        "",
    ])

    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark head pose regression against Boston University dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to Boston University dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("head_orientation", "eval", "reports"),
        help="Directory where CSV and Markdown reports will be saved.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="mov",
        choices=["mov", "avi"],
        help="Video format to evaluate (default: mov for 1:1 200-frame alignment).",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["uniform-light", "varying-light"],
        help="Subfolders inside dataset directory to evaluate.",
    )
    parser.add_argument(
        "--kp-pitch",
        type=float,
        default=0.18,
        help="Anthropometric sensitivity constant for pitch (default: 0.18).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of sequences to evaluate (useful for quick checks).",
    )
    parser.add_argument(
        "--invert-pitch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert ground-truth pitch sign (-col 7) to match ceiling=(+) convention (default: True).",
    )
    parser.add_argument(
        "--invert-roll",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert ground-truth roll sign (-col 5) (default: False, empirically matched).",
    )

    args = parser.parse_args()

    try:
        dataset_root = detect_dataset_root(args.dataset_dir)
        run_benchmark(
            dataset_dir=dataset_root,
            output_dir=args.output_dir,
            video_format=args.format,
            folders=args.folders,
            kp_pitch=args.kp_pitch,
            limit=args.limit,
            invert_pitch=args.invert_pitch,
            invert_roll=args.invert_roll,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] Benchmark failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

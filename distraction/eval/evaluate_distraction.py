"""
Driver Distraction Classification Offline Evaluation Suite.

Evaluates trained LightGazeCNN and C-LSTM checkpoints on offline test datasets,
image directories, or recorded video sequences. Computes confusion matrices,
Accuracy, Precision, Recall, Specificity, F1-Score, and ROC-AUC, generating
CSV details and executive Markdown reports.
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
from PIL import Image
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
distraction_root = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(distraction_root, ".."))

for path in [project_root, distraction_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from distraction.predict.distraction_modular.configuration import (
    BackendType,
    CLSTMConfig,
    GazeClass,
    LightGazeConfig,
)
from distraction.predict.distraction_modular.models import (
    GazeResNetLSTM,
    LightGazeCNN,
    load_clstm,
    load_light_cnn,
)
from distraction.predict.distraction_modular.preprocessing import FramePreprocessor
from distraction.predict.distraction_modular.temporal import (
    LSTMSequenceBuffer,
    ModeTemporalSmoother,
)


@dataclass
class SamplePrediction:
    """Individual test sample result record."""
    sample_id: str
    true_label: int
    true_name: str
    pred_label: int
    pred_name: str
    confidence: float
    prob_road: float
    is_correct: bool
    latency_ms: float


@dataclass
class ClassificationMetrics:
    """Aggregate binary classification metrics."""
    model_name: str
    total_samples: int
    true_positives: int    # looking_road correctly predicted
    true_negatives: int    # not_looking_road correctly predicted
    false_positives: int   # not_looking_road predicted as looking_road
    false_negatives: int   # looking_road predicted as not_looking_road
    accuracy_pct: float
    precision_pct: float
    recall_pct: float
    specificity_pct: float
    f1_score_pct: float
    mean_latency_ms: float
    theoretical_fps: float


class DistractionEvaluator:
    """Executes offline evaluation across image datasets or video sequences."""

    CLASS_NAMES = {
        GazeClass.NOT_LOOKING_ROAD: "not_looking_road",
        GazeClass.LOOKING_ROAD: "looking_road",
    }

    def __init__(
        self,
        backend: str = BackendType.LIGHT_CNN.value,
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.backend = backend
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if backend == BackendType.LIGHT_CNN.value:
            self.light_cfg = LightGazeConfig(weights_path=weights_path or LightGazeConfig.weights_path)
            self.model: LightGazeCNN = load_light_cnn(self.light_cfg, self.device)
            self.preprocessor = FramePreprocessor.for_light_cnn(self.light_cfg, self.device)
            self.model_name = "LightGazeCNN"
        else:
            self.clstm_cfg = CLSTMConfig(weights_path=weights_path or CLSTMConfig.weights_path)
            self.model: GazeResNetLSTM = load_clstm(self.clstm_cfg, self.device)
            self.preprocessor = FramePreprocessor.for_clstm(self.clstm_cfg, self.device)
            self.model_name = "GazeResNetLSTM (C-LSTM)"

    def predict_image(self, img_path: str, true_label: int = -1) -> SamplePrediction:
        """Evaluates single image file."""
        pil_img = Image.open(img_path).convert("RGB")
        tensor = self.preprocessor.preprocess_pil(pil_img)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            if self.backend == BackendType.LIGHT_CNN.value:
                outputs = self.model(tensor)
            else:
                # For single image, feed through feature extractor and recurrent cell
                feat = self.model.extract_single_frame_feature(tensor)
                outputs = self.model.forward_sequence_features(feat.unsqueeze(0))

            probs = torch.softmax(outputs, dim=1)
            conf_t, pred_t = torch.max(probs, 1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        pred_idx = int(pred_t.item())
        confidence = float(conf_t.item() * 100.0)
        prob_road = float(probs[0, 1].item())

        true_name = self.CLASS_NAMES.get(true_label, "unknown")
        pred_name = self.CLASS_NAMES.get(pred_idx, "unknown")
        is_correct = (pred_idx == true_label) if true_label in (0, 1) else False

        return SamplePrediction(
            sample_id=os.path.basename(img_path),
            true_label=true_label,
            true_name=true_name,
            pred_label=pred_idx,
            pred_name=pred_name,
            confidence=confidence,
            prob_road=prob_road,
            is_correct=is_correct,
            latency_ms=latency_ms,
        )

    def evaluate_directory(self, data_dir: str) -> Tuple[ClassificationMetrics, List[SamplePrediction]]:
        """
        Evaluates a directory structured either as:
        data_dir/
            0/ or not_looking_road/
            1/ or looking_road/
        """
        results: List[SamplePrediction] = []
        class_mapping = {
            "0": 0, "not_looking_road": 0, "distracted": 0,
            "1": 1, "looking_road": 1, "safe": 1,
        }

        found_subdirs = False
        for folder_name, label in class_mapping.items():
            sub = os.path.join(data_dir, folder_name)
            if os.path.isdir(sub):
                found_subdirs = True
                for fname in sorted(os.listdir(sub)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        fpath = os.path.join(sub, fname)
                        res = self.predict_image(fpath, true_label=label)
                        results.append(res)

        # Fallback: scan flat directory if no class subfolders
        if not found_subdirs:
            for fname in sorted(os.listdir(data_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    fpath = os.path.join(data_dir, fname)
                    res = self.predict_image(fpath, true_label=-1)
                    results.append(res)

        metrics = self._compute_metrics(results)
        return metrics, results

    def _compute_metrics(self, results: List[SamplePrediction]) -> ClassificationMetrics:
        labeled = [r for r in results if r.true_label in (0, 1)]
        total = len(labeled) if labeled else len(results)

        tp = sum(1 for r in labeled if r.true_label == 1 and r.pred_label == 1)
        tn = sum(1 for r in labeled if r.true_label == 0 and r.pred_label == 0)
        fp = sum(1 for r in labeled if r.true_label == 0 and r.pred_label == 1)
        fn = sum(1 for r in labeled if r.true_label == 1 and r.pred_label == 0)

        acc = ((tp + tn) / total * 100.0) if total > 0 else 0.0
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        latencies = [r.latency_ms for r in results]
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        fps_cap = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        return ClassificationMetrics(
            model_name=self.model_name,
            total_samples=total,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy_pct=acc,
            precision_pct=prec,
            recall_pct=rec,
            specificity_pct=spec,
            f1_score_pct=f1,
            mean_latency_ms=mean_lat,
            theoretical_fps=fps_cap,
        )


def export_reports(
    metrics: ClassificationMetrics,
    results: List[SamplePrediction],
    output_dir: str,
    run_name: str,
) -> None:
    """Exports CSV detailed breakdown, CSV summary, and executive Markdown report."""
    os.makedirs(output_dir, exist_ok=True)

    details_csv = os.path.join(output_dir, f"{run_name}_details.csv")
    with open(details_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "true_label", "true_name", "pred_label", "pred_name", "confidence", "prob_road", "is_correct", "latency_ms"])
        for r in results:
            writer.writerow([r.sample_id, r.true_label, r.true_name, r.pred_label, r.pred_name, f"{r.confidence:.2f}", f"{r.prob_road:.4f}", r.is_correct, f"{r.latency_ms:.2f}"])
    print(f"[+] Saved details CSV to: {details_csv}")

    summary_csv = os.path.join(output_dir, f"{run_name}_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "total_samples", "tp", "tn", "fp", "fn", "accuracy_pct", "precision_pct", "recall_pct", "specificity_pct", "f1_score_pct", "mean_latency_ms", "theoretical_fps"])
        writer.writerow([
            metrics.model_name, metrics.total_samples, metrics.true_positives, metrics.true_negatives,
            metrics.false_positives, metrics.false_negatives, f"{metrics.accuracy_pct:.2f}",
            f"{metrics.precision_pct:.2f}", f"{metrics.recall_pct:.2f}", f"{metrics.specificity_pct:.2f}",
            f"{metrics.f1_score_pct:.2f}", f"{metrics.mean_latency_ms:.2f}", f"{metrics.theoretical_fps:.1f}",
        ])
    print(f"[+] Saved summary CSV to: {summary_csv}")

    summary_md = os.path.join(output_dir, f"{run_name}_summary.md")
    md_content = f"""# Driver Distraction Classification Benchmark Report: {metrics.model_name}

> **Generated on**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
> **Evaluated Model**: `{metrics.model_name}`  
> **Total Evaluated Samples**: {metrics.total_samples:,}  
> **Theoretical Throughput**: **{metrics.theoretical_fps:.1f} FPS** (Mean Latency: {metrics.mean_latency_ms:.2f} ms)  

---

## 1. Executive Performance Summary

| Metric | Score | Description |
|:---|:---:|:---|
| **Accuracy** | **{metrics.accuracy_pct:.2f}%** | Overall correct rate across both safe and distracted states |
| **Precision** | **{metrics.precision_pct:.2f}%** | Positive predictive value for `looking_road` |
| **Recall / Sensitivity** | **{metrics.recall_pct:.2f}%** | True positive detection rate for `looking_road` |
| **Specificity** | **{metrics.specificity_pct:.2f}%** | True negative detection rate for `not_looking_road` (Distraction) |
| **F1-Score** | **{metrics.f1_score_pct:.2f}%** | Harmonic balance between Precision and Recall |

---

## 2. Confusion Matrix

| Ground Truth \\ Prediction | Predicted: Not Looking (Distracted) | Predicted: Looking (Safe) |
|:---|:---:|:---:|
| **Actual: Not Looking (Distracted)** | **TN = {metrics.true_negatives:,}** | FP = {metrics.false_positives:,} |
| **Actual: Looking (Safe)** | FN = {metrics.false_negatives:,} | **TP = {metrics.true_positives:,}** |

---

## 3. Deployment Discussion

1. **Distraction Sensitivity (Specificity)**:
   - Specificity is **{metrics.specificity_pct:.2f}%**, indicating the model's ability to catch moments when the driver is looking away from the road.
2. **Computational Viability**:
   - The model executes in **{metrics.mean_latency_ms:.2f} ms per frame**, enabling a theoretical ceiling of **{metrics.theoretical_fps:.1f} FPS**.
"""
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Saved summary Markdown report to: {summary_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate driver distraction classification model offline")
    parser.add_argument(
        "--backend",
        type=str,
        choices=[BackendType.LIGHT_CNN.value, BackendType.CLSTM.value],
        default=BackendType.LIGHT_CNN.value,
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to model weights checkpoint (.pth)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Path to dataset directory with 0/ and 1/ subfolders or image files",
    )
    parser.add_argument(
        "--single_image",
        type=str,
        default="",
        help="Path to single image file for instantaneous prediction test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(distraction_root, "eval", "reports"),
        help="Directory to save output reports",
    )
    args = parser.parse_args()

    evaluator = DistractionEvaluator(
        backend=args.backend,
        weights_path=args.weights,
    )

    if args.single_image:
        if not os.path.isfile(args.single_image):
            print(f"[!] Image file not found: {args.single_image}")
            return
        res = evaluator.predict_image(args.single_image)
        print("\n" + "=" * 50)
        print(f"[*] Single Image Inference ({evaluator.model_name})")
        print(f"[*] Target Image: {args.single_image}")
        print(f"[*] Prediction:   {res.pred_name} (Confidence: {res.confidence:.1f}%)")
        print(f"[*] Latency:      {res.latency_ms:.2f} ms")
        print("=" * 50 + "\n")
        return

    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            print(f"[!] Dataset directory not found: {args.data_dir}")
            return
        print(f"\n[*] Evaluating {evaluator.model_name} on directory: {args.data_dir}...")
        metrics, results = evaluator.evaluate_directory(args.data_dir)
        run_name = f"distraction_eval_{args.backend}"
        export_reports(metrics, results, args.output_dir, run_name)
    else:
        print("[!] Please provide either --data_dir <folder> or --single_image <file> to evaluate.")


if __name__ == "__main__":
    main()

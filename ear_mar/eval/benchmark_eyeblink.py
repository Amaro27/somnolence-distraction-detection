"""
Eyeblink8 / Eyeblink9 Benchmark Evaluation Tool.

Evaluates adaptive cascaded EAR blink detection against ground truth .tag annotations.
Calculates event-level Precision, Recall, F1, Duration MAE, and frame-level PERCLOS error.
Exports detailed CSV and executive Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import dlib
import numpy as np

# Resolve imports
current_dir = os.path.dirname(os.path.abspath(__file__))
ear_mar_root = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(ear_mar_root, ".."))

for path in [project_root, ear_mar_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from ear_mar.predict.ear_mar_modular.configuration import EarMarLandmarkIndices
from ear_mar.predict.ear_mar_modular.geometry import calculate_ear, extract_facial_regions


@dataclass
class BlinkInterval:
    start_frame: int
    end_frame: int
    frame_count: int
    duration_ms: float


@dataclass
class EyeblinkBenchmarkResult:
    video_path: str
    tag_path: str
    total_frames: int
    fps: float
    gt_blinks_count: int
    pred_blinks_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    mae_duration_ms: float
    mae_perclos_pct: float


def load_ground_truth_tag(tag_path: str, fps: float) -> Tuple[List[BlinkInterval], Dict[int, int]]:
    """Loads eyeblink .tag annotations with strict 'C' (closure) evaluation."""
    gt_blinks: List[BlinkInterval] = []
    current_blink: List[int] = []
    gt_frame_state: Dict[int, int] = {}

    with open(tag_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) < 2 or not parts[0].isdigit():
                continue

            fid = int(parts[0])
            is_closed = 1 if len(parts) >= 7 and "C" in parts[2:7] else 0
            gt_frame_state[fid] = is_closed

            if is_closed == 1:
                current_blink.append(fid)
            else:
                if current_blink:
                    dur = (len(current_blink) / fps) * 1000.0
                    gt_blinks.append(BlinkInterval(current_blink[0], current_blink[-1], len(current_blink), dur))
                    current_blink = []

        if current_blink:
            dur = (len(current_blink) / fps) * 1000.0
            gt_blinks.append(BlinkInterval(current_blink[0], current_blink[-1], len(current_blink), dur))

    return gt_blinks, gt_frame_state


def evaluate_video(
    video_path: str,
    tag_path: str,
    predictor_path: str,
    ear_base_open: float = 0.2981,
    piso_fisiologico: float = 0.1525,
    factor_descenso: float = 0.70,
    sensibilidad_cierre: float = 0.80,
    duracion_min_frames: int = 2,
    cooldown_frames: int = 2,
    iou_thresh: float = 0.20,
) -> Tuple[EyeblinkBenchmarkResult, List[BlinkInterval]]:
    """Runs frame-by-frame evaluation on video and matches predicted blinks with GT."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    gt_blinks, gt_frame_states = load_ground_truth_tag(tag_path, fps)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Initial thresholds
    rango_init = max(0.08, ear_base_open - piso_fisiologico)
    thresh_cierre = piso_fisiologico + ((1.0 - sensibilidad_cierre) * rango_init)
    thresh_descenso = piso_fisiologico + (factor_descenso * rango_init)
    thresh_cierre_static = thresh_cierre

    from collections import deque
    buf_raw: deque[float] = deque(maxlen=5)
    buf_med: deque[float] = deque(maxlen=int(3.0 * fps))

    pred_blinks: List[BlinkInterval] = []
    in_cycle = False
    cierre_efectivo = False
    f_start = -1
    f_end = -1
    last_f_end = -cooldown_frames
    last_valid_ear = ear_base_open

    buf_pred_perclos: deque[int] = deque(maxlen=int(30.0 * fps))
    buf_gt_perclos: deque[int] = deque(maxlen=int(30.0 * fps))
    log_perclos_pred: List[float] = []
    log_perclos_gt: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        if faces:
            shape = predictor(gray, faces[0])
            pts = np.zeros((68, 2), dtype=int)
            for i in range(68):
                pts[i] = (shape.part(i).x, shape.part(i).y)
            eye_r, eye_l, _ = extract_facial_regions(pts)
            ear_val = (calculate_ear(eye_r) + calculate_ear(eye_l)) / 2.0
            last_valid_ear = ear_val
        else:
            ear_val = last_valid_ear

        # Adaptive cascaded update
        if not in_cycle and ear_val > thresh_descenso:
            buf_raw.append(ear_val)
            if len(buf_raw) == buf_raw.maxlen:
                med = float(np.median(buf_raw))
                buf_med.append(med)
                if len(buf_med) >= 15:
                    ear_base_open = float(np.mean(buf_med))
                    r_dyn = max(0.08, ear_base_open - piso_fisiologico)
                    thresh_cierre = piso_fisiologico + ((1.0 - sensibilidad_cierre) * r_dyn)
                    thresh_descenso = piso_fisiologico + (factor_descenso * r_dyn)

        if thresh_cierre < thresh_cierre_static:
            thresh_cierre = thresh_cierre_static

        # Detection logic
        if ear_val < thresh_descenso:
            if not in_cycle and (frame_idx - last_f_end) > cooldown_frames:
                in_cycle = True
                cierre_efectivo = False
                f_start = -1
                f_end = -1

            if in_cycle:
                if ear_val <= thresh_cierre:
                    if not cierre_efectivo:
                        cierre_efectivo = True
                        f_start = frame_idx
                    f_end = frame_idx
        else:
            if in_cycle:
                if cierre_efectivo:
                    dur_f = f_end - f_start + 1
                    if dur_f >= duracion_min_frames:
                        dur_ms = (dur_f / fps) * 1000.0
                        pred_blinks.append(BlinkInterval(f_start, f_end, dur_f, dur_ms))
                        last_f_end = frame_idx - 1
                in_cycle = False
                cierre_efectivo = False

        # PERCLOS tracking
        is_closed_pred = 1 if ear_val <= thresh_cierre else 0
        is_closed_gt = gt_frame_states.get(frame_idx, 0)
        buf_pred_perclos.append(is_closed_pred)
        buf_gt_perclos.append(is_closed_gt)

        log_perclos_pred.append((sum(buf_pred_perclos) / len(buf_pred_perclos)) * 100.0)
        log_perclos_gt.append((sum(buf_gt_perclos) / len(buf_gt_perclos)) * 100.0)

        frame_idx += 1

    cap.release()

    # IoU event matching
    matched_gt: Set[int] = set()
    tp = 0
    fp = 0
    dur_errors: List[float] = []

    for pred in pred_blinks:
        p_set = set(range(pred.start_frame, pred.end_frame + 1))
        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(gt_blinks):
            if idx in matched_gt:
                continue
            gt_set = set(range(gt.start_frame, gt.end_frame + 1))
            inter = len(p_set.intersection(gt_set))
            union = len(p_set.union(gt_set))
            if union > 0:
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gt_idx)
            dur_errors.append(abs(pred.duration_ms - gt_blinks[best_gt_idx].duration_ms))
        else:
            fp += 1

    fn = len(gt_blinks) - len(matched_gt)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mae_dur = float(np.mean(dur_errors)) if dur_errors else 0.0
    mae_perclos = float(np.mean(np.abs(np.array(log_perclos_pred) - np.array(log_perclos_gt))))

    res = EyeblinkBenchmarkResult(
        video_path=video_path,
        tag_path=tag_path,
        total_frames=frame_idx,
        fps=fps,
        gt_blinks_count=len(gt_blinks),
        pred_blinks_count=len(pred_blinks),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        mae_duration_ms=mae_dur,
        mae_perclos_pct=mae_perclos,
    )
    return res, pred_blinks


def export_markdown_report(result: EyeblinkBenchmarkResult, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    md_content = f"""# Eyeblink Ground-Truth Evaluation Benchmark Report

> **Generated on**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
> **Target Video**: `{result.video_path}`  
> **Annotations**: `{result.tag_path}`  
> **Total Frames Evaluated**: {result.total_frames:,} ({result.fps:.1f} FPS)  

---

## 1. Executive Performance Summary

| Metric | Score | Description |
|:---|:---:|:---|
| **F1-Score** | **{result.f1_score * 100.0:.2f}%** | Harmonic balance between Precision and Recall |
| **Precision** | **{result.precision * 100.0:.2f}%** | Correctness of detected blink events (TP / (TP + FP)) |
| **Recall (Sensitivity)** | **{result.recall * 100.0:.2f}%** | Catch rate of actual blink events (TP / (TP + FN)) |
| **Duration MAE** | **{result.mae_duration_ms:.2f} ms** | Mean absolute error in deep occlusion duration |
| **PERCLOS MAE (30s)** | **{result.mae_perclos_pct:.2f}%** | Mean error in 30-second rolling cumulative closure percentage |

---

## 2. Event Detection Breakdown

| Ground Truth Blinks | Detected Blinks | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
|:---:|:---:|:---:|:---:|:---:|
| {result.gt_blinks_count} | {result.pred_blinks_count} | **{result.tp}** | {result.fp} | {result.fn} |
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Saved Eyeblink benchmark report to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark adaptive filter against Eyeblink ground truth")
    parser.add_argument("--video", type=str, required=True, help="Path to .avi video")
    parser.add_argument("--tag", type=str, required=True, help="Path to .tag annotation file")
    parser.add_argument("--predictor", type=str, default="shape_predictor_68_face_landmarks.dat", help="Path to shape predictor .dat")
    parser.add_argument("--output_report", type=str, default="", help="Path to markdown output report")
    args = parser.parse_args()

    out = args.output_report or os.path.join(ear_mar_root, "eval", "reports", "eyeblink_benchmark_summary.md")
    res, _ = evaluate_video(args.video, args.tag, args.predictor)
    export_markdown_report(res, out)


if __name__ == "__main__":
    main()

# Driver Distraction Classification Benchmark Report: GazeResNetLSTM (C-LSTM)

> **Generated on**: 2026-10-29 14:33:23  
> **Evaluated Model**: `GazeResNetLSTM (C-LSTM)`  
> **Total Evaluated Samples**: 100  
> **Theoretical Throughput**: **32.3 FPS** (Mean Latency: 31.00 ms)  

---

## 1. Executive Performance Summary

| Metric | Score | Description |
|:---|:---:|:---|
| **Accuracy** | **96.00%** | Overall correct rate across both safe and distracted states |
| **Precision** | **94.23%** | Positive predictive value for `looking_road` |
| **Recall / Sensitivity** | **98.00%** | True positive detection rate for `looking_road` |
| **Specificity** | **94.00%** | True negative detection rate for `not_looking_road` (Distraction) |
| **F1-Score** | **96.08%** | Harmonic balance between Precision and Recall |

---

## 2. Confusion Matrix

| Ground Truth \ Prediction | Predicted: Not Looking (Distracted) | Predicted: Looking (Safe) |
|:---|:---:|:---:|
| **Actual: Not Looking (Distracted)** | **TN = 47** | FP = 3 |
| **Actual: Looking (Safe)** | FN = 1 | **TP = 49** |

---

## 3. Deployment Discussion

1. **Distraction Sensitivity (Specificity)**:
   - Specificity is **94.00%**, indicating the model's ability to catch moments when the driver is looking away from the road.
2. **Computational Viability**:
   - The model executes in **31.00 ms per frame**, enabling a theoretical ceiling of **32.3 FPS**.

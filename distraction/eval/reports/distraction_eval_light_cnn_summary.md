# Driver Distraction Classification Benchmark Report: LightGazeCNN

> **Generated on**: 2026-10-29 14:55:35  
> **Evaluated Model**: `LightGazeCNN`  
> **Total Evaluated Samples**: 100  
> **Theoretical Throughput**: **209.3 FPS** (Mean Latency: 4.78 ms)  

---

## 1. Executive Performance Summary

| Metric | Score | Description |
|:---|:---:|:---|
| **Accuracy** | **93.00%** | Overall correct rate across both safe and distracted states |
| **Precision** | **89.09%** | Positive predictive value for `looking_road` |
| **Recall / Sensitivity** | **98.00%** | True positive detection rate for `looking_road` |
| **Specificity** | **88.00%** | True negative detection rate for `not_looking_road` (Distraction) |
| **F1-Score** | **93.33%** | Harmonic balance between Precision and Recall |

---

## 2. Confusion Matrix

| Ground Truth \ Prediction | Predicted: Not Looking (Distracted) | Predicted: Looking (Safe) |
|:---|:---:|:---:|
| **Actual: Not Looking (Distracted)** | **TN = 44** | FP = 6 |
| **Actual: Looking (Safe)** | FN = 1 | **TP = 49** |

---

## 3. Deployment Discussion

1. **Distraction Sensitivity (Specificity)**:
   - Specificity is **88.00%**, indicating the model's ability to catch moments when the driver is looking away from the road.
2. **Computational Viability**:
   - The model executes in **4.78 ms per frame**, enabling a theoretical ceiling of **209.3 FPS**.

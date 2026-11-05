# Hardware Latency & Throughput Benchmark Report: LightGazeCNN

> **Generated on**: 2026-10-29 12:55:38  
> **Target Device**: `cpu`  
> **Iterations**: 5 frames (Warmup: 2)  
> **Theoretical Maximum Throughput**: **102.8 FPS**  

---

## 1. Executive Summary & Latency Breakdown

| Pipeline Stage | Mean Latency (ms) | Std Dev (ms) | Median (ms) | 95th Percentile (ms) | Min / Max (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Preprocessing (Resize+Norm)** | 2.65 ms | ±0.32 | 2.50 ms | 3.06 ms | 2.32 / 3.08 ms |
| **CNN Forward Pass** | 6.89 ms | ±2.29 | 5.72 ms | 10.40 ms | 5.42 / 11.45 ms |
| **Mode Smoothing** | 0.18 ms | ±0.17 | 0.11 ms | 0.44 ms | 0.08 / 0.52 ms |
| **Total Pipeline** | **9.73 ms** | **±2.43** | **8.81 ms** | **13.46 ms** | **7.83 / 14.47 ms** |

---

## 2. Real-Time Deployment Assessment

- **Real-Time Viability**: At **102.8 FPS**, the LightGazeCNN model operates well above standard 30 FPS video stream rates.
- **Bottleneck Analysis**:
  - Preprocessing represents approximately 27.3% of execution time.
  - Neural inference represents approximately 70.8% of execution time.
  - Temporal state aggregation represents approximately 1.9% of execution time.

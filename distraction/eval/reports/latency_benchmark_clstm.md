# Hardware Latency & Throughput Benchmark Report: GazeResNetLSTM (C-LSTM)

> **Generated on**: 2026-10-29 12:55:38  
> **Target Device**: `cpu`  
> **Iterations**: 5 frames (Warmup: 2)  
> **Theoretical Maximum Throughput**: **25.3 FPS**  

---

## 1. Executive Summary & Latency Breakdown

| Pipeline Stage | Mean Latency (ms) | Std Dev (ms) | Median (ms) | 95th Percentile (ms) | Min / Max (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Preprocessing (CLAHE+Norm)** | 3.63 ms | ±0.22 | 3.65 ms | 3.91 ms | 3.27 / 3.95 ms |
| **ResNet18 Feature Extractor** | 32.83 ms | ±1.65 | 31.96 ms | 35.35 ms | 31.30 / 35.90 ms |
| **LSTM Sequence Forward** | 3.00 ms | ±0.39 | 2.72 ms | 3.51 ms | 2.65 / 3.53 ms |
| **Total Pipeline** | **39.46 ms** | **±1.75** | **38.84 ms** | **41.96 ms** | **37.67 / 42.30 ms** |

---

## 2. Real-Time Deployment Assessment

- **Real-Time Viability**: At **25.3 FPS**, the GazeResNetLSTM (C-LSTM) model operates below 30 FPS stream rates, requiring strided or threaded execution on this hardware.
- **Bottleneck Analysis**:
  - Preprocessing represents approximately 9.2% of execution time.
  - Neural inference represents approximately 83.2% of execution time.
  - Temporal state aggregation represents approximately 7.6% of execution time.

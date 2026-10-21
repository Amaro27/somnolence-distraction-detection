# Boston University Dataset — Head Pose Regression Benchmark Report

> **Generated on**: 2026-10-20 19:04:26  
> **Video Format Evaluated**: `.mov` (1:1 200-frame alignment)  
> **Total Sequences**: 72 sequences  
> **Success Metrics**: Mean Absolute Error (MAE) for Pitch and Roll in degrees (°)  

---

## 1. Executive Summary & Aggregate Results

The table below summarizes pose estimation performance across lighting conditions. Metrics are reported in **degrees (°)** relative to the Flock of Birds ground truth measurements.

| Group | Sequences | Pitch MAE (Mean ± Std) | Pitch Median [Min, Max] | Roll MAE (Mean ± Std) | Roll Median [Min, Max] | Mean RMSE (Pitch / Roll) | Detection Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OVERALL (All Sequences)** | 72 | 5.48° ± 2.85° | 4.58° [2.01°, 14.80°] | 2.04° ± 0.98° | 1.82° [0.86°, 5.53°] | 6.75° / 2.55° | 100.0% |
| **Folder: uniform-light** | 45 | 5.47° ± 2.55° | 5.30° [2.21°, 13.62°] | 2.07° ± 1.08° | 1.71° [0.86°, 5.53°] | 6.77° / 2.58° | 100.0% |
| **Folder: varying-light** | 27 | 5.50° ± 3.35° | 4.48° [2.01°, 14.80°] | 2.00° ± 0.81° | 1.88° [1.22°, 4.56°] | 6.71° / 2.50° | 99.9% |

---

## 2. Key Insights & Discussion

1. **Roll Estimation Accuracy**:
   - Lateral head tilt (Roll) is measured directly from the eye-to-eye angle in the 2D image plane.
   - Roll estimation shows strong consistency with low standard deviation across sequences.
2. **Pitch Estimation & Vertical Facial Proportions**:
   - Pitch is estimated via the vertical nose position ratio $\rho = d_{\text{top}} / H_{\text{face}}$ normalized against Frame 1.
   - Sequences with extreme out-of-plane head rotations exhibit non-linear foreshortening that affects the linear small-angle assumption of $\arctan2(-\Delta\rho, k_p)$.
3. **Lighting Robustness (`uniform-light` vs `varying-light`)**:
   - Comparing the two condition groups highlights the sensitivity of dlib 68-landmark detection to dynamic illumination and shading changes.

---

## 3. Best & Worst Performing Sequences

### Top 5 Best Sequences (Lowest Combined MAE)

| Sequence | Folder | Pitch MAE | Roll MAE | Mean MAE | Detection Rate |
|:---|:---|:---:|:---:|:---:|:---:|
| `jim1` | `uniform-light` | 2.21° | 0.98° | **1.59°** | 100.0% |
| `llm7` | `uniform-light` | 2.29° | 0.96° | **1.63°** | 100.0% |
| `llm2` | `uniform-light` | 2.33° | 1.27° | **1.80°** | 100.0% |
| `ssl1` | `varying-light` | 2.30° | 1.36° | **1.83°** | 100.0% |
| `llm1` | `uniform-light` | 2.53° | 1.32° | **1.93°** | 100.0% |

### Top 5 Challenging Sequences (Highest Combined MAE)

| Sequence | Folder | Pitch MAE | Roll MAE | Mean MAE | Detection Rate |
|:---|:---|:---:|:---:|:---:|:---:|
| `jal3` | `varying-light` | 14.80° | 3.09° | **8.95°** | 98.5% |
| `jam6` | `uniform-light` | 13.62° | 2.50° | **8.06°** | 100.0% |
| `jal4` | `varying-light` | 12.43° | 1.77° | **7.10°** | 100.0% |
| `mll3` | `varying-light` | 9.56° | 4.56° | **7.06°** | 100.0% |
| `jim8` | `uniform-light` | 11.08° | 2.13° | **6.60°** | 100.0% |

---

## 4. Next Steps & Recommended Algorithmic Refinements

- **Empirical Calibration of $k_p$**: The anthropometric pitch constant ($k_p = 0.18$) can be tuned globally or per-subject by minimizing pitch MAE on a calibration subset.
- **Yaw-Induced Landmark Drift**: In sequences with large lateral head turns, landmark detectors shift slightly on the nose and chin contour, introducing residual pitch drift.
